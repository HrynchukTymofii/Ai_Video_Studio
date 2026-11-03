# How LoRA works in this backend — full walkthrough

A LoRA here is a **small rank-32 adapter trained on the Wan2.2 5B transformer**, keyed to a made-up "trigger word". The system has two independent halves that meet at one directory on disk:

```
                     TRAINING                                 INFERENCE
  POST /api/lora-training                       POST /api/broll  |  POST /api/image
        │                                                  │
        ▼                                                  ▼
  LoraTrainingProcessor  ──installs──►  models/loras/<id>/  ──reads──►  WanPipeline.set_lora()
   (ai-toolkit subprocess)              lora.safetensors                (diffusers + peft)
                                        lora.json
```

`models/loras/` **is** the entire shared state. There is no database table, no cache — the registry re-scans the directory on every call, so a training job that finishes mid-session is immediately visible to the next generation request.

## File map

### Core — LoRA-only files

| File | Lines | Role |
|---|---|---|
| `backend/app/services/loras.py` | 134 | **Registry.** Adapters live as `models/loras/<id>/{lora.safetensors, lora.json}`. Stateless disk scan; `list/get/install/delete`, `_slugify`, `TRIGGER_RE` validation, path-traversal guard in `get()` (`loras.py:67`). |
| `backend/app/services/lora_training.py` | 275 | **Training processor.** Wan2.2 5B + ostris/ai-toolkit. `write_captions`, `build_training_config`, `parse_step_progress`, `find_final_weights`, `_run_trainer` subprocess driver. |
| `backend/app/routes/loras.py` | 51 | `GET /api/loras`, `DELETE /api/loras/{id}` (admin-only delete). |

### Wiring & job plumbing

- `backend/app/routes/generation.py` — `POST /api/lora-training` (`:502`), `_resolve_lora` (`:56`), `_prompt_with_trigger` (`:76`); `lora`/`lora_scale` form fields on the image (`:163`) and B-roll (`:214`) endpoints.
- `backend/app/queue/job.py` — `LoraTrainingParams` (`:93`); `lora_id`/`lora_path`/`lora_scale` on `BrollParams` and `ImageParams`; deserialization at `:149`, `:159`, `:184`.
- `backend/app/routes/jobs.py` — LoRA-training branches for listing/cost/retry; retry re-resolves `lora_path` (`:222-246`).
- `backend/app/main.py` — builds `LoraRegistry` (`:108`), registers `LoraTrainingProcessor` (`:85`), 6h timeout override (`:116`), mounts the router.
- `backend/app/deps.py` — `get_loras` DI provider.
- `backend/app/schemas.py` — `JobKind.LORA_TRAINING`, `LoraStyle`, `LorasResponse`, `lora` output field.
- `backend/app/config.py` — limits, `loras_dir`, `trainer_python`, `trainer_script`.
- `backend/app/models_catalog.py:82` — the `wan22-5b-lora` engine entry.
- `backend/app/services/credits.py:70` — `lora_training_cost()`.
- `backend/app/routes/status.py:55` — surfaces the trained style id in job status.

### Inference-side application

- `backend/app/pipelines/wan_pipeline.py:180-207` — `set_lora()` / `clear_lora()`.
- `backend/app/services/broll.py:44-58` and `backend/app/services/image.py:49-71` — apply around generation, clear in `finally`.

### Infra

- `backend/Dockerfile:112-131` — separate `/opt/trainer-venv` with ai-toolkit.
- `backend/requirements-gpu.txt:11` — `peft`.
- `backend/scripts/download_models.sh:55` — the Wan 5B checkpoint doubles as the training base.

### Tests

`backend/tests/test_loras.py`, `backend/tests/test_lora_training.py`, plus LoRA cases in `backend/tests/test_generation_api.py`.

---

# PART A — Training

## A1. The HTTP entry point

`backend/app/routes/generation.py:502-576`

```python
@router.post("/lora-training", response_model=JobCreatedResponse, responses=_RESPONSES)
async def create_lora_training(
    settings: Annotated[Settings, Depends(get_settings_dep)],
    worker: Annotated[GPUWorker, Depends(get_worker)],
    store: Annotated[JobStore, Depends(get_store)],
    user: Annotated[AuthUser, Depends(get_current_user)],
    name: Annotated[str, Form()],          # display name, e.g. "Zork Comic"
    trigger: Annotated[str, Form()],       # the magic token, e.g. "zork_style"
    images: Annotated[list[UploadFile], File()],   # the whole dataset, multipart
    description: Annotated[str, Form()] = "",      # optional caption suffix
    steps: Annotated[int, Form(ge=0)] = 0,         # 0 == "use the default"
    model: Annotated[str, Form()] = "wan22-5b-lora",
) -> JobCreatedResponse:
```

Everything is `Form`/`File` rather than JSON because the dataset images ride in the same multipart request — you upload the style and its name in one shot.

```python
    # 1. Is this engine id real and enabled on this GPU tier? Raises 422 if not.
    _resolve_engine(JobKind.LORA_TRAINING, model, settings)

    # 2. Charge BEFORE doing any work. A training run is 100 credits (flat) —
    #    by far the most expensive job in the system, hence the up-front check.
    cost = credits.lora_training_cost()
    require_credits(user, cost, store)
```

`credits.lora_training_cost()` (`services/credits.py:70`) is a flat `return 100` — training cost doesn't scale with steps or image count, so the quote is knowable before validation.

```python
    # 3. Validate the display name (trim, length, control chars).
    name = validate_text(name, field="name", max_chars=60)

    # 4. The trigger word is the single most important input. It gets baked
    #    verbatim into every caption AND later into user prompts, so it must
    #    survive tokenization, JSON, and shell round-trips unmangled.
    trigger = trigger.strip()
    if not TRIGGER_RE.match(trigger):     # ^[A-Za-z0-9_]{2,30}$
        raise InputValidationError(
            "trigger must be a single made-up word: 2-30 letters, digits or "
            "underscores (e.g. 'zork_style')"
        )
```

**Why "made-up"**: the whole trick of a style LoRA is teaching the model that one token means the concept. If the trigger is a real word like `cat`, training fights the base model's existing meaning for it and pollutes generations that merely mention cats.

```python
    # 5. Optional free-text hint appended to every caption.
    description = description.strip()
    if description:
        description = validate_text(
            description, field="description", max_chars=settings.max_prompt_chars
        )

    # 6. Steps: 0 means "unset" -> 2000 default; hard ceiling 4000 so a typo
    #    can't occupy the single GPU for a day.
    steps = steps or settings.lora_default_steps      # 2000
    if steps > settings.lora_max_steps:               # 4000
        raise InputValidationError(f"steps must be at most {settings.lora_max_steps}")
```

```python
    # 7. Browsers post empty file parts for untouched inputs — filter those out
    #    before counting, or the count check passes on phantom uploads.
    uploads = [u for u in images if u.filename or u.size]
    if len(uploads) < settings.lora_min_images:        # 5
        raise InputValidationError(
            f"at least {settings.lora_min_images} training images are required "
            f"(got {len(uploads)}) — 20-50 varied images of the character/style "
            "work best"
        )
    if len(uploads) > settings.lora_max_images:        # 60
        raise InputValidationError(
            f"at most {settings.lora_max_images} training images are allowed "
            f"(got {len(uploads)})"
        )
```

5 is the floor where training produces *anything*; the error text steers toward 20-50, which is where style LoRAs actually work. 60 is the ceiling that keeps a 2000-step run inside its time budget.

```python
    # 8. Materialise the dataset on disk under the job's own inputs/ tree.
    job_id = new_job_id()
    dataset_dir = settings.outputs_dir / job_id / "inputs" / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for index, upload in enumerate(uploads):
        # read_image_upload enforces max_upload_bytes and sniffs the real
        # format — the returned extension comes from the decoded image, never
        # from the client-supplied filename (that's the path-safety guarantee).
        image_bytes, extension = await read_image_upload(
            upload,
            max_bytes=settings.max_upload_bytes,
            field=f"image {upload.filename or index + 1!r}",
        )
        # Renamed to a flat, sortable sequence: img_000.png, img_001.jpg, …
        # Original filenames are discarded entirely.
        (dataset_dir / f"img_{index:03d}{extension}").write_bytes(image_bytes)
```

Renaming to `img_NNN` gives the caption writer a deterministic ordering and removes every filename-based attack surface in one move.

```python
    # 9. Hand off to the GPU worker queue and return immediately.
    job = Job(
        id=job_id,
        kind=JobKind.LORA_TRAINING,
        params=LoraTrainingParams(
            name=name, trigger=trigger, dataset_dir=dataset_dir,
            image_count=len(uploads), description=description or None, steps=steps,
        ),
        label=f"{name} · {len(uploads)} images",
        user_id=user.id,
        cost=cost,
    )
    await worker.submit(job)
    return JobCreatedResponse(job_id=job_id)
```

The request returns in milliseconds. Everything after this is the worker's problem, tracked via `GET /api/jobs/{id}`.

## A2. The job payload

`backend/app/queue/job.py:92-101`

```python
@dataclass
class LoraTrainingParams:
    # Style/character LoRA training on the Wan2.2 5B base (ostris/ai-toolkit).
    name: str            # display name for the finished style
    trigger: str         # trigger word baked into every caption
    dataset_dir: Path    # uploaded training images (captions written at run time)
    image_count: int
    description: str | None = None   # optional style hint appended to captions
    steps: int = 2000
    model: str = "wan22-5b-lora"     # engine id from models_catalog
```

Note **captions are not written at upload time**. The route only stores images; caption `.txt` files are generated inside the processor. That keeps the upload path pure I/O and means a retry regenerates captions from the current logic rather than from stale files on disk.

Deserialization (`job.py:184-192`) rebuilds this from the snapshot JSON so jobs survive a backend restart:

```python
    if kind is JobKind.LORA_TRAINING:
        return LoraTrainingParams(
            ...,
            model=str(data.get("model", "wan22-5b-lora")),
        )
```

## A3. Worker dispatch and the special timeout

`backend/app/main.py:78-116`

```python
    loras = LoraRegistry(settings.loras_dir)        # models/loras/
    manager = build_model_manager(settings)
    worker = GPUWorker(
        store,
        build_processors(settings, manager, voices, loras),
        job_timeout_s=settings.job_timeout_s,       # normal jobs: minutes
        failure_hook=manager.after_job_failure,
        # Training legitimately runs for hours — it gets its own ceiling.
        timeout_overrides={JobKind.LORA_TRAINING: settings.lora_timeout_s},  # 21_600 = 6h
    )
```

Every other job kind shares one timeout. Training is the one exception; without this override the watchdog would kill a legitimate 2000-step run.

```python
        JobKind.LORA_TRAINING: LoraTrainingProcessor(manager, loras, settings),
```

The processor gets three things: the **manager** (to evict pipelines off the GPU), the **registry** (to install the result), and **settings** (paths, step limits).

## A4. The processor — the actual training run

`backend/app/services/lora_training.py`

### Step 1 — captions

```python
def write_captions(dataset_dir: Path, trigger: str, description: str | None) -> int:
    """Write a caption .txt next to every image; returns the image count.

    Every caption is the trigger word (plus the optional style description) —
    the standard recipe for a style/character LoRA where the trigger word is
    made to carry the whole concept.
    """
    caption = trigger if not description else f"{trigger}, {description}"
    count = 0
    for image in sorted(dataset_dir.iterdir()):
        if image.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue                       # skip the .txt files we're writing
        image.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")
        count += 1
    return count
```

**Every caption is identical.** This is deliberate and is the core mechanic. Since the trigger token is the only thing constant across all images, gradient descent has no choice but to attach every shared visual property of the dataset to it. If captions described each image individually, the concept would smear across many tokens and the trigger would do nothing.

The suffix filter also makes the function idempotent — re-running it on an already-captioned directory won't produce `img_000.txt.txt`.

### Step 2 — the ai-toolkit config

```python
def build_training_config(*, run_name, training_dir, dataset_dir, model_path, steps) -> dict:
    """ai-toolkit job config for a Wan2.2 5B (arch wan22_5b) LoRA.

    Derived from ai-toolkit's train_lora_wan22_14b_24gb.yaml example. No
    quantization: the 5B transformer (~10 GB) plus the UMT5 encoder fit the
    L40S in bf16, and text embeddings are cached up front anyway. Sampling is
    disabled (sample_every > steps) — previews cost minutes each and the
    result is judged in the studio afterwards.
    """
    return {
        "job": "extension",
        "config": {
            "name": run_name,                       # "style" -> output filenames
            "process": [{
                "type": "sd_trainer",
                "training_folder": str(training_dir),
                "device": "cuda:0",

                # THE LORA ITSELF: rank 32, alpha 32 (alpha/rank == 1.0, so the
                # adapter's learned delta is applied at full strength).
                "network": {"type": "lora", "linear": 32, "linear_alpha": 32},

                "save": {
                    "dtype": "float16",             # half-size weights, no quality cost
                    "save_every": 500,              # crash insurance
                    "max_step_saves_to_keep": 2,    # bound the disk footprint
                },

                "datasets": [{
                    "folder_path": str(dataset_dir),
                    "caption_ext": "txt",           # matches write_captions()
                    "caption_dropout_rate": 0.05,   # 5% of steps train with NO
                                                    # caption — regularisation that
                                                    # stops the trigger from becoming
                                                    # a hard on/off switch
                    "num_frames": 1,                # Wan is a VIDEO model; 1 frame
                                                    # trains it as an image model
                    "resolution": [512, 768, 1024], # multi-res buckets -> the style
                                                    # holds at several output sizes
                }],

                "train": {
                    "batch_size": 1,                # 1 + grad checkpointing = fits 48 GB
                    "steps": steps,
                    "gradient_accumulation": 1,
                    "train_unet": True,             # adapt the transformer only…
                    "train_text_encoder": False,    # …never UMT5, or the trigger's
                                                    # embedding drifts and prompts
                                                    # stop behaving
                    "gradient_checkpointing": True, # trade compute for VRAM
                    "noise_scheduler": "flowmatch",
                    "timestep_type": "linear",
                    "optimizer": "adamw8bit",       # 8-bit states: ~3 GB saved
                    "lr": 1e-4,                     # standard rank-32 style LR
                    "optimizer_params": {"weight_decay": 1e-4},
                    "dtype": "bf16",
                    "cache_text_embeddings": True,  # encode all captions once up
                                                    # front, then free UMT5 — the
                                                    # captions are identical, so this
                                                    # is nearly free
                },

                "model": {
                    "name_or_path": str(model_path),  # the SAME checkpoint inference uses
                    "arch": "wan22_5b",
                    "quantize": False, "quantize_te": False, "low_vram": False,
                },

                "sample": {
                    ...,
                    "sample_every": steps + 1,      # never fires — see docstring
                    ...
                },
            }],
        },
        "meta": {"name": "[name]", "version": "1.0"},
    }
```

Two design notes worth calling out:

- **`sample_every: steps + 1`** — a config-level "off switch" for a setting ai-toolkit has no boolean for. Preview generation costs minutes per sample on a video model.
- **The config is a Python dict written to a `.yaml` file as JSON.** From the module docstring: *"JSON is valid YAML, so ai-toolkit's loader reads it and the backend needs no YAML dependency."* One less dependency in the inference image.
- **`model_path` points at the exact same checkpoint the inference pipeline loads** — `models/wan2.2-{variant}`. Nothing extra to download (`scripts/download_models.sh:55`), and the adapter is guaranteed compatible with the pipeline that'll consume it.

### Step 3 — progress parsing

```python
_STEP_RE = re.compile(r"(\d+)/(\d+)")

def parse_step_progress(line: str, total_steps: int) -> float | None:
    """Extract training progress from a trainer output line.

    Only a step counter whose total matches the configured step count is
    trusted — the trainer prints other N/M counters (latent caching over the
    dataset, downloads) that would otherwise make progress jump around.
    """
    for match in _STEP_RE.finditer(line):
        current, total = int(match.group(1)), int(match.group(2))
        if total == total_steps and current <= total:   # the denominator IS the filter
            return current / total
    return None
```

Screen-scraping a third-party tool's stdout. The `total == total_steps` guard is the whole trick: `47/60` (caching 60 images) is ignored, `812/2000` is accepted, because only the real training loop counts to the configured step total. The `current <= total` check rejects nonsense from interleaved output.

### Step 4 — locating the output

```python
def find_final_weights(run_dir: Path, run_name: str = RUN_NAME) -> Path:
    """The trained LoRA: `<run>.safetensors` written at the end of training,
    or the newest intermediate `<run>_NNNNNNNNN.safetensors` save."""
    final = run_dir / f"{run_name}.safetensors"
    if final.is_file():
        return final                       # clean finish
    saves = sorted(run_dir.glob(f"{run_name}_*.safetensors"))
    if saves:
        return saves[-1]                   # fall back to newest checkpoint
    raise RuntimeError(
        f"training produced no LoRA weights under {run_dir} — see train.log"
    )
```

The zero-padded step numbers (`style_000001500.safetensors`) make lexicographic `sorted()` equal numeric order — that's why the fallback can just take `[-1]`. This salvages a run that hit its 500-step checkpoint but died before finishing. The error message points at the log file rather than swallowing the failure.

### Step 5 — orchestration

```python
class LoraTrainingProcessor:
    async def process(self, job: Job, report: ProgressReporter) -> dict[str, str]:
        params = job.params
        assert isinstance(params, LoraTrainingParams)

        job_dir = self._settings.outputs_dir / job.id
        training_dir = job_dir / "train"
        config_path = job_dir / "train_config.yaml"

        # --- 1-2%: dataset prep -------------------------------------------
        report(1, "preparing dataset")
        image_count = write_captions(params.dataset_dir, params.trigger, params.description)
        if image_count == 0:
            # The route counted uploads; this counts what actually landed as a
            # decodable image. Fail here rather than after the GPU is cleared.
            raise RuntimeError("the training dataset contains no images")

        config = build_training_config(
            run_name=RUN_NAME,                          # "style"
            training_dir=training_dir,
            dataset_dir=params.dataset_dir,
            model_path=self._settings.models_dir / f"wan2.2-{self._settings.wan_variant}",
            steps=params.steps,
        )
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
```

```python
        # --- 2-3%: hand the whole card over -------------------------------
        report(2, "freeing gpu")
        await self._manager.release_gpu()
```

This is the critical resource move. From `pipelines/manager.py:101`:

```python
    async def release_gpu(self) -> None:
        """Fully unload every pipeline — training jobs need the whole card, and
        parking weights in system RAM would squeeze the trainer's own host
        allocations. Reloads afterwards come back from warm page cache."""
```

Not `to_cpu()` — full `unload()`. The module docstring explains why: parking Wan's ~24 GB in system RAM next to the trainer's own copies would squeeze the 64 GB host. The cost of reloading is under a minute from page cache; the cost of *not* doing this is an OOM.

```python
        # --- 3-93%: the run ------------------------------------------------
        report(_TRAIN_START, "training")      # 3

        def on_step(fraction: float) -> None:
            # Map the trainer's 0.0-1.0 into the 3-93 slice of the job's bar,
            # so the UI never rewinds when phases change.
            progress = _TRAIN_START + fraction * (_TRAIN_END - _TRAIN_START)
            report(int(progress), "training")

        await self._run_trainer(config_path, job_dir / "train.log", on_step, params.steps)
```

```python
        # --- 94-100%: install ----------------------------------------------
        report(94, "saving style")
        weights = find_final_weights(training_dir / RUN_NAME)
        # install() is blocking file I/O (copying a multi-MB safetensors) —
        # off the event loop so heartbeats and other requests keep flowing.
        info = await asyncio.to_thread(
            self._registry.install,
            weights,
            name=params.name, trigger=params.trigger,
            steps=params.steps, image_count=params.image_count,
        )
        log.info("style training complete",
                 extra={"job_id": job.id, "lora_id": info.id, "steps": params.steps})
        # The job's output is the new style's id — the frontend links straight to it.
        return {"lora": info.id}
```

### Step 6 — driving the subprocess

```python
    async def _run_trainer(self, config_path, log_path, on_step, total_steps) -> None:
        cmd = [
            str(self._settings.trainer_python),   # /opt/trainer-venv/bin/python
            str(self._settings.trainer_script),   # /opt/ai-toolkit/run.py
            str(config_path),
        ]
        log.info("starting trainer", extra={"cmd": " ".join(cmd)})
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,     # merge streams: one ordered log
            cwd=str(self._settings.trainer_script.parent),   # ai-toolkit needs its
                                                             # own dir as cwd
        )
```

No shell, argv list — a trigger word or style name can never reach a shell parser (and `TRIGGER_RE` already blocked it upstream; belt and braces).

```python
        assert process.stdout is not None
        tail: deque[str] = deque(maxlen=20)      # bounded ring buffer: last 20 lines
                                                 # kept in memory for the error message
        try:
            with log_path.open("w", encoding="utf-8") as log_file:
                pending = ""
                while True:
                    chunk = await process.stdout.read(4096)
                    if not chunk:
                        break                    # EOF: the trainer exited
                    pending += chunk.decode("utf-8", errors="replace")

                    # tqdm redraws with \r; treat both separators as line ends.
                    # Without splitting on \r the whole progress bar arrives as
                    # ONE enormous line and progress never updates.
                    *lines, pending = re.split(r"[\r\n]", pending)
                    # ^ the last element is a partial line — it stays in `pending`
                    #   until the next chunk completes it.

                    for line in lines:
                        if not line.strip():
                            continue
                        log_file.write(line + "\n")   # full log -> train.log
                        tail.append(line.strip())
                        fraction = parse_step_progress(line, total_steps)
                        if fraction is not None:
                            on_step(fraction)

                if pending.strip():               # trailing text with no newline
                    log_file.write(pending + "\n")
                    tail.append(pending.strip())

            code = await process.wait()

        except asyncio.CancelledError:
            # Job timeout or server shutdown — take the trainer down with us.
            # Without this kill, a cancelled job leaves a process holding the
            # entire GPU with nothing tracking it.
            process.kill()
            raise

        if code != 0:
            # Surface the last 3 log lines in the job's error, so the user sees
            # the real cause without SSHing in for train.log.
            detail = " | ".join(list(tail)[-3:]) or "no output"
            raise RuntimeError(f"style training failed (exit {code}): {detail}")
```

Reading fixed 4096-byte chunks rather than `readline()` is required here: tqdm emits carriage-returned progress with no newlines, and `readline()` would block until the bar finished.

## A5. Why the trainer lives in its own venv

`backend/Dockerfile:112-131`

```dockerfile
# ---- Style LoRA trainer (ostris/ai-toolkit) ---------------------------------------
# Trains Wan2.2 5B character/style LoRAs from the checkpoint already mounted at
# /app/models/wan2.2-ti2v-5b (arch wan22_5b). It needs a newer torch than the
# inference stack's pinned 2.4.1 (mmcv wheel matrix above), so it lives in its
# own venv and the backend invokes it as a subprocess
# (app/services/lora_training.py). cu126 wheels bundle their CUDA runtime and
# run fine on this 12.4 base image.
ARG AI_TOOLKIT_REF=main
RUN git clone --depth 1 --branch ${AI_TOOLKIT_REF} \
        https://github.com/ostris/ai-toolkit.git /opt/ai-toolkit \
    && python3.11 -m venv /opt/trainer-venv \
    && /opt/trainer-venv/bin/pip install --no-cache-dir --upgrade pip "setuptools<81" wheel \
    && /opt/trainer-venv/bin/pip install --no-cache-dir \
        torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
        --index-url https://download.pytorch.org/whl/cu126 \
    && /opt/trainer-venv/bin/pip install --no-cache-dir -r /opt/ai-toolkit/requirements.txt \
    # Build-time canary: the trainer venv must import its own torch and the
    # toolkit entry point must parse — fail the build, not the first training job.
    && /opt/trainer-venv/bin/python -c "import torch, torchaudio; print('trainer torch', torch.__version__)" \
    && /opt/trainer-venv/bin/python -m py_compile /opt/ai-toolkit/run.py
```

This is the root reason training is a **subprocess and not a function call**: torch 2.7.1 (trainer) and torch 2.4.1 (inference, pinned by the mmcv wheel matrix MuseTalk needs) cannot coexist in one interpreter. Two venvs, one process boundary, `asyncio.create_subprocess_exec` as the bridge.

The build-time canary is a nice touch — an incompatible ai-toolkit `main` fails `docker build`, not a user's 100-credit job six hours in.

---

# PART B — The registry (shared state)

`backend/app/services/loras.py`

```python
"""Trained style LoRA registry.

Each adapter is a directory under settings.loras_dir:

    loras/<id>/lora.safetensors   — the trained weights
    loras/<id>/lora.json          — metadata (name, trigger word, base, …)

The directory scan is cheap (a handful of small JSON files), so the registry is
stateless and reads the disk on every call — no cache to invalidate when a
training job installs a new style.
"""
```

That last sentence is the key architectural decision. The training processor runs in the worker task; the HTTP routes run in request tasks. A cache would need invalidation across that boundary. Re-reading a handful of tiny JSON files costs less than the synchronisation would.

```python
WEIGHTS_FILENAME = "lora.safetensors"
META_FILENAME = "lora.json"

# Trigger words are baked into captions and prompts verbatim — keep them plain
# tokens so they survive tokenization and shell/JSON round-trips unambiguously.
TRIGGER_RE = re.compile(r"^[A-Za-z0-9_]{2,30}$")
```

`TRIGGER_RE` is defined here, next to the data it protects, and imported by the route — validation and storage share one definition.

```python
@dataclass(frozen=True)
class LoraStyleInfo:
    id: str
    name: str
    trigger: str
    base: str            # engine family the adapter applies to ("wan-5b")
    created_at: datetime
    weights_path: Path   # resolved here so callers never build paths themselves
```

Frozen — a style handed to a job can't be mutated underneath it.

```python
    def list(self) -> list[LoraStyleInfo]:
        if not self._dir.is_dir():
            return []                        # nothing trained yet is not an error
        styles = []
        for entry in sorted(self._dir.iterdir()):
            info = self._read(entry)
            if info is not None:             # skip junk / half-written dirs
                styles.append(info)
        styles.sort(key=lambda s: s.created_at, reverse=True)   # newest first
        return styles
```

```python
    def get(self, lora_id: str) -> LoraStyleInfo | None:
        # ids are generated by _slugify + hex suffix, but guard against path
        # tricks anyway since the id arrives from a form field.
        if not lora_id or Path(lora_id).name != lora_id:
            return None
        return self._read(self._dir / lora_id)
```

`Path(lora_id).name != lora_id` is the traversal guard: `../../etc` has `.name == "etc"`, so it fails the equality and returns `None`. Any id containing a separator is rejected without touching the filesystem.

```python
    def install(self, weights: Path, *, name, trigger,
                base="wan-5b", steps=None, image_count=None) -> LoraStyleInfo:
        """Copy trained weights into the registry and write their metadata."""
        # Readable prefix + 6 hex chars: human-recognisable in URLs, collision-free
        # for same-named styles.
        lora_id = f"{_slugify(name)}-{uuid.uuid4().hex[:6]}"
        target_dir = self._dir / lora_id
        target_dir.mkdir(parents=True, exist_ok=False)   # exist_ok=False: a collision
                                                         # must crash, never overwrite
                                                         # someone's trained weights
        shutil.copyfile(weights, target_dir / WEIGHTS_FILENAME)
        created_at = datetime.now(timezone.utc)
        meta = {
            "id": lora_id, "name": name, "trigger": trigger, "base": base,
            "steps": steps,              # provenance: not read back, but
            "image_count": image_count,  # invaluable when debugging a bad style
            "created_at": created_at.isoformat(),
        }
        (target_dir / META_FILENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        log.info("style installed", extra={"lora_id": lora_id, "style": name})
        return LoraStyleInfo(..., weights_path=target_dir / WEIGHTS_FILENAME)
```

Weights are **copied**, not moved — the job's own `train/` directory keeps its checkpoints so a failed install can be retried, and the registry's copy is independent of output cleanup.

Ordering matters: weights land *before* metadata. `_read()` requires both files, so a crash mid-install leaves a directory that's simply invisible rather than a broken entry.

```python
    def _read(self, entry: Path) -> LoraStyleInfo | None:
        meta_path = entry / META_FILENAME
        weights_path = entry / WEIGHTS_FILENAME
        if not (meta_path.is_file() and weights_path.is_file()):
            return None                  # both-or-nothing: filters in-progress dirs
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return LoraStyleInfo(
                id=meta["id"], name=meta["name"], trigger=meta["trigger"],
                base=meta.get("base", "wan-5b"),   # .get with default = forward
                                                   # compat with pre-`base` metadata
                created_at=datetime.fromisoformat(meta["created_at"]),
                weights_path=weights_path,
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            # One corrupt style must not take down GET /api/loras for everyone.
            log.warning("skipping unreadable style metadata", extra={"path": str(meta_path)})
            return None
```

## B1. The registry's HTTP surface

`backend/app/routes/loras.py` — 51 lines, because the registry does the work.

```python
@router.get("/loras", response_model=LorasResponse)
async def list_loras(registry, _user) -> LorasResponse:
    # Maps LoraStyleInfo -> LoraStyle, which deliberately OMITS weights_path.
    # Server filesystem layout never reaches the client.
    return LorasResponse(loras=[
        LoraStyle(id=s.id, name=s.name, trigger=s.trigger, base=s.base,
                  created_at=s.created_at)
        for s in registry.list()
    ])
```

```python
@router.delete("/loras/{lora_id}", status_code=204, responses={404: {"model": ErrorResponse}})
async def delete_lora(lora_id, registry, user) -> None:
    # Styles are a shared registry with no per-user ownership, so deleting a
    # trained style (a 100-credit run) is admin-only.
    if user.role != "admin":
        raise HTTPException(403, "Only an admin can delete a trained style")
    if not registry.delete(lora_id):
        raise HTTPException(404, "Style not found")
```

The registry is **global, not per-user** — anyone can use any trained style, so no individual can delete one.

---

# PART C — Inference

## C1. Resolving the requested style

`backend/app/routes/generation.py:56-81`

```python
def _resolve_lora(lora: str, engine: Engine, registry: LoraRegistry) -> LoraStyleInfo | None:
    """Validate a requested style adapter against the registry and the engine.
    Styles are trained on (and only apply to) the Wan2.2 5B family."""
    if not lora:
        return None                        # empty string == "no style", not an error
    info = registry.get(lora)
    if info is None:
        raise InputValidationError(
            f"unknown style {lora!r} — see /api/loras for trained styles"
        )
    # Architecture gate. A wan22_5b adapter's tensor names don't match Flux's
    # transformer; without this check the failure would surface deep inside
    # diffusers as an opaque key error, after the GPU had already been claimed.
    if engine.id != "wan-5b":
        raise InputValidationError(
            f"style {info.name!r} was trained for the Wan2.2 5B engine — "
            f"it cannot be applied to model {engine.id!r}"
        )
    return info
```

```python
def _prompt_with_trigger(prompt: str, style: LoraStyleInfo | None) -> str:
    """The trigger word must appear in the prompt for the adapter to fire;
    prepend it when the user leaves it out."""
    if style is None or style.trigger.lower() in prompt.lower():
        return prompt                      # already there — don't double it
    return f"{style.trigger}, {prompt}"
```

This closes the loop on `write_captions`. Training bound the concept to that token; **if the token isn't in the prompt, the loaded adapter contributes almost nothing.** Rather than making users remember, the backend injects it. The case-insensitive substring check avoids `zork_style, zork_style, a castle` when they did remember.

Both are wired into image and B-roll:

```python
    lora: Annotated[str, Form()] = "",                       # style id
    lora_scale: Annotated[float, Form(ge=0.1, le=2.0)] = 1.0,  # bounded by FastAPI
) -> JobCreatedResponse:
    style = _resolve_lora(lora, engine, loras)
    ...
        params=ImageParams(
            ...,
            lora_id=style.id if style else None,             # for retry/audit
            lora_path=style.weights_path if style else None, # what the GPU loads
            lora_scale=lora_scale,
        )
```

`ge=0.1, le=2.0` is enforced by FastAPI before the handler body — 0 would be a no-op load and large values blow out the image.

## C2. Both id and path are stored

`backend/app/queue/job.py:44-64`

```python
@dataclass
class BrollParams:
    prompt: str
    duration_s: int
    image_path: Path | None
    model: str = "wan-5b"
    # Trained style adapter to apply (see services.loras); path resolved at submit.
    lora_id: str | None = None     # stable reference — survives retraining
    lora_path: Path | None = None  # resolved snapshot — what the worker loads
    lora_scale: float = 1.0
```

Carrying both looks redundant but isn't:

- **`lora_path`** is resolved once at submit so the worker never re-consults the registry mid-job — a deletion between submit and execution can't change what runs.
- **`lora_id`** is the durable reference used for retry and for display. Paths in a snapshot go stale; ids don't.

## C3. Applying it around generation

`backend/app/services/image.py:48-71` (B-roll at `services/broll.py:44-58` is identical in shape):

```python
        async with self._manager.acquire(pipeline_name) as pipe:
            if params.lora_path is not None:
                # Blocking: reads a multi-MB safetensors and patches every
                # attention module. to_thread keeps the event loop responsive.
                await asyncio.to_thread(pipe.set_lora, params.lora_path, params.lora_scale)
            try:
                for index in range(count):
                    ...
                    await asyncio.to_thread(pipe.generate_image, ...)
                    urls.append(f"/outputs/{job.id}/{name}")
            finally:
                # NON-NEGOTIABLE. The pipeline is a long-lived shared object
                # inside the manager. A leaked adapter would silently apply
                # someone else's style to every subsequent job on this pipeline.
                if params.lora_path is not None:
                    await asyncio.to_thread(pipe.clear_lora)
```

`set_lora` sits **outside** the `try`, `clear_lora` inside the `finally`. If loading itself throws, `set_lora` has already cleaned up internally (below), so there's nothing to unwind — and the pairing stays exact: cleared once, only when loaded.

Note the adapter is loaded **once for all N variations**, not per image.

## C4. The pipeline mechanics

`backend/app/pipelines/wan_pipeline.py:180-207`

```python
    # ---- style LoRA -------------------------------------------------------------

    _LORA_ADAPTER = "style"        # fixed diffusers adapter name; one slot, always
                                   # the same key, so clearing is unambiguous

    def set_lora(self, path: Path, scale: float = 1.0) -> None:
        """Load a trained style adapter onto the shared transformer (the i2v
        view shares modules, so one load covers both). Call between acquire and
        generate; pair with clear_lora() in a finally block."""
        if self._t2v is None:
            raise RuntimeError("Wan pipeline must be loaded before set_lora()")
        try:
            # peft injects the rank-32 A/B matrices into every targeted Linear.
            self._t2v.load_lora_weights(str(path), adapter_name=self._LORA_ADAPTER)
            # …then set the blend weight. This is where lora_scale lands:
            #   W_effective = W_base + scale * (B @ A)
            self._t2v.set_adapters([self._LORA_ADAPTER], adapter_weights=[scale])
        except Exception as exc:
            # Make sure a half-applied adapter never leaks into the next job.
            # load_lora_weights can fail PARTWAY through module injection —
            # without this the pipeline stays permanently corrupted.
            self.clear_lora()
            raise RuntimeError(
                f"failed to apply style LoRA {path.parent.name!r}: {exc}"
            ) from exc
        log.info("style LoRA applied", extra={"path": str(path), "scale": scale})

    def clear_lora(self) -> None:
        if self._t2v is None:
            return                     # already unloaded — nothing to clear
        try:
            self._t2v.unload_lora_weights()
        except Exception:
            # Never mask the original job error with a cleanup failure —
            # this runs in a `finally`.
            log.exception("failed to unload style LoRA")
```

The `t2v`/`i2v` note matters: those two are *views over the same nn.Modules*, so patching one patches both. Loading twice would double-apply.

`path.parent.name` in the error message is the style's directory name — i.e. the `lora_id` — since the file itself is always just `lora.safetensors`.

The PEFT backend this depends on is pinned explicitly (`requirements-gpu.txt:11`):

```
peft>=0.15,<0.18  # diffusers' LoRA loading (load_lora_weights) requires the PEFT backend
```

---

# PART D — Cross-cutting

**Retry re-resolves the style** — `routes/jobs.py:221-246`:

```python
    # Styles may have been retrained or deleted since — resolve them fresh.
    lora_path = None
    if isinstance(params, (BrollParams, ImageParams)) and params.lora_id:
        style = loras.get(params.lora_id)
        if style is None:
            raise HTTPException(409, "The style this job used has been deleted.")
        lora_path = style.weights_path
    ...
    if lora_path is not None:
        new_params = replace(new_params, lora_path=lora_path)   # overwrite the stale path
```

This is exactly why both `lora_id` and `lora_path` are stored. The snapshot's path may point at a deleted directory; the id is re-resolved, and a 409 with a real explanation beats a `FileNotFoundError` after the job is queued.

**Config** — `app/config.py:59-69`:

```python
    # ---- style LoRA training (ostris/ai-toolkit on the Wan2.2 5B base) ----------
    lora_min_images: int = 5
    lora_max_images: int = 60
    lora_default_steps: int = 2000
    lora_max_steps: int = 4000
    lora_timeout_s: float = 21_600.0       # 6 hours
    # The trainer lives in its own venv — its torch is newer than the inference
    # stack's (see backend/Dockerfile trainer stage).
    trainer_python: Path = Path("/opt/trainer-venv/bin/python")
    trainer_script: Path = Path("/opt/ai-toolkit/run.py")

    @property
    def loras_dir(self) -> Path:
        return self.models_dir / "loras"   # sits beside the base checkpoints, so
                                           # the models volume carries styles too
```

`loras_dir` is included in the startup mkdir list (`config.py:121`), so a fresh deploy has an empty-but-valid registry.

**Status output** — `routes/status.py:55` surfaces `lora=job.outputs.get("lora")`, the id returned by `process()`, so the UI can deep-link the finished style.

**Catalog** — `models_catalog.py:82` registers `wan22-5b-lora` under `JobKind.LORA_TRAINING`, which is what makes `_resolve_engine` and the availability/tier gate work for training like any other engine.

---

# The complete lifecycle

| Phase | Where | What happens |
|---|---|---|
| Upload | `routes/generation.py:502` | Validate trigger/steps/count, charge 100 credits, write `img_NNN.*` into `inputs/dataset/` |
| Queue | `main.py:116` | `JobKind.LORA_TRAINING` gets the 6-hour timeout override |
| Caption (1-2%) | `lora_training.py:43` | Identical `trigger[, description]` caption next to every image |
| Config (2%) | `lora_training.py:60` | rank-32 config written as JSON into `train_config.yaml` |
| Evict (2-3%) | `manager.py:101` | **Every** pipeline fully unloaded — trainer needs the whole L40S |
| Train (3-93%) | `lora_training.py:226` | ai-toolkit subprocess in its own torch-2.7.1 venv; stdout scraped for `N/2000` |
| Install (94-100%) | `loras.py:71` | Weights copied to `models/loras/<slug>-<hex6>/`, then `lora.json` |
| List | `routes/loras.py:15` | Fresh directory scan, newest first, `weights_path` withheld |
| Generate | `generation.py:56` | id → `LoraStyleInfo`, engine gate, trigger injected into prompt |
| Apply | `wan_pipeline.py:184` | `load_lora_weights` + `set_adapters([scale])` on the shared transformer |
| Clear | `image.py:71` | `finally: clear_lora()` — no adapter ever survives its job |

**The three ideas that make it work:** identical captions force the trigger token to absorb the whole concept; the trigger is auto-injected at generation so users can't accidentally disable their own style; and the adapter is always cleared in a `finally` because the pipeline outlives every job that borrows it.
