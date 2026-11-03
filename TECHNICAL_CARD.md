# AI Video Studio — Technical Card

Self-hosted GPU video generation studio: FastAPI backend owning the models and a queue, Next.js frontend owning accounts. Analysis covers 159 tracked files at `6442d7b`.

---

## 1. Stack

### Backend — Python
| Item | Version | Source |
|---|---|---|
| Runtime | Python 3.11 | `backend/Dockerfile:39` |
| Base image | `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` | `backend/Dockerfile:17` |
| Web | FastAPI ≥0.115, uvicorn[standard] ≥0.30, pydantic ≥2.9, pydantic-settings ≥2.5 | `backend/requirements.txt:2-5` |
| ML | torch 2.4.1+cu121, diffusers ≥0.35 <0.36, transformers ≥4.46 <4.58, peft ≥0.15 <0.18 | `backend/Dockerfile:47`, `backend/requirements-gpu.txt:7-11` |
| From git (not pip) | fish-speech `v2.0.0-beta`, MuseTalk `main`, ostris/ai-toolkit `main` | `backend/Dockerfile:20-21,52,70,119-121` |
| mm-stack | mmcv 2.2.0, mmdet 3.3.0, mmpose 1.3.2 | `backend/Dockerfile:60` |
| Trainer venv | torch 2.7.1+cu126, separate from inference torch | `backend/Dockerfile:124-126` |

**(a) Actually imported in code:** `fastapi`, `pydantic`/`pydantic-settings` (`app/config.py`, `app/schemas.py`), `PIL` (`app/services/validation.py`, `app/pipelines/wan_pipeline.py`), `jwt` (`app/services/auth.py`), `torch` + `diffusers` (all pipelines), `transformers` (`app/pipelines/musetalk_pipeline.py:76`), `cv2` (`musetalk_pipeline.py`, `upscale_pipeline.py`), `numpy`, `soundfile`, `torchaudio` (`app/pipelines/s2_pipeline.py`), `basicsr`+`realesrgan` (`app/pipelines/upscale_pipeline.py:62-63`), `fish_speech` (`s2_pipeline.py:120-121`), `musetalk.*` (`musetalk_pipeline.py:74-75`).

**(b) Declared but never imported by app code** (verified by grep over `backend/app` + `backend/assets`):
- `httpx` — declared as "test client dependency" (`requirements.txt:11-12`); needed by `starlette.testclient.TestClient`, used in 7 test files.
- `uvicorn`, `python-multipart` — invoked as CLI (`Dockerfile:161`) / required by FastAPI's `Form`/`File`, never imported.
- `accelerate`, `safetensors`, `peft`, `sentencepiece`, `protobuf`, `librosa`, `einops`, `ftfy`, `imageio` — transitive requirements pinned deliberately with in-file rationale (`requirements-gpu.txt:9-29`); e.g. `peft` is diffusers' LoRA backend, `imageio` backs `diffusers.utils.export_to_video` (`wan_pipeline.py:225`).

### Frontend — TypeScript
Next.js ^15.5.20 (`output: "standalone"`, `frontend/next.config.ts:8`), React 19.2.4, next-auth ^4.24.14, Prisma ^6.19.3, Tailwind v4, `@base-ui/react` ^1.6.0, `jose`, `bcryptjs`, `lucide-react` — all `frontend/package.json`.

**Every frontend dependency is used.** `@base-ui/react` → `components/ui/*.tsx`; `jose` → `app/api/backend-token/route.ts:1`; `bcryptjs` → `lib/auth-config.ts:1`, `app/api/auth/register/route.ts:1`; `shadcn` is not a JS import but a CSS one (`app/globals.css:3`, `@import "shadcn/tailwind.css"`); `tw-animate-css` likewise via `globals.css`.

> `frontend/AGENTS.md` warns this Next.js version diverges from training data and directs readers to `node_modules/next/dist/docs/`.

---

## 2. Architecture

**Entry points**
- Backend: `backend/app/main.py:160` — `app = create_app()`, served by `uvicorn app.main:app` (`Dockerfile:161`).
- Frontend: Next App Router, `frontend/app/layout.tsx` + per-feature pages (`app/talking-head`, `broll`, `image`, `full-video`, `upscale`, `lora`, `voice`, `library`).
- Model download: `backend/scripts/download_models.sh` (host-side, ~55 GB from Hugging Face).

**Module boundaries** (`backend/app/`)
| Layer | Responsibility |
|---|---|
| `routes/` | HTTP only: validate, persist uploads, price, enqueue (`routes/generation.py`) |
| `queue/` | `Job` dataclasses (`job.py`), `JobStore` state+snapshots (`job_store.py`), single `GPUWorker` (`worker.py`) |
| `services/` | Per-kind processors (`talking_head.py`, `broll.py`, `image.py`, `full_video.py`, `upscale.py`, `lora_training.py`) + `ffmpeg.py`, `validation.py`, `script_parser.py`, `credits.py`, `auth.py` |
| `pipelines/` | `ManagedPipeline` state machine (`base.py`) and `ModelManager` VRAM residency (`manager.py`) |
| `models_catalog.py` | Static engine catalog: id → tier, credits hint, pipeline name |

**Data flow (generation request → response)**

1. Browser `POST /api/broll` (multipart) via `frontend/lib/api.ts:119`; same-origin — `next.config.ts:14` rewrites `/api/*` to the backend, nginx does it directly in prod (`deploy/nginx.conf.example:47`).
2. `routes/generation.py:153` resolves the engine (`_resolve_engine`), validates text (`services/validation.py:45`), resolves any style LoRA, prices via `services/credits.py`, gates on `require_credits`.
3. Uploads streamed with a size cap and content-sniffed (`validation.py:28,56,119`), written to `outputs/{jobId}/inputs/`.
4. `worker.submit(job)` → `JobStore.add` (writes `status.json`) → job **id** onto an `asyncio.Queue` (`queue/worker.py:57`). Response is `{jobId}` only.
5. `GPUWorker._run` drains one job at a time (`worker.py:78-88`); processor runs under `asyncio.wait_for(timeout)`.
6. Processor acquires pipelines via `ModelManager.acquire(name)`, which loads/evicts by measured free VRAM (`pipelines/manager.py:80-100`); blocking inference goes through `asyncio.to_thread`.
7. Progress callbacks marshal onto the event loop with `loop.call_soon_threadsafe` (`worker.py:100-103`); persisted to `status.json` at most 1×/s (`job_store.py:25,85-92`).
8. Browser polls `GET /api/status/{id}` every 2 s (`frontend/lib/use-job-polling.ts:15`) until terminal; media is served from the `/outputs` static mount (`main.py:141`).

---

## 3. Data

**Two stores, deliberately split** — `frontend/prisma/schema.prisma:1-2`: "Only account data lives here — jobs and generated files stay on the GPU box".

**Accounts DB:** PostgreSQL (Neon), Prisma ORM, schema at `frontend/prisma/schema.prisma`.
- **2 tables**: `User`, `Account`.
- **Relation**: `Account.userId → User.id`, `onDelete: Cascade` (`schema.prisma:46`).
- **Indexes**: `@id` on `User.id` and `Account.id` (cuid); `@unique` on `User.email` (`:16`); composite `@@unique([provider, providerAccountId])` on `Account` (`:48`). No other `@@index`.
- **Sessions**: none stored — JWT strategy, 60-day maxAge (`lib/auth-config.ts:12-15`).
- **Migrations: none exist.** `frontend/prisma/` contains only `schema.prisma`; the build runs `prisma generate` only (`package.json:7`), and README instructs `npx prisma db push` (`README.md:420`). There is no migration history.

**Job store:** no database. In-memory dict + one `status.json` per job, atomically written via temp-then-rename (`backend/app/queue/job_store.py:116-122`). `rehydrate()` re-reads `outputs/*/status.json` on boot and force-fails anything left queued/processing (`job_store.py:95-114`).

**LoRA registry:** filesystem only — `loras/<id>/lora.safetensors` + `lora.json`, re-scanned on every call, no cache (`backend/app/services/loras.py:1-11`).

**Voices:** `backend/assets/voices.json` + WAVs, validated at startup, broken entries skipped (`backend/app/services/voices.py:30-62`).

---

## 4. API

All backend routes are prefixed `/api`; all require `Depends(get_current_user)`.

| Method | Path | File |
|---|---|---|
| GET | `/health` | `main.py:153` |
| POST | `/api/talking-head` | `routes/generation.py:97` |
| POST | `/api/broll` | `routes/generation.py:152` |
| POST | `/api/image` | `routes/generation.py:203` |
| POST | `/api/full-video` | `routes/generation.py:272` |
| POST | `/api/upscale` | `routes/generation.py:418` |
| POST | `/api/lora-training` | `routes/generation.py:502` |
| GET | `/api/status/{job_id}` | `routes/status.py:60` |
| GET | `/api/jobs` (`?kind`, `?limit` 1–100) | `routes/status.py:93` |
| GET | `/api/credits` | `routes/status.py:78` |
| GET | `/api/jobs/{job_id}` | `routes/jobs.py:108` |
| POST | `/api/jobs/{job_id}/regenerate` | `routes/jobs.py:182` |
| DELETE | `/api/jobs/{job_id}` (204, soft-delete) | `routes/jobs.py:263` |
| GET | `/api/voices` | `routes/voices.py:18` |
| GET | `/api/models` | `routes/models.py:16` |
| GET | `/api/loras` · DELETE `/api/loras/{id}` (admin-only) | `routes/loras.py:15,34` |
| — | `/outputs/*` static mount | `main.py:141` |

Frontend routes: `GET /api/backend-token` (`app/api/backend-token/route.ts:12`), `POST /api/auth/register` (`app/api/auth/register/route.ts:10`), NextAuth catch-all `app/api/auth/[...nextauth]/route.ts`.

**Authentication.** Two-world bridge: NextAuth (Google OAuth + credentials, `lib/auth-config.ts:20-45`) → `/api/backend-token` mints a 30-minute HS256 JWT with `sub/email/name/credits/role` (`backend-token/route.ts:7,26-36`) → FastAPI verifies it with the shared `API_JWT_SECRET`, requiring `exp` and `sub` (`services/auth.py:42-58`). Client caches the token and refreshes 60 s before expiry or after a 401 (`lib/api.ts:36-66`). With `AUTH_ENABLED=false` every request runs as `LOCAL_USER(id="local", role="admin")` (`services/auth.py:39,65`). Ownership: `can_view` = admin-or-owner; foreign/deleted jobs return 404, not 403, to avoid leaking existence (`routes/jobs.py:51-55`).

**Input validation.** Pydantic shapes in `schemas.py`; enforcement in `services/validation.py` — size cap applied *while streaming* so oversized uploads are rejected before full buffering (`:28-42`), format verified by content not filename (PNG/JPEG via Pillow `:56-80`; MP4/MOV by `ftyp` box, WebM by EBML magic `:119-137`). Videos are additionally ffprobe'd for playability (`generation.py:354-359`). Script markers parsed and rejected with user-safe messages, including path-traversal guards on `[CLIP: …]` names (`services/script_parser.py:118-123`). `InputValidationError` → HTTP 422 via one app-level handler (`main.py:149-151`).

---

## 5. External Integrations

| Integration | Where |
|---|---|
| Hugging Face Hub (model download, `hf` CLI; two license-gated repos need `HF_TOKEN`) | `backend/scripts/download_models.sh:33,39,65` |
| GitHub releases (Real-ESRGAN `.pth` weights via curl) | `download_models.sh:83-88` |
| GitHub clones at build time (MuseTalk, fish-speech, ai-toolkit) | `backend/Dockerfile:52,70,120` |
| `adrianbulat.com` S3FD face-detector weights, pre-cached at build | `backend/Dockerfile:84-85` |
| Google OAuth | `frontend/lib/auth-config.ts:21-24` |
| Neon Postgres | `frontend/lib/db.ts`, `frontend/prisma/schema.prisma:9-11` |
| FFmpeg / ffprobe (subprocess) | `backend/app/services/ffmpeg.py:33-46,89-102` |
| ai-toolkit trainer (subprocess, separate venv) | `backend/app/services/lora_training.py:239-244` |

**Not present:** no payment provider, no Redis/Celery/RabbitMQ/SQS, no S3 or object storage (outputs are a Docker volume, `docker-compose.yml:14`), no third-party LLM/inference API, no email/SMS, no telemetry or error-tracking SDK. All inference is local.

---

## 6. AI / ML

All models run **in-process on one GPU**; nothing calls a hosted API.

| Engine id | Model | Key call | Settings |
|---|---|---|---|
| `musetalk` / `musetalk-animate` | MuseTalk 1.5 + whisper-tiny + sd-vae + DWPose | `pipelines/musetalk_pipeline.py:139` | 25 fps, batch 8 |
| — (TTS) | Fish Audio S2 Pro (fish-speech dual-AR + DAC codec) | `pipelines/s2_pipeline.py:200` | 44.1 kHz, `top_p=0.8`, `temperature=0.8`, `repetition_penalty=1.1` (`:288-291`) |
| `wan-5b` | Wan2.2 TI2V-5B (diffusers), T2V + I2V views over shared weights | `pipelines/wan_pipeline.py:211,268` | 40 steps, guidance 5.0, 704×1280@24fps, frames must be 4k+1 (`:22-26,78-82`) |
| `flux-schnell` | FLUX.1-schnell | `pipelines/flux_pipeline.py` | 4 steps, guidance 0.0 (`:21-22`) — chosen over FLUX.1-dev for Apache-2.0 license safety (`:3-4`) |
| `flux-kontext` | FLUX.1-Kontext-dev (reference editing) | `pipelines/flux_kontext_pipeline.py:107` | 28 steps, guidance 2.5, `max_sequence_length=512` |
| `realesrgan-photo/-anime` | RealESRGAN_x4plus / _anime_6B | `pipelines/upscale_pipeline.py:119` | tile 512, fp16 on GPU |
| `wan22-5b-lora` | ostris/ai-toolkit LoRA on Wan2.2 5B | `services/lora_training.py:60-139` | rank 32/alpha 32, adamw8bit, lr 1e-4, bf16, flowmatch, 2000 default steps |
| `wan-a14b`, `wan-s2v-14b`, `wan-animate-14b` | **Scaffolds**, `implemented=False` or premium-gated | `pipelines/wan_a14b_pipeline.py`, `wan_s2v_pipeline.py:34-38`, `wan_animate_pipeline.py` | never registered unless `PREMIUM_ENABLED` (`main.py:66-70`) |

- **Structured output:** N/A — no LLM, no JSON-schema/tool-calling anywhere. Structure comes from Pydantic response models (`app/schemas.py`) and a discriminated status union (`schemas.py:103-106`).
- **Retries:** none for generation. A failed job is terminal (`worker.py:130-138`); the user re-runs it via `POST /api/jobs/{id}/regenerate`. The only retry logic is the frontend's single token-refresh retry on 401 (`lib/api.ts:60-64`) and `curl --retry 3` in the download script.
- **Caching:** page-cache warming of checkpoints before load (`s2_pipeline.py:130-150`); tensor `.clone()` to defeat lazy mmap paging on EBS (`:163-172`); reference-audio encodings cached per clip mtime (`:110-112,253`); pipelines are left `ON_GPU` after a job so back-to-back jobs of one kind pay no moves (`manager.py:84-87`).
- **Batching:** MuseTalk UNet batches 8 frames (`musetalk_pipeline.py:28`). The full-video assembler batches **by pipeline rather than timeline order** so each VRAM peak is satisfied once (`services/full_video.py:3-9`). Image jobs loop 1–4 variations inside one residency (`services/image.py:52`).
- **Rate limiting:** none at HTTP level. The de facto limiter is the single sequential worker (`queue/worker.py:1-6`) plus per-job timeouts: 7200 s default, 21600 s for training (`config.py:50,65`; `main.py:114-116`).
- **Cost control:** credit ledger — priced at submission in `services/credits.py`, stored on the `Job`, spend = sum of non-failed job costs, so failed jobs self-refund and price changes never rewrite history (`credits.py:1-10`, `job_store.py:56-63`). Admins are exempt (`credits.py:75-76`). Structural caps: `full_video_max_segments=12`, `upscale_max_image_mp=4`, `upscale_max_video_s=120`, `lora_max_steps=4000` (`config.py:46,56-57,63`).

---

## 7. Infra

- **Docker Compose** (`docker-compose.yml`): `backend` (GPU reservation `driver: nvidia, count: all`, `shm_size: 8gb` because Docker's 64 MB default kills DataLoader workers during training, `:6-8`) + `frontend`. Both bind to `127.0.0.1` only — "nginx (or an SSH tunnel) is the front door, never the raw port" (`:9`). Model weights are **not** baked into the image; they're a mounted volume (`backend/Dockerfile:4-5`, compose `:13-15`).
- **Backend image**: single-stage CUDA image with a documented version matrix and three build-time canaries that fail the build instead of the first job (`Dockerfile:7-15,101,130-131,144-152`). Healthcheck on `/health` (`:158-159`).
- **Frontend image**: 3-stage node:22-alpine → Next standalone, runs as `USER node` (`frontend/Dockerfile:31`).
- **Hosting**: single GPU host (L40S referenced throughout config comments), systemd oneshot unit wrapping compose (`deploy/ai-video-studio.service`), nginx reverse proxy with carve-outs for `/api/auth/` and `/api/backend-token` (`deploy/nginx.conf.example:30-44`), certbot for TLS.
- **Background jobs**: one in-process `asyncio` worker (`backend/app/queue/worker.py`). No external broker, no scheduler, no cron.
- **CI/CD: none.** No `.github/`, `.gitlab-ci.yml`, or any other CI config exists in the repo.
- **Environment variables** (names only) — `.env.example`, `docker-compose.yml`, `frontend/Dockerfile`:
`BACKEND_PORT`, `FRONTEND_PORT`, `BACKEND_URL`, `MODELS_DIR`, `OUTPUTS_DIR`, `ASSETS_DIR`, `MAX_UPLOAD_MB`, `MAX_SCRIPT_CHARS`, `MAX_PROMPT_CHARS`, `JOB_TIMEOUT_S`, `RECENT_JOBS_LIMIT`, `LORA_MIN_IMAGES`, `LORA_MAX_IMAGES`, `LORA_DEFAULT_STEPS`, `LORA_MAX_STEPS`, `LORA_TIMEOUT_S`, `TRAINER_PYTHON`, `TRAINER_SCRIPT`, `WAN_VARIANT`, `VRAM_RESERVE_GB`, `S2_OFFLOAD`, `MUSETALK_OFFLOAD`, `WAN_OFFLOAD`, `FLUX_OFFLOAD`, `KONTEXT_OFFLOAD`, `UPSCALE_OFFLOAD`, `UPSCALE_MAX_IMAGE_MP`, `UPSCALE_MAX_VIDEO_S`, `PREMIUM_ENABLED`, `AUTH_ENABLED`, `API_JWT_SECRET`, `DATABASE_URL`, `NEXTAUTH_URL`, `NEXTAUTH_SECRET`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `NEXT_PUBLIC_AUTH_ENABLED`, `LOG_FORMAT`, `HF_TOKEN` (download script), `MAX_CLIP_UPLOAD_MB` / `FULL_VIDEO_MAX_SEGMENTS` (in `config.py` but absent from `.env.example`).
- A `.env` file exists at the repo root and is gitignored (`.gitignore`); its contents were not read.

---

## 8. Tests

**Backend only: pytest 8.3 + pytest-asyncio in auto mode** (`backend/requirements-dev.txt`, `backend/pytest.ini`). **209 test functions across 17 files.**

| Area | File | Tests |
|---|---|---|
| Generation API (all 6 kinds, validation, credits) | `tests/test_generation_api.py` | 33 |
| Script parser grammar | `tests/test_script_parser.py` | 19 |
| Upload/text validation | `tests/test_validation.py` | 18 |
| FFmpeg helpers | `tests/test_ffmpeg.py` | 14 |
| Queue/worker lifecycle | `tests/test_queue.py` | 14 |
| VRAM eviction policy | `tests/test_model_manager.py` | 13 |
| Auth/ownership | `tests/test_auth.py` | 12 |
| Processors (fakes) | `tests/test_processors.py` | 12 |
| Upscale | `tests/test_upscale.py` | 12 |
| Jobs API, full-video, loras, voices, catalog, pause markers, misc API | 7 files | 55 |

Design that makes this testable: the VRAM probe and cache-clear are injectable, so eviction is fully covered without a GPU (`pipelines/manager.py:8-9,60-74`). Seven files drive the real app through `TestClient`.

**Not covered:**
- **No frontend tests at all** — `frontend/package.json` has no test script and no test framework; there are no `.test.*`/`.spec.*` files.
- **No CI**, so nothing runs these automatically.
- Real model inference: pipelines are exercised through fakes; correctness on the GPU is validated by Dockerfile build canaries and manual runs, not tests.
- The premium scaffolds (`wan_s2v_pipeline.py`, `wan_animate_pipeline.py`) raise `NotImplementedError` and have no tests.
- Prisma/NextAuth layer: no tests.

---

## 9. Top 7 Non-Trivial Engineering Decisions

**1. VRAM-measured LRU model manager instead of framework offloading**
`backend/app/pipelines/manager.py`. Pipelines are a 3-state machine (`base.py:1-13`); the manager probes real free VRAM via `torch.cuda.mem_get_info` and evicts least-recently-used pipelines until `vram_peak - resident + reserve` fits (`:134-156`). Solves: seven models totalling far more than 44 GB sharing one card, with no job ever OOMing on admission. Rejected: static assumptions — the file states VRAM "is measured, not assumed" (`:8`), and the probe is injected so the policy is testable GPU-free. Also rejected implicitly: a naive `gc`-free `empty_cache()`, because diffusers pipelines hold reference cycles and eviction would free almost nothing without an explicit `gc.collect()` first (`:37-50`).

**2. Per-pipeline offload policy, with `unload` chosen over `cpu` for the big models**
`backend/app/config.py:74-88`. S2, FLUX and Kontext drop their weights entirely on eviction; MuseTalk, Wan and Real-ESRGAN park in RAM. The rejected alternative is documented with its failure: parking S2 in RAM while Wan (~24 GB) also had to be resident "swap-froze the 64 GB host"; FLUX at ~34 GB bf16 "would blow the 64 GB host". Reload comes back from warm page cache in under a minute.

**3. Frame-grid quantization for A/V alignment in the full-video assembler**
`backend/app/services/full_video.py:11-14,150-161`. Each segment's video is built to exactly `ceil(audio_duration × 24)/24` seconds and its audio silence-padded to the same figure, so concatenated video and audio align at every cut with zero cumulative drift. Rejected: muxing per segment or trimming to audio length — both accumulate sub-frame error across up to 12 segments. This also enables decision **4**.

**4. One shared `_H264_ARGS` constant so the final concat is a lossless stream copy**
`backend/app/services/ffmpeg.py:18-27,285-289`. Every segment encode uses identical codec parameters at one size/fps, which is precisely the precondition for the concat demuxer's `-c copy`. Rejected: `filter_complex concat`, which would re-encode the whole timeline — the comment states the invariant is "guaranteed for our segments by `_H264_ARGS` at one size/fps".

**5. LoRA trainer isolated in its own venv, driven as a subprocess**
`backend/Dockerfile:112-131`, `backend/app/services/lora_training.py:1-14,239-244`. The trainer needs torch 2.7.1+cu126; the inference stack is pinned to torch 2.4.1 because that is the newest torch with prebuilt mmcv wheels and the runtime image has no nvcc. Rejected: upgrading the inference torch (would force mmcv to build from source) and installing ai-toolkit into the main environment (version conflict). Related micro-decision: the trainer config is written as JSON into a `.yaml` file — "JSON is valid YAML, so ai-toolkit's loader reads it and the backend needs no YAML dependency" (`lora_training.py:12-13`).

**6. Credits as a derived quantity, not a stored balance**
`backend/app/services/credits.py:1-10`, `backend/app/queue/job_store.py:56-63`. Allowance lives in `User.credits`; spend is recomputed as the sum of non-failed job costs, with the price frozen on each `Job` at submission. Consequence: failed jobs refund themselves and price changes never rewrite history. Rejected: a mutable balance column with debit/credit transactions — which would need explicit refund logic on every failure path. Soft-delete preserves this: `status.json` is kept when a job is deleted because "the snapshot IS the spend record" (`routes/jobs.py:287`).

**7. Monkeypatching diffusers' NATIVE attention backend rather than moving torch**
`backend/app/pipelines/wan_pipeline.py:43-76`, reused by both FLUX pipelines (`flux_kontext_pipeline.py:22,58`). diffusers 0.35 passes `enable_gqa` unconditionally to `scaled_dot_product_attention`, which torch <2.5 rejects; the backend is re-registered without it, with the justification that `enable_gqa` is a perf hint and Wan's attention isn't grouped-query. The alternative — bumping torch — was rejected upstream by the mmcv wheel matrix (decision 5). A sibling of the same class: `.to(device)` alone silently upcast bf16 to fp32 in this environment (23 GB → 43 GB, OOM), so every move passes dtype explicitly — evidence preserved as `backend/assets/diag_wan_vram.py` (`wan_pipeline.py:168-178`).

---

## 10. Technical Debt

**Weakest part: the credit/accounting layer, because it spans three trust domains with no transactional boundary.**

1. **Check-then-act race.** `require_credits` reads the balance (`services/credits.py:78`) and the cost is only recorded when `JobStore.add` runs inside `worker.submit`. Between those two points every route `await`s file reads and writes (e.g. `generation.py:126-148`), so two concurrent submissions from one user can both observe the same `spent` value and both pass. There is no reservation, lock, or unique constraint anywhere in that path.
2. **The balance is asserted by the client's token.** `user.credits` and `role` are JWT claims minted by the frontend (`app/api/backend-token/route.ts:26-31`) and trusted verbatim by the backend (`services/auth.py:52-58`). The backend never reads the accounts DB. Anyone holding `API_JWT_SECRET` can mint `role: "admin"`, which is unlimited credits (`credits.py:75-76`) plus visibility of every user's jobs (`routes/status.py:105`) and delete rights on shared LoRAs (`routes/loras.py:46`).
3. **Spend lives on the filesystem.** `credits_spent` sums the in-memory job dict, rehydrated from `outputs/*/status.json` (`job_store.py:95-114`). Clearing the outputs volume resets everyone's spend to zero. The code defends the delete path but not the volume.
4. **Prices are duplicated in two languages.** `frontend/lib/pricing.ts` is an explicit hand-maintained mirror of `services/credits.py` — its own header says "Keep the two in sync when prices change". Nothing enforces it.

**Other debt, in rough order of severity:**

- **Non-reproducible builds.** `MUSETALK_REF=main` and `AI_TOOLKIT_REF=main` (`backend/Dockerfile:21,119`) are moving targets, despite the file's own instruction to "pin these to the exact refs validated on your instance" (`:19`). The build also `sed`s three third-party packages in site-packages (mmdet's version ceiling `:64`, basicsr's torchvision import `:99`, fish-speech's torch pin `:72-73`) and uninstalls whatever provides flash-attn by introspection (`:107-110`). It is well-commented and canary-guarded, but any upstream drift breaks it silently at build time.
- **No CI and no frontend tests.** 209 backend tests exist and nothing runs them automatically; ~2,900 lines of TypeScript UI have zero test coverage.
- **No DB migration history.** Schema changes are applied with `prisma db push` (`README.md:420`), so there is no way to review, replay, or roll back a schema change.
- **Job store scalability.** Every job ever created stays in one process dict; `list_recent` sorts the whole collection per request (`job_store.py:43-54`), `queued_position` re-sorts on every status poll (`:65-76`), and each poll is a 2-second-interval request per active client. Fine for one GPU box, not for growth.
- **Queued work is lost on restart** — `rehydrate()` rewrites anything non-terminal as failed (`job_store.py:106-110`). Correct and honest, but it means a deploy silently kills the backlog.
- **Dead-weight scaffolds.** Three catalog entries (`wan-s2v-14b`, `wan-animate-14b`, plus the untested `wan-a14b`) are listed in the UI but raise `NotImplementedError` or have never run on hardware (`models_catalog.py:39-57`, `wan_s2v_pipeline.py:34-38`). Honestly labelled, but it's shipped surface area that can't be exercised.
- **Duplicated `_image_urls` helper**, byte-identical in `routes/status.py:28-37` and `routes/jobs.py:92-101`.
