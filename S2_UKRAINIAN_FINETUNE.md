# Fine-tuning S2 Pro for Ukrainian

S2 Pro reads Ukrainian with Polish-style stress and turns digits into Russian.
The base model has seen too little Ukrainian, and it ignores stress marks
(U+0301) because it never saw them in training. A LoRA fine-tune on Ukrainian
speech is the only fix that changes how the model itself pronounces the language.

This guide covers the whole process: data, patches, training, testing, and loading the result
in the app. Nothing here has been run yet. Each step lists what to check before
moving on.

> **License first.** S2 Pro is free for research and non-commercial use only.
> A monetized channel needs a commercial license from Fish Audio. That applies to
> the base model and to any fine-tune of it.

---

## 0. What we're building

- **Base:** `models/s2-pro` (the checkpoint the app already uses).
- **Method:** LoRA on the attention and MLP layers only. Embeddings and
  output layers stay frozen.
- **Data:** 50–100 h of clean Ukrainian speech, about half of it with stress-marked transcripts.
- **Output:** a merged checkpoint `models/s2-pro-uk`, used automatically for
  voices with `"language": "uk"`.

**Bonus: stress control.** If part of the training text has stress marks, the
fine-tuned model learns what they mean. After that, `обгі́н` in a script actually
changes the stress, which the base model can't do.

Reference: someone did the same for Hebrew. See the
[Hebrew S2 Pro model card](https://huggingface.co/notmax123/Fish-Audio-S2-Pro-He)
and its [fork](https://github.com/maxmelichov/fish-speech) (`tools/hebrew/`).
Their settings are the starting point below.

---

## 1. Two ways to do it

| | A. Upstream fish-speech `v2.0.0-beta` + patches | B. Adapt the Hebrew fork |
|---|---|---|
| Base code | the version already in our Docker image | `maxmelichov/fish-speech` |
| Proven on S2 Pro? | **No.** The training code was written for the older S1 model and needs 3 patches (section 3) | **Yes**, for Hebrew |
| Extra work | patches + checking the training text format | remove the Hebrew-only parts (IPA G2P, `<ipa_*>` tokens) |

**Recommendation:** read `tools/hebrew/README.md` and `run_hebrew_pipeline.sh`
in the fork first. If their pipeline runs with plain text (`--hebrew-mode raw`),
use it with Ukrainian data, which is path B. Otherwise use path A, described below.
The data preparation (section 2) is the same for both.

---

## 2. Data

### Sources (Hugging Face, `speech-uk` organisation)

| Dataset | Utterances | Notes |
|---|---|---|
| `speech-uk/opentts-lada`, `-tetiana`, `-kateryna`, `-mykyta`, `-oleksa` | ~7k each | Studio quality, one speaker each, has a **`transcription_stressed`** column. Apache-2.0. The core of the dataset. |
| `speech-uk/voa` | 326k | Voice of America broadcasts, professional speakers. Columns: `audio`, `duration`, `transcription`. No speaker labels. No license tag, so check the dataset card. |
| `speech-uk/broadcast` | 137k | Same format as `voa`. |
| `speech-uk/cv22` | — | Common Voice 22 (CC0). Gated: accept on the website and use an HF token. Quality varies, so filter hard. |

Columns were checked with the HF datasets server. Recheck if a script fails.

### Target mix (~60–80 h)

- All 5 opentts voices (~35k utterances).
- 40–60k utterances from `voa` + `broadcast`.
- Filter: 2–15 s long, transcript not empty, no digits (or digits converted
  to words, since we want the model to see numbers as words).

### Format fish-speech expects

```
data/
  lada/      0001.flac  0001.lab   0002.flac  0002.lab ...
  tetiana/   ...
  voa_00000/ 0001.flac  0001.lab
```

One folder = one speaker. The dataset loader joins **several clips from the same
folder** into one training sample: one clip acts as the voice reference for the
others, which is how the model learns cloning. So:

- opentts: one folder per voice. Good.
- voa/broadcast: no speaker labels. **One clip per folder** (it teaches the
  language only, not cloning). Don't put mixed speakers in one folder.

### Stress marks

- opentts: use `transcription_stressed` for **50%** of samples and
  `transcription` for the other 50%. Check the format first: if it marks stress with
  `+`, convert it to a combining acute accent (U+0301) after the vowel, the
  same format you'll type in scripts.
- voa/broadcast: add stress automatically to 50% of samples with
  [`ukrainian-word-stress`](https://github.com/lang-uk/ukrainian-word-stress)
  (`pip install ukrainian-word-stress`):

  ```python
  from ukrainian_word_stress import Stressifier, StressSymbol
  stressify = Stressifier(stress_symbol=StressSymbol.CombiningAcuteAccent)
  stressify("Сьогодні маємо дві ситуації")
  ```

  It skips ambiguous words (e.g. за́мок / замо́к), which is fine.

fish-speech's `clean_text` keeps U+0301 (it only converts `’` to `'`,
removes emoji and repeated commas), so the marks reach the model.

### Prep script (sketch)

```python
# prep_uk.py — run on the GPU box, inside the backend container or a venv
import random, re
from pathlib import Path
import soundfile as sf
from datasets import load_dataset, Audio
from ukrainian_word_stress import Stressifier, StressSymbol

OUT = Path("data"); random.seed(0)
stress = Stressifier(stress_symbol=StressSymbol.CombiningAcuteAccent)
DIGITS = re.compile(r"\d")

def keep(row):
    return 2.0 <= row["duration"] <= 15.0 and row["transcription"].strip() and not DIGITS.search(row["transcription"])

def write(folder: Path, idx: int, audio, text: str):
    folder.mkdir(parents=True, exist_ok=True)
    sf.write(folder / f"{idx:06d}.flac", audio["array"], audio["sampling_rate"])
    (folder / f"{idx:06d}.lab").write_text(text.strip(), encoding="utf-8")

for voice in ["lada", "tetiana", "kateryna", "mykyta", "oleksa"]:
    ds = load_dataset(f"speech-uk/opentts-{voice}", split="train").filter(keep)
    for i, row in enumerate(ds):
        text = row["transcription_stressed"] if random.random() < 0.5 else row["transcription"]
        write(OUT / voice, i, row["audio"], text)   # TODO: normalise '+' → U+0301 if needed

for name, limit in [("voa", 40_000), ("broadcast", 15_000)]:
    ds = load_dataset(f"speech-uk/{name}", split="train", streaming=True)
    n = 0
    for row in ds:
        if not keep(row):
            continue
        text = stress(row["transcription"]) if random.random() < 0.5 else row["transcription"]
        write(OUT / f"{name}_{n:06d}", 0, row["audio"], text)
        n += 1
        if n >= limit:
            break
```

Then normalise loudness (recommended by the official guide):

```bash
pip install fish-audio-preprocess
fap loudness-norm data data-norm --clean
```

**Check before going on:** listen to 20 random clips and read their `.lab`
files. If the text doesn't match the audio, the model learns mistakes.

---

## 3. Patches (path A only)

All in `/opt/fish-speech` (or your own clone of `v2.0.0-beta`).

### 3.1 Embedding scale mismatch (required)

S2 Pro has `scale_codebook_embeddings=True`. When generating speech,
`forward_generate()` divides the audio-token embeddings by √(num_codebooks+1) ≈ 3.3.
The training path in `embed()` doesn't. A fine-tune trained without this fix
generates fine for the first word and then falls apart. This is the bug the
Hebrew project found.

`fish_speech/models/text2semantic/llama.py`, in `BaseTransformer.embed()`, change
the end to:

```python
        x = self.embeddings(inp[:, 0]) + vq_embeds_sum

        if self.config.scale_codebook_embeddings:
            x = torch.where(
                is_semantic.unsqueeze(-1).expand_as(x),
                x / math.sqrt(self.config.num_codebooks + 1),
                x,
            )

        return x
```

### 3.2 LoRA targets (required)

`fish_speech/models/text2semantic/lora.py` wraps `model.output`, but S2 Pro's text
model has `tie_word_embeddings: true`, so there is no `output` layer and
`setup_lora` crashes. It also puts LoRA on the embedding tables, which the
tied output can't see. Replace `setup_lora` so it only touches attention and MLP:

```python
def setup_lora(model, lora_config):
    linears = []
    layer_groups = [model.layers] + ([model.fast_layers] if hasattr(model, "fast_layers") else [])
    for layers in layer_groups:
        for layer in layers:
            linears += [(layer.attention, "wqkv"), (layer.attention, "wo"),
                        (layer.feed_forward, "w1"), (layer.feed_forward, "w2"),
                        (layer.feed_forward, "w3")]
    for module, name in linears:
        old = getattr(module, name)
        new = lora.Linear(old.in_features, old.out_features, bias=old.bias is not None,
                          r=lora_config.r, lora_alpha=lora_config.lora_alpha,
                          lora_dropout=lora_config.lora_dropout)
        new.weight.data = old.weight.data   # keep the pretrained weights
        setattr(module, name, new)
    lora.mark_only_lora_as_trainable(model, bias="none")
```

Check that LoRA is applied **after** the pretrained weights are loaded, or that
the loader maps `wqkv.weight` into the wrapped layer. Print the number of trainable
parameters at startup. It should be tens of millions, not billions and not zero.

### 3.3 Training text format (check)

`fish_speech/datasets/semantic.py` builds samples as
`<|speaker:user|> text` → `<|speaker:assistant|> <|voice|> codes`. Compare that
with the prompt `generate_long()` builds when generating speech (in
`fish_speech/models/text2semantic/inference.py`). If they differ, training teaches
a format the app never uses. Make training match inference. The Hebrew fork has
already solved this, so check their dataset code.

### 3.4 LoRA config

Create `fish_speech/configs/lora/r_32_alpha_16.yaml`:

```yaml
_target_: fish_speech.models.text2semantic.lora.LoraConfig
r: 32
lora_alpha: 16
lora_dropout: 0.05
```

---

## 4. Training

### Config overrides (`text2semantic_finetune.yaml`)

| Key | Default | Set to | Why |
|---|---|---|---|
| `pretrained_ckpt_path` | `checkpoints/openaudio-s1-mini` | `checkpoints/s2-pro` | the guide was written for the old model |
| `tokenizer.model_path` | `.../tokenizer.tiktoken` | `${pretrained_ckpt_path}` | S2 Pro ships `tokenizer.json` (loaded via `AutoTokenizer`) |
| `model.optimizer.lr` | `1e-4` | `5e-5` | Hebrew run setting, safer for an RL-trained model |
| `data.batch_size` | 4 | 4 (lower if out of memory) | |
| `trainer.accumulate_grad_batches` | 1 | 3 | effective batch of 12 |
| `trainer.max_steps` | 10000 | 3000 | the Hebrew run stopped at 2,200 and was still improving |
| `trainer.val_check_interval` | 100 | 250 | checkpoint every 250 steps |

### Commands

```bash
# on the L40S (g6e). The 24 GB L4 is too small for this.
cd /opt/fish-speech
ln -s /app/models/s2-pro checkpoints/s2-pro

python tools/vqgan/extract_vq.py data-norm \
    --num-workers 1 --batch-size 16 \
    --config-name modded_dac_vq \
    --checkpoint-path checkpoints/s2-pro/codec.pth

python tools/llama/build_dataset.py \
    --input data-norm --output data/protos \
    --text-extension .lab --num-workers 16

# smoke test first: 20 steps, check loss goes down and nothing crashes
python fish_speech/train.py --config-name text2semantic_finetune \
    project=s2_uk pretrained_ckpt_path=checkpoints/s2-pro \
    +lora@model.model.lora_config=r_32_alpha_16 trainer.max_steps=20

# full run (inside tmux)
python fish_speech/train.py --config-name text2semantic_finetune \
    project=s2_uk pretrained_ckpt_path=checkpoints/s2-pro \
    +lora@model.model.lora_config=r_32_alpha_16
```

Checkpoints land in `results/s2_uk/checkpoints/step_XXXXXX.ckpt`.

---

## 5. Testing checkpoints

Merge a checkpoint and listen. Start with **early** ones (the official guide: the earliest
checkpoint that sounds right usually works best on new text).

```bash
python tools/llama/merge_lora.py \
    --lora-config r_32_alpha_16 \
    --base-weight checkpoints/s2-pro \
    --lora-weight results/s2_uk/checkpoints/step_001000.ckpt \
    --output checkpoints/s2-pro-uk-1000
cp checkpoints/s2-pro/codec.pth checkpoints/s2-pro-uk-1000/   # if merge didn't copy it
```

Make sure the output folder has the same files as `s2-pro` (config, tokenizer,
safetensors, `codec.pth`). Copy any that are missing.

**Test set:** use the same texts every time and compare against the base model:

1. The ПДР intro and solution scripts (with and without stress marks).
2. Numbers written as words: "чотири", "двадцять п'ять", "п'ятнадцять".
3. Words that came out Polish-sounding in the base model.
4. A minimal pair with stress marks: `за́мок` vs `замо́к`.
5. One English sentence. It should still sound English (catches the model forgetting other languages).

What to listen for:
- **Generation collapse** (fades or mumbles after the first word): patch 3.1 is
  missing or wrong.
- **Voice no longer sounds like you** (the Hebrew run found pitch cloning got weaker): try an
  earlier checkpoint, or `lora_alpha: 64` (the Hebrew author's untested suggestion
  α = 2r).
- **Stress marks still ignored:** more stressed data, or train longer.

---

## 6. Loading it in the app

Planned changes (not built yet):

1. `models/s2-pro-uk/`: the chosen merged checkpoint.
2. A second S2 pipeline entry (`s2-uk`) in the model manager, pointing at that
   folder. Same class (`S2Pipeline`), different `checkpoint_dir`.
3. Voice routing: voices with `"language": "uk"` use `s2-uk`, all others
   use `s2`. Only one is on the GPU at a time. The manager already evicts, so
   switching language costs one reload (~30–60 s).
4. Optional: an "auto-stress" checkbox that runs `ukrainian-word-stress` on
   the script before generating, so you only fix the words it gets wrong.

---

## 7. Time and cost estimate

| Step | Where | Rough time |
|---|---|---|
| Download + prep 60–80 h | any instance (CPU) | 1–3 h |
| `extract_vq` | L40S | ~1 h |
| Training 3,000 steps | L40S | several hours (measure step time in the smoke test) |
| Merging + testing 3–4 checkpoints | L40S | ~1 h |

Stop the instance between steps. The data and checkpoints stay on EBS.
