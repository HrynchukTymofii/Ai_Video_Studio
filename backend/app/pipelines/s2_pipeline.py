"""Fish Audio S2 Pro as a managed pipeline (in-process fish-speech inference).

The fish-speech package (pinned in the Docker image) provides two models:
the dual-AR text-to-semantic transformer and the codec decoder. Both are owned
directly by this wrapper so the ModelManager can move them between devices.
All fish-speech/torch imports are lazy so the backend boots without them.

Long scripts are split at sentence boundaries and synthesized chunk by chunk,
which keeps memory bounded and makes progress reporting meaningful; chunks are
joined with short silences and written as one 44.1 kHz mono WAV.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.config import OffloadPolicy
from app.pipelines.base import ManagedPipeline

log = logging.getLogger(__name__)

# Sentence-boundary split that also swallows the trailing delimiter; CJK-safe.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")
_MAX_CHUNK_CHARS = 400
_INTER_CHUNK_SILENCE_S = 0.25
_OUTPUT_SAMPLE_RATE = 44_100

# Author-controlled silences: [pause] = default, [pause:2] / [pause:1.5] = that
# many seconds (capped). The marker never reaches the TTS model.
_PAUSE_RE = re.compile(r"\[\s*pause(?:\s*[:\s]\s*(\d+(?:\.\d+)?)\s*s?)?\s*\]", re.IGNORECASE)
_DEFAULT_PAUSE_S = 0.6
_MAX_PAUSE_S = 10.0


def parse_pause_markers(text: str) -> list[tuple[str, float]]:
    """Split a script at [pause] markers. Returns (text, silence_after_s) parts;
    the last part always ends with 0.0. Adjacent/leading markers accumulate into
    the surrounding gaps and empty parts are dropped."""
    parts: list[tuple[str, float]] = []
    cursor = 0
    pending_text = ""
    for match in _PAUSE_RE.finditer(text):
        pending_text += text[cursor : match.start()]
        cursor = match.end()
        seconds = min(float(match.group(1) or _DEFAULT_PAUSE_S), _MAX_PAUSE_S)
        if pending_text.strip():
            parts.append((pending_text.strip(), seconds))
            pending_text = ""
        elif parts:
            # marker directly after another: extend the previous gap
            previous_text, previous_pause = parts[-1]
            parts[-1] = (previous_text, min(previous_pause + seconds, _MAX_PAUSE_S))
    pending_text += text[cursor:]
    if pending_text.strip():
        parts.append((pending_text.strip(), 0.0))
    elif parts:
        parts[-1] = (parts[-1][0], 0.0)  # trailing pause is pointless — drop it
    return parts


def split_script(text: str, max_chars: int = _MAX_CHUNK_CHARS) -> list[str]:
    """Split a script into synthesis chunks at sentence boundaries, merging
    short sentences up to max_chars. Oversized single sentences are hard-split."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        while len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sentence[:max_chars])
            sentence = sentence[max_chars:].strip()
        if not sentence:
            continue
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def _cap_context_length(model: Any, max_seq_len: int) -> None:
    """Shrink fish-speech's context window from the checkpoint's 32k tokens.

    The window sizes three GPU-resident allocations that never shrink between
    jobs: the KV cache allocated on first generate (36 layers x 32k ≈ 4.8 GB),
    a 32k x 32k bool causal mask (1 GB) and the RoPE table. Our chunks (≤400
    chars plus a ≤30 s reference ≈ 2-3k tokens) never come close, and on a
    24 GB card that dead weight leaves no headroom for the next job. Must run
    before the first generate(), which sizes the KV cache from config.
    """
    if model.config.max_seq_len <= max_seq_len:
        return
    model.config.max_seq_len = max_seq_len
    # registered (non-persistent) buffers: reassigning updates _buffers
    model.causal_mask = model.causal_mask[:max_seq_len, :max_seq_len].clone()
    model.freqs_cis = model.freqs_cis[:max_seq_len].clone()
    log.info("S2 context window capped", extra={"max_seq_len": max_seq_len})


class S2Pipeline(ManagedPipeline):
    def __init__(
        self,
        checkpoint_dir: Path,
        *,
        offload_policy: OffloadPolicy,
        max_seq_len: int = 8192,
        # Resident = ~9 GB bf16 weights + codec + KV cache sized by max_seq_len
        # (~1.2 GB at 8k); peak adds per-chunk activations and codec decode.
        vram_estimate_gb: float = 11.5,
        vram_peak_gb: float = 14.0,
    ) -> None:
        super().__init__(
            "s2",
            vram_estimate_gb=vram_estimate_gb,
            vram_peak_gb=vram_peak_gb,
            offload_policy=offload_policy,
        )
        self._checkpoint_dir = checkpoint_dir
        self._max_seq_len = max_seq_len
        self._text2semantic: Any = None
        self._decode_one_token: Any = None
        self._decoder: Any = None
        self._device = "cpu"
        # Reference audio encodings are tiny; cache per clip path so repeat jobs
        # with the same voice skip re-encoding.
        self._reference_cache: dict[str, Any] = {}

    # ---- ManagedPipeline ----------------------------------------------------

    def load(self) -> None:
        if self._text2semantic is not None:
            return
        import torch
        from fish_speech.models.dac.inference import load_model as load_decoder
        from fish_speech.models.text2semantic.inference import init_model

        if not self._checkpoint_dir.is_dir():
            raise RuntimeError(
                f"S2 Pro checkpoint not found at {self._checkpoint_dir} — "
                "run scripts/download_models.sh first"
            )
        precision = torch.bfloat16
        # Warm the page cache with plain sequential reads first: the weight
        # files get memory-mapped with random-access madvise, so letting the
        # loader (or the clone below) fault pages in from disk runs at EBS
        # random-IOPS speed (~12 MB/s observed). Sequential reads run at full
        # volume throughput and everything afterwards hits RAM.
        import time

        warm_start = time.perf_counter()
        warmed_bytes = 0
        for path in sorted(self._checkpoint_dir.rglob("*")):
            if path.is_file() and path.suffix in {".safetensors", ".pth"}:
                with open(path, "rb") as fh:
                    while fh.read(32 << 20):
                        pass
                warmed_bytes += path.stat().st_size
        log.info(
            "model files warmed into cache",
            extra={
                "gb": round(warmed_bytes / 1e9, 1),
                "took_s": round(time.perf_counter() - warm_start, 1),
            },
        )
        log.info("loading S2 Pro", extra={"checkpoint": str(self._checkpoint_dir)})
        self._text2semantic, self._decode_one_token = init_model(
            checkpoint_path=str(self._checkpoint_dir),
            device="cpu",
            precision=precision,
            compile=False,
        )
        _cap_context_length(self._text2semantic, self._max_seq_len)
        self._decoder = load_decoder(
            config_name="modded_dac_vq",
            checkpoint_path=str(self._checkpoint_dir / "codec.pth"),
            device="cpu",
        )
        # fish-speech loads safetensors memory-mapped, so weights stay on disk
        # until first touched — which would turn the CPU->GPU move into an
        # IOPS-bound page-fault crawl on EBS. Clone every tensor now to pull
        # the weights into RAM sequentially while we're still in the (already
        # slow-budgeted) load phase.
        import itertools

        for module in (self._text2semantic, self._decoder):
            for tensor in itertools.chain(module.parameters(), module.buffers()):
                tensor.data = tensor.data.clone()
        self._device = "cpu"

    def to_gpu(self) -> None:
        self._move("cuda")

    def to_cpu(self) -> None:
        self._move("cpu")
        import torch

        torch.cuda.empty_cache()

    def unload(self) -> None:
        self._text2semantic = None
        self._decode_one_token = None
        self._decoder = None
        self._reference_cache.clear()
        self._device = "cpu"

    def _move(self, device: str) -> None:
        if self._text2semantic is None:
            raise RuntimeError("S2 pipeline is not loaded")
        self._text2semantic.to(device)
        self._decoder.to(device)
        self._device = device

    # ---- generation ------------------------------------------------------------

    def generate(
        self,
        text: str,
        reference_audio: Path,
        out_path: Path,
        on_progress: Callable[[float], None],
        reference_text: str | None = None,
    ) -> Path:
        """Synthesize `text` in the voice cloned from `reference_audio` and write
        a 44.1 kHz mono WAV to `out_path`. Blocking; run in a worker thread."""
        import numpy as np
        import soundfile as sf

        if self._text2semantic is None or self._device != "cuda":
            raise RuntimeError("S2 pipeline must be ON_GPU before generate()")

        prompt_tokens, prompt_text = self._encode_reference(reference_audio, reference_text)
        # (chunk, silence_after_s): [pause] markers control the gap between
        # parts, sentence groups inside a part get the default short silence.
        plan: list[tuple[str, float]] = []
        for part_text, pause_after in parse_pause_markers(text):
            chunks = split_script(part_text)
            for index, chunk in enumerate(chunks):
                after = _INTER_CHUNK_SILENCE_S if index < len(chunks) - 1 else pause_after
                plan.append((chunk, after))
        if not plan:
            raise ValueError("script contains no speakable text")

        pieces: list[np.ndarray] = []
        for index, (chunk, silence_s) in enumerate(plan):
            pieces.append(self._synthesize_chunk(chunk, prompt_tokens, prompt_text))
            if silence_s > 0 and index < len(plan) - 1:
                pieces.append(
                    np.zeros(int(silence_s * _OUTPUT_SAMPLE_RATE), dtype=np.float32)
                )
            on_progress((index + 1) / len(plan))

        waveform = np.concatenate(pieces)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), waveform, _OUTPUT_SAMPLE_RATE, subtype="PCM_16")
        log.info(
            "speech synthesized",
            extra={"chunks": len(plan), "duration_s": round(len(waveform) / _OUTPUT_SAMPLE_RATE, 1)},
        )
        return out_path

    def _encode_reference(
        self, reference_audio: Path, reference_text: str | None
    ) -> tuple[Any, str]:
        """Encode the voice reference clip to codec tokens (cached per clip)."""
        import torch
        import torchaudio

        cache_key = f"{reference_audio}:{reference_audio.stat().st_mtime_ns}"
        if cache_key in self._reference_cache:
            return self._reference_cache[cache_key], reference_text or ""

        audio, sample_rate = torchaudio.load(str(reference_audio))
        audio = audio.mean(dim=0, keepdim=True)  # downmix to mono
        target_rate = self._decoder.sample_rate
        if sample_rate != target_rate:
            audio = torchaudio.functional.resample(audio, sample_rate, target_rate)
        with torch.inference_mode():
            encoded = self._decoder.encode(audio.to(self._device)[None])
        # encode returns (indices, feature_lengths); generate_long expects each
        # prompt token tensor as 2D (num_codebooks, time), no batch dim.
        if isinstance(encoded, (tuple, list)):
            encoded = encoded[0]
        tokens = encoded[0] if encoded.ndim == 3 else encoded
        self._reference_cache[cache_key] = tokens
        return tokens, reference_text or ""

    def _synthesize_chunk(self, chunk: str, prompt_tokens: Any, prompt_text: str) -> Any:
        """One sentence-group → semantic tokens → waveform (float32 at 44.1 kHz)."""
        import torch
        import torchaudio
        from fish_speech.models.text2semantic.inference import generate_long

        segments: list[Any] = []
        with torch.inference_mode():
            for response in generate_long(
                model=self._text2semantic,
                device=self._device,
                decode_one_token=self._decode_one_token,
                text=chunk,
                prompt_tokens=[prompt_tokens] if prompt_tokens is not None else None,
                prompt_text=[prompt_text] if prompt_text else None,
                max_new_tokens=0,
                top_p=0.8,
                temperature=0.8,
                repetition_penalty=1.1,
                iterative_prompt=True,
            ):
                if response.action == "sample":
                    segments.append(response.codes)
            if not segments:
                raise RuntimeError("S2 produced no audio tokens for a text chunk")
            codes = torch.cat(segments, dim=1)
            # from_indices dequantizes (B, codebooks, time) integer codes and
            # decodes to audio (B, 1, time); decode() alone expects latents.
            decoded = self._decoder.from_indices(codes[None].to(self._device))
            if isinstance(decoded, (tuple, list)):
                decoded = decoded[0]
            waveform = decoded[0]

        waveform = waveform.float().cpu()
        if self._decoder.sample_rate != _OUTPUT_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform, self._decoder.sample_rate, _OUTPUT_SAMPLE_RATE
            )
        return waveform.flatten().numpy()
