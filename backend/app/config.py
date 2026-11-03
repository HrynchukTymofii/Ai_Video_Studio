"""Application settings, loaded from environment variables and the project .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKEND_DIR.parent

OffloadPolicy = Literal["cpu", "unload"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Project-root .env is shared with docker-compose; a backend-local .env wins if present.
        env_file=(PROJECT_DIR / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    backend_port: int = 8000

    # ---- auth --------------------------------------------------------------------
    # false = single implicit local user (dev, tests, basic-auth-only deploys).
    # true = every /api request needs a Bearer token minted by the Next.js
    # frontend from the NextAuth session, signed with api_jwt_secret (set the
    # same value as the frontend's API_JWT_SECRET).
    auth_enabled: bool = False
    api_jwt_secret: str = ""

    models_dir: Path = BACKEND_DIR / "models"
    outputs_dir: Path = BACKEND_DIR / "outputs"
    assets_dir: Path = BACKEND_DIR / "assets"

    max_upload_mb: int = 20
    # Uploaded [CLIP: …] footage for full-video jobs can legitimately be large.
    max_clip_upload_mb: int = 200
    # ~20 minutes of spoken English at a natural pace (~900 chars/minute).
    max_script_chars: int = 20_000
    max_prompt_chars: int = 1_000
    # 12 all-b-roll segments is ~90-100 min of diffusion on the L40S — still
    # inside job_timeout_s with margin.
    full_video_max_segments: int = 12

    # Talking head runs at roughly 1-3 minutes of processing per minute of script,
    # so a 20-minute script can legitimately take over an hour.
    job_timeout_s: float = 7_200.0
    recent_jobs_limit: int = 20

    # ---- upscaling (Real-ESRGAN) --------------------------------------------------
    # Inputs are upscaled 4x natively; caps keep output sizes and per-frame
    # video runtimes sane on the L40S.
    upscale_max_image_mp: float = 4.0  # input megapixels (4 MP -> 64 MP at 4x)
    upscale_max_video_s: float = 120.0

    # ---- style LoRA training (ostris/ai-toolkit on the Wan2.2 5B base) ----------
    lora_min_images: int = 5
    lora_max_images: int = 60
    lora_default_steps: int = 2000
    lora_max_steps: int = 4000
    # ~1.5-3 s/step on the L40S: 2000 steps ≈ 1-2 h, plus dataset caching.
    lora_timeout_s: float = 21_600.0
    # The trainer lives in its own venv — its torch is newer than the inference
    # stack's (see backend/Dockerfile trainer stage).
    trainer_python: Path = Path("/opt/trainer-venv/bin/python")
    trainer_script: Path = Path("/opt/ai-toolkit/run.py")

    wan_variant: str = "ti2v-5b"
    vram_reserve_gb: float = 2.0
    # S2 evicted-from-GPU weights are dropped, not parked in RAM: the animate
    # flow needs Wan (~24 GB) in RAM at the same moment S2 is evicted, and the
    # pair swap-froze the 64 GB host. Reload from warm page cache is <1 min.
    s2_offload: OffloadPolicy = "unload"
    # fish-speech context window (checkpoint default 32k). Sizes the resident KV
    # cache + causal mask; 8k is ample for our ≤400-char chunks.
    s2_max_seq_len: int = 8192
    musetalk_offload: OffloadPolicy = "cpu"
    wan_offload: OffloadPolicy = "cpu"
    # Real-ESRGAN is ~100 MB total — parking it in RAM is free.
    upscale_offload: OffloadPolicy = "cpu"
    # FLUX is ~34 GB in bf16 — parking it in CPU RAM next to Wan would blow the
    # 64 GB host, so it drops to disk when evicted (page cache keeps reloads fast).
    flux_offload: OffloadPolicy = "unload"
    # Kontext is FLUX-dev-sized (~34 GB bf16) — same RAM constraint as FLUX.
    kontext_offload: OffloadPolicy = "unload"
    # Premium (H100-tier) engines: registers their pipelines and lists them as
    # selectable. Leave false on the L40S — the 14B/A14B models don't fit.
    premium_enabled: bool = False

    log_format: Literal["pretty", "json"] = "pretty"

    @field_validator("models_dir", "outputs_dir", "assets_dir")
    @classmethod
    def _resolve_relative_to_backend(cls, value: Path) -> Path:
        return value if value.is_absolute() else (BACKEND_DIR / value).resolve()

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def max_clip_upload_bytes(self) -> int:
        return self.max_clip_upload_mb * 1024 * 1024

    @property
    def voices_dir(self) -> Path:
        return self.assets_dir / "voices"

    @property
    def loras_dir(self) -> Path:
        # Trained style adapters live with the model weights, not the job outputs:
        # they survive output cleanups and ship on the same volume.
        return self.models_dir / "loras"

    def ensure_dirs(self) -> None:
        for directory in (
            self.models_dir,
            self.outputs_dir,
            self.assets_dir,
            self.voices_dir,
            self.loras_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
