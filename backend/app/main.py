"""FastAPI application factory and lifespan wiring."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings, get_settings
from app.logging_config import setup_logging
from app.pipelines.flux_kontext_pipeline import FluxKontextPipeline
from app.pipelines.flux_pipeline import FluxPipeline
from app.pipelines.manager import ModelManager
from app.pipelines.musetalk_pipeline import MuseTalkPipeline
from app.pipelines.s2_pipeline import S2Pipeline
from app.pipelines.upscale_pipeline import UpscalePipeline
from app.pipelines.wan_a14b_pipeline import WanA14BPipeline
from app.pipelines.wan_pipeline import WanPipeline
from app.queue.job_store import JobStore
from app.queue.worker import GPUWorker, JobProcessor
from app.routes import generation as generation_routes
from app.routes import jobs as jobs_routes
from app.routes import loras as loras_routes
from app.routes import models as models_routes
from app.routes import status as status_routes
from app.routes import voices as voices_routes
from app.schemas import JobKind
from app.services.broll import BrollProcessor
from app.services.full_video import FullVideoProcessor
from app.services.image import ImageProcessor
from app.services.lora_training import LoraTrainingProcessor
from app.services.loras import LoraRegistry
from app.services.talking_head import TalkingHeadProcessor
from app.services.upscale import UpscaleProcessor
from app.services.validation import InputValidationError
from app.services.voices import VoiceRegistry

log = logging.getLogger(__name__)


def build_model_manager(settings: Settings) -> ModelManager:
    pipelines = [
        S2Pipeline(
            settings.models_dir / "s2-pro",
            offload_policy=settings.s2_offload,
            max_seq_len=settings.s2_max_seq_len,
        ),
        MuseTalkPipeline(
            settings.models_dir / "musetalk", offload_policy=settings.musetalk_offload
        ),
        WanPipeline(
            settings.models_dir / f"wan2.2-{settings.wan_variant}",
            offload_policy=settings.wan_offload,
        ),
        FluxPipeline(
            settings.models_dir / "flux.1-schnell",
            offload_policy=settings.flux_offload,
        ),
        FluxKontextPipeline(
            settings.models_dir / "flux.1-kontext-dev",
            offload_policy=settings.kontext_offload,
        ),
        UpscalePipeline(
            settings.models_dir / "realesrgan",
            offload_policy=settings.upscale_offload,
        ),
    ]
    if settings.premium_enabled:
        # H100-tier engines; never registered on the L40S (they don't fit).
        pipelines.append(
            WanA14BPipeline(settings.models_dir / "wan2.2-t2v-a14b", offload_policy="unload")
        )
    return ModelManager(pipelines, vram_reserve_gb=settings.vram_reserve_gb)


def build_processors(
    settings: Settings,
    manager: ModelManager,
    voices: VoiceRegistry,
    loras: LoraRegistry,
) -> dict[JobKind, JobProcessor]:
    return {
        JobKind.TALKING_HEAD: TalkingHeadProcessor(manager, voices, settings),
        JobKind.BROLL: BrollProcessor(manager, settings),
        JobKind.IMAGE: ImageProcessor(manager, settings),
        JobKind.FULL_VIDEO: FullVideoProcessor(manager, voices, settings),
        JobKind.LORA_TRAINING: LoraTrainingProcessor(manager, loras, settings),
        JobKind.UPSCALE: UpscaleProcessor(manager, settings),
    }


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    log.info(
        "AI Video Studio backend starting",
        extra={
            "outputs_dir": str(settings.outputs_dir),
            "models_dir": str(settings.models_dir),
            "log_format": settings.log_format,
        },
    )

    store = JobStore(settings.outputs_dir)
    store.rehydrate()

    voices = VoiceRegistry(settings.assets_dir)
    voices.load()

    loras = LoraRegistry(settings.loras_dir)
    manager = build_model_manager(settings)
    worker = GPUWorker(
        store,
        build_processors(settings, manager, voices, loras),
        job_timeout_s=settings.job_timeout_s,
        failure_hook=manager.after_job_failure,
        # Training legitimately runs for hours — it gets its own ceiling.
        timeout_overrides={JobKind.LORA_TRAINING: settings.lora_timeout_s},
    )
    worker.start()

    app.state.store = store
    app.state.worker = worker
    app.state.voices = voices
    app.state.loras = loras
    app.state.model_manager = manager

    yield

    await worker.stop()
    await manager.shutdown()
    log.info("AI Video Studio backend shut down")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_format)
    settings.ensure_dirs()

    app = FastAPI(title="AI Video Studio", lifespan=lifespan)
    app.state.settings = settings

    app.mount("/outputs", StaticFiles(directory=settings.outputs_dir), name="outputs")
    app.include_router(status_routes.router)
    app.include_router(jobs_routes.router)
    app.include_router(voices_routes.router)
    app.include_router(generation_routes.router)
    app.include_router(models_routes.router)
    app.include_router(loras_routes.router)

    @app.exception_handler(InputValidationError)
    async def _invalid_input(request: Request, exc: InputValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
