"""ModelManager: owns GPU residency for all pipelines.

The worker acquires a pipeline per job; the manager guarantees it is ON_GPU with
enough free VRAM before yielding it, evicting idle pipelines least-recently-used
first. Nothing is moved when a job finishes — the most recently used pipelines
stay hot on the GPU, and eviction only ever happens on demand.

VRAM is measured (torch.cuda.mem_get_info), not assumed; the probe is injectable
so the eviction policy is fully testable without a GPU.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import time
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager

from app.pipelines.base import ManagedPipeline, PipelineState

log = logging.getLogger(__name__)


class InsufficientVRAMError(RuntimeError):
    """Raised when the GPU cannot hold a pipeline even after evicting everything else."""


def _torch_free_vram_gb() -> float:
    import torch

    free_bytes, _total = torch.cuda.mem_get_info()
    return free_bytes / 1024**3


def _torch_clear_cuda_cache() -> None:
    # Diffusers pipelines contain reference cycles, so dropping the last
    # reference in unload() does NOT free their tensors immediately — they wait
    # for the cyclic GC, and empty_cache() can only return memory of tensors
    # that are already collected. Without the explicit collect, evicting a
    # pipeline leaves most of its VRAM occupied by uncollected garbage.
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_cuda_oom(exc: BaseException) -> bool:
    """True for CUDA out-of-memory errors, without requiring torch at import time."""
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    return "out of memory" in str(exc).lower()


class ModelManager:
    def __init__(
        self,
        pipelines: Sequence[ManagedPipeline],
        *,
        vram_reserve_gb: float,
        vram_probe: Callable[[], float] | None = None,
        clear_cuda_cache: Callable[[], None] | None = None,
    ) -> None:
        names = [p.name for p in pipelines]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate pipeline names: {names}")
        self._pipelines: dict[str, ManagedPipeline] = {p.name: p for p in pipelines}
        self._reserve_gb = vram_reserve_gb
        self._probe = vram_probe or _torch_free_vram_gb
        self._clear_cache = clear_cuda_cache or _torch_clear_cuda_cache
        self._lock = asyncio.Lock()

    def free_vram_gb(self) -> float:
        return self._probe()

    @asynccontextmanager
    async def acquire(self, name: str) -> AsyncIterator[ManagedPipeline]:
        """Yield the named pipeline loaded and ON_GPU with peak headroom available.

        Held for the duration of a generation. The lock serialises access as
        defense in depth — the single worker is already sequential. On exit the
        pipeline is left ON_GPU so back-to-back jobs of one kind pay no moves.
        """
        pipeline = self._pipelines.get(name)
        if pipeline is None:
            raise KeyError(f"unknown pipeline {name!r}; have {sorted(self._pipelines)}")

        async with self._lock:
            if pipeline.state is PipelineState.UNLOADED:
                await self._transition(pipeline, "load", PipelineState.ON_CPU, reason="first use")
            await self._ensure_headroom(pipeline)
            if pipeline.state is PipelineState.UNLOADED:
                # the headroom pass self-evicted an "unload"-policy target
                await self._transition(pipeline, "load", PipelineState.ON_CPU, reason="reload")
            if pipeline.state is not PipelineState.ON_GPU:
                await self._transition(pipeline, "to_gpu", PipelineState.ON_GPU, reason="acquired")
            pipeline.last_used_at = time.monotonic()
            yield pipeline

    async def release_gpu(self) -> None:
        """Fully unload every pipeline — training jobs need the whole card, and
        parking weights in system RAM would squeeze the trainer's own host
        allocations. Reloads afterwards come back from warm page cache."""
        async with self._lock:
            for pipeline in self._pipelines.values():
                if pipeline.state is not PipelineState.UNLOADED:
                    await self._transition(
                        pipeline, "unload", PipelineState.UNLOADED, reason="gpu released"
                    )
            self._clear_cache()

    async def after_job_failure(self, exc: BaseException) -> None:
        """GPU hygiene after any failed job: drop cached allocations, and after an
        OOM offload every resident pipeline so the next job starts clean."""
        async with self._lock:
            self._clear_cache()
            if not (is_cuda_oom(exc) or isinstance(exc, InsufficientVRAMError)):
                return
            log.warning("GPU memory exhausted — offloading all resident pipelines")
            for pipeline in self._pipelines.values():
                if pipeline.state is PipelineState.ON_GPU:
                    await self._evict(pipeline, reason="post-oom cleanup")

    async def shutdown(self) -> None:
        async with self._lock:
            for pipeline in self._pipelines.values():
                if pipeline.state is not PipelineState.UNLOADED:
                    await self._transition(
                        pipeline, "unload", PipelineState.UNLOADED, reason="shutdown"
                    )
            self._clear_cache()

    async def _ensure_headroom(self, target: ManagedPipeline) -> None:
        while self._probe() < self._needed_free_gb(target):
            victim = self._pick_victim(exclude=target.name)
            if victim is None:
                # Last resort before failing the job: stray tensors may be held
                # by uncollected garbage (cycles, a failed job's traceback).
                self._clear_cache()
                if self._probe() >= self._needed_free_gb(target):
                    return
                if target.state is PipelineState.ON_GPU:
                    # The resident target holds more than its estimate (KV
                    # caches, fragmentation, leaked activations). Refusing here
                    # would fail every later job until a restart, so evict the
                    # target itself and bring it back fresh.
                    await self._evict(target, reason="self-evict: resident footprint over estimate")
                    continue
                raise InsufficientVRAMError(
                    f"pipeline {target.name!r} needs "
                    f"{self._needed_free_gb(target):.1f} GB free VRAM but only "
                    f"{self._probe():.1f} GB is available with nothing left to evict"
                )
            await self._evict(victim, reason=f"vram_needed_by={target.name}")

    def _needed_free_gb(self, target: ManagedPipeline) -> float:
        # If the target's weights are already resident, only its activation
        # headroom (peak minus weights) must be free.
        weights_resident = (
            target.vram_estimate_gb if target.state is PipelineState.ON_GPU else 0.0
        )
        return target.vram_peak_gb - weights_resident + self._reserve_gb

    def _pick_victim(self, exclude: str) -> ManagedPipeline | None:
        candidates = [
            p
            for p in self._pipelines.values()
            if p.state is PipelineState.ON_GPU and p.name != exclude
        ]
        return min(candidates, key=lambda p: p.last_used_at) if candidates else None

    async def _evict(self, pipeline: ManagedPipeline, reason: str) -> None:
        if pipeline.offload_policy == "cpu":
            await self._transition(pipeline, "to_cpu", PipelineState.ON_CPU, reason=reason)
        else:
            await self._transition(pipeline, "unload", PipelineState.UNLOADED, reason=reason)
        self._clear_cache()

    async def _transition(
        self,
        pipeline: ManagedPipeline,
        method: str,
        new_state: PipelineState,
        reason: str,
    ) -> None:
        old_state = pipeline.state
        started = time.perf_counter()
        await asyncio.to_thread(getattr(pipeline, method))
        pipeline.state = new_state
        log.info(
            "pipeline transition",
            extra={
                "pipeline": pipeline.name,
                "action": method,
                "from": old_state.value,
                "to": new_state.value,
                "took_s": round(time.perf_counter() - started, 2),
                "reason": reason,
                "free_vram_gb": round(self._safe_probe(), 1),
            },
        )

    def _safe_probe(self) -> float:
        try:
            return self._probe()
        except Exception:
            return -1.0
