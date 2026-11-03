"""ModelManager residency and eviction policy, exercised with dummy pipelines
and a simulated VRAM probe — no GPU or torch required."""

import pytest

from app.pipelines.base import ManagedPipeline, PipelineState
from app.pipelines.manager import InsufficientVRAMError, ModelManager, is_cuda_oom


class DummyPipeline(ManagedPipeline):
    def __init__(self, name: str, estimate: float, peak: float, policy: str = "cpu") -> None:
        super().__init__(
            name,
            vram_estimate_gb=estimate,
            vram_peak_gb=peak,
            offload_policy=policy,  # type: ignore[arg-type]
        )
        self.calls: list[str] = []

    def load(self) -> None:
        self.calls.append("load")

    def to_gpu(self) -> None:
        self.calls.append("to_gpu")

    def to_cpu(self) -> None:
        self.calls.append("to_cpu")

    def unload(self) -> None:
        self.calls.append("unload")


class FakeGPU:
    """Free VRAM = total minus the weight estimates of everything ON_GPU."""

    def __init__(self, total_gb: float, pipelines: list[ManagedPipeline]) -> None:
        self.total_gb = total_gb
        self.pipelines = pipelines
        self.cache_clears = 0

    def free_gb(self) -> float:
        used = sum(
            p.vram_estimate_gb for p in self.pipelines if p.state is PipelineState.ON_GPU
        )
        return self.total_gb - used

    def clear_cache(self) -> None:
        self.cache_clears += 1


def build(total_gb: float, *pipelines: DummyPipeline, reserve: float = 2.0):
    gpu = FakeGPU(total_gb, list(pipelines))
    manager = ModelManager(
        pipelines,
        vram_reserve_gb=reserve,
        vram_probe=gpu.free_gb,
        clear_cuda_cache=gpu.clear_cache,
    )
    return manager, gpu


async def test_lazy_load_then_stays_hot() -> None:
    pipe = DummyPipeline("s2", estimate=10, peak=12)
    manager, _ = build(48, pipe)

    async with manager.acquire("s2") as acquired:
        assert acquired is pipe
        assert pipe.state is PipelineState.ON_GPU
    assert pipe.calls == ["load", "to_gpu"]

    # released but still resident: second acquire moves nothing
    assert pipe.state is PipelineState.ON_GPU
    async with manager.acquire("s2"):
        pass
    assert pipe.calls == ["load", "to_gpu"]


async def test_all_pipelines_coexist_when_vram_allows() -> None:
    s2 = DummyPipeline("s2", 10, 12)
    musetalk = DummyPipeline("musetalk", 6, 8)
    wan = DummyPipeline("wan", 18, 26)
    manager, _ = build(48, s2, musetalk, wan)

    for name in ("s2", "musetalk", "wan"):
        async with manager.acquire(name):
            pass

    # L40S-sized card: everything stays resident, zero evictions
    assert all(p.state is PipelineState.ON_GPU for p in (s2, musetalk, wan))
    assert "to_cpu" not in s2.calls + musetalk.calls + wan.calls


async def test_lru_eviction_when_vram_tight() -> None:
    a = DummyPipeline("a", 10, 12)
    b = DummyPipeline("b", 10, 12)
    manager, _ = build(20, a, b)  # can hold one pipeline's peak, not two

    async with manager.acquire("a"):
        pass
    async with manager.acquire("b"):
        pass
    assert a.state is PipelineState.ON_CPU  # evicted per cpu policy
    assert b.state is PipelineState.ON_GPU

    async with manager.acquire("a"):
        pass
    assert b.state is PipelineState.ON_CPU
    assert a.state is PipelineState.ON_GPU
    assert a.calls == ["load", "to_gpu", "to_cpu", "to_gpu"]  # reloaded from RAM, not disk


async def test_least_recently_used_is_evicted_first() -> None:
    a = DummyPipeline("a", 8, 9)
    b = DummyPipeline("b", 8, 9)
    c = DummyPipeline("c", 8, 9)
    manager, _ = build(20, a, b, c)  # room for two resident, not three

    async with manager.acquire("a"):
        pass
    async with manager.acquire("b"):
        pass
    async with manager.acquire("c"):  # must evict exactly one: a (least recent)
        pass

    assert a.state is PipelineState.ON_CPU
    assert b.state is PipelineState.ON_GPU
    assert c.state is PipelineState.ON_GPU


async def test_unload_policy_frees_instead_of_offloading() -> None:
    a = DummyPipeline("a", 10, 12, policy="unload")
    b = DummyPipeline("b", 10, 12)
    manager, _ = build(20, a, b)

    async with manager.acquire("a"):
        pass
    async with manager.acquire("b"):
        pass
    assert a.state is PipelineState.UNLOADED
    assert a.calls == ["load", "to_gpu", "unload"]

    async with manager.acquire("a"):  # comes back from disk this time
        pass
    assert a.calls == ["load", "to_gpu", "unload", "load", "to_gpu"]


async def test_activation_headroom_when_already_resident() -> None:
    # 28 GB card, 2 GB reserve. wan (18 GB weights, 26 GB peak) fits alone exactly.
    # With musetalk (3 GB) also resident, free = 7 GB — but re-acquiring wan needs
    # activation headroom of 26-18+2 = 10 GB, so musetalk must be evicted even
    # though wan itself never left the GPU.
    wan = DummyPipeline("wan", 18, 26)
    musetalk = DummyPipeline("musetalk", 3, 4)
    manager, _ = build(28, wan, musetalk, reserve=2.0)

    async with manager.acquire("wan"):
        pass
    async with manager.acquire("musetalk"):
        pass
    assert wan.state is PipelineState.ON_GPU and musetalk.state is PipelineState.ON_GPU

    # acquiring wan again: weights resident, but activation headroom requires
    # evicting musetalk
    async with manager.acquire("wan"):
        pass
    assert musetalk.state is PipelineState.ON_CPU
    assert wan.calls.count("to_gpu") == 1  # never left the GPU


async def test_insufficient_vram_raises_clear_error() -> None:
    huge = DummyPipeline("huge", 30, 40)
    manager, _ = build(20, huge)

    with pytest.raises(InsufficientVRAMError, match="huge"):
        async with manager.acquire("huge"):
            pass


class LeakyGPU(FakeGPU):
    """Resident pipelines use `extra_gb` more than their estimate (e.g. a KV cache
    the estimate doesn't cover), released only when they leave the GPU."""

    def __init__(self, total_gb: float, pipelines: list[ManagedPipeline], extra_gb: float) -> None:
        super().__init__(total_gb, pipelines)
        self.extra_gb = extra_gb

    def free_gb(self) -> float:
        resident = [p for p in self.pipelines if p.state is PipelineState.ON_GPU]
        return super().free_gb() - self.extra_gb * len(resident)


@pytest.mark.parametrize("policy", ["cpu", "unload"])
async def test_resident_target_over_estimate_self_evicts_instead_of_failing(policy: str) -> None:
    # 22 GB card (L4-class): s2 fits on first acquire, but once resident it holds
    # 8 GB more than estimated, so the second acquire lacks activation headroom
    # with nothing else to evict. It must recover, not fail until a restart.
    s2 = DummyPipeline("s2", 10, 13, policy=policy)
    gpu = LeakyGPU(22, [s2], extra_gb=8)
    manager = ModelManager(
        [s2], vram_reserve_gb=2.0, vram_probe=gpu.free_gb, clear_cuda_cache=gpu.clear_cache
    )
    for _ in range(3):
        async with manager.acquire("s2") as pipe:
            assert pipe.state is PipelineState.ON_GPU
    assert s2.calls.count("to_gpu") == 3
    if policy == "unload":
        assert s2.calls.count("load") == 3


async def test_after_job_failure_insufficient_vram_offloads_everything() -> None:
    s2 = DummyPipeline("s2", 10, 12)
    manager, _ = build(48, s2)
    async with manager.acquire("s2"):
        pass
    await manager.after_job_failure(InsufficientVRAMError("no room"))
    assert s2.state is PipelineState.ON_CPU


async def test_unknown_pipeline_name() -> None:
    manager, _ = build(48, DummyPipeline("s2", 10, 12))
    with pytest.raises(KeyError, match="nope"):
        async with manager.acquire("nope"):
            pass


async def test_after_job_failure_plain_error_only_clears_cache() -> None:
    pipe = DummyPipeline("s2", 10, 12)
    manager, gpu = build(48, pipe)
    async with manager.acquire("s2"):
        pass

    await manager.after_job_failure(ValueError("some model bug"))
    assert pipe.state is PipelineState.ON_GPU  # not evicted
    assert gpu.cache_clears >= 1


async def test_after_job_failure_oom_offloads_everything() -> None:
    s2 = DummyPipeline("s2", 10, 12)
    wan = DummyPipeline("wan", 18, 26, policy="unload")
    manager, _ = build(48, s2, wan)
    for name in ("s2", "wan"):
        async with manager.acquire(name):
            pass

    await manager.after_job_failure(RuntimeError("CUDA out of memory. Tried to allocate..."))
    assert s2.state is PipelineState.ON_CPU     # cpu policy
    assert wan.state is PipelineState.UNLOADED  # unload policy


async def test_shutdown_unloads_all() -> None:
    s2 = DummyPipeline("s2", 10, 12)
    wan = DummyPipeline("wan", 18, 26)
    manager, _ = build(48, s2, wan)
    for name in ("s2", "wan"):
        async with manager.acquire(name):
            pass

    await manager.shutdown()
    assert s2.state is PipelineState.UNLOADED
    assert wan.state is PipelineState.UNLOADED


def test_is_cuda_oom_detection() -> None:
    assert is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB"))
    assert not is_cuda_oom(ValueError("bad input shape"))

    class OutOfMemoryError(RuntimeError):  # mimics torch.cuda.OutOfMemoryError
        pass

    assert is_cuda_oom(OutOfMemoryError("whatever"))


def test_duplicate_pipeline_names_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        ModelManager(
            [DummyPipeline("x", 1, 1), DummyPipeline("x", 1, 1)],
            vram_reserve_gb=2.0,
            vram_probe=lambda: 48.0,
            clear_cuda_cache=lambda: None,
        )
