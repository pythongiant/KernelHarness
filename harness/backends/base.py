"""Backend protocol and the profile schema shared by every backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


PROFILE_KEYS = (
    "variant_id", "gpu", "device_name", "arch", "p50_ms", "p90_ms", "min_ms", "std_ms",
    "reps", "warmup", "bytes_moved", "flops", "eff_bw_gbps", "tflops", "peak_bw_gbps",
    "peak_tflops", "sol_pct", "bound", "regs_per_thread", "smem_bytes",
    "max_active_blocks_per_sm", "occupancy_pct", "occupancy_limiter", "grid", "block",
    "numeric", "max_abs_err", "max_rel_err", "counters", "ncu", "baseline", "simulated",
    "raw_log_path",
)


@dataclass
class Caps:
    gpu: str
    device_name: str
    arch: str
    sm_count: int
    peak_bw_gbps: float
    peak_tflops: float
    ncu: bool = False

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class ProfileJob:
    variant_id: str
    gpu: str
    source: str
    harness_header: str
    shape: dict
    tol: dict
    reps: int
    warmup: int
    baseline: str
    bytes_moved: int
    flops: int
    sol_kind: str
    out_index: int
    buffers: list[str]
    dtype: str
    family: str
    grid: int
    block: int
    extra: dict = field(default_factory=dict)


@dataclass
class ProfileResult:
    data: dict

    def __getitem__(self, k):
        return self.data[k]

    def get(self, k, default=None):
        return self.data.get(k, default)

    @property
    def p50_ms(self) -> float:
        return float(self.data["p50_ms"])

    @property
    def sol_pct(self) -> float:
        return float(self.data["sol_pct"])


class Backend(Protocol):
    def caps(self) -> Caps: ...

    def profile(self, job: ProfileJob) -> ProfileResult: ...
