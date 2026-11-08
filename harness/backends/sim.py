"""Deterministic cost-model backend (offline development, tests, CI).

Mirrors the real profile schema exactly. `simulated: true` means the SLA gate can
never pass (`measured` check in the orchestrator).
"""

from __future__ import annotations

from .base import Caps, ProfileJob, ProfileResult
from .peaks import compute_peak, peaks

#: Fraction of peak achieved per family/level before any transform.
LEVEL_EFFICIENCY: dict[str, list[float]] = {
    "elementwise": [0.10, 0.34, 0.52, 0.71, 0.80],
    "gemm": [0.02, 0.08, 0.24, 0.46, 0.58],
    "attention": [0.01, 0.06, 0.18, 0.33],
}

REG_PENALTY = {0: 1.0}


class SimBackend:
    def __init__(self, gpu: str = "H100", latency_scale: float = 1.0):
        self.gpu = gpu.upper()
        self.scale = latency_scale
        self._caps: Caps | None = None

    def caps(self) -> Caps:
        p = peaks(self.gpu)
        return Caps(gpu=self.gpu, device_name=p["device"] + " (simulated)",
                    arch=p["arch"], sm_count=int(p["sm_count"]),
                    peak_bw_gbps=float(p["bw"]),
                    peak_tflops=float(p["tflops_tf32"]), ncu=False)

    def _efficiency(self, job: ProfileJob) -> float:
        level = int(job.extra.get("level", 0))
        table = LEVEL_EFFICIENCY.get(job.extra.get("family", "elementwise"), [0.1])
        base = table[min(level, len(table) - 1)]
        if "vectorize_x4" in job.extra.get("transforms", []):
            base = min(0.92, base * 1.9)
        if "streaming_stores" in job.extra.get("transforms", []):
            base = min(0.95, base * 1.08)
        if "cp_async_double_buffer" in job.extra.get("transforms", []):
            base = min(0.9, base * 1.35)
        if "mma_sync" in job.extra.get("transforms", []):
            base = min(0.85, base * 2.4)
        return base

    def profile(self, job: ProfileJob) -> ProfileResult:
        caps = self.caps()
        eff = self._efficiency(job)
        peak_tflops = compute_peak(self.gpu, job.dtype, bool(
            {"mma_sync", "wgmma"} & set(job.extra.get("transforms", []))))
        bw_time = (job.bytes_moved / 1e9) / (caps.peak_bw_gbps * eff) * 1e3 if job.bytes_moved else 0.0
        flop_time = (job.flops / 1e12) / (peak_tflops * eff) * 1e3 if job.flops else 0.0
        p50 = max(bw_time, flop_time, 1e-4) * self.scale
        intensity = (job.flops / job.bytes_moved) if job.bytes_moved else 0.0
        bound = "compute" if intensity > (peak_tflops * 1e12) / (caps.peak_bw_gbps * 1e9) else "memory"
        eff_bw = (job.bytes_moved / 1e9) / (p50 / 1e3) if job.bytes_moved else 0.0
        tflops = (job.flops / 1e12) / (p50 / 1e3) if job.flops else 0.0
        sol = max(eff_bw / caps.peak_bw_gbps if job.bytes_moved else 0.0,
                  tflops / peak_tflops if job.flops else 0.0) * 100.0
        occupancy = min(100.0, 12.5 * max(1, int(job.extra.get("level", 0)) + 2))
        data = {
            "variant_id": job.variant_id,
            "gpu": self.gpu,
            "device_name": caps.device_name,
            "arch": caps.arch,
            "p50_ms": round(p50, 6),
            "p90_ms": round(p50 * 1.08, 6),
            "min_ms": round(p50 * 0.94, 6),
            "std_ms": round(p50 * 0.03, 6),
            "reps": job.reps,
            "warmup": job.warmup,
            "bytes_moved": job.bytes_moved,
            "flops": job.flops,
            "eff_bw_gbps": round(eff_bw, 2),
            "tflops": round(tflops, 3),
            "peak_bw_gbps": caps.peak_bw_gbps,
            "peak_tflops": peak_tflops,
            "sol_pct": round(sol, 2),
            "bound": bound,
            "regs_per_thread": 32 + 8 * int(job.extra.get("level", 0)),
            "smem_bytes": 0 if job.extra.get("family") != "gemm" else 1024 * (2 ** int(job.extra.get("level", 0))),
            "max_active_blocks_per_sm": 4,
            "occupancy_pct": round(occupancy, 1),
            "occupancy_limiter": "simulated",
            "grid": job.grid,
            "block": job.block,
            "numeric": "SKIPPED",
            "max_abs_err": None,
            "max_rel_err": None,
            "counters": "unavailable",
            "ncu": None,
            "baseline": {"name": job.baseline, "p50_ms": round(p50 * 2.4, 6),
                         "speedup": 1.0 if job.baseline != "none" else None},
            "simulated": True,
            "raw_log_path": None,
        }
        return ProfileResult(data)
