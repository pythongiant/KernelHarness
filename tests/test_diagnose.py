from harness.config import RunConfig  # noqa: F401  (keeps config import warm)
from harness.registry import get
from harness.stages.diagnose import diagnose


def _profile(**over):
    p = {
        "sol_pct": 31.0, "bound": "memory", "occupancy_pct": 25.0,
        "regs_per_thread": 40, "smem_bytes": 0, "grid": 4096, "block": 256,
        "eff_bw_gbps": 1000.0, "peak_bw_gbps": 3350.0, "bytes_moved": 12582912,
        "p50_ms": 0.04, "tflops": 0.0, "peak_tflops": 989.0, "flops": 0,
        "occupancy_limiter": "registers", "ncu": {}, "simulated": False,
        "baseline": {"name": "torch", "speedup": 1.2},
        "numeric": "PASS",
    }
    p.update(over)
    return p


def test_low_occupancy_and_memory_bound_codes():
    k = get("elementwise_affine")
    codes = {b["code"] for b in diagnose(_profile(), None, k,
                                         proof={"status": "VERIFIED"},
                                         feats={"instr_mix": {"ld": 1, "st": 1}})}
    assert codes == {"LOW_OCCUPANCY", "MEM_BW_BOUND"}


def test_tensor_core_and_async_rules_for_gemm():
    k = get("matmul_tiled")
    prof = _profile(bound="compute", sol_pct=12.0, occupancy_pct=75.0, smem_bytes=4096,
                    grid=1024, regs_per_thread=80)
    feats = {"instr_mix": {"ld": 3, "st": 1}, "has_mma": False, "has_async": False,
             "max_stride": 1}
    codes = {b["code"] for b in diagnose(prof, None, k, proof={"status": "VERIFIED"},
                                         feats=feats)}
    assert {"COMPUTE_BOUND", "NO_TENSOR_CORES", "NO_ASYNC_COPY"} <= codes


def test_evidence_quotes_the_metric_that_fired():
    k = get("elementwise_affine")
    rows = diagnose(_profile(occupancy_pct=25.0), None, k, proof={"status": "VERIFIED"},
                    feats={"instr_mix": {}})
    low = [r for r in rows if r["code"] == "LOW_OCCUPANCY"][0]
    assert "occupancy_pct = 25.0%" in low["evidence_md"]
    assert low["severity"] == "high"


def test_unverified_and_baseline_rules():
    k = get("elementwise_affine")
    prof = _profile(sol_pct=80.0, occupancy_pct=90.0,
                    baseline={"name": "none", "speedup": None})
    codes = {b["code"] for b in diagnose(prof, None, k,
                                         proof={"status": "PARTIAL", "open_goals": ["g"]},
                                         feats={"instr_mix": {}})}
    assert "UNVERIFIED" in codes and "BASELINE_UNAVAILABLE" in codes
