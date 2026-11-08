"""Peak-memory-bandwidth and peak-tensor-throughput table per GPU name."""

from __future__ import annotations

# Bandwidth in GB/s, FP32/TF32/FP16-tensor in TFLOPs (dense, no sparsity).
PEAKS: dict[str, dict] = {
    "H100": {"device": "NVIDIA H100 SXM", "arch": "sm_90a", "sm_count": 132,
             "bw": 3350.0, "tflops_fp32": 67.0, "tflops_tf32": 989.0,
             "tflops_fp16": 1979.0},
    "H200": {"device": "NVIDIA H200 SXM", "arch": "sm_90a", "sm_count": 132,
             "bw": 4800.0, "tflops_fp32": 67.0, "tflops_tf32": 989.0,
             "tflops_fp16": 1979.0},
    "B200": {"device": "NVIDIA B200", "arch": "sm_100a", "sm_count": 148,
             "bw": 8000.0, "tflops_fp32": 80.0, "tflops_tf32": 2200.0,
             "tflops_fp16": 4500.0},
    "A100": {"device": "NVIDIA A100 SXM", "arch": "sm_80", "sm_count": 108,
             "bw": 2039.0, "tflops_fp32": 19.5, "tflops_tf32": 156.0,
             "tflops_fp16": 312.0},
    "L4": {"device": "NVIDIA L4", "arch": "sm_89", "sm_count": 60,
           "bw": 300.0, "tflops_fp32": 30.3, "tflops_tf32": 121.0,
           "tflops_fp16": 242.0},
}

#: Fallbacks for the simulator when a name is not tabulated.
DEFAULT = {"device": "simulated GPU", "arch": "sim", "sm_count": 132, "bw": 3350.0,
           "tflops_fp32": 67.0, "tflops_tf32": 989.0, "tflops_fp16": 1979.0}


def peaks(gpu: str) -> dict:
    return dict(PEAKS.get(gpu.upper(), DEFAULT))


def compute_peak(gpu: str, dtype: str, tensor_cores: bool) -> float:
    p = peaks(gpu)
    if not tensor_cores:
        return p["tflops_fp32"]
    if dtype in ("f16", "bf16"):
        return p["tflops_fp16"]
    return p["tflops_tf32"]
