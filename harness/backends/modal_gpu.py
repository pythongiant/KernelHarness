"""Modal GPU backend: H100/B200 profiling with optional `ncu` counters.

Deploy once with `uv run modal deploy harness/backends/modal_gpu.py`. The local
backend calls the deployed functions by name, so a run does not need to re-upload
the image.

`probe()` decides once whether Nsight Compute counters work in the sandbox; when
they do not, profiles carry `counters: "unavailable"` and the diagnosis falls back
to analytical rules and static PTX analysis.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..config import STATE_DIR
from ..runner.gen import bench_driver
from .base import Caps, ProfileJob, ProfileResult
from .peaks import compute_peak, peaks

APP_NAME = "kernel-harness"
GPU_FUNCTIONS = {"H100": "run_job", "B200": "run_job_b200", "H200": "run_job_h200",
                 "L4": "run_job_l4", "A100": "run_job_a100"}
PROBE_FUNCTIONS = {"H100": "probe", "B200": "probe_b200", "H200": "probe_h200",
                   "L4": "probe_l4", "A100": "probe_a100"}


def _caps_path(gpu: str) -> Path:
    return STATE_DIR / f"caps-{gpu.upper()}.json"


SM_THREADS = 2048
SM_REGISTERS = 65536
SM_SMEM_BYTES = 228 * 1024
SM_BLOCK_LIMIT = 32


def _occupancy(regs: int, smem: int, block: int) -> tuple[float, str]:
    """Warp-occupancy ceiling from registers, shared memory and block size.

    A kernel launched via a wrapper cannot be queried with `cudaOccupancy*` without its
    function pointer, so the ceiling is computed analytically from `ptxas` numbers.
    """
    block = max(1, block)
    by_threads = min(SM_THREADS // block, SM_BLOCK_LIMIT)
    by_regs = (SM_REGISTERS // (max(1, regs) * block)) if regs else by_threads
    by_smem = (SM_SMEM_BYTES // smem) if smem else by_threads
    blocks = max(1, min(by_threads, by_regs, by_smem))
    if blocks == by_regs and by_regs < by_threads and by_regs <= by_smem:
        limiter = "registers"
    elif blocks == by_smem and by_smem < by_threads:
        limiter = "shared memory"
    else:
        limiter = "threads (block size)"
    return round(100.0 * blocks * block / SM_THREADS, 1), limiter


class ModalBackend:
    def __init__(self, gpu: str = "H100"):
        self.gpu = gpu.upper()
        self._caps: Caps | None = None

    # -- caps ------------------------------------------------------------
    def caps(self) -> Caps:
        if self._caps is not None:
            return self._caps
        cached = _caps_path(self.gpu)
        if cached.exists():
            try:
                data = json.loads(cached.read_text())
                self._caps = Caps(**{k: data[k] for k in
                                     ("gpu", "device_name", "arch", "sm_count",
                                      "peak_bw_gbps", "peak_tflops", "ncu")})
                return self._caps
            except Exception:
                pass
        try:
            data = self._call("probe", {})
            self._caps = Caps(
                gpu=self.gpu, device_name=data["device_name"], arch=data["arch"],
                sm_count=int(data["sm_count"]), peak_bw_gbps=float(data["peak_bw_gbps"]),
                peak_tflops=float(data["peak_tflops"]), ncu=bool(data["ncu"]))
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(data, indent=2))
            return self._caps
        except Exception:
            p = peaks(self.gpu)
            return Caps(gpu=self.gpu, device_name=p["device"] + " (unprobed)",
                        arch=p["arch"], sm_count=int(p["sm_count"]),
                        peak_bw_gbps=float(p["bw"]), peak_tflops=float(p["tflops_tf32"]),
                        ncu=False)

    # -- transport -------------------------------------------------------
    def _call(self, kind: str, payload: dict) -> dict:
        import modal

        if kind == "probe":
            fn = modal.Function.from_name(APP_NAME, PROBE_FUNCTIONS[self.gpu])
        else:
            name = GPU_FUNCTIONS.get(self.gpu)
            if name is None:
                raise ValueError(f"no Modal function for GPU {self.gpu}")
            fn = modal.Function.from_name(APP_NAME, name)
        return fn.remote(json.dumps(payload))

    # -- profile ---------------------------------------------------------
    def profile(self, job: ProfileJob) -> ProfileResult:
        caps = self.caps()
        payload = {
            "variant_id": job.variant_id,
            "source": job.source,
            "harness_header": job.harness_header,
            "reps": job.reps,
            "warmup": job.warmup,
            "baseline": job.baseline,
            "dtype": job.dtype,
            "bytes_moved": job.bytes_moved,
            "flops": job.flops,
            "sol_kind": job.sol_kind,
            "grid": job.grid,
            "block": job.block,
            "out_index": job.out_index,
            "buffers": job.buffers,
            "shape": job.shape,
            "tol": job.tol,
            "ncu": caps.ncu,
            "driver": bench_driver(job.source, job.variant_id),
            "tensor_cores": bool({"mma_sync", "wgmma"} & set(job.extra.get("transforms", []))),
        }
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                raw = self._call("job", payload)
                if raw.get("error"):
                    raise RuntimeError(f"modal job error: {raw['error']} {raw.get('log', '')[:800]}")
                data = self._shape_profile(raw, job, caps)
                self._apply_numeric(raw, job, data)
                return ProfileResult(data)
            except Exception as exc:  # transient Modal/network errors only
                last_err = exc
                if attempt < 2:
                    time.sleep(30)
        raise RuntimeError(f"modal job failed after 3 attempts: {last_err}")

    def _apply_numeric(self, raw: dict, job: ProfileJob, data: dict) -> None:
        """Compare the downloaded output buffer with the numpy reference."""
        import base64

        import numpy as np

        from ..runner import reference

        blob = raw.get("out_b64")
        if not blob:
            data["numeric"] = "SKIPPED"
            return
        if job.dtype in ("s32", "i32"):
            got = np.frombuffer(base64.b64decode(blob), dtype="<i4")
        else:
            got = np.frombuffer(base64.b64decode(blob), dtype="<f4")
        want = reference.reference_for_job(job)
        status, abs_err, rel_err = reference.compare_for_job(job, got, )
        data["numeric"] = status
        data["max_abs_err"] = abs_err
        data["max_rel_err"] = rel_err

    def _shape_profile(self, raw: dict, job: ProfileJob, caps: Caps) -> dict:
        p50 = float(raw["p50_ms"])
        peak_tflops = compute_peak(self.gpu, job.dtype, bool(raw.get("tensor_cores")))
        eff_bw = (job.bytes_moved / 1e9) / (p50 / 1e3) if job.bytes_moved and p50 > 0 else 0.0
        tflops = (job.flops / 1e12) / (p50 / 1e3) if job.flops and p50 > 0 else 0.0
        sol = max(eff_bw / caps.peak_bw_gbps if job.bytes_moved else 0.0,
                  tflops / peak_tflops if job.flops else 0.0) * 100.0
        intensity = (job.flops / job.bytes_moved) if job.bytes_moved else 0.0
        knee = (peak_tflops * 1e12) / (caps.peak_bw_gbps * 1e9)
        regs = int(raw.get("regs_per_thread") or 0)
        smem = int(raw.get("smem_bytes") or 0)
        occupancy, limiter = _occupancy(regs, smem, job.block)
        if raw.get("occupancy_pct") is not None:
            occupancy = float(raw["occupancy_pct"])
        baseline = raw.get("baseline") or {"name": job.baseline, "p50_ms": None, "speedup": None}
        if baseline.get("p50_ms"):
            baseline["speedup"] = round(float(baseline["p50_ms"]) / p50, 4) if p50 > 0 else None
        return {
            "variant_id": job.variant_id,
            "gpu": self.gpu,
            "device_name": caps.device_name,
            "arch": caps.arch,
            "p50_ms": p50,
            "p90_ms": float(raw.get("p90_ms") or p50),
            "min_ms": float(raw.get("min_ms") or p50),
            "std_ms": float(raw.get("std_ms") or 0.0),
            "reps": int(raw.get("reps") or job.reps),
            "warmup": int(raw.get("warmup") or job.warmup),
            "bytes_moved": job.bytes_moved,
            "flops": job.flops,
            "eff_bw_gbps": round(eff_bw, 2),
            "tflops": round(tflops, 3),
            "peak_bw_gbps": caps.peak_bw_gbps,
            "peak_tflops": peak_tflops,
            "sol_pct": round(sol, 2),
            "bound": "compute" if intensity > knee else "memory",
            "regs_per_thread": regs,
            "smem_bytes": smem,
            "max_active_blocks_per_sm": int(raw.get("max_active_blocks_per_sm") or 0),
            "occupancy_pct": float(occupancy if occupancy is not None else 0.0),
            "occupancy_limiter": raw.get("occupancy_limiter") or limiter,
            "grid": job.grid,
            "block": job.block,
            "numeric": raw.get("numeric", "SKIPPED"),
            "max_abs_err": raw.get("max_abs_err"),
            "max_rel_err": raw.get("max_rel_err"),
            "counters": "available" if raw.get("ncu") else "unavailable",
            "ncu": raw.get("ncu_counters"),
            "baseline": baseline,
            # Artifact channel (not a metric): the compiled PTX, vendored into the ledger.
            "ptx": raw.get("ptx") or "",
            "simulated": False,
            "raw_log_path": raw.get("log_path"),
        }


# ---------------------------------------------------------------------------
# Modal application (deployed once with `uv run modal deploy <this file>`)
# ---------------------------------------------------------------------------

import base64  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402

import modal  # noqa: E402

app = modal.App(APP_NAME)

# `nsight-compute` is not in the plain Ubuntu archive: it comes from NVIDIA's repo.
# The install is best-effort (`|| true`) because counters are optional — `probe()`
# records `ncu: available|unavailable` and the diagnosis degrades to analytical rules.
IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "curl", "ca-certificates", "gnupg")
    .run_commands(
        "curl -fsSL -o /tmp/cuda-keyring.deb "
        "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/"
        "cuda-keyring_1.1-1_all.deb && dpkg -i /tmp/cuda-keyring.deb && apt-get update && "
        "(apt-get install -y nsight-compute || echo 'ncu unavailable')"
    )
    .pip_install("torch", "numpy", "pydantic")
)

_ARCH = {"H100": "sm_90a", "H200": "sm_90a", "B200": "sm_100a", "A100": "sm_80", "L4": "sm_89"}


def _probe_impl(gpu: str) -> dict:
    import subprocess

    name = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,clocks.max.memory,clocks.max.sm",
         "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    parts = [p.strip() for p in name.split(",")]
    device = parts[0] if parts else "unknown"
    clock_mhz = float(parts[2]) if len(parts) > 2 and parts[2].replace('.', '', 1).isdigit() else 0.0
    p = peaks(gpu)
    # bus width is not exposed by nvidia-smi; use the tabulated peak bandwidth.
    bw = p["bw"]
    ncu_ok = False
    try:
        src = "#include <cstdio>\n__global__ void k(){}\nint main(){k<<<1,1>>>();return 0;}\n"
        open("/tmp/p.cu", "w").write(src)
        subprocess.run(["nvcc", "-arch", _ARCH.get(gpu, "sm_90a"), "/tmp/p.cu", "-o", "/tmp/p"],
                       check=True, capture_output=True)
        r = subprocess.run(["ncu", "--csv", "--metrics", "gpu__time_duration.sum",
                            "/tmp/p"], capture_output=True, text=True)
        ncu_ok = r.returncode == 0 and "gpu__time_duration" in (r.stdout or "")
    except Exception:
        ncu_ok = False
    return {
        "gpu": gpu,
        "device_name": device,
        "arch": _ARCH.get(gpu, "sm_90a"),
        "sm_count": p["sm_count"],
        "peak_bw_gbps": bw,
        "peak_tflops": p["tflops_fp16"],
        "memory_clock_mhz": clock_mhz,
        "ncu": ncu_ok,
    }


def _run_job_impl(payload: dict) -> dict:
    import json
    import subprocess
    import tempfile
    from pathlib import Path

    gpu = payload["gpu"]
    arch = _ARCH.get(gpu, "sm_90a")
    work = Path(tempfile.mkdtemp(prefix="harness-"))
    (work / "harness_gen.h").write_text(payload["harness_header"])
    (work / "kernel.cu").write_text(payload["source"])
    driver = payload["driver"]
    (work / "bench_main.cu").write_text(driver)

    compile_cmd = ["nvcc", "-O3", f"-arch={arch}", "-lineinfo", "-Xptxas", "-v",
                   "bench_main.cu", "-o", "bench", "-lm"]
    comp = subprocess.run(compile_cmd, cwd=work, capture_output=True, text=True)
    if comp.returncode != 0:
        return {"error": "compile failed", "log": comp.stderr[-6000:]}
    ptxas = comp.stderr
    regs, smem = 0, 0
    for line in ptxas.splitlines():
        if "Used" in line and "registers" in line:
            try:
                regs = int(line.split("Used")[1].split("registers")[0].strip().split()[0])
            except Exception:
                pass
        if "bytes smem" in line:
            try:
                smem = int(line.split("bytes smem")[0].strip().split()[-1])
            except Exception:
                pass

    # The variant's PTX artifact: compile the kernel together with the generated
    # header (the source alone does not know `HarnessParams`).
    (work / "kernel_tu.cu").write_text('#include "harness_gen.h"\n' + payload["source"])
    subprocess.run(["nvcc", f"-arch={arch}", "-ptx", "kernel_tu.cu", "-o", "kernel.ptx"],
                   cwd=work, capture_output=True, text=True)
    ptx_text = (work / "kernel.ptx").read_text() if (work / "kernel.ptx").exists() else ""

    run = subprocess.run(["./bench", "out.bin", str(payload["warmup"]), str(payload["reps"])],
                         cwd=work, capture_output=True, text=True, timeout=1800)
    if run.returncode != 0:
        return {"error": f"run failed (exit {run.returncode})",
                "log": ("STDOUT:\n" + run.stdout + "\nSTDERR:\n" + run.stderr)[-6000:],
                "compile_log": comp.stderr[-2000:], "ptx": ptx_text}
    result = {}
    for line in run.stdout.splitlines():
        if line.startswith("HARNESS_RESULT "):
            result = json.loads(line[len("HARNESS_RESULT "):])
    if not result:
        return {"error": "no result line", "log": run.stdout[-4000:], "ptx": ptx_text}

    counters = None
    if payload.get("ncu"):
        metrics = ",".join([
            "gpu__time_duration.sum", "dram__bytes.sum",
            "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
            "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
            "sm__warps_active.avg.pct_of_peak_sustained_active"])
        ncur = subprocess.run(
            ["ncu", "--csv", "--metrics", metrics, "./bench", "out_ncu.bin",
             str(payload["warmup"]), str(payload["reps"])],
            cwd=work, capture_output=True, text=True, timeout=1800)
        vals = {}
        for line in ncur.stdout.splitlines():
            parts = [p.strip().strip('"') for p in line.split(",")]
            if len(parts) >= 4 and parts[-2].startswith(("gpu__", "dram__", "l1tex__", "smsp__", "sm__")):
                try:
                    vals[parts[-2]] = float(parts[-1])
                except ValueError:
                    pass
        if vals:
            counters = {
                "sectors_per_request": vals.get(
                    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio"),
                "long_scoreboard_pct": vals.get(
                    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"),
                "warps_active_pct": vals.get("sm__warps_active.avg.pct_of_peak_sustained_active"),
                "dram_bytes": vals.get("dram__bytes.sum"),
            }

    baseline = None
    try:
        import torch
        baseline = _torch_baseline(payload)
    except Exception as exc:  # baseline is optional; the gate fails closed
        baseline = {"name": "torch", "p50_ms": None, "error": str(exc)[:200]}

    out_bytes = (work / "out.bin").read_bytes() if (work / "out.bin").exists() else b""
    return {
        **result,
        "regs_per_thread": regs,
        "smem_bytes": smem,
        "tensor_cores": bool(payload.get("tensor_cores")),
        "ncu": counters is not None,
        "ncu_counters": counters,
        "baseline": baseline,
        "ptx": ptx_text,
        "out_b64": base64.b64encode(out_bytes).decode(),
        "log_path": str(work),
    }


def _torch_baseline(payload: dict) -> dict:
    """Time the equivalent torch operation on the same GPU."""
    import time

    import torch

    shape = payload["shape"]
    dtype = torch.int32 if payload["dtype"] in ("s32", "i32") else torch.float32
    dev = "cuda"
    n = int(shape.get("N", 1024))
    if "N_rows" in shape:
        rows, t, d = int(shape["N_rows"]), int(shape["T"]), int(shape["D"])
        q = torch.randn(rows, d, device=dev, dtype=torch.float32)
        k = torch.randn(t, d, device=dev, dtype=torch.float32)
        v = torch.randn(t, d, device=dev, dtype=torch.float32)
        fn = lambda: torch.softmax(q @ k.transpose(0, 1) / d ** 0.5, dim=-1) @ v
    elif "M" in shape:
        m, kk = int(shape["M"]), int(shape["K"])
        nn = int(shape.get("N_out", kk))
        td = torch.float32 if dtype != torch.int32 else torch.float32
        a = torch.randn(m, kk, device=dev, dtype=td)
        b = torch.randn(kk, nn, device=dev, dtype=td)
        fn = lambda: a @ b
    elif dtype == torch.int32:
        x = torch.randint(-100, 100, (n,), device=dev, dtype=torch.int32)
        fn = lambda: x * 10 + 3
    else:
        x = torch.randn(n, device=dev, dtype=dtype)
        fn = lambda: 10 * x + 3
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(True)
    stop = torch.cuda.Event(True)
    times = []
    for _ in range(30):
        start.record()
        fn()
        stop.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(stop))
    times.sort()
    return {"name": "torch", "p50_ms": times[len(times) // 2]}


@app.function(image=IMAGE, gpu="H100", timeout=900)
def probe(payload: str = "{}") -> dict:
    """H100 device probe: name, SMs, peak bandwidth, `ncu` availability."""
    return _probe_impl("H100")


@app.function(image=IMAGE, gpu="B200", timeout=900)
def probe_b200(payload: str = "{}") -> dict:
    return _probe_impl("B200")


@app.function(image=IMAGE, gpu="H200", timeout=900)
def probe_h200(payload: str = "{}") -> dict:
    return _probe_impl("H200")


@app.function(image=IMAGE, gpu="L4", timeout=900)
def probe_l4(payload: str = "{}") -> dict:
    return _probe_impl("L4")


@app.function(image=IMAGE, gpu="A100", timeout=900)
def probe_a100(payload: str = "{}") -> dict:
    return _probe_impl("A100")


def _job(payload: str, gpu: str) -> dict:
    import json

    data = json.loads(payload)
    data["gpu"] = gpu
    try:
        return _run_job_impl(data)
    except Exception as exc:  # surface infra errors to the client
        return {"error": f"{type(exc).__name__}: {exc}"}


@app.function(image=IMAGE, gpu="H100", timeout=3600)
def run_job(payload: str) -> dict:
    return _job(payload, "H100")


@app.function(image=IMAGE, gpu="B200", timeout=3600)
def run_job_b200(payload: str) -> dict:
    return _job(payload, "B200")


@app.function(image=IMAGE, gpu="H200", timeout=3600)
def run_job_h200(payload: str) -> dict:
    return _job(payload, "H200")


@app.function(image=IMAGE, gpu="L4", timeout=3600)
def run_job_l4(payload: str) -> dict:
    return _job(payload, "L4")


@app.function(image=IMAGE, gpu="A100", timeout=3600)
def run_job_a100(payload: str) -> dict:
    return _job(payload, "A100")


@app.local_entrypoint()
def main(payload_file: str = "") -> None:
    """Probe H100; with `--payload-file` also run one profiling job and print it.

    `uv run modal run -m harness.backends.modal_gpu`
    `uv run modal run -m harness.backends.modal_gpu --payload-file /tmp/job.json`
    """
    print(probe.remote("{}"))
    if payload_file:
        print(run_job.remote(open(payload_file).read()))
