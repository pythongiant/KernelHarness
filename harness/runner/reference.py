"""Deterministic input fill and reference implementations.

The fill is bit-identical to `harness/runner/bench_main.cu`'s xorshift32, so input
bytes on the GPU and in numpy agree exactly. `s32` kernels are checked exactly
(rtol=atol=0); float kernels use the tolerance from `kernel.toml`.
"""

from __future__ import annotations

import numpy as np

SEED = 0x9E3779B9
MASK = 0xFFFFFFFF


def xorshift_words(n: int, seed: int = SEED) -> list[int]:
    """The first `n` outputs of the harness's xorshift32 generator."""
    x = seed & MASK
    out = []
    for _ in range(n):
        x ^= (x << 13) & MASK
        x ^= x >> 17
        x ^= (x << 5) & MASK
        x &= MASK
        out.append(x)
    return out


def fill_s32(n: int, seed: int = SEED) -> np.ndarray:
    """Integer input values in `[-100, 100]`, exact in int32."""
    return fill_buffers_s32([n], seed)[0]


def fill_f32(n: int, seed: int = SEED) -> np.ndarray:
    """Float input values in `[-1, 1)` with 24-bit mantissas."""
    return fill_buffers_f32([n], seed)[0]


def _as_s32(words) -> np.ndarray:
    w = np.array(words, dtype=np.uint64)
    v = ((w >> np.uint64(8)) % np.uint64(201)).astype(np.int64) - 100
    return v.astype(np.int32)


def _as_f32(words) -> np.ndarray:
    w = np.array(words, dtype=np.uint64)
    v = (w >> np.uint64(8)).astype(np.float64) / 16777216.0
    return (v * 2.0 - 1.0).astype(np.float32)


def fill_buffers_s32(elems: list[int], seed: int = SEED) -> list[np.ndarray]:
    """Fill buffers from one continuous xorshift32 stream, like `bench_main.cu`.

    The driver never reseeds between buffers, so buffer `k` starts where buffer
    `k-1` stopped. The output buffer is zeroed and consumes no randomness.
    """
    total = int(sum(elems))
    words = xorshift_words(total, seed)
    out, start = [], 0
    for e in elems:
        out.append(_as_s32(words[start : start + int(e)]))
        start += int(e)
    return out


def fill_buffers_f32(elems: list[int], seed: int = SEED) -> list[np.ndarray]:
    total = int(sum(elems))
    words = xorshift_words(total, seed)
    out, start = [], 0
    for e in elems:
        out.append(_as_f32(words[start : start + int(e)]))
        start += int(e)
    return out


def bytes_of_s32(arr: np.ndarray) -> list[int]:
    return arr.astype("<i4").tobytes()


def bytes_of_f32(arr: np.ndarray) -> list[int]:
    return arr.astype("<f4").tobytes()


# --- per-family references ------------------------------------------------
def reference_elementwise(kernel, env: dict[str, int] | None = None) -> np.ndarray:
    """`out[i] = a*x[i] + b` for `N` elements."""
    env = env or kernel.shape_env()
    n = int(env["N"])
    a = int(env.get("a", kernel.extra.get("a", 10)))
    b = int(env.get("b", kernel.extra.get("b", 3)))
    if kernel.dtype in ("s32", "i32"):
        x = fill_buffers_s32([n, n])[0]
        return (np.int32(a) * x + np.int32(b)).astype(np.int32)
    x = fill_buffers_f32([n, n])[0]
    return (np.float32(a) * x + np.float32(b)).astype(np.float32)


def reference_matmul(kernel, env: dict[str, int] | None = None) -> np.ndarray:
    env = env or kernel.shape_env()
    m, k = int(env.get("M", 8)), int(env.get("K", 8))
    n = int(env.get("N_out", 8))
    if kernel.dtype in ("s32", "i32"):
        A, B = fill_buffers_s32([m * k, k * n])
        return (A.reshape(m, k).astype(np.int32) @ B.reshape(k, n).astype(np.int32)).astype(
            np.int32
        )
    A, B = fill_buffers_f32([m * k, k * n])
    return (A.reshape(m, k) @ B.reshape(k, n)).astype(np.float32)


def reference_attention(kernel, env: dict[str, int] | None = None) -> np.ndarray:
    """Scalar online-softmax attention over `N_rows` rows of length `T`."""
    env = env or kernel.shape_env()
    rows, t, d = int(env["N_rows"]), int(env["T"]), int(env["D"])
    qb, kb, vb = fill_buffers_f32([rows * d, t * d, t * d])
    q, k, v = qb.reshape(rows, d), kb.reshape(t, d), vb.reshape(t, d)
    scale = np.float32(1.0 / np.sqrt(d))
    logits = (q @ k.T) * scale
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits.astype(np.float32))
    p = e / e.sum(axis=1, keepdims=True)
    return (p @ v).astype(np.float32)


REFERENCES = {
    "elementwise": reference_elementwise,
    "gemm": reference_matmul,
    "attention": reference_attention,
}


class _JobKernel:
    """Adapter so a `ProfileJob` can be used with the reference functions."""

    def __init__(self, job):
        self.family = job.family
        self.dtype = job.dtype
        self.shape = {k: str(v) for k, v in job.shape.items()}
        self.tol = job.tol
        self.extra = {"reference": job.extra.get("reference")}

    def shape_env(self) -> dict[str, int]:
        return {k: int(v) for k, v in self.shape.items()}


def reference_for_job(job) -> np.ndarray:
    """Reference output for a profile job (same dtype/layout as the harness)."""
    return reference(_JobKernel(job))


def compare_for_job(job, got: np.ndarray) -> tuple[str, float, float]:
    """Compare a flat output buffer with the reference (both flattened)."""
    return compare(_JobKernel(job), np.ravel(got), np.ravel(reference_for_job(job)))


def reference(kernel, env: dict[str, int] | None = None) -> np.ndarray:
    """Reference output for a kernel.

    A `family = "user"` kernel names the reference it implements in its manifest
    (`reference = "elementwise" | "gemm" | "attention"`).
    """
    fn = REFERENCES.get(kernel.family)
    if fn is None:
        named = (getattr(kernel, "extra", {}) or {}).get("reference")
        fn = REFERENCES.get(named)
    if fn is None:
        raise NotImplementedError(f"no reference for family {kernel.family}")
    return fn(kernel, env)


def compare(kernel, got: np.ndarray, want: np.ndarray) -> tuple[str, float, float]:
    """Return (PASS|FAIL, max_abs_err, max_rel_err) using the kernel tolerance."""
    if got.shape != want.shape:
        return "FAIL", float("inf"), float("inf")
    if kernel.dtype in ("s32", "i32"):
        g = got.astype(np.int64)
        w = want.astype(np.int64)
        if np.array_equal(g, w):
            return "PASS", 0.0, 0.0
        err = np.abs(g - w)
        return "FAIL", float(err.max()), float("inf")
    g = got.astype(np.float64)
    w = want.astype(np.float64)
    err = np.abs(g - w)
    denom = np.maximum(np.abs(w), 1e-30)
    rel = err / denom
    rtol = kernel.tol.get("rtol", 1e-4)
    atol = kernel.tol.get("atol", 1e-4)
    ok = bool(np.all(err <= atol + rtol * np.abs(w)))
    return ("PASS" if ok else "FAIL"), float(err.max()), float(rel.max())
