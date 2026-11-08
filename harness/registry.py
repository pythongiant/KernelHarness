"""Kernel registry: manifests, level ladders, and per-variant launch geometry."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .config import KERNEL_DIR, TEMPLATE_DIR
from .launch import eval_all, evaluate


@dataclass
class Sol:
    kind: str  # bandwidth | compute
    bytes: str = "0"
    flops: str = "0"


@dataclass
class Kernel:
    id: str
    family: str
    entry: str
    source: str
    shape: dict[str, str]
    dtype: str
    sol: Sol
    buffers: list[str]
    out_index: int
    tol: dict[str, float]
    spec_lean: str
    baseline: str
    level: int
    path: Path
    extra: dict = field(default_factory=dict)

    # -- derived -----------------------------------------------------------
    @property
    def user_supplied(self) -> bool:
        return self.family == "user"

    def shape_env(self) -> dict[str, int]:
        return eval_all(self.shape)

    def sol_numbers(self, env: dict[str, int] | None = None) -> tuple[int, int]:
        e = dict(env or self.shape_env())
        return evaluate(self.sol.bytes, e), evaluate(self.sol.flops, e)

    def levels(self) -> list[int]:
        d = TEMPLATE_DIR / self.family
        if not d.is_dir():
            return []
        out = []
        for p in d.iterdir():
            if p.is_dir() and p.name.startswith("L") and p.name[1:].isdigit():
                out.append(int(p.name[1:]))
        return sorted(out)

    def template_path(self, level: int | None = None) -> Path | None:
        lvl = self.level if level is None else level
        p = TEMPLATE_DIR / self.family / f"L{lvl}" / "kernel.cu"
        return p if p.exists() else None


def load_kernel(path: Path | str) -> Kernel:
    p = Path(path)
    if p.is_dir():
        p = p / "kernel.toml"
    data = tomllib.loads(p.read_text())
    sol = data.get("sol", {})
    return Kernel(
        id=data["id"],
        family=data["family"],
        entry=data.get("entry", "launch_kernel"),
        source=data.get("source", "@@TEMPLATE@@"),
        shape={k: str(v) for k, v in data.get("shape", {}).items()},
        dtype=data.get("dtype", "f32"),
        sol=Sol(
            kind=sol.get("kind", "bandwidth"),
            bytes=str(sol.get("bytes", "0")),
            flops=str(sol.get("flops", "0")),
        ),
        buffers=list(data.get("buffers", [])),
        out_index=int(data.get("out_index", 0)),
        tol={k: float(v) for k, v in data.get("tol", {}).items()},
        spec_lean=data.get("spec_lean", ""),
        baseline=data.get("baseline", "none"),
        level=int(data.get("level", 0)),
        path=p,
        extra={
            k: v
            for k, v in data.items()
            if k
            not in {
                "id",
                "family",
                "entry",
                "source",
                "shape",
                "dtype",
                "sol",
                "buffers",
                "out_index",
                "tol",
                "spec_lean",
                "baseline",
                "level",
            }
        },
    )


def registry() -> list[Kernel]:
    """All kernels in `harness/kernels/`, sorted by id."""
    if not KERNEL_DIR.is_dir():
        return []
    out = []
    for p in sorted(KERNEL_DIR.glob("*/kernel.toml")):
        try:
            out.append(load_kernel(p))
        except Exception:  # pragma: no cover - a broken manifest must not hide others
            continue
    return out


def get(kernel_id: str) -> Kernel:
    for k in registry():
        if k.id == kernel_id:
            return k
    raise KeyError(f"unknown kernel: {kernel_id}")


def geometry(k: Kernel) -> dict[str, int]:
    """Grid/block geometry for a kernel, derived from its shape."""
    env = k.shape_env()
    n = env.get("N", 1)
    if k.family == "elementwise":
        block = env.get("block", 256)
        return {"block": block, "grid": evaluate("ceildiv(N, block)", {**env, "block": block})}
    if k.family == "gemm":
        m, nn = env.get("M", n), env.get("N_out", env.get("K", 16))
        tm, tn = env.get("tile_m", 16), env.get("tile_n", 16)
        return {
            "block": env.get("block", 128),
            "grid": evaluate(
                "ceildiv(M, tile_m) * ceildiv(N_out, tile_n)",
                {**env, "M": m, "N_out": nn, "tile_m": tm, "tile_n": tn},
            ),
        }
    if k.family == "attention":
        row_tile = env.get("row_tile", 1)
        return {
            "block": env.get("block", 64),
            "grid": evaluate("ceildiv(N_rows, row_tile)", {**env, "row_tile": row_tile}),
        }
    return {"block": env.get("block", 128), "grid": evaluate("ceildiv(N, 128)", env)}


def render_source(k: Kernel, level: int | None = None, params: dict | None = None) -> str:
    """Render the CUDA source for a level, substituting `@@NAME@@` placeholders.

    A kernel whose manifest names an explicit file (`source = "kernel.cu"`, the
    `family = "user"` case) is read from disk; templates substitute placeholders.
    Only `@@NAME@@` tokens are replaced (CUDA braces are preserved verbatim).
    """
    if k.source and not k.source.startswith("@@"):
        user_path = (k.path.parent / k.source).resolve()
        if not user_path.exists():
            raise FileNotFoundError(f"kernel {k.id}: source file not found: {user_path}")
        return user_path.read_text()
    path = k.template_path(level)
    if path is None:
        raise FileNotFoundError(f"no template for family={k.family} level={level or k.level}")
    text = path.read_text()
    env = {**k.shape_env(), **(params or {})}
    for key, val in env.items():
        text = text.replace(f"@@{key.upper()}@@", str(val))
    text = text.replace("@@N@@", str(env.get("N", 1)))
    text = text.replace("@@DTYPE@@", "int" if k.dtype in ("s32", "i32") else "float")
    leftover = [chunk.split("@@")[0] for chunk in text.split("@@")[1::2]]
    if leftover:
        raise KeyError(f"unsubstituted placeholders in {path}: {leftover}")
    return text
