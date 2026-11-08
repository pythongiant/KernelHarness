"""Lower a variant's CUDA template to the harness instruction model.

This is the bridge the verifier consumes: a template id plus transform parameters
becomes a concrete `Program` over a small launch instance (`lean_n` elements), which
`emit.py` renders to Lean. Vectorized memory accesses are expanded into one scalar
access per element, so the modelled program is extensionally the kernel's memory
behaviour.

Lowering is used when real PTX is unavailable (offline/sim runs). When a Modal job
returns the compiled PTX, `parse_ptx` is preferred and the same emitter runs on it.
"""

from __future__ import annotations

from . import interp

IN_BASE = 0x1000
OUT_BASE = 0x2000
A_BASE = 0x1000
B_BASE = 0x2000
C_BASE = 0x3000


def _reg(n: int) -> dict:
    return {"kind": "reg", "n": n}


def _imm(v: int) -> dict:
    return {"kind": "imm", "v": int(v)}


def _mem(space: str, base: int, off: int, stride: int, row_size: int = 0,
         row_stride: int = 0, col_stride: int = 0) -> dict:
    return {"kind": "mem", "space": space, "base": base, "off": off, "stride": stride,
            "rowSize": row_size, "rowStride": row_stride, "colStride": col_stride}


def _instr(op: str, dst: list | None = None, src: list | None = None) -> dict:
    return {"op": op, "dst": dst or [], "src": src or [], "pred": None, "line": 0,
            "opcode": op.split(" ")[0]}


def lower_elementwise(params: dict, lean_n: int = 16, a: int = 10, b: int = 3) -> tuple[list[dict], dict]:
    """`per_thread` elements per thread, one block, straight-line program."""
    vec = int(params.get("vector_width", 1))
    vec = max(1, min(vec, 4))
    threads = (lean_n + vec - 1) // vec
    elems = lean_n
    stride = 4 * vec
    prog: list[dict] = [
        _instr("Op.mov .s32", [_reg(0)], [_imm(a)]),
        _instr("Op.mov .s32", [_reg(1)], [_imm(b)]),
    ]
    for j in range(vec):
        off = 4 * j
        if (j * threads) >= elems and vec > 1:
            break
        prog += [
            _instr("Op.ld .global .s32 .v1 .ca", [_reg(2)],
                   [_mem("global", IN_BASE, off, stride)]),
            _instr("Op.mul .s32", [_reg(3)], [_reg(2), _reg(0)]),
            _instr("Op.add .s32", [_reg(4)], [_reg(3), _reg(1)]),
            _instr("Op.st .global .s32 .v1 .cs", [],
                   [_mem("global", OUT_BASE, off, stride), _reg(4)]),
        ]
    prog.append(_instr("Op.ret"))
    return prog, {"threads": threads, "block": threads, "grid": 1}


def lower_matmul_scalar(params: dict, m: int = 4, n: int = 4, k: int = 4) -> tuple[list[dict], dict]:
    """One thread per output element, `k` unrolled multiply-adds."""
    threads = m * n
    prog: list[dict] = [_instr("Op.mov .s32", [_reg(0)], [_imm(0)])]
    for kk in range(k):
        prog += [
            _instr("Op.ld .global .s32 .v1 .ca", [_reg(1)],
                   [_mem("global", A_BASE, 4 * kk, 0, row_size=n, row_stride=4 * k,
                         col_stride=0)]),
            _instr("Op.ld .global .s32 .v1 .ca", [_reg(2)],
                   [_mem("global", B_BASE, 4 * n * kk, 0, row_size=n, row_stride=0,
                         col_stride=4)]),
            _instr("Op.mad .s32", [_reg(0)], [_reg(1), _reg(2), _reg(0)]),
        ]
    prog += [
        _instr("Op.st .global .s32 .v1 .cs", [], [_mem("global", C_BASE, 0, 4), _reg(0)]),
        _instr("Op.ret"),
    ]
    return prog, {"threads": threads, "block": threads, "grid": 1}


def lower(variant) -> tuple[list[dict], dict]:
    """Lower a variant to (program, geometry) over the verification instance."""
    family = variant.kernel.family
    params = variant.params
    if family == "elementwise":
        return lower_elementwise(params, lean_n=int(params.get("lean_n", 16)),
                                 a=int(variant.kernel.shape_env().get("a", 10)),
                                 b=int(variant.kernel.shape_env().get("b", 3)))
    if family == "gemm":
        return lower_matmul_scalar(params, m=int(params.get("lean_m", 4)),
                                   n=int(params.get("lean_n", 4)),
                                   k=int(params.get("lean_k", 4)))
    raise NotImplementedError(f"no lowering for family {family}")


def build_launch(program: list[dict], geom: dict, init: dict[int, int], a: int = 10,
                 b: int = 3) -> interp.Launch:
    return interp.Launch(program=program, threads=geom["threads"], init=init, a=a, b=b)


def expected_observe(launch: interp.Launch) -> dict[int, int]:
    return interp.run(launch)["observe"]
