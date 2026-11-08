"""Reference interpreter for the harness instruction model.

Mirrors `Harness.PTX.step` for the opcode subset used by lowered kernels. It is
used for two things only: computing the expected output bytes that the generated
Lean `post` asserts, and computing static features. It is *not* the verifier — the
verifier is Lean.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Launch:
    program: list[dict]
    threads: int
    init: dict[int, int]
    params: dict[int, int] = field(default_factory=dict)
    a: int = 10
    b: int = 3


def _m32(v: int) -> int:
    return v & 0xFFFFFFFF


def _reg_get(regs: dict[int, int], n: int) -> int:
    return regs.get(n, 0)


def _addr(operand: dict, tid: int) -> int:
    row_size = operand.get("rowSize", 0)
    row = 0 if not row_size else (tid // row_size) * operand.get("rowStride", 0) \
        + (tid % row_size) * operand.get("colStride", 0)
    return operand.get("base", 0) + tid * operand.get("stride", 0) + row + operand.get("off", 0)


def _bytes_of(value: int, size: int) -> list[int]:
    return [(value >> (8 * i)) & 0xFF for i in range(size)]


def run(launch: Launch, steps: int | None = None) -> dict:
    """Execute every thread to completion and return the observable byte map."""
    writes: dict[int, int] = {}
    events: list[dict] = []
    ev_id = 0
    n_steps = steps if steps is not None else len(launch.program)

    for tid in range(launch.threads):
        regs: dict[int, int] = {}
        pc = 0
        for _ in range(n_steps):
            if pc >= len(launch.program):
                break
            instr = launch.program[pc]
            op = instr["op"]
            name = op.split(" ")[0]
            dst, src = instr["dst"], instr["src"]
            nxt = pc + 1

            if name in ("Op.mov", "Op.cvt"):
                if dst and src:
                    regs[dst[0]["n"]] = (
                        _reg_get(regs, src[0]["n"]) if src[0]["kind"] == "reg" else src[0].get("v", 0)
                    )
            elif name in ("Op.add", "Op.sub", "Op.mul", "Op.div", "Op.rem", "Op.shl",
                          "Op.shr", "Op.and_", "Op.or_", "Op.xor_", "Op.min_", "Op.max_"):
                a = _reg_get(regs, src[0]["n"]) if src[0]["kind"] == "reg" else src[0].get("v", 0)
                b = _reg_get(regs, src[1]["n"]) if src[1]["kind"] == "reg" else src[1].get("v", 0)
                val = {
                    "Op.add": a + b, "Op.sub": a - b, "Op.mul": a * b,
                    "Op.div": 0 if b == 0 else a // b, "Op.rem": 0 if b == 0 else a % b,
                    "Op.shl": a << (b & 31), "Op.shr": a >> (b & 31),
                    "Op.and_": a & b, "Op.or_": a | b, "Op.xor_": a ^ b,
                    "Op.min_": min(a, b), "Op.max_": max(a, b),
                }[name]
                regs[dst[0]["n"]] = _m32(val)
            elif name == "Op.mad":
                a = _reg_get(regs, src[0]["n"])
                b = _reg_get(regs, src[1]["n"])
                c = _reg_get(regs, src[2]["n"])
                regs[dst[0]["n"]] = _m32(a * b + c)
            elif name in ("Op.not_", "Op.neg", "Op.abs"):
                a = _reg_get(regs, src[0]["n"])
                regs[dst[0]["n"]] = _m32({"Op.not_": ~a, "Op.neg": -a, "Op.abs": a}[name])
            elif name in ("Op.ld", "Op.ldGlobalNc", "Op.paramLd", "Op.ldmatrixX1",
                          "Op.ldmatrixX2", "Op.ldmatrixX4", "Op.tcgen05Ld"):
                if not src:
                    pc = nxt
                    continue
                addr = _addr(src[0], tid)
                size = _access_bytes(op, launch, addr)
                value = 0
                for i in range(min(size, 4)):
                    value |= (launch.init.get(addr + i, 0) & 0xFF) << (8 * i)
                if dst:
                    regs[dst[0]["n"]] = value
                events.append({"id": ev_id, "tid": tid, "kind": "read", "addr": addr, "size": size})
                ev_id += 1
            elif name in ("Op.st", "Op.stGlobalCs", "Op.paramSt"):
                if not src:
                    pc = nxt
                    continue
                addr = _addr(src[0], tid)
                size = _access_bytes(op, launch, addr)
                value = _reg_get(regs, src[1]["n"]) if len(src) > 1 and src[1]["kind"] == "reg" \
                    else (src[1].get("v", 0) if len(src) > 1 else 0)
                for i, by in enumerate(_bytes_of(value, size)):
                    writes[addr + i] = by
                events.append({"id": ev_id, "tid": tid, "kind": "write", "addr": addr, "size": size})
                ev_id += 1
            elif name in ("Op.barSync", "Op.barArrive", "Op.clusterSync",
                          "Op.barrierClusterArrive", "Op.barrierClusterWait"):
                events.append({"id": ev_id, "tid": tid, "kind": "barArrive", "addr": 0, "size": 0})
                ev_id += 1
                events.append({"id": ev_id, "tid": tid, "kind": "barReturn", "addr": 0, "size": 0})
                ev_id += 1
            elif name in ("Op.ret", "Op.exit"):
                events.append({"id": ev_id, "tid": tid, "kind": "exit", "addr": 0, "size": 0})
                ev_id += 1
                break
            elif name in ("Op.bra", "Op.braUni"):
                nxt = int(op.split(" ")[1]) if " " in op else 0
            elif name in ("Op.barWarpSync", "Op.shflSyncIdx", "Op.shflSyncUp",
                          "Op.shflSyncDown", "Op.shflSyncBfly", "Op.electSync"):
                events.append({"id": ev_id, "tid": tid, "kind": "shfl", "addr": 0, "size": 0})
                ev_id += 1
            elif name in ("Op.cpAsyncCa", "Op.cpAsyncCg", "Op.cpAsyncCommitGroup",
                          "Op.cpAsyncWaitGroup", "Op.cpAsyncBulkTensor2d",
                          "Op.cpAsyncBulkCommitGroup", "Op.cpAsyncBulkWaitGroup",
                          "Op.cpAsyncWaitAll", "Op.tcgen05Cp", "Op.tcgen05Alloc"):
                events.append({"id": ev_id, "tid": tid, "kind": "asyncIssue", "addr": 0, "size": 0})
                ev_id += 1
            elif name in ("Op.mmaSyncM16n8k16", "Op.wgmmaMmaAsyncM64n8k16",
                          "Op.wgmmaFence", "Op.wgmmaCommitGroup", "Op.tcgen05Mma",
                          "Op.tcgen05Fence"):
                events.append({"id": ev_id, "tid": tid, "kind": "wgmmaIssue", "addr": 0, "size": 0})
                ev_id += 1
            elif name in ("Op.mbarrierInit", "Op.mbarrierArrive", "Op.mbarrierArriveExpectTx",
                          "Op.mbarrierTryWaitParity", "Op.mbarrierWaitParity"):
                kind = "mbarInit" if name == "Op.mbarrierInit" else (
                    "mbarWait" if "Wait" in name else "mbarArrive")
                events.append({"id": ev_id, "tid": tid, "kind": kind, "addr": 0, "size": 0})
                ev_id += 1
            elif name in ("Op.fenceSc", "Op.fenceProxyAsync", "Op.fenceProxyTensormap",
                          "Op.fenceProxyAsyncShared", "Op.membar", "Op.membarGl", "Op.membarCta"):
                events.append({"id": ev_id, "tid": tid, "kind": "fence", "addr": 0, "size": 0})
                ev_id += 1
            pc = nxt
    observ: dict[int, int] = {}
    for a in sorted(set(launch.init) | set(writes)):
        observ[a] = writes.get(a, launch.init.get(a, 0))
    return {"observe": observ, "writes": writes, "events": events}


_TY_SIZE = {"u8": 1, "s8": 1, "u16": 2, "s16": 2, "f16": 2, "bf16": 2,
            "u32": 4, "s32": 4, "f32": 4, "u64": 8, "s64": 8, "f64": 8}


def _access_bytes(op: str, launch: Launch, addr: int) -> int:
    parts = op.split(" ")
    name = parts[0]
    if name in ("Op.cpAsyncCa", "Op.cpAsyncCg"):
        return int(parts[1]) if len(parts) > 1 else 16
    if name in ("Op.ldmatrixX1", "Op.ldmatrixX2", "Op.ldmatrixX4"):
        return 16 * {"Op.ldmatrixX1": 1, "Op.ldmatrixX2": 2, "Op.ldmatrixX4": 4}[name]
    for p in parts[1:]:
        if p.startswith(".") and p[1:] in _TY_SIZE:
            size = _TY_SIZE[p[1:]]
            vec = next((int(q[1:]) for q in parts if q in (".v2", ".v4")), 1)
            return size * vec
    return 4
