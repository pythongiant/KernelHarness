"""PTX → harness instruction model.

The form table (`ptx_forms.toml`, 81 forms) is authoritative. An opcode whose
dotted prefix matches no form raises :class:`UnsupportedInstruction` — opcodes are
never silently skipped. Qualifier tokens after the matched form are kept as
qualifiers (`ld.global.v4.f32` → form `ld`, quals `global v4 f32`).
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

FORMS_PATH = Path(__file__).with_name("ptx_forms.toml")


class UnsupportedInstruction(Exception):
    def __init__(self, opcode: str, line: int, text: str = ""):
        self.opcode = opcode
        self.line = line
        self.text = text
        super().__init__(f"unsupported PTX instruction {opcode!r} at line {line}: {text.strip()}")


class ParseError(Exception):
    pass


SPACES = {"global", "shared", "local", "param", "const"}
SCOPES = {"cta", "cluster", "gpu", "sys"}
SEMS = {"relaxed", "acquire", "release", "acqrel", "sc"}
CACHES = {"ca", "cg", "cs", "lu", "cv"}
VECS = {"v1", "v2", "v4"}
TYS = {"u8", "u16", "u32", "u64", "s8", "s16", "s32", "s64", "f16", "bf16", "f32", "f64", "pred"}
ATOMOPS = {"add": "add_", "and": "and_", "or": "or_", "xor": "xor_", "cas": "cas", "exch": "exch"}
CMPS = {"eq", "ne", "lt", "le", "gt", "ge", "ltu", "leu", "gtu", "geu"}

#: Number of destination operand *groups* per form (0 for pure stores/control).
DST_GROUPS = {
    "st": 0,
    "red": 0,
    "ret": 0,
    "exit": 0,
    "bra": 0,
    "bra.uni": 0,
    "bar.sync": 0,
    "bar.arrive": 0,
    "bar.warp.sync": 0,
    "cluster.sync": 0,
    "barrier.cluster.arrive": 0,
    "barrier.cluster.wait": 0,
    "fence.sc": 0,
    "fence.proxy.async": 0,
    "fence.proxy.tensormap": 0,
    "fence.proxy.async.shared::cta": 0,
    "membar": 0,
    "membar.gl": 0,
    "membar.cta": 0,
    "cp.async.commit_group": 0,
    "cp.async.wait_group": 0,
    "cp.async.wait_all": 0,
    "cp.async.bulk.commit_group": 0,
    "cp.async.bulk.wait_group": 0,
    "wgmma.fence": 0,
    "wgmma.commit_group": 0,
    "wgmma.wait_group": 0,
    "tcgen05.commit": 0,
    "tcgen05.wait": 0,
    "tcgen05.fence": 0,
    "mbarrier.init": 1,
    "mbarrier.arrive": 1,
    "mbarrier.arrive.expect_tx": 1,
    "mbarrier.try_wait.parity": 1,
    "mbarrier.wait.parity": 1,
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32": 2,
    "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16": 2,
    "tcgen05.mma": 1,
    "tcgen05.ld": 1,
    "tcgen05.st": 0,
    "tcgen05.alloc": 1,
    "tcgen05.cp": 1,
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16": 1,
    "ldmatrix.sync.aligned.m8n8.x2.shared.b16": 1,
    "ldmatrix.sync.aligned.m8n8.x1.shared.b16": 1,
    "cp.async.ca.shared.global": 1,
    "cp.async.cg.shared.global": 1,
    "cp.async.bulk.tensor.2d.shared::cluster.global.tile": 1,
    "elect.sync": 1,
    "mapa": 1,
    "clusterlaunchcontrol.try_cancel": 1,
    "st.param": 0,
    "st.global.cs": 0,
    "st.global": 0,
}


@dataclass
class Form:
    op: str
    quals: list[str]
    operands: str
    semantics: str
    lean_ctor: str


def load_forms(path: Path | None = None) -> dict[str, Form]:
    data = tomllib.loads((path or FORMS_PATH).read_text())
    return {
        f["op"]: Form(f["op"], f["quals"], f["operands"], f["semantics"], f["lean_ctor"])
        for f in data["form"]
    }


FORMS = load_forms()


def match_form(opcode: str) -> tuple[str, list[str]]:
    """Split a PTX opcode into (form key, remaining qualifier tokens)."""
    parts = opcode.split(".")
    for i in range(len(parts), 0, -1):
        cand = ".".join(parts[:i])
        if cand in FORMS:
            return cand, parts[i:]
    raise KeyError(opcode)


def op_term(opcode: str) -> str:
    """Render the Lean `Op` term for a PTX opcode."""
    form, quals = match_form(opcode)
    ctor = FORMS[form].lean_ctor
    if form in ("ld", "st"):
        sp = next((q for q in quals if q in SPACES), "global")
        ty = next((q for q in quals if q in TYS), "u32")
        vec = next((q for q in quals if q in VECS), "v1")
        cache = next((q for q in quals if q in CACHES), "cs" if form == "st" else "ca")
        return f"{ctor} .{sp} .{ty} .{vec} .{cache}"
    if form in ("ld.global.nc", "st.global.cs"):
        ty = next((q for q in quals if q in TYS), "u32")
        vec = next((q for q in quals if q in VECS), "v1")
        return f"{ctor} .{ty} .{vec}"
    if form == "membar":
        return f"{ctor} .{next((q for q in quals if q in SCOPES), 'gpu')}"
    if form in ("atom", "red"):
        sem = next((q for q in quals if q in SEMS), "relaxed")
        sc = next((q for q in quals if q in SCOPES), "gpu")
        aop = next((q for q in quals if q in ATOMOPS), "add")
        ty = next((q for q in quals if q in TYS), "u32")
        return f"{ctor} .{sem} .{sc} .{ATOMOPS[aop]} .{ty}"
    if form == "setp":
        ty = next((q for q in quals if q in TYS), "u32")
        cmp = next((q for q in quals if q in CMPS), "eq")
        return f"{ctor} .{ty} .{cmp}"
    if form == "cvt":
        tys = [q for q in quals if q in TYS]
        src = tys[0] if tys else "u32"
        dst = tys[1] if len(tys) > 1 else src
        return f"{ctor} .{src} .{dst}"
    if form in ("cp.async.ca.shared.global", "cp.async.cg.shared.global"):
        size = next((q for q in quals if q.isdigit()), "16")
        return f"{ctor} {size}"
    if form in ("cp.async.wait_group", "cp.async.bulk.wait_group", "wgmma.wait_group"):
        return f"{ctor} {next((q for q in quals if q.isdigit()), '0')}"
    if form in ("bar.sync", "bar.arrive"):
        return f"{ctor} {next((q for q in quals if q.isdigit()), '0')}"
    if form in (
        "mov",
        "add",
        "sub",
        "mul",
        "mad",
        "div",
        "rem",
        "selp",
        "shl",
        "shr",
        "and",
        "or",
        "xor",
        "not",
        "neg",
        "abs",
        "min",
        "max",
        "ld.param",
        "st.param",
    ):
        return f"{ctor} .{next((q for q in quals if q in TYS), 'u32')}"
    return ctor


def dst_group_count(opcode: str) -> int:
    form, _ = match_form(opcode)
    if opcode in DST_GROUPS:
        return DST_GROUPS[opcode]
    return DST_GROUPS.get(form, 1)


_ADDR_OFF_RE = re.compile(r"^(?P<base>[A-Za-z0-9_%]+)\s*(?:\+\s*(?P<off>-?(?:0x)?[0-9a-fA-F]+))?$")


def _mem_space(opcode: str) -> str:
    _, quals = match_form(opcode)
    for q in quals:
        if q in SPACES:
            return q
    return "global"


def _parse_addr(expr: str, opcode: str) -> dict:
    """Parse `[base]`, `[base+off]`, `[base+tid*stride]` into a symbolic operand."""
    e = expr.strip()
    m = re.match(
        r"^(?P<base>[A-Za-z0-9_%]+)\s*\+\s*[A-Za-z0-9_%]*tid[A-Za-z0-9_.]*\s*\*\s*(?P<stride>\d+)\s*(?:\+\s*(?P<off>\d+))?$",
        e,
    )
    if m:
        return {
            "kind": "mem",
            "space": _mem_space(opcode),
            "base": 0,
            "off": int(m.group("off") or 0),
            "stride": int(m.group("stride")),
            "rowSize": 0,
            "rowStride": 0,
            "colStride": 0,
        }
    m = re.match(r"^(?P<base>[A-Za-z0-9_%]+)\s*\*\s*(?P<stride>\d+)\s*(?:\+\s*(?P<off>\d+))?$", e)
    if m:
        return {
            "kind": "mem",
            "space": _mem_space(opcode),
            "base": 0,
            "off": int(m.group("off") or 0),
            "stride": int(m.group("stride")),
            "rowSize": 0,
            "rowStride": 0,
            "colStride": 0,
        }
    m = _ADDR_OFF_RE.match(e)
    if m:
        base = m.group("base")
        off = m.group("off")
        b = 0
        if re.fullmatch(r"(0x[0-9a-fA-F]+|\d+)", base):
            b = int(base, 0)
        return {
            "kind": "mem",
            "space": _mem_space(opcode),
            "base": b,
            "off": int(off, 0) if off else 0,
            "stride": 0,
            "rowSize": 0,
            "rowStride": 0,
            "colStride": 0,
        }
    raise ParseError(f"cannot parse address expression: {expr!r}")


class _Decls:
    def __init__(self) -> None:
        self.table: dict[str, int] = {}

    def index(self, name: str) -> int:
        if name not in self.table:
            self.table[name] = len(self.table)
        return self.table[name]


def _split_groups(operand_text: str) -> list[str]:
    """Split a PTX operand list on top-level commas, keeping `{...}` groups."""
    groups: list[str] = []
    depth = 0
    cur = ""
    for ch in operand_text:
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "," and depth == 0:
            groups.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        groups.append(cur)
    return [g.strip() for g in groups if g.strip()]


def _operand(tok: str, decls: _Decls, opcode: str) -> dict:
    t = tok.strip()
    if t.startswith("["):
        return _parse_addr(t[1 : t.rindex("]")], opcode)
    if t.startswith("%"):
        return {"kind": "reg", "n": decls.index(t)}
    if re.fullmatch(r"-?(?:0x[0-9a-fA-F]+|\d+)", t):
        return {"kind": "imm", "v": int(t, 0)}
    return {"kind": "sym", "name": t}


def parse_ptx(text: str) -> dict:
    """Parse a PTX listing.

    Returns ``{"program": [...], "labels": {name: pc}, "decls": {name: index}}``
    where each instruction is ``{"op": <Lean Op term>, "dst": [...], "src": [...],
    "pred": int | None, "line": int}``.
    """
    decls = _Decls()
    labels: dict[str, int] = {}
    program: list[dict] = []
    in_signature = False

    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("//")[0].rstrip()
        s = line.strip()
        if not s:
            continue
        if s.startswith(".visible .entry") or s.startswith(".entry"):
            in_signature = not s.rstrip().endswith(")")
            continue
        if in_signature:
            if s.rstrip().endswith(")"):
                in_signature = False
            continue
        if (
            s in ("{", "}", "};")
            or s.startswith(".version")
            or s.startswith(".target")
            or s.startswith(".address_size")
            or s.startswith(".param")
            or s.startswith(".shared")
        ):
            continue
        if s.startswith(".reg"):
            for part in s.split()[1:]:
                m = re.match(r"%([A-Za-z0-9_]+)", part)
                if m:
                    decls.index("%" + m.group(1))
            continue
        if s.endswith(":"):
            labels[s[:-1]] = len(program)
            continue

        pred = None
        if s.startswith("@"):
            head, _, rest = s.partition(" ")
            pred = decls.index(head[1:].rstrip("!"))
            s = rest.strip()
        if not s:
            continue
        opcode, _, operand_text = s.partition(" ")
        opcode = opcode.strip().rstrip(";")
        operand_text = operand_text.strip().rstrip(";")
        try:
            op = op_term(opcode)
        except KeyError:
            raise UnsupportedInstruction(opcode, line_no, raw) from None

        groups = _split_groups(operand_text)
        ndst = dst_group_count(opcode)
        dst: list[dict] = []
        src: list[dict] = []
        for gi, g in enumerate(groups):
            inner = g.strip()
            if inner.startswith("{"):
                toks = [t for t in inner.strip("{}").split(",") if t.strip()]
                parsed = [_operand(t, decls, opcode) for t in toks]
            else:
                parsed = [_operand(inner, decls, opcode)]
            (dst if gi < ndst else src).extend(parsed)

        branch_target = None
        if opcode in ("bra", "bra.uni") and src and src[0].get("kind") == "sym":
            branch_target = src[0]["name"]
            op = f"{op} {labels.get(branch_target, 0)}"
        program.append(
            {"op": op, "dst": dst, "src": src, "pred": pred, "line": line_no, "opcode": opcode}
        )
    # second pass: resolve branch labels known only after the full scan
    for instr in program:
        if instr.get("opcode") in ("bra", "bra.uni"):
            base = instr["op"].split(" ")[0]
            tgt = instr["op"].split(" ")[1] if " " in instr["op"] else "0"
            if instr["src"] and instr["src"][0].get("kind") == "sym":
                tgt = str(labels.get(instr["src"][0]["name"], 0))
            instr["op"] = f"{base} {tgt}"
    return {"program": program, "labels": labels, "decls": decls.table}


def parse_ptx_file(path: Path | str) -> dict:
    return parse_ptx(Path(path).read_text())
