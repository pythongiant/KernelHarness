"""Emit Lean modules for a variant and turn the Lean build into a proof status.

`emit_lean` renders the lowered program, its initial memory, the reference output
bytes and the four obligations (`canonConsistent`, `canonPost`, `admissible`,
`correct`/`nonvacuous`) into `lean/Harness/Generated/<Variant>.lean`.

`check` runs Lean on the emitted module (not the whole library), scans for `sorry`,
and reports the axiom audit; the resulting status is `VERIFIED`, `PARTIAL`, `FAILED`
or `UNSUPPORTED`. A status is never upgraded: all three checks must pass.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ..config import LEAN_DIR, NUMERICAL_SCOPE

GENERATED_DIR = LEAN_DIR / "Harness" / "Generated"


def _lean_name(variant_id: str) -> str:
    parts = re.split(r"[^A-Za-z0-9]+", variant_id)
    name = "".join(p[:1].upper() + p[1:] for p in parts if p)
    if not name or not name[0].isalpha():
        name = "V" + name
    return name


def _operand(o: dict) -> str:
    k = o["kind"]
    if k == "reg":
        return f"Operand.reg {o['n']}"
    if k == "imm":
        return f"Operand.imm {o['v']}"
    if k == "mem":
        return (
            f"Operand.mem .{o.get('space', 'global')} {o.get('base', 0)} {o.get('off', 0)} "
            f"{o.get('stride', 0)} {o.get('rowSize', 0)} {o.get('rowStride', 0)} "
            f"{o.get('colStride', 0)}"
        )
    raise ValueError(f"cannot emit operand {o!r}")


def _instr(i: dict) -> str:
    dst = ", ".join(_operand(o) for o in i.get("dst", []))
    src = ", ".join(_operand(o) for o in i.get("src", []))
    pred = "none" if i.get("pred") is None else f"some {i['pred']}"
    return f"  Instr.mk ({i['op']}) [{dst}] [{src}] {pred}"


def _match_arms(mapping: dict[int, int], default: int = 0) -> str:
    arms = "".join(f"  | {k} => {v}\n" for k, v in sorted(mapping.items()))
    return arms + f"  | _ => {default}"


def emit_lean(
    *,
    variant_id: str,
    family: str,
    program: list[dict],
    geom: dict,
    init: dict[int, int],
    out_base: int,
    out_bytes: list[int],
    meta: dict | None = None,
) -> tuple[str, str]:
    """Return (module name, Lean source) for a variant."""
    name = _lean_name(variant_id)
    meta = meta or {}
    instrs = ",\n".join(_instr(i) for i in program)
    threads = int(geom.get("threads", 1))
    block = int(geom.get("block", threads))
    grid = int(geom.get("grid", 1))
    refs = ", ".join(str(b) for b in out_bytes)
    src = f"""import Harness.PTX.Observer
import Harness.Lib.Core
import Harness.Audit

/-!
# Generated verification module: `{variant_id}`

Family `{family}`. Instance: {threads} threads (block {block}, grid {grid}), output
region starts at byte {out_base} ({len(out_bytes)} bytes).
Numerical scope: {NUMERICAL_SCOPE}.
Statement: every consistent, complete execution of this launch that reads memory
through the canonical oracle writes exactly the reference bytes below.
-/

namespace Harness.Generated.{name}

open Harness.PTX

set_option maxRecDepth 400000
set_option maxHeartbeats 2000000

abbrev instrs : Array Instr := #[
{instrs}
]

abbrev grid : Grid := ⟨({grid}, 1, 1), ({block}, 1, 1), (1, 1, 1), {threads}⟩

/-- Initial byte map: the deterministic input fill, zero elsewhere. -/
abbrev initBytes : Addr → Byte := fun a =>
  match a with
{_match_arms(init)}

abbrev launch : Launch := ⟨{{ instrs := instrs }}, grid, ⟨initBytes⟩, fun _ => .u32 0⟩

/-- Reference bytes of the output region. -/
abbrev refBytes : List Byte := [{refs}]

abbrev post : Post := fun m =>
  (List.range refBytes.length).all (fun i => m.get ({out_base} + i) == refBytes.getD i 0) = true

theorem canonConsistent : Consistent (Candidate.canonical launch) := by decide

theorem canonPost : post (observe (Candidate.canonical launch)) := by decide

theorem admissible : launch.Admissible :=
  ⟨by decide, fun a => ⟨launch.init.get a, rfl⟩, fun a => ⟨launch.params a, rfl⟩,
    canonConsistent⟩

theorem nonvacuous : ∃ e : Candidate launch, Consistent e ∧ e.Complete :=
  Harness.PTX.nonvacuous launch admissible

theorem correct : Correct launch post :=
  correct_of_canonical launch post canonConsistent canonPost

run_cmd Harness.Audit.audit #[``correct, ``nonvacuous]

#eval "AUDIT_OK"

end Harness.Generated.{name}
"""
    return name, src


def lean_binary() -> str | None:
    exe = shutil.which("lake")
    if exe:
        return exe
    for cand in (Path.home() / ".elan" / "bin" / "lake", Path("/usr/local/bin/lake")):
        if cand.exists():
            return str(cand)
    return None


@dataclass
class ProofResult:
    status: str
    model: str
    open_goals: list[str] = field(default_factory=list)
    axiom_audit: dict = field(default_factory=dict)
    modules: dict = field(default_factory=dict)
    log_path: str | None = None
    log: str = ""
    instantiation: str = ""

    def as_row(self, variant_id: str) -> dict:
        return {
            "variant_id": variant_id,
            "status": self.status,
            "model": self.model,
            "open_goals": self.open_goals,
            "modules": self.modules,
            "axiom_audit": self.axiom_audit,
            "log_path": self.log_path,
        }

    def summary(self) -> dict:
        """The proof record written to the ledger (`proof/status.json`)."""
        return {
            "status": self.status,
            "model": self.model,
            "open_goals": self.open_goals,
            "axiom_audit": self.axiom_audit,
            "modules": self.modules,
            "instantiation": self.instantiation,
            "numerical_scope": NUMERICAL_SCOPE,
        }

    def module_sources(self) -> dict[str, str]:
        """Generated Lean sources keyed by module name (empty string if missing)."""
        out = {}
        for rel in (self.modules or {}).values():
            path = LEAN_DIR / rel
            out[Path(rel).stem] = path.read_text() if path.exists() else ""
        return out


_ERROR_RE = re.compile(r"^.*?:\d+:\d+: error(?:\(([a-z_.]+)\))?: (.*)$")


def check(
    module_name: str, source: str, *, log_dir: Path | None = None, timeout: int = 900
) -> ProofResult:
    """Write the module, run Lean, and classify the result."""
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_DIR / f"{module_name}.lean"
    path.write_text(source)
    log_dir = log_dir or (LEAN_DIR / ".proof-logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{module_name}.log"

    lake = lean_binary()
    if lake is None:
        result = ProofResult(
            status="UNSUPPORTED",
            model="none",
            open_goals=["Lean toolchain not found on PATH"],
            instantiation="none",
        )
        log_path.write_text(result.open_goals[0])
        result.log_path = str(log_path)
        return result

    env_path = str(Path(lake).parent)
    cmd = [lake, "env", "lean", str(path.relative_to(LEAN_DIR))]
    proc = subprocess.run(
        cmd,
        cwd=LEAN_DIR,
        env={**__import__("os").environ, "PATH": env_path + ":/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    log = (proc.stdout or "") + (proc.stderr or "")
    log_path.write_text(log)

    errors = []
    for line in log.splitlines():
        m = _ERROR_RE.match(line)
        if m:
            errors.append(line.strip())
    sorry = "sorry" in log and ("declaration uses 'sorry'" in log or "'sorry'" in log)
    audit_ok = "AUDIT_OK" in log
    audit_bad = "audit:" in log

    if proc.returncode == 0 and audit_ok and not errors:
        status = "VERIFIED"
        open_goals: list[str] = []
    elif errors and not sorry:
        status = "FAILED"
        open_goals = errors[:20]
    else:
        status = "PARTIAL"
        open_goals = (
            errors[:20]
            or [l.strip() for l in log.splitlines() if "unsolved goals" in l or "sorry" in l][:20]
        )

    if audit_bad and status == "VERIFIED":
        status = "FAILED"
        open_goals = ["axiom audit failed"]

    return ProofResult(
        status=status,
        model="paper-full",
        open_goals=open_goals,
        axiom_audit={
            "ok": status == "VERIFIED" and audit_ok,
            "allowed": ["propext", "Classical.choice", "Quot.sound"],
            "checked": ["correct", "nonvacuous"],
            "run": audit_ok or audit_bad,
        },
        modules={module_name: str(path.relative_to(LEAN_DIR))},
        log_path=str(log_path),
        log=log,
        instantiation=f"{module_name}: concrete instance, exact int arithmetic",
    )
