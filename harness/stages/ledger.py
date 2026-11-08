"""LEDGER stage: commit the round's artifacts to the git tree.

Each round writes `ledgers/runs/<run_id>/round-<n>/` with the profile, diagnosis,
hypotheses, kernel source/PTX, diff and proof artifacts, then commits with a fixed
message format, and regenerates `ledgers/REPORT.md` and `ledgers/index.json`.
"""

from __future__ import annotations

import difflib
import json
import os
import shutil
import subprocess
from pathlib import Path

from ..config import LEDGER_DIR, ROOT

LEDGER_RUNS = LEDGER_DIR / "runs"


def _git(*args: str, check: bool = True) -> str:
    proc = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def ensure_repo() -> None:
    """Make sure the ledger tree is a git repository.

    The operator's identity is never overridden: defaults are supplied only when git
    has none, from `HARNESS_GIT_NAME` / `HARNESS_GIT_EMAIL` (or a neutral fallback).
    """
    if not (ROOT / ".git").exists():
        _git("init")
    if not _git("config", "user.name", check=False):
        _git("config", "user.name", os.environ.get("HARNESS_GIT_NAME", "kernel-harness"))
    if not _git("config", "user.email", check=False):
        _git("config", "user.email",
             os.environ.get("HARNESS_GIT_EMAIL", "kernel-harness@localhost"))


def diff_text(before: str, after: str, name: str = "kernel.cu") -> str:
    if before == after:
        return ""
    return "".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile=f"a/{name}", tofile=f"b/{name}"))


def write_round_dir(run_id: str, n: int, artifacts: dict) -> Path:
    d = LEDGER_RUNS / run_id / f"round-{n:02d}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "profile.json").write_text(json.dumps(artifacts["profile"], indent=2) + "\n")
    (d / "diagnosis.md").write_text(artifacts["diagnosis_md"])
    (d / "hypotheses.json").write_text(json.dumps(artifacts["hypotheses"], indent=2) + "\n")
    (d / "kernel.cu").write_text(artifacts["kernel_cu"])
    (d / "kernel.ptx").write_text(artifacts.get("kernel_ptx", ""))
    (d / "diff.patch").write_text(artifacts.get("diff", ""))
    proof_dir = d / "proof"
    proof_dir.mkdir(exist_ok=True)
    (proof_dir / "status.json").write_text(json.dumps(artifacts["proof"], indent=2) + "\n")
    for name, src in (artifacts.get("lean_modules") or {}).items():
        (proof_dir / f"{name}.lean").write_text(src)
    log = artifacts.get("proof_log")
    if log:
        (proof_dir / "build.log").write_text(log)
    return d


def commit_round(run_id: str, n: int, kernel_id: str, codes: list[str], transform_id: str | None,
                 sol_before: float, sol_after: float, p50_before: float, p50_after: float,
                 proof_status: str, artifacts: dict) -> tuple[str, str]:
    """Write artifacts, commit them, and return (sha, message)."""
    ensure_repo()
    write_round_dir(run_id, n, artifacts)
    msg = (f"round {n}: {kernel_id} [{','.join(codes) or 'none'}] {transform_id or 'baseline'} — "
           f"sol {sol_before:.1f}%→{sol_after:.1f}%, "
           f"p50 {p50_before:.3f}ms→{p50_after:.3f}ms, proof {proof_status}")
    body = "\n".join([
        f"variant: {artifacts['variant_id']}",
        f"hypotheses: {', '.join(artifacts.get('hypothesis_ids') or ['-'])}",
        f"artifacts: ledgers/runs/{run_id}/round-{n:02d}/",
    ])
    _git("add", "-A")
    proc = subprocess.run(["git", "commit", "-m", msg, "-m", body], cwd=ROOT,
                          capture_output=True, text=True)
    if proc.returncode != 0 and "nothing to commit" not in (proc.stdout + proc.stderr):
        raise RuntimeError(f"git commit failed: {proc.stderr.strip()}")
    sha = _git("rev-parse", "HEAD", check=False)
    return sha, msg


def rebuild_index(preferred: str | None = None) -> None:
    """Rewrite `ledgers/REPORT.md` and `ledgers/index.json` from the run tree."""
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    index = []
    for d in (sorted(LEDGER_RUNS.iterdir()) if LEDGER_RUNS.is_dir() else []):
        if not d.is_dir():
            continue
        m = _run_meta(d)
        if not m:
            continue
        index.append({
            "run_id": d.name,
            "kernel_id": m.get("kernel_id"),
            "gpu": m.get("gpu"),
            "backend": m.get("backend"),
            "status": m.get("status"),
            "stop_reason": m.get("stop_reason"),
            "rounds": m.get("rows"),
            "best_sol_pct": m.get("best_sol_pct"),
            "verified_rounds": m.get("verified"),
            "report": f"runs/{d.name}/REPORT.md",
        })
    if preferred:
        index.sort(key=lambda r: (r["run_id"] != preferred, r["run_id"]))
    else:
        index.sort(key=lambda r: r["run_id"])

    top = [
        "# Kernel discovery ledger",
        "",
        "Every round of every harness run is committed here: the measured profile, the "
        "diagnosis, the ranked hypotheses, the kernel source and PTX, the diff, and the "
        "Lean verification status with its module. Numbers come from a real GPU "
        "(`backend: modal`) or from the deterministic simulator (`backend: sim`, which "
        "can never satisfy the SLA).",
        "",
        "| run | kernel | backend | status | rounds | verified | best SOL % | report |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in index:
        best = r["best_sol_pct"]
        best_s = f"{best:.1f}" if best is not None else ""
        status = r["status"] or "n/a"
        if r["stop_reason"]:
            status += f" / {r['stop_reason']}"
        top.append(
            f"| `{r['run_id']}` | {r['kernel_id'] or 'n/a'} | {r['backend'] or 'n/a'} | "
            f"{status} | {r['rounds'] if r['rounds'] is not None else ''} | "
            f"{r['verified_rounds'] if r['verified_rounds'] is not None else ''} | "
            f"{best_s} | [{r['run_id']}]({r['report']}) |")
    (LEDGER_DIR / "REPORT.md").write_text("\n".join(top) + "\n")
    (LEDGER_DIR / "index.json").write_text(json.dumps(index, indent=2) + "\n")


def _round_table(rows: list[dict]) -> list[str]:
    lines = [
        "| round | transform | sol % | p50 ms | proof | commit |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['n']} | `{r.get('transform_id') or 'baseline'}` | {r.get('sol_pct', 0):.1f} | "
            f"{r.get('p50_ms', 0):.3f} | {r.get('proof')} | `{(r.get('sha') or '')[:12]}` |")
    return lines


def _run_meta(run_dir: Path) -> dict:
    """Read a run's summary from `summary.json` (written by `regenerate_report`)."""
    path = run_dir / "summary.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def regenerate_report(rows: list[dict], run_id: str, *, meta: dict | None = None) -> None:
    """Write this run's `REPORT.md`/`index.json` and the ledger-wide index.

    A run owns `ledgers/runs/<run>/REPORT.md`; `ledgers/REPORT.md` lists every run in
    the tree so the repository is browsable on GitHub.
    """
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = LEDGER_RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {"run_id": run_id, **(meta or {})}
    summary["rows"] = len(rows)
    summary["best_sol_pct"] = max((r.get("sol_pct", 0.0) for r in rows), default=0.0)
    summary["p50_ms"] = rows[-1].get("p50_ms") if rows else None
    summary["verified"] = sum(1 for r in rows if r.get("proof") == "VERIFIED")

    lines = [
        f"# Run `{run_id}`",
        "",
        f"Kernel `{summary.get('kernel_id', 'n/a')}` · GPU `{summary.get('gpu', 'n/a')}` · "
        f"backend `{summary.get('backend', 'n/a')}` · status **{summary.get('status', 'n/a')}**"
        f"{' (' + summary['stop_reason'] + ')' if summary.get('stop_reason') else ''}",
        "",
        "Each row is one committed round; `proof` is the Lean verification status and "
        "`commit` its git sha.",
        "",
    ] + _round_table(rows)
    (run_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    (run_dir / "index.json").write_text(json.dumps(rows, indent=2) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    rebuild_index(preferred=run_id)
