import asyncio

from harness.backends.base import ProfileJob, ProfileResult
from harness.backends.sim import SimBackend
from harness.config import RunConfig, Sla
from harness.db import create_run, run_snapshot
from harness.orchestrator import run_loop, sla_met


def test_sla_gate_blocks_simulated_numbers():
    prof = {
        "sol_pct": 99.0,
        "p50_ms": 0.001,
        "numeric": "SKIPPED",
        "simulated": True,
        "baseline": {"name": "torch", "speedup": 9.0},
    }
    ok, checks = sla_met(
        prof, {"status": "VERIFIED"}, RunConfig(kernel_id="elementwise_affine", backend="sim")
    )
    assert checks["measured"] is False
    assert ok is False


def test_sim_loop_commits_improvements_and_stalls(isolated):
    conn, ledger_dir = isolated
    cfg = RunConfig(
        kernel_id="elementwise_affine",
        gpu="H100",
        backend="sim",
        sla=Sla(max_rounds=2, stall_rounds=2),
    )
    run_id = create_run(conn, cfg)
    stop = asyncio.run(run_loop(conn, cfg, run_id, backend=SimBackend("H100")))
    snap = run_snapshot(conn, run_id)
    # round 0 is the committed baseline; 1..max_rounds are optimisation attempts
    assert [r["n"] for r in snap["rounds"]] == [0, 1, 2]
    assert snap["status"] in ("max_rounds", "stalled", "sla_met")
    assert stop in ("MAX_ROUNDS", "STALLED", "SLA_MET")
    commits = snap["commits"]
    assert len(commits) == 3
    assert all(c["sha"] for c in commits)
    p50 = [p["profile"]["p50_ms"] for p in snap["profiles"]]
    assert len(p50) >= 2
    assert p50[1] <= p50[0]
    # artifacts landed in the (isolated) ledger tree and were committed
    rounds = list((ledger_dir / "runs" / run_id).glob("round-*"))
    assert rounds and (rounds[0] / "profile.json").exists()
    assert (rounds[0] / "proof" / "status.json").exists()
    assert (ledger_dir / "REPORT.md").exists()


class _FailingNumericBackend(SimBackend):
    def profile(self, job: ProfileJob) -> ProfileResult:
        data = dict(super().profile(job).data)
        data["numeric"] = "FAIL"
        data["max_abs_err"] = 3.0
        return ProfileResult(data)


def test_numeric_failure_blocks_without_commits(isolated):
    conn, ledger_dir = isolated
    cfg = RunConfig(
        kernel_id="elementwise_affine", gpu="H100", backend="sim", sla=Sla(max_rounds=3)
    )
    run_id = create_run(conn, cfg)
    stop = asyncio.run(run_loop(conn, cfg, run_id, backend=_FailingNumericBackend("H100")))
    snap = run_snapshot(conn, run_id)
    assert stop == "BLOCKED"
    assert snap["status"] == "blocked"
    assert snap["commits"] == []
    errors = [e for e in snap["events"] if e["kind"] == "error"]
    assert errors


def test_stop_event_ends_run(isolated):
    conn, ledger_dir = isolated
    cfg = RunConfig(
        kernel_id="elementwise_affine", gpu="H100", backend="sim", sla=Sla(max_rounds=5)
    )
    run_id = create_run(conn, cfg)
    cancel = asyncio.Event()
    cancel.set()
    stop = asyncio.run(run_loop(conn, cfg, run_id, cancel_evt=cancel, backend=SimBackend("H100")))
    assert stop == "STOPPED_BY_USER"


def test_candidate_meeting_the_sla_is_kept_even_within_noise(isolated):
    """A variant that satisfies the gate must be adopted, even if its latency gain is
    under the noise tolerance (otherwise real progress is thrown away)."""
    conn, ledger_dir = isolated
    cfg = RunConfig(
        kernel_id="elementwise_affine",
        gpu="H100",
        backend="sim",
        sla=Sla(max_rounds=2, tolerance_pct=50.0, sol_pct=1.0, speedup=0.1),
    )
    run_id = create_run(conn, cfg)
    asyncio.run(run_loop(conn, cfg, run_id, backend=SimBackend("H100")))
    snap = run_snapshot(conn, run_id)
    assert snap["rounds"][1]["status"] == "improved"


def test_ledger_is_its_own_repository(isolated, tmp_path, monkeypatch):
    """A run commits only ledger files, into the ledger's own repository, and never
    stages or commits anything in the source tree around it."""
    import subprocess

    import harness.stages.ledger as ledger

    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    (source / "work_in_progress.py").write_text("x = 1\n")
    ledger_dir = source / "ledgers"
    monkeypatch.setattr(ledger, "LEDGER_DIR", ledger_dir)
    monkeypatch.setattr(ledger, "LEDGER_RUNS", ledger_dir / "runs")

    conn, _ = isolated
    cfg = RunConfig(kernel_id="elementwise_affine", backend="sim", sla=Sla(max_rounds=1))
    run_id = create_run(conn, cfg)
    asyncio.run(run_loop(conn, cfg, run_id, backend=SimBackend("H100")))

    def git(cwd, *args):
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True).stdout

    assert (ledger_dir / ".git").is_dir()
    assert git(ledger_dir, "rev-list", "--count", "HEAD").strip() == "3"  # 2 rounds + report
    tracked = git(ledger_dir, "ls-files").split()
    assert "REPORT.md" in tracked and "index.json" in tracked
    assert all(p.startswith("runs/") or p in ("REPORT.md", "index.json") for p in tracked)
    # the surrounding repository is untouched: no commits, nothing staged
    assert git(source, "rev-list", "--all") == ""
    assert git(source, "diff", "--cached", "--name-only") == ""
