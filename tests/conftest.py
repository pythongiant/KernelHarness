import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    """Isolate the DB, ledger tree and git history inside a temp directory."""
    import harness.config as config
    import harness.db as db
    import harness.stages.ledger as ledger

    state = tmp_path / "state"
    ledger_dir = tmp_path / "ledgers"
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)

    monkeypatch.setattr(config, "STATE_DIR", state)
    monkeypatch.setattr(db, "STATE_DIR", state)
    monkeypatch.setattr(db, "db_path", lambda: state / "harness.db")
    monkeypatch.setattr(ledger, "LEDGER_DIR", ledger_dir)
    monkeypatch.setattr(ledger, "LEDGER_RUNS", ledger_dir / "runs")
    monkeypatch.setattr(ledger, "ROOT", repo)

    state.mkdir(parents=True, exist_ok=True)
    conn = db.init_db()
    yield conn, repo, ledger_dir
    conn.close()


@pytest.fixture()
def conn(isolated):
    return isolated[0]
