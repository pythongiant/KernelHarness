import pytest


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    """Isolate the state directory, the database and the ledger repository in tmp_path.

    Yields `(conn, ledger_dir)`. The ledger directory does not exist yet: the ledger
    stage creates it and initialises it as its own git repository on first commit.
    """
    import harness.config as config
    import harness.db as db
    import harness.stages.ledger as ledger

    state = tmp_path / "state"
    ledger_dir = tmp_path / "ledgers"

    monkeypatch.setattr(config, "STATE_DIR", state)
    monkeypatch.setattr(db, "STATE_DIR", state)
    monkeypatch.setattr(db, "db_path", lambda: state / "harness.db")
    monkeypatch.setattr(ledger, "LEDGER_DIR", ledger_dir)
    monkeypatch.setattr(ledger, "LEDGER_RUNS", ledger_dir / "runs")
    monkeypatch.setenv("GIT_AUTHOR_NAME", "test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "test@example.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "test@example.invalid")

    state.mkdir(parents=True, exist_ok=True)
    conn = db.init_db()
    yield conn, ledger_dir
    conn.close()


@pytest.fixture()
def conn(isolated):
    return isolated[0]
