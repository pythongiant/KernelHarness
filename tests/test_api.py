import json
import time

from fastapi.testclient import TestClient


def _client(monkeypatch):
    import harness.server as server
    monkeypatch.setattr(server, "_conn", None)
    monkeypatch.setattr(server, "_subscribers", {})
    return TestClient(server.app)


def test_registry_endpoint(monkeypatch):
    with _client(monkeypatch) as c:
        r = c.get("/api/registry")
        assert r.status_code == 200
        ids = {k["id"] for k in r.json()["kernels"]}
        assert {"elementwise_affine", "matmul_scalar", "matmul_tiled",
                "attention_scalar"} <= ids
        ew = [k for k in r.json()["kernels"] if k["id"] == "elementwise_affine"][0]
        # the traffic model is read x plus write out only: 8 bytes per s32 element
        assert ew["sol"]["bytes_resolved"] == 8 * 4194304
        assert ew["geometry"]["grid"] == 4194304 // 256


def test_run_lifecycle_and_artifacts(isolated, monkeypatch):
    conn, repo, ledger_dir = isolated
    with _client(monkeypatch) as c:
        r = c.post("/api/runs", json={"kernel_id": "elementwise_affine", "gpu": "H100",
                                      "backend": "sim",
                                      "sla": {"max_rounds": 2, "stall_rounds": 2}})
        assert r.status_code == 200
        run_id = r.json()["run_id"]

        deadline = time.time() + 180
        snap = None
        while time.time() < deadline:
            snap = c.get(f"/api/runs/{run_id}").json()
            if snap["status"] != "running":
                break
            time.sleep(0.5)
        assert snap["status"] != "running", snap["status"]
        assert snap["rounds"], "no rounds recorded"
        assert snap["commits"], "no ledger commit"
        variant_id = snap["commits"][0]["variant_id"] if "variant_id" in snap["commits"][0] \
            else snap["variants"][-1]["id"]

        diff = c.get(f"/api/variants/{variant_id}/diff")
        assert diff.status_code == 200
        assert "---" in diff.text and "+++" in diff.text

        src = c.get(f"/api/variants/{variant_id}/source")
        assert src.status_code == 200 and "launch_kernel" in src.text

        proof = c.get(f"/api/variants/{variant_id}/proof")
        assert proof.status_code == 200
        body = proof.json()
        assert body["status"] in ("VERIFIED", "PARTIAL", "FAILED", "UNSUPPORTED")
        assert body["numerical_scope"].startswith("exact real arithmetic")

        report = c.get(f"/api/runs/{run_id}/report")
        assert report.status_code == 200 and "numerical_scope" in report.text

        # the snapshot's events cover every stage of the loop
        stages = {e["stage"] for e in snap["events"]}
        assert {"DIAGNOSE", "RESEARCH", "OPTIMIZE", "VERIFY", "PROFILE",
                "LEDGER"} <= stages

        # SSE replay yields at least six stage frames
        frames = []
        with c.stream("GET", f"/api/runs/{run_id}/stream") as stream:
            for line in stream.iter_lines():
                if line.startswith("event: "):
                    frames.append(line[len("event: "):])
                if frames.count("stage") >= 6:
                    break
        assert frames.count("stage") >= 6
        assert "sync" in frames


def test_stop_unknown_run_is_404(monkeypatch):
    with _client(monkeypatch) as c:
        assert c.post("/api/runs/nope/stop").status_code == 404
        assert c.get("/api/runs/nope").status_code == 404
