import harness.stages.research as research
from harness.model import Variant
from harness.registry import get


def _variant(level=0, params=None):
    k = get("elementwise_affine")
    return Variant.build(
        variant_id="v",
        run_id="r",
        round_n=0,
        kernel=k,
        level=level,
        transform_id=None,
        params=params or {},
    )


def test_catalog_picks_vectorize_for_memory_bound_elementwise():
    b = [{"code": "MEM_BW_BOUND", "severity": "high", "evidence_md": "x"}]
    hyps = research.propose(b, {"sol_pct": 30.0}, get("elementwise_affine"), _variant())
    assert hyps[0]["transform_id"] == "vectorize_x4"
    assert hyps[0]["source"] == "catalog"
    assert hyps[0]["refs"]


def test_applied_transform_is_not_proposed_again():
    b = [{"code": "MEM_BW_BOUND", "severity": "high", "evidence_md": "x"}]
    hyps = research.propose(
        b, {}, get("elementwise_affine"), _variant(level=1, params={"vector_width": 4})
    )
    assert all(h["transform_id"] != "vectorize_x4" for h in hyps)


def test_llm_unknown_transform_is_rejected(monkeypatch):
    class Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"hypotheses":[{"transform_id":"magic_wand","rationale_md":"no",'
                            '"refs":[],"expected_effect":"?"}]}'
                        }
                    }
                ]
            }

    class Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url):
            return Resp()

        def post(self, url, json=None):
            return Resp()

    import httpx

    monkeypatch.setattr(httpx, "Client", Client)
    monkeypatch.setenv("HARNESS_LLM_BASE_URL", "http://127.0.0.1:1/v1")
    b = [{"code": "MEM_BW_BOUND", "severity": "high", "evidence_md": "x"}]
    hyps = research.propose(b, {}, get("elementwise_affine"), _variant(), llm=True)
    assert all(h["transform_id"] in research.KNOWN_TRANSFORMS for h in hyps)
    assert not any(h["transform_id"] == "magic_wand" for h in hyps)


def test_no_hypotheses_when_all_transforms_applied():
    b = [{"code": "MEM_BW_BOUND", "severity": "low", "evidence_md": "x"}]
    v = _variant(level=3, params={"vector_width": 4, "streaming": True, "grid_stride": True})
    assert research.propose(b, {}, get("elementwise_affine"), v) == []
