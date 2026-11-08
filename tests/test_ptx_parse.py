from pathlib import Path

import pytest

from harness.translator.parse import UnsupportedInstruction, parse_ptx_file

FIX = Path(__file__).parent / "fixtures"


def test_affine_fixture_counts_and_registers():
    r = parse_ptx_file(FIX / "affine.ptx")
    assert len(r["program"]) == 9
    ops = [i["op"].split(" ")[0] for i in r["program"]]
    assert ops[0] == "Op.paramLd" and ops[-1] == "Op.ret"
    assert any(o == "Op.mul" for o in ops)
    assert any(o == "Op.st" for o in ops)
    assert len(r["decls"]) >= 6


def test_label_resolution_and_async_forms():
    text = """
    .version 8.0
    .visible .entry k()
    {
        .reg .u32 %r<4>;
        mov.u32 %r1, 0;
    loop:
        add.u32 %r1, %r1, 1;
        setp.lt.u32 %p1, %r1, 4;
        @%p1 bra loop;
        bar.sync 0;
        cp.async.ca.shared.global [%r1], [%r2], 16;
        cp.async.commit_group;
        cp.async.wait_group 0;
        ret;
    }
    """
    r = parse_ptx_file.__globals__["parse_ptx"](text)
    assert r["labels"]["loop"] == 1
    branch = [i for i in r["program"] if i["op"].startswith("Op.bra")][0]
    assert branch["op"] == "Op.bra 1"
    assert branch["pred"] is not None
    ops = {i["op"].split(" ")[0] for i in r["program"]}
    assert {"Op.barSync", "Op.cpAsyncCa", "Op.cpAsyncCommitGroup", "Op.cpAsyncWaitGroup"} <= ops


def test_unknown_opcode_is_rejected_with_name():
    with pytest.raises(UnsupportedInstruction) as exc:
        parse_ptx_file(FIX / "unknown_op.ptx")
    assert exc.value.opcode == "fancy.new.opcode"
    assert exc.value.line == 8


def test_tiled_mm_tensor_core_forms():
    r = parse_ptx_file(FIX / "tiled_mm.ptx")
    ops = [i["op"].split(" ")[0] for i in r["program"]]
    assert "Op.mmaSyncM16n8k16" in ops
    assert "Op.ldmatrixX4" in ops
