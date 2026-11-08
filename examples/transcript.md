# Transcript: one full run on an H100

Captured from the committed run
[`run_69cbe7246706`](../ledgers/runs/run_69cbe7246706/REPORT.md). The kernel starts at 63.6
percent of the memory roofline, climbs to 82.8 and then 94.3 percent as each rung of the
ladder is applied, and stops when every gate passes (`SLA_MET`).

Below is what `harness run` printed, with long diffs and logs removed. The same events
are the rows the web UI renders, so the transcript, the UI and the ledger cannot drift
apart.

```text
$ uv run harness run --kernel elementwise_affine --gpu H100 --backend modal --max-rounds 4 --sol 40

[-        ] log         run started
            kernel `elementwise_affine`, gpu `H100`, backend `modal`, max_rounds 4.
[PROFILE  ] measurement baseline measurement
            `p50_ms=0.016`, `sol_pct=63.6`, `bound=memory`, `numeric=PASS`, `simulated=False`.
[VERIFY   ] stage_start VERIFY: baseline proof
            Checking the unmodified kernel against the model before any optimisation, so the track record starts from a verified baseline.
[VERIFY   ] decision    baseline proof VERIFIED
            Status `VERIFIED`; open goals: none
[LEDGER   ] stage_start LEDGER: commit baseline
            `ledgers/runs/run_69cbe7246706/round-00/`
[LEDGER   ] artifact    committed baseline
            `9fe3c5a937e8`, round 0: elementwise_affine [none] baseline, sol 63.6%→63.6%, p50 0.016ms→0.016ms, proof VERIFIED
[PROFILE  ] reasoning   gate check before round 1
            SLA not met yet: `sol`=pass, `speedup`=fail, `latency`=pass, `verified`=pass, `numeric`=pass, `measured`=pass.
[DIAGNOSE ] stage_start DIAGNOSE: diagnose round 1
            Analysing the parent variant `run_69cbe7246706-v0` (`sol_pct=63.6`, `p50_ms=0.016`, `bound=memory`).
[DIAGNOSE ] stage_end   DIAGNOSE done
[DIAGNOSE ] reasoning   bottlenecks
            - **MEM_BW_BOUND** (high): Roofline says memory-bound at `63.62%` of peak: `eff_bw_gbps = 2131.25` of `3350.00` GB/s, moving `33554432` bytes in `
[RESEARCH ] stage_start RESEARCH: rank fix candidates
            1 diagnosed bottleneck(s) -> catalog lookup, severity-ranked.
[RESEARCH ] reasoning   ranked hypotheses
            **vectorize_x4 for MEM_BW_BOUND** (source `catalog`)
            
            The kernel issues one 4-byte load and one 4-byte store per element, so the request
            count (not the byte count) is the limit: 32-byte sectors are half-used. Loading and
[RESEARCH ] stage_end   RESEARCH produced 1 hypotheses
[OPTIMIZE ] stage_start OPTIMIZE: materialize variant
            Taking the top-ranked hypothesis `vectorize_x4`.
[OPTIMIZE ] stage_end   OPTIMIZE built `run_69cbe7246706-v0-vectorize_x4-r1`
            level 1, params `{'vector_width': 4}`, sha256 `cc6bc1a5ffbf`.
[OPTIMIZE ] artifact    variant `run_69cbe7246706-v0-vectorize_x4-r1`
            Applied `vectorize_x4` at level 1.
            
            ```diff
            --- a/kernel.cu
[VERIFY   ] stage_start VERIFY: emit Lean and check
            Lowering the template to a concrete instruction program, computing reference bytes, emitting the Lean module and running `lake env lean`.
[VERIFY   ] decision    proof VERIFIED
            Status `VERIFIED` (model `paper-full`, instantiation `elementwise N=8, a=10, b=3`).
            
            Open goals: none
            
[PROFILE  ] stage_start PROFILE: measure candidate
            Running 50 reps after 25 warmup on `H100` (`modal` backend).
[PROFILE  ] measurement round 1 measurement
            `p50_ms` 0.016 → 0.012, `sol_pct` 63.6 → 82.8, `numeric` PASS, `simulated` False.
[LEDGER   ] stage_start LEDGER: commit round 1
            vendoring profile/diagnosis/hypotheses/kernel/diff/proof into `ledgers/runs/run_69cbe7246706/round-01/`.
[LEDGER   ] artifact    committed round 1
            `a38b60db1a99`, round 1: elementwise_affine [MEM_BW_BOUND] vectorize_x4, sol 63.6%→82.8%, p50 0.016ms→0.012ms, proof VERIFIED
[PROFILE  ] reasoning   gate check before round 2
            SLA not met yet: `sol`=pass, `speedup`=fail, `latency`=pass, `verified`=pass, `numeric`=pass, `measured`=pass.
[DIAGNOSE ] stage_start DIAGNOSE: diagnose round 2
            Analysing the parent variant `run_69cbe7246706-v0-vectorize_x4-r1` (`sol_pct=82.8`, `p50_ms=0.012`, `bound=memory`).
[DIAGNOSE ] stage_end   DIAGNOSE done
[DIAGNOSE ] reasoning   bottlenecks
            - **MEM_BW_BOUND** (low): Roofline says memory-bound at `82.81%` of peak: `eff_bw_gbps = 2774.01` of `3350.00` GB/s, moving `33554432` bytes in `0
[RESEARCH ] stage_start RESEARCH: rank fix candidates
            1 diagnosed bottleneck(s) -> catalog lookup, severity-ranked.
[RESEARCH ] reasoning   ranked hypotheses
            **streaming_stores for MEM_BW_BOUND** (source `catalog`)
            
            The output buffer is never read again, but the default store allocates the line in L2
            first (write-allocate). `st.global.cs` (via `__stcs`) marks the line as streaming so it
[RESEARCH ] stage_end   RESEARCH produced 1 hypotheses
[OPTIMIZE ] stage_start OPTIMIZE: materialize variant
            Taking the top-ranked hypothesis `streaming_stores`.
[OPTIMIZE ] stage_end   OPTIMIZE built `run_69cbe7246706-v0-vectorize_x4-r1-streaming_stores-r2`
            level 3, params `{'vector_width': 4, 'streaming': True}`, sha256 `50abc7be49c1`.
[OPTIMIZE ] artifact    variant `run_69cbe7246706-v0-vectorize_x4-r1-streaming_stores-r2`
            Applied `streaming_stores` at level 3.
            
            ```diff
            --- a/kernel.cu
[VERIFY   ] stage_start VERIFY: emit Lean and check
            Lowering the template to a concrete instruction program, computing reference bytes, emitting the Lean module and running `lake env lean`.
[VERIFY   ] decision    proof VERIFIED
            Status `VERIFIED` (model `paper-full`, instantiation `elementwise N=8, a=10, b=3`).
            
            Open goals: none
            
[PROFILE  ] stage_start PROFILE: measure candidate
            Running 50 reps after 25 warmup on `H100` (`modal` backend).
[PROFILE  ] measurement round 2 measurement
            `p50_ms` 0.012 → 0.011, `sol_pct` 82.8 → 94.3, `numeric` PASS, `simulated` False.
[LEDGER   ] stage_start LEDGER: commit round 2
            vendoring profile/diagnosis/hypotheses/kernel/diff/proof into `ledgers/runs/run_69cbe7246706/round-02/`.
[LEDGER   ] artifact    committed round 2
            `8cc786565bee`, round 2: elementwise_affine [MEM_BW_BOUND] streaming_stores, sol 82.8%→94.3%, p50 0.012ms→0.011ms, proof VERIFIED
[-        ] decision    run finished: SLA_MET
            Final gate: `sol`=pass, `speedup`=pass, `latency`=pass, `verified`=pass, `numeric`=pass, `measured`=pass.
            
            `sol_pct=94.3`, `p50_ms=0.011`, proof `VERIFIED`.

profile round 0 (baseline        ): sol= 63.6%  p50= 15.74 us  bw= 2131.2 GB/s  numeric=PASS  measured=True
profile round 1 (vectorize_x4    ): sol= 82.8%  p50= 12.10 us  bw= 2774.0 GB/s  numeric=PASS  measured=True
profile round 2 (streaming_stores): sol= 94.3%  p50= 10.62 us  bw= 3158.4 GB/s  numeric=PASS  measured=True
```

## What to read in the ledger

```text
ledgers/runs/run_69cbe7246706/
  round-00/                 baseline, committed before any change
    profile.json            measured numbers, numeric check, baseline speedup
    diagnosis.md            "baseline: no diagnosis before the first measurement"
    kernel.cu               the scalar kernel that was measured
    kernel.ptx              the PTX the GPU actually compiled
    proof/status.json       VERIFIED, including the axiom audit
    proof/*.lean            the generated module Lean checked
    proof/build.log         the Lean log
  round-01/ round-02/
    profile.json            each rung's numbers
    diagnosis.md            the finding, with the metric that fired it
    hypotheses.json         the chosen fix, its rationale and references
    diff.patch              unified diff against the parent
    kernel.cu, kernel.ptx   the variant, and its compiled PTX
  REPORT.md                 the round table for this run
```

## The gate, as printed

```text
Final gate: `sol`=pass, `speedup`=pass, `latency`=pass,
            `verified`=pass, `numeric`=pass, `measured`=pass.
```

Six checks, all required. `verified` needs a clean Lean build, no `sorry` and a clean
axiom audit. `numeric` compares the GPU output with a bit identical numpy reference.
`measured` is false for `--backend sim`, which is why a simulated run can show 71
percent SOL and still not satisfy the gate.
