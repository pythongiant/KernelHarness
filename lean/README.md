# Lean model of PTX

Core Lean 4 only (`leanprover/lean4:v4.35.0`). No Mathlib: the model and the specs
are byte-level (`ByteMap : Addr → UInt8`), so nothing needs `ℝ` or the Mathlib
analysis library.

## Build and check

```bash
cd lean
lake build                                 # library: PTX model, Lib, tests, audit
lake env lean Harness/Tests/BarrierExample.lean
lake env lean Harness/Tests/ObserverExample.lean
lake build Harness.Audit                   # prints "AUDIT_OK"
```

Generated per-variant modules (`Harness/Generated/<Variant>.lean`) are *not* part of
the library glob: the verifier checks each one with
`lake env lean Harness/Generated/<Variant>.lean`, so a failed proof for one variant
never breaks the library build.

## Optional Mathlib manifest

`lakefile.mathlib.toml` adds Mathlib at tag `v4.35.0` for environments that have disk
for the cache (`lake exe cache get` needs ~10 GB; building from source needs more).
It is not required and is not used by the harness: with it you can restate specs over
`ℝ` and prove a floating-point correspondence argument, which is exactly the part the
harness reports as `numerical_scope: exact real arithmetic; IEEE rounding not modeled`.

## Module map

| module | contents |
|---|---|
| `Harness.PTX.Basic` | `Addr`, `Byte`, `ByteMap`, `Value`, `RegFile`, `Oracle` |
| `Harness.PTX.Instr` | qualifier enums and the 81 `Op` forms |
| `Harness.PTX.Program` | `Program`, `Grid`, `Launch` |
| `Harness.PTX.Event` | `Event`, `EventKind`, executable footprint tests |
| `Harness.PTX.Thread` | `step`, `run`, `traceOf` (the §3.3 machine) |
| `Harness.PTX.Execution` | `Candidate`, relations, the eight consistency axioms |
| `Harness.PTX.Observer` | `observe`, `apiSync`, `compose`, `Correct`, `nonvacuous`, `correct` |
| `Harness.PTX.Tcgen05` / `.Cluster` | Blackwell tensor memory, DSMEM/cluster barrier rules |
| `Harness.Lib.Core` | lemmas the generated proofs use |
| `Harness.Audit` | axiom audit (`propext`, `Classical.choice`, `Quot.sound`) |
| `Harness.Tactic` | `harness_auto`, `harness_lib` |

## Modelled scope

* A `Candidate` is a per-thread step-count schedule plus an oracle; its event trace is
  derived by `step`, so the axioms constrain the program's real executions.
* The step machine is thread-local: a load returns its oracle byte. Cross-thread
  visibility therefore enters through the consistency axioms (`valueAgreement` +
  `coherenceAxiom` pin every read to the latest write to its byte). A kernel whose
  threads communicate *within* a launch is rejected rather than accepted unsoundly.
* `Complete` means every thread ran its program to the end; `Correct` quantifies over
  consistent, complete executions that read memory through the launch's canonical
  oracle.
