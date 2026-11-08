# The Lean model of PTX

`lean/Harness/` is self-contained Lean 4 (`leanprover/lean4:v4.35.0`) with **no
Mathlib**: memory is bytes, specs are predicates over bytes, and the proofs are
therefore checkable offline in seconds.

## Layers

| module | contents |
|---|---|
| `PTX.Basic` | `Addr`, `Byte`, `ByteMap` (total map + `get`/`set`/`ext`), `Value` (`b8`, `u32`, `u64`, `f32`), `RegFile`, `Oracle` |
| `PTX.Instr` | qualifier enums (`Space`, `Proxy`, `Scope`, `Sem`, `Cache`, `Vec`, `Ty`) and the **81** `Op` forms, one constructor per form; `Operand` with the affine memory addressing |
| `PTX.Program` | `Program`, `Grid`, `Launch` |
| `PTX.Event` | `Event` (id, tid, kind, proxy, space, addr, size, value, sem, gen, group, lane) and executable footprint tests |
| `PTX.Thread` | `step`, `run`, `traceOf`, the small-step machine |
| `PTX.Execution` | `Candidate`, relations, the eight consistency axioms |
| `PTX.Observer` | `observe`, `apiSync`, `compose`, `Correct`, `Admissible`, `nonvacuous`, `correct`, `correct_of_canonical` |
| `PTX.Tcgen05` | tensor-memory allocations, disjointness, completion via mbarrier rules |
| `PTX.Cluster` | `mapa`/DSMEM addressing and disjointness, cluster barrier matching, cancellation uniqueness |
| `Lib.Core` | lemmas generated proofs call: `lastWriteB_writes`, `lastWriteB_mem`, `observer_latest_write`, `observe_untouched`, `loop_invariant_frame`, `bar_sync_visible`, `ticket_uniqueness`, `disjoint_no_overlap`, `disjoint_no_mixed_size_race` |
| `Tests.BarrierExample` | paper §3.1: stale read rejected, observing read accepted (both by `decide`) |
| `Tests.ObserverExample` | paper Fig. 1: initialization-sourced observer read rejected |
| `Audit` | axiom audit + `#eval "AUDIT_OK"` |
| `Tactic` | `harness_auto` (never `sorry`, never `admit`) |

## Executions are derived, not supplied

```lean
structure Candidate (l : Launch) where
  steps  : Nat → Nat            -- per-thread step counts
  oracle : Nat → Nat → Value    -- per-thread external data

abbrev Candidate.events (c : Candidate l) : List Event := traceOf l c.steps c.oracle
```

Because events come out of `step`, the axioms constrain *the program's* executions
instead of an author's event list. `Complete c` means every thread ran its program to
the end.

Relations: `po`, `co` (writes to one address), `rf` (the latest write to a byte before
the read), `sc`, `sw` (barrier arrive → return, same generation), `morallyStrong`,
`causality = (po ∪ sw)⁺`, `proxyPreservedCausality`.

## The eight consistency axioms

Each is an executable `Bool` checker, so a concrete execution is machine-checkable by
`decide` and a failure names the axiom:

| axiom | statement |
|---|---|
| `valueAgreement` | every read byte equals a write's value or the initial byte |
| `coherenceAxiom` | writes to one address are ordered by `id`; a read observes the latest preceding write |
| `atomicityAxiom` | atomic RMW events are unique per (thread, generation) |
| `scPerLocation` | accesses to one address are totally ordered |
| `causalityAxiom` | `(po ∪ sw)⁺` is acyclic, checked as "every edge strictly increases `id`", which is equivalent on a finite trace with distinct ids |
| `noMixedSizeRace` | no conflicting access pair of different width |
| `constructRules` | `barrierGenerationMatch ∧ mbarrierPhaseComplete ∧ asyncCompletionGroup ∧ shuffleGroupMatch ∧ clusterCancelUniqueness` |
| `proxyFenceRule` | a cross-proxy read of a written byte needs an intervening fence |

## Top-level statements

```lean
def Correct (l : Launch) (post : Post) : Prop :=
  ∀ (e : Candidate l), e.oracle = (Candidate.canonical l).oracle →
    Consistent e → e.Complete → post (observe e)

theorem nonvacuous (l : Launch) (h : l.Admissible) :
    ∃ e : Candidate l, Consistent e ∧ e.Complete

theorem correct (l : Launch) (post : Post) (h : l.Admissible)
    (hpost : ∀ (e : Candidate l), Consistent e → e.Complete → post (observe e)) :
    Correct l post := hpost

theorem correct_of_canonical (l : Launch) (post : Post)
    (hc : Consistent (Candidate.canonical l))
    (hpost : post (observe (Candidate.canonical l))) : Correct l post
```

`Admissible` bundles the well-formedness hypotheses (non-empty grid, initialized
inputs, defined parameters) **and** `canonicalConsistent`, which generated proofs
*prove* by `decide` rather than assume. `correct_of_canonical` is what makes the
per-variant obligation concrete: complete executions of a launch observe the same
memory, so the canonical trace is representative.

## How a generated proof looks

`lean/Harness/Generated/<Variant>.lean` (abridged):

```lean
abbrev instrs : Array Instr := #[
  Instr.mk (Op.mov .s32) [Operand.reg 0] [Operand.imm 10] none,
  …
]

abbrev launch : Launch := ⟨{ instrs := instrs }, grid, ⟨initBytes⟩, fun _ => .u32 0⟩
abbrev refBytes : List Byte := [ … ]
abbrev post : Post := fun m =>
  (List.range refBytes.length).all (fun i => m.get (8192 + i) == refBytes.getD i 0) = true

theorem canonConsistent : Consistent (Candidate.canonical launch) := by decide
theorem canonPost : post (observe (Candidate.canonical launch)) := by decide
theorem admissible : launch.Admissible := ⟨by decide, …, canonConsistent⟩
theorem nonvacuous : ∃ e : Candidate launch, Consistent e ∧ e.Complete :=
  Harness.PTX.nonvacuous launch admissible
theorem correct : Correct launch post :=
  correct_of_canonical launch post canonConsistent canonPost

run_cmd Harness.Audit.audit #[``correct, ``nonvacuous]
#eval "AUDIT_OK"
```

`decide` is used on *concrete* traces: Lean evaluates the program's own arithmetic, so
the emitted `refBytes` are checked by the kernel, not copied from it.

## Status decision

The verifier (`harness/translator/emit.py:check`) never upgrades a result:

| observation | status |
|---|---|
| module compiles, no `sorry`, audit printed `AUDIT_OK` | `VERIFIED` |
| open goals or `sorry` remain | `PARTIAL` (goals listed) |
| Lean rejects the module | `FAILED` (errors listed) |
| opcode or dtype outside the model | `UNSUPPORTED` (reason recorded) |

## Known modelling limits

* **Thread-local step machine.** Loads return the oracle byte; cross-thread
  visibility is axiomatic. Intra-launch producer/consumer kernels therefore fail
  `coherenceAxiom` (conservative) instead of being accepted unsoundly.
* **`exact real arithmetic; IEEE rounding not modeled`.** No float equality in
  proofs; `f32`/tensor-core/softmax kernels are `UNSUPPORTED`.
* **Bounded instance.** Proofs are about a concrete small launch of the same program;
  the `instantiation` field records it.
* **`Complete` fixes step counts**, so a proof is about the modelled program run to
  completion, not about a preempted or partial execution.
