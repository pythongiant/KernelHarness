import Harness.PTX.Observer
import Harness.Lib.Core

/-!
# `harness_auto`

The tactic used by generated per-kernel proofs. It never closes a goal with
`sorry`/`admit`: whatever it cannot prove stays as an open goal, which the verifier
reports as `PARTIAL`.
-/

namespace Harness.Tactic

open Lean Meta Elab Tactic

/-- Simplification set contributed by `Harness.Lib`. -/
macro "harness_lib" : tactic =>
  `(tactic|
    simp only [Harness.Lib.observer_latest_write, Harness.Lib.observe_untouched,
      Harness.Lib.loop_invariant_frame, Harness.PTX.observe, Harness.PTX.lastWrite])

end Harness.Tactic

/--
Discharge a generated per-kernel obligation, in order:

1. the goal is already a hypothesis (the `hpost` shape),
2. `decide` — consistency/post obligations over the concrete trace,
3. `simp_all` + the `Harness.Lib` simp set,
4. `omega` for the residual arithmetic.

On failure the remaining goals are reported, never closed by assumption.
-/
macro "harness_auto" : tactic =>
  `(tactic|
    (first
      | (intros; assumption)
      | decide
      | (intros; simp_all only [Harness.PTX.observe, Harness.PTX.lastWrite,
          Harness.Lib.observer_latest_write, Harness.Lib.observe_untouched])
      | omega
      | (intros; done)))
