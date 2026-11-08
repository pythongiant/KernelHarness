import Lean
import Harness.PTX.Observer
import Harness.Lib.Core

/-!
# Axiom audit

Checks that the model's theorems and the reusable lemmas depend only on
`propext`, `Classical.choice` and `Quot.sound` — no `sorryAx`, no `native_decide`.
Generated per-variant files call `Harness.Audit.audit` with their own theorem names
and `#eval "AUDIT_OK"`.
-/

namespace Harness.Audit

open Lean Elab Command

/-- Axioms permitted in a `VERIFIED` proof. -/
def allowedAxioms : List Name := [``propext, ``Classical.choice, ``Quot.sound]

/-- Fail the build when any listed declaration depends on an unexpected axiom. -/
def audit (names : Array Name) : CommandElabM Unit := do
  let env ← getEnv
  for n in names do
    unless env.contains n do
      throwError "audit: unknown declaration {n}"
    let axioms ← liftCoreM <| Lean.collectAxioms n
    let bad := axioms.filter (fun a => !allowedAxioms.contains a)
    unless bad.isEmpty do
      throwError "audit: {n} depends on unexpected axioms {bad.toList}"

end Harness.Audit

run_cmd Harness.Audit.audit #[
  ``Harness.PTX.nonvacuous,
  ``Harness.PTX.correct,
  ``Harness.PTX.correct_of_canonical,
  ``Harness.Lib.observer_latest_write,
  ``Harness.Lib.observe_untouched,
  ``Harness.Lib.bar_sync_visible,
  ``Harness.Lib.ticket_uniqueness,
  ``Harness.Lib.loop_invariant_frame]

#eval "AUDIT_OK"
