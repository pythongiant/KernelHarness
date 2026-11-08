import Harness.PTX.Observer

/-!
# Paper Fig. 1 sanity proof: an initialization-sourced observer read

One thread stores `7` to global byte 8; another thread loads that byte. An
execution in which the load returns the initial value (a *stale* observer read) is
inconsistent, while the execution that observes the store is consistent, and its
observable memory is `7` at byte 8.
-/

namespace Harness.Tests.Observer

open Harness.PTX

set_option maxRecDepth 200000

def prog : Program := ⟨#[
  ⟨.st .global .u32 .v1 .cg, [], [.mem .global 8 0 0 0 0 0, .imm 7], none⟩,
  ⟨.ld .global .u32 .v1 .ca, [.reg 0], [.mem .global 8 0 0 0 0 0], none⟩
]⟩

def initMap : ByteMap := ⟨fun _ => 3⟩

def launch : Launch :=
  ⟨prog, ⟨(1, 1, 1), (2, 1, 1), (1, 1, 1), 2⟩, initMap, fun _ => .ofU32 0⟩

/-- Loads return the initial byte (stale). -/
def stale : Candidate launch := ⟨fun _ => launch.codeLength, fun _ _ => .ofU32 3⟩

/-- Loads return the stored value. -/
def fresh : Candidate launch := ⟨fun _ => launch.codeLength, fun _ _ => .ofU32 7⟩

theorem stale_is_complete : stale.Complete := fun _ => rfl

theorem fresh_is_complete : fresh.Complete := fun _ => rfl

/-- A stale, initialization-sourced read is rejected by the consistency axioms. -/
theorem stale_observer_read_inconsistent : ¬ Consistent stale := by decide

/-- Observing the store is consistent. -/
theorem fresh_observer_read_consistent : Consistent fresh := by decide

/-- The observable memory of the consistent execution has the stored byte. -/
theorem observe_has_store : (observe fresh).get 8 = 7 := by decide

end Harness.Tests.Observer
