import Harness.PTX.Observer

/-!
# Paper §3.1 sanity proof: `st` / `bar.sync` / `ld`

Thread 0 stores `1` to global byte 0 and then hits `bar.sync`; thread 1 hits
`bar.sync` and then loads the byte. The paper's claim is that the load must
observe the store. In this model the load's value is supplied by the execution's
oracle, and the consistency axioms (`coherenceAxiomB`: a read observes the latest
preceding write to its byte) decide whether such an execution is allowed:

* an execution whose load returns the *initial* value `0` is **inconsistent**,
* an execution whose load returns the stored value `1` is **consistent**.

Both are machine-checked below by evaluation of the consistency checkers.
-/

namespace Harness.Tests.Barrier

open Harness.PTX

set_option maxRecDepth 200000

/-- `st.global.u32 [0], 1 ; bar.sync 0` for thread 0 and `bar.sync 0 ;
ld r0, [0]` for thread 1 (both run the same program). -/
def prog : Program := ⟨#[
  ⟨.st .global .u32 .v1 .cg, [], [.mem .global 0 0 0 0 0 0, .imm 1], none⟩,
  ⟨.barSync 0, [], [], none⟩,
  ⟨.ld .global .u32 .v1 .ca, [.reg 0], [.mem .global 0 0 0 0 0 0], none⟩
]⟩

def initMap : ByteMap := ⟨fun _ => 0⟩

def launch : Launch :=
  ⟨prog, ⟨(1, 1, 1), (2, 1, 1), (1, 1, 1), 2⟩, initMap, fun _ => .ofU32 0⟩

/-- An execution whose loads all return the initial bytes (stale read). -/
def stale : Candidate launch := ⟨fun _ => launch.codeLength, fun _ _ => .ofU32 0⟩

/-- An execution whose loads return the value the store wrote. -/
def fresh : Candidate launch := ⟨fun _ => launch.codeLength, fun _ _ => .ofU32 1⟩

theorem stale_is_complete : stale.Complete := fun _ => rfl

theorem fresh_is_complete : fresh.Complete := fun _ => rfl

/-- The stale read is rejected: the read must observe the preceding store. -/
theorem stale_read_inconsistent : ¬ Consistent stale := by decide

/-- The read that observes the store is consistent. -/
theorem fresh_read_consistent : Consistent fresh := by decide

end Harness.Tests.Barrier
