import Harness.PTX.Program

/-!
# Events emitted by the small-step semantics
-/


namespace Harness.PTX

inductive EventKind where
  | read | write | barArrive | barReturn | mbarInit | mbarArrive | mbarWait
  | asyncIssue | asyncComplete | fence | wgmmaIssue | wgmmaCommit | wgmmaWait
  | shfl | atomRmw | clusterCancel | exit
  deriving Repr, DecidableEq, Inhabited

/-- One observable step of one thread. -/
structure Event where
  id : Nat
  tid : Nat
  kind : EventKind
  proxy : Proxy
  space : Space
  addr : Addr
  size : Nat
  value : Addr → Option Byte
  sem : Option Sem
  gen : Nat
  group : Nat
  lane : Option Nat

namespace Event

/-- Is byte `a` inside this event's footprint? -/
abbrev covers (e : Event) (a : Addr) : Bool :=
  decide (e.addr ≤ a && a < e.addr + e.size)

/-- Executable form of `writesByte`. -/
abbrev writesBool (e : Event) (a : Addr) : Bool :=
  e.kind == EventKind.write && e.covers a

/-- Executable form of `readsByte`. -/
abbrev readsBool (e : Event) (a : Addr) : Bool :=
  e.kind == EventKind.read && e.covers a

/-- Does the event access byte `a` at all? -/
abbrev touches (e : Event) (a : Addr) : Bool := e.covers a

/-- Executable overlap test for two footprints. -/
abbrev overlapB (a b : Event) : Bool :=
  a.addr < b.addr + b.size && b.addr < a.addr + a.size

/-- Program order, executable. -/
abbrev poB (a b : Event) : Bool := a.tid == b.tid && a.id < b.id

/-- Does this event write byte `a`? -/
def writesByte (e : Event) (a : Addr) : Prop :=
  e.kind = EventKind.write ∧ e.addr ≤ a ∧ a < e.addr + e.size

/-- Does this event read byte `a`? -/
def readsByte (e : Event) (a : Addr) : Prop :=
  e.kind = EventKind.read ∧ e.addr ≤ a ∧ a < e.addr + e.size

/-- Byte written by this event at address `a`: the event's value function,
falling back to none outside its footprint. -/
def valueAt (e : Event) (a : Addr) : Option Byte :=
  if e.addr ≤ a ∧ a < e.addr + e.size then e.value a else none

theorem writesByte_kind {e : Event} {a : Addr} (h : e.writesByte a) : e.kind = EventKind.write :=
  h.1

theorem writesByte_erase (e : Event) (a : Addr) (b : Byte) (h : e.value a = some b) :
    ({ e with value := fun a' => if a' = a then some b else e.value a' } : Event).value a = some b := by
  simp [h]

end Event

end Harness.PTX
