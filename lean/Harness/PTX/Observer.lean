import Harness.PTX.Execution

/-!
# Observer reads, launch admissibility, composition, and correctness

`observe` is the paper's §3.5 observable: after all `apiSync` edges have retired,
the value of every allocated global byte is the value of the last write to it (or
the initial byte, if the launch never wrote it).

`Correct l post` quantifies over every consistent, complete execution of `l`.
-/


namespace Harness.PTX

open Event

/-- The latest write to byte `a` in a trace, ignoring ordering bounds. -/
abbrev lastWrite (evs : List Event) (a : Addr) : Option Event :=
  lastWriteB evs a (evs.length + 1)

/-- Observer-visible memory of an execution: last write per byte, else the
initial byte. -/
abbrev observe {l : Launch} (c : Candidate l) : ByteMap where
  bytes a :=
    match lastWrite c.events a with
    | some w => (w.value a).getD 0
    | none => l.init.get a

/-- Two executions with the same schedule and oracle observe the same memory. -/
theorem observe_congr {l : Launch} {c c' : Candidate l} (h : c.steps = c'.steps)
    (ho : c.oracle = c'.oracle) : observe c = observe c' := by
  have he : c.events = c'.events := Candidate.events_congr h ho
  simp [observe, he]

/-- All complete executions of a launch that share an oracle observe the same
memory (the derived-trace construction). -/
theorem complete_observe_unique {l : Launch} {c c' : Candidate l}
    (hc : c.Complete) (hc' : c'.Complete) (ho : c.oracle = c'.oracle) :
    observe c = observe c' :=
  observe_congr (Candidate.complete_steps hc hc') ho

/-- The `apiSync` edges of a launch: the tail of its trace is visible to the next
kernel of the pipeline. -/
def apiSync {l : Launch} (c : Candidate l) : List Event :=
  c.events.filter (fun e => e.kind == .exit)

/-- Sequential multi-kernel composition (paper §3.5): the successor's global bytes
are equated with the predecessor's observed output. -/
def compose (k1 k2 : Launch) : Launch where
  prog := k2.prog
  grid := k2.grid
  init := observe (Candidate.canonical k1)
  params := k2.params

/-- Post-conditions are predicates on observable memory. -/
abbrev Post := ByteMap → Prop

/-- Correctness: every consistent, complete execution of the launch, reading memory
through the launch's canonical oracle (loads observe initial bytes), satisfies the
post-condition on observable memory.

Kernels whose threads communicate *within* a launch are outside this quantifier;
they are checked directly against the consistency axioms (see
`Harness.Tests.BarrierExample`). -/
def Correct (l : Launch) (post : Post) : Prop :=
  ∀ (e : Candidate l), e.oracle = (Candidate.canonical l).oracle →
    Consistent e → e.Complete → post (observe e)

/-- Admissibility of a launch. Besides the well-formedness predicates (non-empty
grid, initialized input bytes, defined parameters), it states that the launch's
canonical execution satisfies the consistency axioms. That last hypothesis is what
the generated proofs *prove* (by `decide`) for a concrete launch; it is a
hypothesis here, never an axiom. -/
structure Launch.Admissible (l : Launch) : Prop where
  /-- The grid launches at least one thread. -/
  gridNonEmpty : 0 < l.grid.threads
  /-- Every byte read is initialized (the byte map is total). -/
  inputsInitialized : ∀ a, ∃ v, l.init.get a = v
  /-- Kernel parameters are defined. -/
  paramsDefined : ∀ a, ∃ v, l.params a = v
  /-- The canonical execution of the launch is consistent. -/
  canonicalConsistent : Consistent (Candidate.canonical l)

/-- Model non-vacuity: an admissible launch has a consistent, complete execution. -/
theorem nonvacuous (l : Launch) (h : l.Admissible) :
    ∃ e : Candidate l, Consistent e ∧ e.Complete :=
  ⟨Candidate.canonical l, h.canonicalConsistent, Candidate.canonical_complete l⟩

/-- Correctness bridge: if every consistent complete execution satisfies `post`, the
launch is correct with respect to `post`. -/
theorem correct (l : Launch) (post : Post) (h : l.Admissible)
    (hpost : ∀ (e : Candidate l), e.oracle = (Candidate.canonical l).oracle →
      Consistent e → e.Complete → post (observe e)) :
    Correct l post :=
  hpost

/-- Discharging `Correct` from a single verified canonical trace: it suffices that
the canonical execution is complete, consistent and satisfies `post`, because all
complete executions of a launch observe the same memory. -/
theorem correct_of_canonical (l : Launch) (post : Post)
    (hc : Consistent (Candidate.canonical l))
    (hpost : post (observe (Candidate.canonical l))) : Correct l post := by
  intro e horacle _ hcomp
  rw [complete_observe_unique hcomp (Candidate.canonical_complete l) horacle]
  exact hpost

end Harness.PTX
