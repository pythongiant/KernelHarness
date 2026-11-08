import Harness.PTX.Thread

/-!
# Executions, consistency axioms, and the correctness statement

A `Candidate` is a schedule (per-thread step count) plus an oracle. Its event
trace is *derived* by the small-step machine (`Candidate.events`), following the
paper's §3.3 construction: events are not free data, so the axioms below constrain
the program's actual executions.

The eight axioms are executable checkers (`Bool`) with their mathematical
statement in the docstring, so that a concrete execution is machine-checkable by
`decide` and so that a report can name the axiom that failed. Relation helpers
(`po`, `co`, `rf`, `sw`, `causality`) are also provided in `Prop` form for
readability of the model.

Relation conventions (all with an explicit `id`-order component, so the trace's
`id` field induces a strict linear order):

* `po` — program order within a thread,
* `co` — coherence order on writes to one address,
* `rf` — read-from: the latest write to a byte before the read,
* `sc` — sequential-consistency order on one address,
* `sw` — synchronizes-with (barrier arrive → barrier return, same generation),
* `causality` — `(po ∪ sw)⁺`.
-/


namespace Harness.PTX

open Event

/-- An execution candidate: a per-thread step-count schedule and an oracle. -/
structure Candidate (l : Launch) where
  /-- Steps taken by each thread. Out-of-range entries do not affect the trace. -/
  steps : Nat → Nat
  /-- External data source (thread index → query index → value). -/
  oracle : Nat → Nat → Value

namespace Candidate

variable {l : Launch}

/-- The derived event trace. -/
abbrev events (c : Candidate l) : List Event := traceOf l c.steps c.oracle

/-- An execution is complete when every thread has run its program to the end. -/
abbrev Complete (c : Candidate l) : Prop := ∀ t, c.steps t = l.codeLength

theorem events_congr {c c' : Candidate l} (h : c.steps = c'.steps) (ho : c.oracle = c'.oracle) :
    c.events = c'.events := by
  simp [events, h, ho]

theorem complete_steps {c c' : Candidate l} (hc : c.Complete) (hc' : c'.Complete) :
    c.steps = c'.steps := by
  funext t; rw [hc t, hc' t]

/-- The oracle that answers every load with the initial bytes at that address. -/
def initOracle (l : Launch) : Nat → Nat → Value :=
  fun _ a => Value.ofU32 (loadNat l.init a 4)

/-- The canonical (full-length) execution of a launch. -/
abbrev canonical (l : Launch) : Candidate l :=
  ⟨fun _ => l.codeLength, initOracle l⟩

@[simp] theorem canonical_steps (l : Launch) : (canonical l).steps = fun _ => l.codeLength := rfl

theorem canonical_complete (l : Launch) : (canonical l).Complete := fun _ => rfl

end Candidate

/-- Two events overlap when their byte footprints intersect. -/
def overlap (a b : Event) : Prop :=
  a.addr < b.addr + b.size ∧ b.addr < a.addr + a.size

/-- Program order. -/
def po (a b : Event) : Prop := a.tid = b.tid ∧ a.id < b.id

/-- Coherence order on writes to the same address. -/
def co (a b : Event) : Prop :=
  a.kind = .write ∧ b.kind = .write ∧ a.addr = b.addr ∧ a.id < b.id

/-- Sequential-consistency order on accesses to the same address. -/
def sc (a b : Event) : Prop := a.addr = b.addr ∧ a.id < b.id

/-- Synchronizes-with: a barrier arrival releasing a matching return. -/
def sw (a b : Event) : Prop :=
  a.kind = .barArrive ∧ b.kind = .barReturn ∧ a.gen = b.gen ∧ a.id < b.id

/-- One fold step of `lastWriteB`: remember `e` when it writes the byte and is early
enough. -/
abbrev lastWriteStep (a : Addr) (bound : Nat) (acc : Option Event) (e : Event) : Option Event :=
  if e.writesBool a && decide (e.id < bound) then some e else acc

/-- The latest write to byte `a` before `bound`, as data. -/
abbrev lastWriteB (evs : List Event) (a : Addr) (bound : Nat) : Option Event :=
  evs.foldl (lastWriteStep a bound) none

/-- Read-from for one byte: the latest preceding write to that byte. -/
def rfByte {l : Launch} (c : Candidate l) (w r : Event) (a : Addr) : Prop :=
  w.writesByte a ∧ r.readsByte a ∧ w.id < r.id ∧ lastWriteB c.events a r.id = some w

/-- Read-from. -/
def rf {l : Launch} (c : Candidate l) (w r : Event) : Prop := ∃ a, rfByte c w r a

/-- Morally strong relation (paper §3.4): `po ∪ co ∪ rf ∪ sw`. -/
def morallyStrong (a b : Event) : Prop := po a b ∨ co a b ∨ sw a b

/-- A conflicting access pair of different width with no program-order between them. -/
def mixedSizeRace (a b : Event) : Prop :=
  a.id ≠ b.id ∧ a.size ≠ b.size ∧ overlap a b ∧ ¬ po a b ∧ ¬ po b a

/-- Causality: `(po ∪ sw)⁺`. -/
def causality (a b : Event) : Prop := Relation.TransGen (fun x y => po x y ∨ sw x y) a b

/-- Causality preserves proxy identity (or crosses an acquire/release event). -/
def proxyPreservedCausality (a b : Event) : Prop :=
  causality a b → a.proxy = b.proxy ∨ a.sem = some .acqrel ∨ b.sem = some .acqrel

/-- A relation is acyclic. -/
def Acyclic (r : Event → Event → Prop) : Prop := ∀ a, ¬ Relation.TransGen r a a

/-- If every edge strictly increases `id`, so does its transitive closure. -/
theorem transGen_id_lt {r : Event → Event → Prop} (h : ∀ a b, r a b → a.id < b.id) :
    ∀ a b, Relation.TransGen r a b → a.id < b.id := by
  intro a b htr
  induction htr with
  | single hstep => exact h _ _ hstep
  | tail _ hstep ih => exact Nat.lt_trans ih (h _ _ hstep)

/-- Causality is acyclic on any trace with increasing ids (see `transGen_id_lt`). -/
theorem causalityAcyclic_of_ids (hid : ∀ a b : Event, (po a b ∨ sw a b) → a.id < b.id) :
    Acyclic (fun x y => po x y ∨ sw x y) := by
  intro a ha
  exact absurd (transGen_id_lt hid a a ha) (Nat.lt_irrefl a.id)

-- ---------------------------------------------------------------------------
-- Executable consistency checkers
-- ---------------------------------------------------------------------------

/-- Edges of `po ∪ sw` over a trace. -/
abbrev poSwEdges (evs : List Event) : List (Nat × Nat) :=
  evs.flatMap (fun a => evs.filterMap (fun b =>
    if (a.tid == b.tid && a.id < b.id) ||
        (a.kind == .barArrive && b.kind == .barReturn && a.gen == b.gen && a.id < b.id)
      then some (a.id, b.id) else none))

/-- (1) `valueAgreement`: every read byte matches a write's value, or the initial byte. -/
abbrev valueAgreementB {l : Launch} (c : Candidate l) : Bool :=
  c.events.all (fun r =>
    if r.kind == .read then
      (List.range r.size).all (fun i =>
        let a := r.addr + i
        (c.events.any (fun w => w.writesBool a && (r.value a == w.value a))) ||
          (r.value a == some (l.init.get a)))
    else true)

/-- (2) `coherenceAxiom`: writes to one address are ordered by `id`, and a read sees
the latest preceding write to its byte. -/
abbrev coherenceAxiomB {l : Launch} (c : Candidate l) : Bool :=
  (c.events.all (fun w1 => c.events.all (fun w2 =>
    if w1.kind == .write && w2.kind == .write && w1.addr == w2.addr && w1.id != w2.id
      then decide (w1.id < w2.id || w2.id < w1.id) else true))) &&
  (c.events.all (fun r =>
    if r.kind == .read then
      (List.range r.size).all (fun i =>
        let a := r.addr + i
        match lastWriteB c.events a r.id with
        | some w => r.value a == w.value a
        | none => true)
    else true))

/-- (3) `atomicityAxiom`: atomic RMW events are unique per (thread, generation). -/
abbrev atomicityAxiomB {l : Launch} (c : Candidate l) : Bool :=
  c.events.all (fun e1 => c.events.all (fun e2 =>
    if e1.kind == .atomRmw && e2.kind == .atomRmw && e1.tid == e2.tid && e1.gen == e2.gen
      then decide (e1.id = e2.id) else true))

/-- (4) `scPerLocation`: accesses to one address are totally ordered by `id`. -/
abbrev scPerLocationB {l : Launch} (c : Candidate l) : Bool :=
  c.events.all (fun a => c.events.all (fun b =>
    if a.addr == b.addr && a.id != b.id then decide (a.id < b.id || b.id < a.id) else true))

/-- (5) `causalityAxiom`: `(po ∪ sw)⁺` is irreflexive.

On a finite set of events carrying distinct ids, the transitive closure of a
relation is cyclic iff some edge does not strictly increase the id: every cycle has
a non-positive step, and a relation whose edges all increase a well-founded measure
has no cycle. It therefore suffices to check that each `po ∪ sw` edge increases
`id` (and ids in a trace are distinct by construction). -/
abbrev causalityAxiomB {l : Launch} (c : Candidate l) : Bool :=
  (poSwEdges c.events).all (fun p => p.1 < p.2)

/-- (6) `noMixedSizeRace`: no conflicting access pair of different width. -/
abbrev noMixedSizeRaceB {l : Launch} (c : Candidate l) : Bool :=
  c.events.all (fun a => c.events.all (fun b =>
    if a.id != b.id && a.size != b.size && a.overlapB b
        && !(a.poB b) && !(b.poB a)
      then false else true))

/-- (7a) Barrier returns are matched by an arrival of the same thread and generation. -/
abbrev barrierGenerationMatchB {l : Launch} (c : Candidate l) : Bool :=
  c.events.all (fun e =>
    if e.kind == .barReturn then
      c.events.any (fun a =>
        a.kind == .barArrive && a.tid == e.tid && a.gen == e.gen && a.id < e.id)
    else true)

/-- (7b) Every mbarrier wait is preceded by a matching arrival. -/
abbrev mbarrierPhaseCompleteB {l : Launch} (c : Candidate l) : Bool :=
  c.events.all (fun e =>
    if e.kind == .mbarWait then
      c.events.any (fun a => a.kind == .mbarArrive && a.gen == e.gen && a.id < e.id)
    else true)

/-- (7c) Async completion events are preceded by an issue in the same thread. -/
abbrev asyncCompletionGroupB {l : Launch} (c : Candidate l) : Bool :=
  c.events.all (fun e =>
    if e.kind == .asyncComplete then
      c.events.any (fun a => a.kind == .asyncIssue && a.tid == e.tid && a.id < e.id)
    else true)

/-- (7d) Shuffle events of one generation agree on the participating group. -/
abbrev shuffleGroupMatchB {l : Launch} (c : Candidate l) : Bool :=
  c.events.all (fun e1 => c.events.all (fun e2 =>
    if e1.kind == .shfl && e2.kind == .shfl && e1.gen == e2.gen && e1.tid == e2.tid
      then decide (e1.group = e2.group) else true))

/-- (7e) A cluster-launch-control cancellation is unique per generation. -/
abbrev clusterCancelUniquenessB {l : Launch} (c : Candidate l) : Bool :=
  c.events.all (fun e1 => c.events.all (fun e2 =>
    if e1.kind == .clusterCancel && e2.kind == .clusterCancel && e1.gen == e2.gen
      then decide (e1.id = e2.id) else true))

/-- (7) `constructRules`, split into five sub-rules. -/
abbrev constructRulesB {l : Launch} (c : Candidate l) : Bool :=
  barrierGenerationMatchB c && mbarrierPhaseCompleteB c && asyncCompletionGroupB c &&
    shuffleGroupMatchB c && clusterCancelUniquenessB c

/-- (8) `proxyFenceRule`: a cross-proxy read of a written byte requires an
intervening fence in the same thread. -/
abbrev proxyFenceRuleB {l : Launch} (c : Candidate l) : Bool :=
  c.events.all (fun r =>
    if r.kind == .read then
      c.events.all (fun w =>
        if w.writesBool r.addr && w.id < r.id && w.proxy != r.proxy then
          c.events.any (fun f => f.kind == .fence && w.id < f.id && f.id < r.id)
        else true)
    else true)

/-- The eight consistency axioms, in the order the UI reports them. -/
abbrev Consistent {l : Launch} (c : Candidate l) : Prop :=
  valueAgreementB c = true ∧ coherenceAxiomB c = true ∧ atomicityAxiomB c = true ∧
    scPerLocationB c = true ∧ causalityAxiomB c = true ∧ noMixedSizeRaceB c = true ∧
    constructRulesB c = true ∧ proxyFenceRuleB c = true

/-- Named projections, so a report can say *which* axiom failed. -/
theorem Consistent.valueAgreement {l : Launch} {c : Candidate l} (h : Consistent c) :
    valueAgreementB c = true := h.1

theorem Consistent.coherenceAxiom {l : Launch} {c : Candidate l} (h : Consistent c) :
    coherenceAxiomB c = true := h.2.1

theorem Consistent.atomicityAxiom {l : Launch} {c : Candidate l} (h : Consistent c) :
    atomicityAxiomB c = true := h.2.2.1

theorem Consistent.scPerLocation {l : Launch} {c : Candidate l} (h : Consistent c) :
    scPerLocationB c = true := h.2.2.2.1

theorem Consistent.causalityAxiom {l : Launch} {c : Candidate l} (h : Consistent c) :
    causalityAxiomB c = true := h.2.2.2.2.1

theorem Consistent.noMixedSizeRace {l : Launch} {c : Candidate l} (h : Consistent c) :
    noMixedSizeRaceB c = true := h.2.2.2.2.2.1

theorem Consistent.constructRules {l : Launch} {c : Candidate l} (h : Consistent c) :
    constructRulesB c = true := h.2.2.2.2.2.2.1

theorem Consistent.proxyFenceRule {l : Launch} {c : Candidate l} (h : Consistent c) :
    proxyFenceRuleB c = true := h.2.2.2.2.2.2.2

end Harness.PTX
