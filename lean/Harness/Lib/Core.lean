import Harness.PTX.Observer

/-!
# Reusable lemmas for generated proofs

`Harness.Lib` collects the facts that per-kernel obligations (`hpost`) discharge,
so generated proofs stay short. Every lemma is proved from the model's
definitions; none is assumed.
-/

namespace Harness.Lib

open Harness.PTX Event

/-! ## Last-write structure -/

/-- One `lastWriteStep` either keeps the accumulated event or stores `e`, and `e`
writes the byte when it is stored. -/
theorem lastWriteStep_writes {a bound : Nat} {acc : Option Event} {e w : Event}
    (hacc : ∀ w, acc = some w → w.writesBool a = true) (h : lastWriteStep a bound acc e = some w) :
    w.writesBool a = true := by
  unfold lastWriteStep at h
  by_cases c : (e.writesBool a && decide (e.id < bound)) = true
  · rw [if_pos c] at h
    simp only [Option.some.injEq] at h
    subst h
    simp [writesBool, Bool.and_eq_true] at c
    simp [writesBool, Bool.and_eq_true]
    exact c.1
  · rw [if_neg c] at h
    exact hacc w h

/-- Fold specification for `lastWriteB`: the result is either the seed or an element
of the list, and it always writes the byte. -/
theorem foldl_lastWrite_spec (evs : List Event) (a bound : Nat) :
    ∀ acc : Option Event, (∀ w, acc = some w → w.writesBool a = true) →
      (∀ w, evs.foldl (lastWriteStep a bound) acc = some w → w.writesBool a = true) ∧
      (∀ w, evs.foldl (lastWriteStep a bound) acc = some w → w ∈ evs ∨ acc = some w) := by
  induction evs with
  | nil =>
    intro acc hacc
    constructor
    · intro w h; exact hacc w h
    · intro w h; exact Or.inr h
  | cons e es ih =>
    intro acc hacc
    obtain ⟨hw1, hw2⟩ := ih (lastWriteStep a bound acc e)
      (fun w h => lastWriteStep_writes hacc h)
    constructor
    · intro w h
      simp only [List.foldl_cons] at h
      exact hw1 w h
    · intro w h
      simp only [List.foldl_cons] at h
      rcases hw2 w h with h' | h'
      · exact Or.inl (List.mem_cons_of_mem _ h')
      · unfold lastWriteStep at h'
        by_cases c : (e.writesBool a && decide (e.id < bound)) = true
        · rw [if_pos c] at h'
          simp only [Option.some.injEq] at h'
          subst h'
          exact Or.inl (by simp)
        · rw [if_neg c] at h'
          exact Or.inr h'

/-- `lastWriteB` only ever returns an event that writes the byte. -/
theorem lastWriteB_writes {evs : List Event} {a bound : Nat} {w : Event}
    (h : lastWriteB evs a bound = some w) : w.writesBool a = true :=
  (foldl_lastWrite_spec evs a bound none (by intro w' h'; simp at h')).1 w h

/-- `lastWriteB` only ever returns an element of the trace. -/
theorem lastWriteB_mem {evs : List Event} {a bound : Nat} {w : Event}
    (h : lastWriteB evs a bound = some w) : w ∈ evs := by
  rcases (foldl_lastWrite_spec evs a bound none (by intro w' h'; simp at h')).2 w h with h' | h'
  · exact h'
  · simp at h'

/-! ## Observability -/

/-- Observer memory at a byte is the value of the last write to that byte. -/
theorem observer_latest_write {l : Launch} (c : Candidate l) {a : Addr} {w : Event}
    (h : lastWrite c.events a = some w) : (observe c).get a = (w.value a).getD 0 := by
  simp [observe, ByteMap.get, h]

/-- A byte that the execution never writes keeps its initial value. -/
theorem observe_untouched {l : Launch} (c : Candidate l) (a : Addr)
    (h : ∀ e ∈ c.events, e.writesBool a = false) : (observe c).get a = l.init.get a := by
  have hn : lastWrite c.events a = none := by
    unfold lastWrite
    cases hb : lastWriteB c.events a (c.events.length + 1) with
    | none => rfl
    | some w =>
      have hw := lastWriteB_writes hb
      have hfalse := h w (lastWriteB_mem hb)
      exact absurd hfalse (by simp [hw])
  simp [observe, ByteMap.get, hn]

/-- Frame rule for loop bodies: bytes outside the writes of a loop body are not
modified by it. -/
theorem loop_invariant_frame {l : Launch} (c : Candidate l) (a : Addr)
    (h : ∀ e ∈ c.events, e.writesBool a = false) :
    (observe c).get a = l.init.get a :=
  observe_untouched c a h

/-! ## Memory-model rules -/

/-- Barrier returns are matched by an arrival of the same thread and generation. -/
theorem bar_sync_visible {l : Launch} {c : Candidate l}
    (hc : constructRulesB c = true) :
    ∀ e ∈ c.events, e.kind = .barReturn →
      ∃ a ∈ c.events, a.kind = .barArrive ∧ a.tid = e.tid ∧ a.gen = e.gen ∧ a.id < e.id := by
  intro e he hk
  simp only [constructRulesB, Bool.and_eq_true] at hc
  obtain ⟨⟨⟨⟨hb, _⟩, _⟩, _⟩, _⟩ := hc
  have hstep := (List.all_eq_true.mp hb) e he
  simp [hk] at hstep
  simpa [and_assoc] using hstep

/-- Atomic read-modify-write events are unique per (thread, generation). -/
theorem ticket_uniqueness {l : Launch} {c : Candidate l}
    (hc : atomicityAxiomB c = true) :
    ∀ e1 ∈ c.events, ∀ e2 ∈ c.events, e1.kind = .atomRmw → e2.kind = .atomRmw →
      e1.tid = e2.tid → e1.gen = e2.gen → e1.id = e2.id := by
  intro e1 h1 e2 h2 k1 k2 t g
  have h2' := (List.all_eq_true.mp hc) e1 h1
  have h3 := (List.all_eq_true.mp h2') e2 h2
  simp [k1, k2, t, g] at h3
  exact h3

/-- Disjoint footprints cannot overlap. -/
theorem disjoint_no_overlap (a b : Event)
    (h : a.addr + a.size ≤ b.addr ∨ b.addr + b.size ≤ a.addr) : ¬ overlap a b := by
  unfold overlap
  rcases h with h | h
  · rintro ⟨_, h2⟩
    exact absurd (Nat.lt_of_lt_of_le h2 h) (Nat.lt_irrefl b.addr)
  · rintro ⟨h1, _⟩
    exact absurd (Nat.lt_of_lt_of_le h1 h) (Nat.lt_irrefl a.addr)

/-- Disjoint footprints cannot produce a mixed-size race. -/
theorem disjoint_no_mixed_size_race (a b : Event)
    (h : a.addr + a.size ≤ b.addr ∨ b.addr + b.size ≤ a.addr) : ¬ mixedSizeRace a b := by
  intro hm
  unfold mixedSizeRace at hm
  exact disjoint_no_overlap a b h hm.2.2.1

end Harness.Lib
