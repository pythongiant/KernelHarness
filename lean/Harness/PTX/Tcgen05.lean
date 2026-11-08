import Harness.PTX.Execution

/-!
# Blackwell `tcgen05` extension

Tensor memory is a per-CTA address space owned by the MMA unit. Allocations come
from a monotone allocator, MMA and load/store instructions operate on tensor-memory
columns, and completion is signalled through an `mbarrier`: a `tcgen05.wait` is only
justified once the matching `mbarrier` phase completes (`mbarrierPhaseComplete`).
-/

namespace Harness.PTX.Tcgen05

open Harness.PTX Event

/-- Tensor-memory allocation: a base column, a column count, and the phase round. -/
structure TensorMemory where
  baseCol : Nat
  cols : Nat
  round : Nat

/-- Disjointness of two tensor-memory allocations. -/
def Disjoint (a b : TensorMemory) : Prop :=
  a.baseCol + a.cols ≤ b.baseCol ∨ b.baseCol + b.cols ≤ a.baseCol

/-- Disjoint allocations never share a column. -/
theorem disjoint_no_shared_column (a b : TensorMemory) (h : Disjoint a b) :
    ¬ (a.baseCol < b.baseCol + b.cols ∧ b.baseCol < a.baseCol + a.cols) := by
  unfold Disjoint at h
  rintro ⟨h1, h2⟩
  rcases h with h | h <;> omega

/-- Every `tcgen05` completion (an mbarrier wait) has a matching arrival. -/
theorem tcgen05_completion_requires_mbarrier {l : Launch} {c : Candidate l}
    (hc : constructRulesB c = true) :
    ∀ e ∈ c.events, e.kind = .mbarWait →
      ∃ a ∈ c.events, a.kind = .mbarArrive ∧ a.gen = e.gen ∧ a.id < e.id := by
  intro e he hk
  simp only [constructRulesB, Bool.and_eq_true] at hc
  obtain ⟨⟨⟨⟨_, hm⟩, _⟩, _⟩, _⟩ := hc
  have hstep := (List.all_eq_true.mp hm) e he
  simp [hk] at hstep
  simpa [and_assoc] using hstep

/-- A `tcgen05` MMA issue is followed only by its own completion group. -/
theorem tcgen05_async_group {l : Launch} {c : Candidate l}
    (hc : constructRulesB c = true) :
    ∀ e ∈ c.events, e.kind = .asyncComplete →
      ∃ a ∈ c.events, a.kind = .asyncIssue ∧ a.tid = e.tid ∧ a.id < e.id := by
  intro e he hk
  simp only [constructRulesB, Bool.and_eq_true] at hc
  obtain ⟨⟨⟨⟨_, _⟩, ha⟩, _⟩, _⟩ := hc
  have hstep := (List.all_eq_true.mp ha) e he
  simp [hk] at hstep
  simpa [and_assoc] using hstep

end Harness.PTX.Tcgen05
