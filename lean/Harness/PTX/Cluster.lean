import Harness.PTX.Execution

/-!
# Cluster / DSMEM extension

`mapa` maps a cluster-rank-local shared address into the distributed shared memory
space. Cluster barriers (`cluster.sync`, `barrier.cluster.arrive/wait`) are recorded
as barrier pairs by the small-step machine, so `barrierGenerationMatch` and
`causalityAxiom` cover them; a cluster-launch-control cancellation is a control
event, unique per generation (`clusterCancelUniqueness`) and never a memory write.
-/

namespace Harness.PTX.Cluster

open Harness.PTX Event

/-- Size of one CTA's shared-memory window in the DSMEM address space. -/
def dsmemWindow : Nat := 0x100000

/-- `mapa.shared::cluster` address of offset `off` in the CTA with rank `rank`. -/
def mapa (rank off : Nat) : Addr := rank * dsmemWindow + off

/-- A lower rank with an in-window offset maps below any higher rank's window. -/
theorem mapa_lt {r₁ r₂ o₁ o₂ : Nat} (h : r₁ < r₂) (h₁ : o₁ < dsmemWindow) :
    mapa r₁ o₁ < mapa r₂ o₂ := by
  have hw : 0 < dsmemWindow := by decide
  have hkey : r₁ * dsmemWindow + o₁ < r₂ * dsmemWindow := by
    have hs : r₁ * dsmemWindow + dsmemWindow ≤ r₂ * dsmemWindow := by
      calc r₁ * dsmemWindow + dsmemWindow = (r₁ + 1) * dsmemWindow :=
            (Nat.succ_mul r₁ dsmemWindow).symm
        _ ≤ r₂ * dsmemWindow := Nat.mul_le_mul_right _ h
    omega
  have : r₁ * dsmemWindow + o₁ < r₂ * dsmemWindow + o₂ := by omega
  simpa [mapa] using this

/-- Distinct ranks with in-window offsets map to distinct DSMEM addresses. -/
theorem mapa_disjoint {r₁ r₂ o₁ o₂ : Nat} (h : r₁ ≠ r₂)
    (h₁ : o₁ < dsmemWindow) (h₂ : o₂ < dsmemWindow) : mapa r₁ o₁ ≠ mapa r₂ o₂ := by
  rcases Nat.lt_or_gt_of_ne h with hlt | hgt
  · have hlt' := mapa_lt (r₂ := r₂) (o₂ := o₂) hlt h₁
    intro heq
    rw [heq] at hlt'
    exact Nat.lt_irrefl _ hlt'
  · have hlt' := mapa_lt (r₂ := r₁) (o₂ := o₁) hgt h₂
    intro heq
    rw [heq] at hlt'
    exact Nat.lt_irrefl _ hlt'

/-- A cluster barrier return is matched by an arrival of the same generation. -/
theorem cluster_barrier_matched {l : Launch} {c : Candidate l}
    (hc : constructRulesB c = true) :
    ∀ e ∈ c.events, e.kind = .barReturn →
      ∃ a ∈ c.events, a.kind = .barArrive ∧ a.tid = e.tid ∧ a.gen = e.gen ∧ a.id < e.id := by
  intro e he hk
  simp only [constructRulesB, Bool.and_eq_true] at hc
  obtain ⟨⟨⟨⟨hb, _⟩, _⟩, _⟩, _⟩ := hc
  have hstep := (List.all_eq_true.mp hb) e he
  simp [hk] at hstep
  simpa [and_assoc] using hstep

/-- A cluster-launch-control cancellation is unique per generation. -/
theorem cancel_unique {l : Launch} {c : Candidate l}
    (hc : constructRulesB c = true) :
    ∀ e₁ ∈ c.events, ∀ e₂ ∈ c.events, e₁.kind = .clusterCancel →
      e₂.kind = .clusterCancel → e₁.gen = e₂.gen → e₁.id = e₂.id := by
  intro e₁ h₁ e₂ h₂ k₁ k₂ g
  simp only [constructRulesB, Bool.and_eq_true] at hc
  obtain ⟨⟨⟨⟨_, _⟩, _⟩, _⟩, hu⟩ := hc
  have hstep := (List.all_eq_true.mp hu) e₁ h₁
  have hstep₂ := (List.all_eq_true.mp hstep) e₂ h₂
  simp [k₁, k₂, g] at hstep₂
  exact hstep₂

end Harness.PTX.Cluster
