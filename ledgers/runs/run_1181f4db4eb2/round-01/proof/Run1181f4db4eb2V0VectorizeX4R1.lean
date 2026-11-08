import Harness.PTX.Observer
import Harness.Lib.Core
import Harness.Audit

/-!
# Generated verification module: `run_1181f4db4eb2-v0-vectorize_x4-r1`

Family `elementwise`. Instance: 2 threads (block 2, grid 1), output
region starts at byte 8192 (32 bytes).
Numerical scope: exact real arithmetic; IEEE rounding not modeled.
Statement: every consistent, complete execution of this launch that reads memory
through the canonical oracle writes exactly the reference bytes below.
-/

namespace Harness.Generated.Run1181f4db4eb2V0VectorizeX4R1

open Harness.PTX

set_option maxRecDepth 400000
set_option maxHeartbeats 2000000

abbrev instrs : Array Instr := #[
  Instr.mk (Op.mov .s32) [Operand.reg 0] [Operand.imm 10] none,
  Instr.mk (Op.mov .s32) [Operand.reg 1] [Operand.imm 3] none,
  Instr.mk (Op.ld .global .s32 .v1 .ca) [Operand.reg 2] [Operand.mem .global 4096 0 16 0 0 0] none,
  Instr.mk (Op.mul .s32) [Operand.reg 3] [Operand.reg 2, Operand.reg 0] none,
  Instr.mk (Op.add .s32) [Operand.reg 4] [Operand.reg 3, Operand.reg 1] none,
  Instr.mk (Op.st .global .s32 .v1 .cs) [] [Operand.mem .global 8192 0 16 0 0 0, Operand.reg 4] none,
  Instr.mk (Op.ld .global .s32 .v1 .ca) [Operand.reg 2] [Operand.mem .global 4096 4 16 0 0 0] none,
  Instr.mk (Op.mul .s32) [Operand.reg 3] [Operand.reg 2, Operand.reg 0] none,
  Instr.mk (Op.add .s32) [Operand.reg 4] [Operand.reg 3, Operand.reg 1] none,
  Instr.mk (Op.st .global .s32 .v1 .cs) [] [Operand.mem .global 8192 4 16 0 0 0, Operand.reg 4] none,
  Instr.mk (Op.ld .global .s32 .v1 .ca) [Operand.reg 2] [Operand.mem .global 4096 8 16 0 0 0] none,
  Instr.mk (Op.mul .s32) [Operand.reg 3] [Operand.reg 2, Operand.reg 0] none,
  Instr.mk (Op.add .s32) [Operand.reg 4] [Operand.reg 3, Operand.reg 1] none,
  Instr.mk (Op.st .global .s32 .v1 .cs) [] [Operand.mem .global 8192 8 16 0 0 0, Operand.reg 4] none,
  Instr.mk (Op.ld .global .s32 .v1 .ca) [Operand.reg 2] [Operand.mem .global 4096 12 16 0 0 0] none,
  Instr.mk (Op.mul .s32) [Operand.reg 3] [Operand.reg 2, Operand.reg 0] none,
  Instr.mk (Op.add .s32) [Operand.reg 4] [Operand.reg 3, Operand.reg 1] none,
  Instr.mk (Op.st .global .s32 .v1 .cs) [] [Operand.mem .global 8192 12 16 0 0 0, Operand.reg 4] none,
  Instr.mk (Op.ret) [] [] none
]

abbrev grid : Grid := ⟨(1, 1, 1), (2, 1, 1), (1, 1, 1), 2⟩

/-- Initial byte map: the deterministic input fill, zero elsewhere. -/
abbrev initBytes : Addr → Byte := fun a =>
  match a with
  | 4096 => 33
  | 4097 => 0
  | 4098 => 0
  | 4099 => 0
  | 4100 => 187
  | 4101 => 255
  | 4102 => 255
  | 4103 => 255
  | 4104 => 247
  | 4105 => 255
  | 4106 => 255
  | 4107 => 255
  | 4108 => 47
  | 4109 => 0
  | 4110 => 0
  | 4111 => 0
  | 4112 => 89
  | 4113 => 0
  | 4114 => 0
  | 4115 => 0
  | 4116 => 85
  | 4117 => 0
  | 4118 => 0
  | 4119 => 0
  | 4120 => 17
  | 4121 => 0
  | 4122 => 0
  | 4123 => 0
  | 4124 => 236
  | 4125 => 255
  | 4126 => 255
  | 4127 => 255
  | _ => 0

abbrev launch : Launch := ⟨{ instrs := instrs }, grid, ⟨initBytes⟩, fun _ => .u32 0⟩

/-- Reference bytes of the output region. -/
abbrev refBytes : List Byte := [77, 1, 0, 0, 81, 253, 255, 255, 169, 255, 255, 255, 217, 1, 0, 0, 125, 3, 0, 0, 85, 3, 0, 0, 173, 0, 0, 0, 59, 255, 255, 255]

abbrev post : Post := fun m =>
  (List.range refBytes.length).all (fun i => m.get (8192 + i) == refBytes.getD i 0) = true

theorem canonConsistent : Consistent (Candidate.canonical launch) := by decide

theorem canonPost : post (observe (Candidate.canonical launch)) := by decide

theorem admissible : launch.Admissible :=
  ⟨by decide, fun a => ⟨launch.init.get a, rfl⟩, fun a => ⟨launch.params a, rfl⟩,
    canonConsistent⟩

theorem nonvacuous : ∃ e : Candidate launch, Consistent e ∧ e.Complete :=
  Harness.PTX.nonvacuous launch admissible

theorem correct : Correct launch post :=
  correct_of_canonical launch post canonConsistent canonPost

run_cmd Harness.Audit.audit #[``correct, ``nonvacuous]

#eval "AUDIT_OK"

end Harness.Generated.Run1181f4db4eb2V0VectorizeX4R1
