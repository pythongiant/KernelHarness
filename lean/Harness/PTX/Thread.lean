import Harness.PTX.Event

/-!
# Small-step thread semantics (paper §3.3 `step`)

The semantics is *thread-local*: a thread's step depends on its own state, the
program and the initial byte map. Cross-thread visibility is expressed by the
`Consistent` axioms in `Harness.PTX.Execution`, not by the local machine. This is
disclosed as a scope limitation of the model: a kernel whose threads read data
written by other threads *within the same launch* is reported conservatively
(readers always observe initial bytes, so such a kernel fails consistency).

Consequence used by the generated proofs: `Candidate.events` is a function of the
per-thread step counts alone (`traceOf`), hence two executions with the same step
counts have identical event traces and identical observable memory.
-/


namespace Harness.PTX

/-- Per-thread machine state. -/
structure State where
  pc : Nat
  regs : RegFile
  k : Nat

/-- Initial thread state. -/
def State.initial : State := ⟨0, fun _ => none, 0⟩

namespace Value

/-- Interpret a value as a natural number. -/
def toNat : Value → Nat
  | .b8 b => b.toNat
  | .u32 v => v.toNat
  | .u64 v => v.toNat
  | .f32 v => v.toNat

/-- A `u32` value. -/
def ofU32 (n : Nat) : Value := .u32 (UInt32.ofNat n)

end Value

/-- Bytes of a value, little-endian. -/
abbrev valueBytes (v : Value) (i : Nat) : Byte :=
  match v with
  | .b8 b => if i = 0 then b else 0
  | .u32 x => ⟨x.toNat >>> (8 * i) &&& 0xFF⟩
  | .u64 x => ⟨x.toNat >>> (8 * i) &&& 0xFF⟩
  | .f32 x => ⟨x.toNat >>> (8 * i) &&& 0xFF⟩

/-- Byte `i` of value `v` at footprint offset `i`. -/
abbrev byteOf (v : Value) (i : Nat) : Byte := valueBytes v i

namespace State

/-- Read a register. -/
def getReg (s : State) (r : Reg) : Option Value := s.regs r

/-- Write a register. -/
def setReg (s : State) (r : Reg) (v : Value) : State :=
  ⟨s.pc, fun r' => if r' = r then some v else s.regs r', s.k⟩

end State

/-- Resolve a source operand to a value. -/
def resolve (s : State) (o : Operand) : Value :=
  match o with
  | .imm n => Value.ofU32 n
  | .reg r => (s.regs r).getD (.ofU32 0)
  | .mem _ _ _ _ _ _ _ => Value.ofU32 0

/-- Apply a binary arithmetic form. -/
def binOp (op : Op) (a b : Nat) : Nat :=
  match op with
  | .add _ => a + b
  | .sub _ => a - b
  | .mul _ => a * b
  | .div _ => if b = 0 then 0 else a / b
  | .rem _ => if b = 0 then 0 else a % b
  | .shl _ => a <<< b
  | .shr _ => a >>> b
  | .and_ _ => a &&& b
  | .or_ _ => a ||| b
  | .xor_ _ => a ^^^ b
  | .min_ _ => if a ≤ b then a else b
  | .max_ _ => if a ≤ b then b else a
  | _ => a

/-- Apply a unary arithmetic form. -/
def unOp (op : Op) (a : Nat) : Nat :=
  match op with
  | .not_ _ => a ^^^ 0xFFFFFFFF
  | .neg _ => 0 - a
  | .abs _ => a
  | _ => a

/-- Comparison used by `setp`. -/
def cmpOp (cmp : CmpOp) (a b : Nat) : Bool :=
  match cmp with
  | .eq => a == b
  | .ne => a != b
  | .lt => a < b
  | .le => a ≤ b
  | .gt => b < a
  | .ge => b ≤ a
  | .ltu => a < b
  | .leu => a ≤ b
  | .gtu => b < a
  | .geu => b ≤ a

/-- Footprint of a memory operand in bytes. -/
def footprint (op : Op) : Nat := op.accessBytes

/-- Emit a read event covering `size` bytes at `addr`, whose values come from `m`. -/
def readEvent (tid : Nat) (sp : Space) (addr size gen : Nat) (m : ByteMap) : Event :=
  { id := 0, tid := tid, kind := .read, proxy := .generic, space := sp, addr := addr,
    size := size, value := fun a => if addr ≤ a ∧ a < addr + size then some (m.get a) else none,
    sem := none, gen := gen, group := 0, lane := none }

/-- Emit a write event covering `size` bytes at `addr`, values from `v`. -/
def writeEvent (tid : Nat) (sp : Space) (addr size gen : Nat) (v : Value) : Event :=
  { id := 0, tid := tid, kind := .write, proxy := .generic, space := sp, addr := addr,
    size := size, value := fun a => if addr ≤ a ∧ a < addr + size then some (byteOf v (a - addr)) else none,
    sem := none, gen := gen, group := 0, lane := none }

/-- Emit a control event. -/
def ctrlEvent (tid : Nat) (kind : EventKind) (gen : Nat) (sp : Space := .global)
    (addr size : Nat := 0) : Event :=
  { id := 0, tid := tid, kind := kind, proxy := .generic, space := sp, addr := addr, size := size,
    value := fun _ => none, sem := none, gen := gen, group := 0, lane := none }

/-- The value a `ld` places in its destination register: the little-endian
assembly of the footprint bytes of the initial map. -/
abbrev loadNat (m : ByteMap) (addr size : Nat) : Nat :=
  (m.get addr).toNat
    + (if size > 1 then (m.get (addr + 1)).toNat <<< 8 else 0)
    + (if size > 2 then (m.get (addr + 2)).toNat <<< 16 else 0)
    + (if size > 3 then (m.get (addr + 3)).toNat <<< 24 else 0)

/-- The value a `ld` places in its destination register (little-endian). -/
abbrev loadValue (m : ByteMap) (addr size : Nat) : Value :=
  Value.ofU32 (loadNat m addr size)

/-- One step of thread `tid` in launch `l`. Deterministic in the state. -/
abbrev step (l : Launch) (o : Nat → Value) (tid : Nat) (s : State) : State × List Event :=
  match l.instrAt s.pc with
  | none => ({ s with pc := s.pc + 1 }, [])
  | some i =>
    let adv : State := { s with pc := s.pc + 1 }
    match i.op with
    | .mov _ | .cvt _ _ =>
      match i.src with
      | src :: _ =>
        let st := match i.dst with
          | .reg r :: _ => adv.setReg r (resolve s src)
          | _ => adv
        (st, [])
      | [] => (adv, [])
    | .add _ | .sub _ | .mul _ | .div _ | .rem _ | .shl _ | .shr _ | .and_ _ | .or_ _
    | .xor_ _ | .min_ _ | .max_ _ =>
      match i.dst, i.src with
      | .reg r :: _, a :: b :: _ =>
        (adv.setReg r (.ofU32 (binOp i.op (resolve s a).toNat (resolve s b).toNat)), [])
      | _, _ => (adv, [])
    | .mad _ =>
      match i.dst, i.src with
      | .reg r :: _, a :: b :: c :: _ =>
        (adv.setReg r (.ofU32 ((resolve s a).toNat * (resolve s b).toNat
          + (resolve s c).toNat)), [])
      | _, _ => (adv, [])
    | .not_ _ | .neg _ | .abs _ =>
      match i.dst, i.src with
      | .reg r :: _, a :: _ => (adv.setReg r (.ofU32 (unOp i.op (resolve s a).toNat)), [])
      | _, _ => (adv, [])
    | .setp _ cmp =>
      match i.dst, i.src with
      | .reg r :: _, a :: b :: _ =>
        (adv.setReg r (.ofU32 (if cmpOp cmp (resolve s a).toNat (resolve s b).toNat then 1 else 0)), [])
      | _, _ => (adv, [])
    | .selp _ =>
      match i.dst, i.src with
      | .reg r :: _, c :: a :: b :: _ =>
        (adv.setReg r (if (resolve s c).toNat ≠ 0 then resolve s a else resolve s b), [])
      | _, _ => (adv, [])
    | .bra tgt | .braUni tgt => ({ s with pc := tgt }, [])
    | .ret | .exit => (s, [ctrlEvent tid .exit s.k])
    | .barSync n | .barArrive n =>
      (adv, [ctrlEvent tid .barArrive s.k, ctrlEvent tid .barReturn s.k])
    | .clusterSync | .barrierClusterArrive | .barrierClusterWait =>
      (adv, [ctrlEvent tid .barArrive s.k, ctrlEvent tid .barReturn s.k])
    | .barWarpSync => (adv, [ctrlEvent tid .shfl s.k])
    | .ld _ _ _ _ | .ldGlobalNc _ _ | .paramLd _ | .ldmatrixX1 | .ldmatrixX2 | .ldmatrixX4 =>
      match i.dst, i.src with
      | .reg r :: _, m :: _ =>
        let a := m.addr tid
        let sz := if footprint i.op = 0 then 4 else footprint i.op
        let v := o a
        (adv.setReg r v,
          [{ readEvent tid ((i.op.space?).getD .global) a sz s.k l.init with
             value := fun b => if a ≤ b ∧ b < a + sz then some (byteOf v (b - a)) else none }])
      | _, _ => (adv, [])
    | .st _ _ _ _ | .stGlobalCs _ _ | .paramSt _ =>
      match i.src with
      | m :: v :: _ =>
        let a := m.addr tid
        let sz := if footprint i.op = 0 then 4 else footprint i.op
        (adv, [writeEvent tid ((i.op.space?).getD .global) a sz s.k (resolve s v)])
      | _ => (adv, [])
    | .atom _ _ _ _ | .red _ _ _ _ =>
      match i.dst, i.src with
      | .reg r :: _, m :: v :: _ =>
        let a := m.addr tid
        let sz := if footprint i.op = 0 then 4 else footprint i.op
        (adv.setReg r (o a),
          [readEvent tid .global a sz s.k l.init,
           { writeEvent tid .global a sz s.k (resolve s v) with kind := .atomRmw }])
      | _, _ => (adv, [])
    | .cpAsyncCa n | .cpAsyncCg n =>
      match i.src with
      | m :: _ => (adv, [ctrlEvent tid .asyncIssue s.k .shared (m.addr tid) n])
      | _ => (adv, [])
    | .cpAsyncCommitGroup | .cpAsyncBulkCommitGroup | .wgmmaCommitGroup | .tcgen05Commit =>
      (adv, [ctrlEvent tid .asyncIssue s.k])
    | .cpAsyncWaitGroup n | .cpAsyncBulkWaitGroup n | .wgmmaWaitGroup n | .cpAsyncWaitAll =>
      (adv, [ctrlEvent tid .asyncComplete s.k])
    | .cpAsyncBulkTensor2d _ _ | .tcgen05Cp =>
      (adv, [ctrlEvent tid .asyncIssue s.k ((i.op.space?).getD .shared)])
    | .mbarrierInit => (adv, [ctrlEvent tid .mbarInit s.k])
    | .mbarrierArrive | .mbarrierArriveExpectTx => (adv, [ctrlEvent tid .mbarArrive s.k])
    | .mbarrierTryWaitParity | .mbarrierWaitParity => (adv, [ctrlEvent tid .mbarWait s.k])
    | .wgmmaFence | .fenceProxyAsync | .fenceProxyTensormap | .fenceProxyAsyncShared
    | .fenceSc | .membar _ | .membarGl | .membarCta => (adv, [ctrlEvent tid .fence s.k])
    | .wgmmaMmaAsyncM64n8k16 | .mmaSyncM16n8k16 | .tcgen05Mma =>
      (adv, [ctrlEvent tid .wgmmaIssue s.k])
    | .shflSyncIdx | .shflSyncUp | .shflSyncDown | .shflSyncBfly | .electSync =>
      (adv, [ctrlEvent tid .shfl s.k])
    | .mapa | .clusterlaunchcontrolTryCancel =>
      (adv, [ctrlEvent tid .clusterCancel s.k])
    | .tcgen05Alloc => (adv, [ctrlEvent tid .asyncIssue s.k])
    | .tcgen05Wait => (adv, [ctrlEvent tid .wgmmaWait s.k])
    | .tcgen05Ld => (adv, [readEvent tid .shared 0 4 s.k l.init])
    | .tcgen05St => (adv, [ctrlEvent tid .write s.k])
    | .tcgen05Fence => (adv, [ctrlEvent tid .fence s.k])

/-- Run `n` steps of thread `tid`, collecting events. -/
abbrev run (l : Launch) (o : Nat → Value) (tid : Nat) : Nat → State → List Event
  | 0, _ => []
  | n + 1, s =>
    let (s', evs) := step l o tid s
    evs ++ run l o tid n s'

/-- Assign consecutive ids to a trace. -/
abbrev assignIds : Nat → List Event → List Event
  | _, [] => []
  | i, e :: es => { e with id := i } :: assignIds (i + 1) es

/-- Trace of a launch under a per-thread step-count schedule. Event ids are
assigned in trace order so that `id` is a strict linear order. -/
abbrev traceOf (l : Launch) (steps : Nat → Nat) (oracle : Nat → Nat → Value) : List Event :=
  assignIds 0 ((List.range l.grid.threads).flatMap
    (fun t => run l (fun a => oracle t a) t (steps t) State.initial))

end Harness.PTX
