import Harness.PTX.Basic

/-!
# PTX instruction forms

Exactly 81 instruction forms, mirroring `harness/translator/ptx_forms.toml`
(`lean_ctor` column). Qualifier fields are parameters of the constructor so that
one form covers its qualifier space (`ld.global.v4.f32` is `ld .global .f32 .v4`).
-/

namespace Harness.PTX

inductive Space where
  | global | shared | local | param | const
  deriving Repr, DecidableEq, Inhabited

inductive Proxy where
  | generic | async_ | tensormap
  deriving Repr, DecidableEq, Inhabited

inductive Scope where
  | cta | cluster | gpu | sys
  deriving Repr, DecidableEq, Inhabited

inductive Sem where
  | relaxed | acquire | release | acqrel | sc
  deriving Repr, DecidableEq, Inhabited

inductive Cache where
  | ca | cg | cs | lu | cv
  deriving Repr, DecidableEq, Inhabited

inductive Vec where
  | v1 | v2 | v4
  deriving Repr, DecidableEq, Inhabited

inductive Ty where
  | u8 | u16 | u32 | u64 | s8 | s16 | s32 | s64
  | f16 | bf16 | f32 | f64 | pred
  deriving Repr, DecidableEq, Inhabited

/-- Width in bytes of a type. -/
def Ty.size : Ty → Nat
  | .u8 | .s8 | .pred => 1
  | .u16 | .s16 | .f16 | .bf16 => 2
  | .u32 | .s32 | .f32 => 4
  | .u64 | .s64 | .f64 => 8

/-- Number of lanes of a vector. -/
def Vec.width : Vec → Nat
  | .v1 => 1
  | .v2 => 2
  | .v4 => 4

inductive AtomOp where
  | add | and_ | or_ | xor_ | cas | exch
  deriving Repr, DecidableEq, Inhabited

inductive CmpOp where
  | eq | ne | lt | le | gt | ge | ltu | leu | gtu | geu
  deriving Repr, DecidableEq, Inhabited

/-- Classification used for static feature extraction and for the memory model. -/
inductive OpClass where
  | read | write | atom | bar | async_ | wgmma | shfl | branch | arith | mbar
  deriving Repr, DecidableEq, Inhabited

/-- The 81 PTX forms modelled by this harness. -/
inductive Op where
  -- scalar / control (25)
  | mov (ty : Ty)
  | add (ty : Ty)
  | sub (ty : Ty)
  | mul (ty : Ty)
  | mad (ty : Ty)
  | div (ty : Ty)
  | rem (ty : Ty)
  | setp (ty : Ty) (cmp : CmpOp)
  | selp (ty : Ty)
  | shl (ty : Ty)
  | shr (ty : Ty)
  | and_ (ty : Ty)
  | or_ (ty : Ty)
  | xor_ (ty : Ty)
  | not_ (ty : Ty)
  | neg (ty : Ty)
  | abs (ty : Ty)
  | min_ (ty : Ty)
  | max_ (ty : Ty)
  | cvt (srcTy dstTy : Ty)
  | bra (target : Nat)
  | braUni (target : Nat)
  | ret
  | exit
  | barWarpSync
  -- memory (10)
  | ld (sp : Space) (ty : Ty) (v : Vec) (c : Cache)
  | st (sp : Space) (ty : Ty) (v : Vec) (c : Cache)
  | ldGlobalNc (ty : Ty) (v : Vec)
  | stGlobalCs (ty : Ty) (v : Vec)
  | membar (sc : Scope)
  | fenceSc
  | fenceProxyAsync
  | fenceProxyTensormap
  | atom (sem : Sem) (sc : Scope) (aop : AtomOp) (ty : Ty)
  | red (sem : Sem) (sc : Scope) (aop : AtomOp) (ty : Ty)
  -- Ampere (8)
  | cpAsyncCa (bytes : Nat)
  | cpAsyncCg (bytes : Nat)
  | cpAsyncCommitGroup
  | cpAsyncWaitGroup (n : Nat)
  | mmaSyncM16n8k16
  | ldmatrixX4
  | barSync (n : Nat)
  | barArrive (n : Nat)
  -- Hopper (16)
  | cpAsyncBulkTensor2d (dest : Space) (proxy : Proxy)
  | cpAsyncBulkCommitGroup
  | cpAsyncBulkWaitGroup (n : Nat)
  | mbarrierInit
  | mbarrierArrive
  | mbarrierArriveExpectTx
  | mbarrierTryWaitParity
  | mbarrierWaitParity
  | wgmmaFence
  | wgmmaMmaAsyncM64n8k16
  | wgmmaCommitGroup
  | wgmmaWaitGroup (n : Nat)
  | shflSyncIdx
  | shflSyncUp
  | shflSyncDown
  | shflSyncBfly
  -- cluster (5)
  | mapa
  | clusterSync
  | barrierClusterArrive
  | barrierClusterWait
  | clusterlaunchcontrolTryCancel
  -- Blackwell (7)
  | tcgen05Alloc
  | tcgen05Mma
  | tcgen05Commit
  | tcgen05Wait
  | tcgen05Ld
  | tcgen05St
  | tcgen05Fence
  -- parameter / fence / misc forms (10)
  | paramLd (ty : Ty)
  | paramSt (ty : Ty)
  | membarGl
  | membarCta
  | fenceProxyAsyncShared
  | cpAsyncWaitAll
  | ldmatrixX1
  | ldmatrixX2
  | tcgen05Cp
  | electSync
  deriving Repr, DecidableEq, Inhabited

/-- Operand of an instruction. -/
inductive Operand where
  | reg (r : Reg)
  | imm (i : Nat)
  /-- A memory operand:
  `base + tid * stride + (tid / rowSize) * rowStride + (tid % rowSize) * colStride + off`.
  Addresses may depend on the thread index (thread-local addressing); `rowSize = 0`
  disables the row/column terms. -/
  | mem (sp : Space) (base off stride : Nat) (rowSize rowStride colStride : Nat)
  deriving Repr, DecidableEq, Inhabited

namespace Operand

/-- Concrete address of an operand for thread `tid`. -/
def addr (o : Operand) (tid : Nat) : Addr :=
  match o with
  | .mem _ base off stride rowSize rowStride colStride =>
    base + tid * stride +
      (if rowSize = 0 then 0 else tid / rowSize * rowStride + tid % rowSize * colStride) + off
  | _ => 0

end Operand

structure Instr where
  op : Op
  dst : List Operand
  src : List Operand
  pred : Option Reg
  deriving Repr, DecidableEq, Inhabited

namespace Op

/-- Static classification of an instruction form. -/
def cls : Op → OpClass
  | .ld _ _ _ _ | .ldGlobalNc _ _ | .paramLd _ => .read
  | .st _ _ _ _ | .stGlobalCs _ _ | .paramSt _ => .write
  | .atom _ _ _ _ | .red _ _ _ _ => .atom
  | .barSync _ | .barArrive _ | .barWarpSync | .clusterSync
  | .barrierClusterArrive | .barrierClusterWait | .membar _ | .membarGl
  | .membarCta | .fenceSc => .bar
  | .cpAsyncCa _ | .cpAsyncCg _ | .cpAsyncCommitGroup | .cpAsyncWaitGroup _
  | .cpAsyncBulkTensor2d _ _ | .cpAsyncBulkCommitGroup | .cpAsyncBulkWaitGroup _
  | .cpAsyncWaitAll => .async_
  | .mmaSyncM16n8k16 | .ldmatrixX1 | .ldmatrixX2 | .ldmatrixX4 => .read
  | .wgmmaFence | .wgmmaMmaAsyncM64n8k16 | .wgmmaCommitGroup | .wgmmaWaitGroup _ => .wgmma
  | .shflSyncIdx | .shflSyncUp | .shflSyncDown | .shflSyncBfly | .electSync => .shfl
  | .bra _ | .braUni _ | .ret | .exit => .branch
  | .mbarrierInit | .mbarrierArrive | .mbarrierArriveExpectTx
  | .mbarrierTryWaitParity | .mbarrierWaitParity => .mbar
  | .fenceProxyAsync | .fenceProxyTensormap | .fenceProxyAsyncShared => .bar
  | .tcgen05Alloc | .tcgen05Mma | .tcgen05Commit | .tcgen05Wait | .tcgen05Ld
  | .tcgen05St | .tcgen05Fence | .tcgen05Cp => .wgmma
  | .mapa | .clusterlaunchcontrolTryCancel => .arith
  | .mov _ | .add _ | .sub _ | .mul _ | .mad _ | .div _ | .rem _ | .setp _ _
  | .selp _ | .shl _ | .shr _ | .and_ _ | .or_ _ | .xor_ _ | .not_ _ | .neg _
  | .abs _ | .min_ _ | .max_ _ | .cvt _ _ => .arith

/-- Space touched by a memory instruction, if any. -/
def space? : Op → Option Space
  | .ld sp _ _ _ | .st sp _ _ _ => some sp
  | .ldGlobalNc _ _ | .stGlobalCs _ _ => some .global
  | .paramLd _ | .paramSt _ => some .param
  | .cpAsyncCa _ | .cpAsyncCg _ => some .shared
  | .cpAsyncBulkTensor2d dest _ => some dest
  | .ldmatrixX1 | .ldmatrixX2 | .ldmatrixX4 | .tcgen05Ld | .tcgen05St => some .shared
  | _ => none

/-- Does the form write memory? -/
def isWrite : Op → Bool
  | .st _ _ _ _ | .stGlobalCs _ _ | .paramSt _ | .atom _ _ _ _ => true
  | .red _ _ _ _ => true
  | _ => false

/-- Does the form read memory? -/
def isRead : Op → Bool
  | .ld _ _ _ _ | .ldGlobalNc _ _ | .paramLd _ | .ldmatrixX1 | .ldmatrixX2
  | .ldmatrixX4 | .atom _ _ _ _ => true
  | _ => false

/-- Access width in bytes for a `ld`/`st` form. -/
def accessBytes : Op → Nat
  | .ld _ ty v _ | .st _ ty v _ => ty.size * v.width
  | .ldGlobalNc ty v | .stGlobalCs ty v => ty.size * v.width
  | .paramLd ty | .paramSt ty => ty.size
  | .atom _ _ _ ty => ty.size
  | .red _ _ _ ty => ty.size
  | .cpAsyncCa n | .cpAsyncCg n => n
  | _ => 0

end Op

end Harness.PTX
