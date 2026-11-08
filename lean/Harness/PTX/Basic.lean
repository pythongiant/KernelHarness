/-!
# PTX basic types

Byte-addressed memory, register values, and the oracle used for external data.

Numerical scope: values are bit patterns. Floating point is represented as
`BitVec 32` / `BitVec 64`; the model reasons about exact arithmetic on these
bit patterns and does **not** model IEEE rounding of real-number arithmetic.
-/

namespace Harness.PTX

/-- A byte address. -/
abbrev Addr := Nat

/-- A byte. -/
abbrev Byte := UInt8

/-- Byte-addressable memory. Total function, so every address is mapped. -/
structure ByteMap where
  bytes : Addr → Byte

namespace ByteMap

@[simp] theorem eta (m : ByteMap) : (⟨m.bytes⟩ : ByteMap) = m := by cases m; rfl

/-- Read a byte. -/
def get (m : ByteMap) (a : Addr) : Byte := m.bytes a

/-- Write a byte. -/
def set (m : ByteMap) (a : Addr) (b : Byte) : ByteMap :=
  ⟨fun a' => if a' = a then b else m.bytes a'⟩

@[simp] theorem get_set_same (m : ByteMap) (a : Addr) (b : Byte) : (m.set a b).get a = b := by
  simp [set, get]

theorem get_set_other (m : ByteMap) (a a' : Addr) (b : Byte) (h : a' ≠ a) :
    (m.set a b).get a' = m.get a' := by
  simp [set, get, h]

@[ext] theorem ext {m m' : ByteMap} (h : ∀ a, m.get a = m'.get a) : m = m' := by
  cases m; cases m'; congr; funext a; exact h a

/-- Two maps agree on a set of addresses. -/
def agreeOn (m m' : ByteMap) (s : Addr → Prop) : Prop := ∀ a, s a → m.get a = m'.get a

end ByteMap

/-- A machine value. -/
inductive Value where
  | b8 (v : Byte)
  | u32 (v : UInt32)
  | u64 (v : UInt64)
  | f32 (v : BitVec 32)
  deriving Repr, DecidableEq, Inhabited

/-- Register index. -/
abbrev Reg := Nat

/-- Register file: partial map from register index to value. -/
abbrev RegFile := Reg → Option Value

/-- External data source: one query yields one value (loads, shuffles, mma). -/
abbrev Oracle := Nat → Value

/-- Little-endian byte of a `BitVec 32`. -/
def f32Byte (v : BitVec 32) (i : Nat) : Byte :=
  ⟨v.toNat >>> (8 * i) &&& 0xFF⟩

/-- Assemble a `BitVec 32` from four little-endian bytes. -/
def f32OfBytes (b0 b1 b2 b3 : Byte) : BitVec 32 :=
  (b0.toNat : BitVec 32) ||| ((b1.toNat : BitVec 32) <<< 8)
    ||| ((b2.toNat : BitVec 32) <<< 16) ||| ((b3.toNat : BitVec 32) <<< 24)

/-- Little-endian byte of a `UInt64`. -/
def u64Byte (v : UInt64) (i : Nat) : Byte :=
  ⟨v.toNat >>> (8 * i) &&& 0xFF⟩

end Harness.PTX
