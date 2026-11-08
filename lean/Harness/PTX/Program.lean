import Harness.PTX.Instr

/-!
# Programs, grids and launches
-/

namespace Harness.PTX

structure Program where
  instrs : Array Instr

/-- Thread-block-cluster grid configuration. -/
structure Grid where
  gridDim : Nat × Nat × Nat
  blockDim : Nat × Nat × Nat
  clusterDim : Nat × Nat × Nat
  /-- Total number of threads in a 1-D flattening of the grid. -/
  launchOrder : Nat
  deriving Repr, DecidableEq, Inhabited

abbrev Grid.threads (g : Grid) : Nat := g.launchOrder

/-- A kernel launch: program, grid, initial memory, kernel parameters. -/
structure Launch where
  prog : Program
  grid : Grid
  init : ByteMap
  params : Addr → Value

/-- Number of instructions in the program. -/
abbrev Launch.codeLength (l : Launch) : Nat := l.prog.instrs.size

/-- Read the instruction at `pc`, if in range. -/
abbrev Launch.instrAt (l : Launch) (pc : Nat) : Option Instr := l.prog.instrs[pc]?

end Harness.PTX
