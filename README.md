# KernelHarness

**Make your GPU code faster, and prove it still gives the right answer.**

[![ci](https://github.com/pythongiant/KernelHarness/actions/workflows/ci.yml/badge.svg)](https://github.com/pythongiant/KernelHarness/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)
![status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)

A **CUDA kernel** is a small program that runs on an NVIDIA GPU. Making one fast usually
means trying tricks by hand, and every trick risks quietly breaking the result.

KernelHarness does this for you, one step at a time:

1. It **runs** your kernel on a real GPU and measures how fast it is.
2. It **works out** what is slowing it down, from the measured numbers.
3. It **applies one** well-known speed-up trick.
4. It **proves**, with a math proof checked by the [Lean](https://lean-lang.org) proof
   checker, that the new kernel still produces exactly the right output.
5. It **saves** everything to git, so you can see what was tried and why.

Then it repeats until the kernel is as fast as you asked for.

## Highlights

- **Faster kernels:** in the example below, a simple kernel goes from 64% to 94% of the GPU's top speed in two steps.
- **Proved correct:** every faster version comes with a checked proof, not just a passing test.
- **Explains itself:** each change states the number that triggered it and why the trick should help.
- **Nothing hidden:** failed attempts are saved too, and simulated numbers can never count as a pass.
- **Try it without a GPU:** a built-in simulator runs the whole loop on your laptop.
- **Real GPUs on demand:** H100, H200, B200, A100 and L4 through [Modal](https://modal.com).
- **Web UI:** watch a run live and read each step.

## What it looks like

The included `elementwise_affine` kernel computes `out[i] = 10 * x[i] + 3` for 4 million
numbers. Here is a real run on an NVIDIA H100:

| step | what changed | speed (% of GPU max) | time | vs PyTorch | proof |
|---|---|---|---|---|---|
| start | the original kernel | 63.6% | 15.7 µs | 1.10× | ✅ verified |
| 1 | read and write 4 numbers at a time | 82.8% | 12.1 µs | 1.49× | ✅ verified |
| 2 | streaming stores and a grid-stride loop | 94.3% | 10.6 µs | 1.71× | ✅ verified |

This is the change it made in step 1, together with its reasoning (output trimmed):

```diff
-    out[i] = 10 * x[i] + 3;
+    int4 v = x[t];             // load 4 numbers in one instruction
+    v.x = 10 * v.x + 3;
+    v.y = 10 * v.y + 3;
+    v.z = 10 * v.z + 3;
+    v.w = 10 * v.w + 3;
+    out[t] = v;                // store 4 numbers in one instruction
```

```text
[DIAGNOSE ] MEM_BW_BOUND: memory-bound at 63.62% of peak
            (2131 of 3350 GB/s, moving 33554432 bytes in 0.02 ms)
[RESEARCH ] vectorize_x4: the kernel issues one 4-byte load and one 4-byte store per
            element, so 32-byte memory sectors are half used. int4 moves 16 bytes
            per instruction.
[VERIFY   ] proof VERIFIED (instance N=8, a=10, b=3; axiom audit ok)
[PROFILE  ] time 0.016 -> 0.012 ms, speed 63.6% -> 82.8%, output matches reference
```

All the files from this run are in [`examples/runs/elementwise_affine-h100/`](examples/runs/elementwise_affine-h100/).

## Installation

You need Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/pythongiant/KernelHarness.git
cd KernelHarness
uv sync
```

Optional extras:

| for | install |
|---|---|
| proofs | Lean: `curl -sSfL https://elan.lean-lang.org/elan-init.sh \| sh -s -- -y`, then `cd lean && lake build` |
| real GPUs | a Modal account: `uv run modal setup`, then `uv run modal deploy -m harness.backends.modal_gpu` |
| web UI | [bun](https://bun.sh): `cd web && bun install && bun run build` |

## Getting started

Try the whole loop with no GPU, using the simulator:

```console
$ uv run harness init
$ uv run harness run --kernel elementwise_affine --backend sim --max-rounds 3
[PROFILE  ] measurement baseline measurement
[VERIFY   ] decision    baseline proof VERIFIED
[DIAGNOSE ] reasoning   bottlenecks
            - MEM_BW_BOUND (high): memory-bound at 10.00% of peak ...
[RESEARCH ] reasoning   ranked hypotheses
            vectorize_x4 for MEM_BW_BOUND (source catalog)
[VERIFY   ] decision    proof VERIFIED
[PROFILE  ] measurement round 1 measurement
            p50_ms 0.100 -> 0.029, sol_pct 10.0 -> 34.0, simulated True
...
stop reason: MAX_ROUNDS  (run run_8cbda8457609)
```

The run ends with `MAX_ROUNDS` and never "passes". That is on purpose: simulated numbers
are not real measurements, so they are never accepted.

See what happened:

```bash
uv run harness runs          # list runs
uv run harness report        # the latest run's steps, as JSON
uv run harness serve         # open http://127.0.0.1:8787 in a browser
```

Run it for real on an H100, aiming for 80% of the GPU's max speed. A run passes only when
every target is met: speed, a 1.5× speed-up over PyTorch, and a verified proof.

```bash
uv run harness run --kernel elementwise_affine --gpu H100 --sol 80
```

## What you get from a run

Each run is saved to `ledgers/`, which is a git repository of its own, separate from
this one. Each step gets one commit and one folder:

```text
ledgers/runs/<run>/round-01/
  profile.json      the measured numbers
  diagnosis.md      what is slow, and the number that shows it
  hypotheses.json   the trick that was chosen, and why
  kernel.cu         the new kernel source
  kernel.ptx        what the GPU actually ran
  diff.patch        the change from the previous step
  proof/            the generated Lean proof and whether it checked
```

## Using your own kernel

Each kernel is described by a short `kernel.toml` file in `harness/kernels/<name>/`,
which gives its input size, data type and expected output. Run `uv run harness registry`
to see the built-in ones (elementwise, matrix multiply, attention).

To measure a kernel you wrote yourself, copy [`examples/custom_kernel/`](examples/custom_kernel/)
and point `source` at your `.cu` file. The harness measures it, checks its output and
diagnoses it. Automatic speed-ups and proofs need a kernel family with templates;
[CONTRIBUTING.md](CONTRIBUTING.md#adding-a-kernel) shows how to add one.

## FAQ

**Do I need a GPU?**
No, not to try it. `--backend sim` runs everything locally. You need a GPU (through
Modal) for real measurements and for a run to finish with `SLA_MET` (every target met).

**Do I need to know Lean?**
No. The harness writes the proofs and runs the checker for you. You only need Lean
installed. Without it, proofs are marked `UNSUPPORTED` and the run cannot pass.

**What exactly is proved?**
That every valid way the GPU could run the kernel's PTX (its low-level code) ends with
exactly the expected output bytes. The proof uses a small version of the same launch
(for example 8 numbers instead of 4 million). The full-size output is checked on the GPU
against a reference.

**What can't it prove?**
Floating-point kernels, because rounding is not modelled, and kernels whose threads read
each other's results within one launch. These are reported as `UNSUPPORTED`. They are
never wrongly marked as verified.

**What does "% of GPU max" (SOL) mean?**
"Speed of light": how close the kernel gets to the hardware limit. For memory-heavy
kernels that limit is memory bandwidth, and for compute-heavy kernels it is arithmetic
throughput.

**Does it use an LLM?**
Not by default. Speed-up ideas come from a fixed catalogue of known techniques. You can
point `HARNESS_LLM_BASE_URL` at any OpenAI-compatible server to suggest extra ideas. They
go through the same measuring and proving steps, and are labelled as LLM suggestions.

## Configuration

Run `harness run --help` for all flags. The main ones are `--kernel`, `--gpu`,
`--backend`, `--max-rounds`, `--sol` (target % of max speed) and `--speedup` (target
speed-up over PyTorch, default 1.5×).

| environment variable | default | what it does |
|---|---|---|
| `HARNESS_STATE_DIR` | `.harness` | where the database and logs go |
| `HARNESS_LEDGER_DIR` | `ledgers` | where the run history (a git repo) goes |
| `HARNESS_LLM_BASE_URL` | unset | optional LLM server for extra ideas |
| `HARNESS_LLM_MODEL` | `qwen/qwen3-coder-30b` | model name for that server |

## Documentation

- [docs/reference.md](docs/reference.md): how it works inside, all status codes, the
  Lean model, and troubleshooting
- [CONTRIBUTING.md](CONTRIBUTING.md): development setup, and how to add kernels, tricks
  and PTX instructions

## License

[MIT](LICENSE)
