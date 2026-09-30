# Phase 3 plan: LeNet-5 inference, developed on the Mac and deployed to the Nano

PLAN.md Phase 3 says to hand-write LeNet-5 in CUDA C. This document breaks that down into steps.
Most of the work can happen on the M2 Mac, which has no CUDA. The GPU-specific part is kept thin, so
by the time code reaches the Nano, only the question of how the GPU performs is still open, and the
questions of whether the math is right and whether the kernels are split correctly are already answered.

## The three implementations and how they relate

```
            ┌──────────────── Mac (M2) ────────────────┐     ┌──── Nano ────┐
PyTorch ──► weights.bin + golden.bin ──► C reference ──► C kernel emulator ──► CUDA kernels
(train)     (the contract)               (plain loops)   (same kernel code,     (same kernel code,
                                                          CPU launch loop)       cuLaunchKernel)
```

Each stage is checked against the one to its left using the same test harness and the same files:

1. **PyTorch** trains the network and exports weights, test images, and per-layer activations.
2. **The C reference** (`cpu_ref.c`) implements plain nested loops in NCHW layout. It has no
   parallelism concepts and gives the "obviously correct" answer that everything else is diffed against.
3. **The C kernel emulator** (`cpu_emu.c`) runs the kernel bodies from step 4 on the CPU. A small
   launcher iterates over `blockIdx` and `threadIdx`. It checks the *decomposition* (indexing,
   bounds, and what each thread owns) without needing a GPU.
4. **CUDA** (`kernels/lenet.cu`) compiles the same kernel bodies into `__global__` wrappers. They're
   compiled to PTX by clang-12 and launched with the driver API, exactly as in Phase 2.

Steps 1 to 3 run on the Mac. On the Nano, step 3 also runs, and it's a useful sanity check that the
aarch64 Linux build agrees with macOS.

---

## 1. The Python reference (Mac only)

### The linked notebook

[lychengrex/LeNet-5-Implementation-Using-Pytorch](https://github.com/lychengrex/LeNet-5-Implementation-Using-Pytorch/blob/master/LeNet-5%20Implementation%20Using%20Pytorch.ipynb)
has the right **architecture**, but it isn't a good base to build on:

| | Notebook | Why it matters here |
|---|---|---|
| Architecture | conv5×5(1→6, pad 2) → ReLU → maxpool2 → conv5×5(6→16) → ReLU → maxpool2 → FC 400→120→84→10 | ✅ This is the standard "modern LeNet-5" (ReLU and max pool instead of tanh and subsampling). Keep it. |
| Data loading | `from MNISTtools import load`, a course helper that isn't in the repo | ❌ It doesn't run as-is. `torchvision.datasets.MNIST` replaces it. |
| Evaluation | The "Evaluate on the testing dataset" cell calls `backprop_deep(xtest, ltest, ...)`, which **trains on the test set** before measuring test accuracy | ❌ The 95.42% figure is contaminated. Only the 97.98% GPU figure is valid. |
| Accuracy | 97.98% after 10 epochs of SGD with lr 0.001 | ⚠️ Low for LeNet. Adam gets about 99% in 3 to 5 epochs. |
| Device | `cuda` or `cpu` | ⚠️ No `mps`. It works on the M2 CPU, but at this size, CPU and MPS perform about the same anyway. |
| Export | No weight saving, no ONNX | ❌ Export is the most important output for this project. |
| Normalization | `x*2/255 - 1` (range [-1, 1]) | ✅ It's fine, but it's part of the contract: the camera preprocessing must produce the same range. |

**Verdict:** copy the `LeNet` class (it's 15 lines) and write the rest in about 150 lines. No
better-known repo fits well. The official `pytorch/examples` MNIST network isn't LeNet: it uses
dropout and a 9216-wide FC layer. The notebook's structure is fine as a reading reference.

### What `phase3/train/` produces

```
train/train.py     torchvision MNIST, Adam, 5 epochs, model.eval() test accuracy → lenet.pt
train/export.py    lenet.pt → weights.bin, golden.bin, lenet.onnx
```

- **`weights.bin`**: a small header (magic `LNT5`, version, tensor count), then for each tensor its
  name, shape, and float32 little-endian data in **PyTorch's own layout** (`conv.weight` is
  `[out][in][kh][kw]` and `fc.weight` is `[out][in]`). The order is fixed: conv1.w, conv1.b, conv2.w,
  conv2.b, fc1.w, fc1.b, fc2.w, fc2.b, fc3.w, fc3.b. That's 61,706 floats, about 241 KB. Keep the
  PyTorch layout at first. Any relayout for coalescing later becomes an explicit, testable step in C.
- **`golden.bin`**: for the first N test images (for example 64), the normalized input plus the
  output of **every layer** (conv1, relu1, pool1, ..., logits). When the C code disagrees, this tells
  you *which layer* is wrong instead of just "wrong digit". It's generated with forward hooks.
- **The full MNIST test set** (10k images, 7.8 MB in idx format) is used for the accuracy number. It
  stays out of git (`data/`), is downloaded by torchvision, and is synced to the Nano with `sync.sh`.
- **`lenet.onnx`** (opset 11) is for the `trtexec` baseline on the Nano (TensorRT 8.2).

Python rules: CLAUDE.md's "3.8, stdlib + gi" rule applies to the **Nano**. `train/` is Mac-only
and uses a venv with `torch` and `torchvision`. The Nano never runs it. Only its output files cross over.

### The operation count, for scale

| Layer | Output | MACs |
|---|---|---|
| conv1 5×5, 1→6, pad 2 | 6×28×28 | 117,600 |
| pool1 | 6×14×14 | |
| conv2 5×5, 6→16 | 16×10×10 | 240,000 |
| pool2 | 16×5×5 = 400 | |
| fc1 400→120 | 120 | 48,000 |
| fc2 120→84 | 84 | 10,080 |
| fc3 84→10 | 10 | 840 |
| **total** | | **≈ 417k MACs per image** |

That's about 0.1 ms of naive C on one M2 core, and not much more on the Nano's A57. The GPU will not
win on raw math for one image. Launch overhead (about 10 to 20 µs per launch on the Nano) is on the same
order as all the work. This shapes the CUDA plan in section 4, and it's the most important performance
lesson of the phase.

---

## 2. The C reference (`cpu_ref.c`)

Plain C11, no SIMD intrinsics, and no Accelerate. Write one function per layer, with explicit
dimensions and an `out[c][y][x]` NCHW layout:

```c
void conv2d_ref(const float *in, int C, int H, int W,
                const float *w, const float *b, int K, int pad, int OC,
                float *out);          /* out: OC × (H+2p-K+1) × (W+2p-K+1) */
void relu_ref(float *x, int n);
void maxpool2_ref(const float *in, int C, int H, int W, float *out);
void linear_ref(const float *in, int N, const float *w, const float *b, int M, float *out);
```

It's done when it matches `golden.bin` layer by layer to within `1e-4` max absolute error, and when
its test-set accuracy equals PyTorch's to within one or two images.

**Floating-point parity:** results won't be bit-identical to PyTorch or to CUDA. Summation order
differs, and CUDA fuses `a*b+c` into FMA by default. Compare activations with a tolerance, and require
the **argmax** (the predicted digit) to match exactly. On the full test set, a handful of
near-tie images may flip. Count them instead of hiding them.

---

## 3. The kernel emulator: your "CPU with a kernel boundary" idea

Your instinct is right, and it's the core of this phase: **write the code in the shape it will run in
on the GPU, then run that shape on the CPU**. Some refinements make it pay off:

### 3.1 One source of truth for the kernel math

Put each kernel's per-thread body in a header that compiles as both C (host) and CUDA:

```c
/* kernels/lenet_kernels.h: included by cpu_emu.c (C11) and kernels/lenet.cu (clang CUDA) */
#ifdef __CUDACC__
#  define KDEV __device__ __forceinline__
#else
#  define KDEV static inline
#endif

typedef struct { unsigned bx, by, bz, tx, ty, tz, bdx, bdy, bdz; } KIdx;

/* One thread per output element: out[oc][y][x]. Launch: grid (ceil(OW/bx), ceil(OH/by), OC). */
KDEV void conv1_body(KIdx t, const float *in, const float *w, const float *b, float *out) { ... }
```

- **CUDA:** `extern "C" __global__ void conv1(...) { conv1_body(cuda_idx(), ...); }`
- **Emulator:** `emu_launch(grid, block, conv1_thunk, args)` is a six-deep loop that builds a
  `KIdx` and calls the same body.

If the emulator passes and CUDA fails, the bug is in the launch, memory, or GPU, never in the
math. That's the clean kernel boundary you're after.

### 3.2 Emulate the host API too

Mirror the driver API calls that `phase2/src/gpu.c` already uses: `emu_mem_alloc`, `emu_memcpy_htod`,
`emu_memcpy_dtoh`, and `emu_launch`. Put the network behind one interface:

```c
/* src/lenet.h */
typedef struct Lenet Lenet;
Lenet *lenet_create(const char *weights_path);          /* uploads weights once */
int    lenet_forward(Lenet *, const float in[28*28], float logits[10], LenetTimes *t);
```

The same `lenet_test` harness then links against `lenet_ref.c`, `lenet_emu.c`, or (on the Nano)
`lenet_cuda.c`. Host code that differs only in which backend it links against is exactly what the
Rust port wants later.

### 3.3 Shared memory and `__syncthreads`: split into phases

A serial loop over threads can't pause at a barrier. The standard fix, used by CPU CUDA
implementations such as MCUDA and POCL, is **loop fission**: split the kernel at each barrier into
phase functions. The emulator runs phase 0 for every thread in the block, then phase 1 for every
thread, and so on. `__shared__` becomes a per-block buffer passed in by the launcher. On CUDA, the
wrapper calls `phase0(); __syncthreads(); phase1();`. Writing kernels this way is a good discipline:
you have to state exactly what crosses the barrier.

The naive version (section 4, step A) needs no barriers. Add the phase machinery when you reach tiling.

### 3.4 What the emulator can and can't tell you

| It catches | It can't tell you |
|---|---|
| Indexing, bounds, and off-by-one errors in the grid mapping | Speed: coalescing, occupancy, bank conflicts |
| Tail blocks (sizes that don't divide evenly) | Warp divergence cost |
| Wrong layer wiring or weight layout | Real data races (serial execution hides them) |
| Reads of another thread's output with no barrier (see below) | FMA and rounding differences from the GPU |

**Race detection:** add `EMU_ORDER=reverse|shuffle` to run the threads within a phase in a different
order. A correct kernel gives bit-identical output in every order (except for float atomics).
Any difference means a thread read something another thread wrote in the same phase, which would be
a real race on the GPU. It costs one line of code and finds real bugs.

### 3.5 Should you use Apple's matrix hardware?

**Not for the mimic.** Accelerate's `cblas_sgemm` (which runs on the M2's AMX units) would hide the kernel
boundary you're trying to emulate. At batch size 1, LeNet is too small to benefit anyway. One place
it earns a spot is as a **separate, optional backend** (`lenet_gemm.c`): im2col + sgemm. That's the
other classic way to structure convolutions, and you may try it in CUDA too. Having it on the Mac
gives you a correctness reference for the im2col layout. Treat it as a side quest, not the main path.

The M2's real GPU equivalent is **Metal compute**. A threadgroup maps to a block, threadgroup memory
maps to `__shared__`, and a SIMD-group of 32 maps to a warp. It's the only way to learn *performance*
lessons on the Mac, but it needs Objective-C or Swift on the host, and the performance lessons
wouldn't transfer to a single Maxwell SM on shared LPDDR4. Skip it unless you're curious.

---

## 4. CUDA on the Nano (`kernels/lenet.cu`)

Each step is a separate kernel set behind the same `lenet.h`, so they can be benchmarked against
each other with a `kbench`-style tool (`tools/lbench`). Every step is emulated on the Mac first.

| Step | Kernels | What it teaches |
|---|---|---|
| **A. Naive, one kernel per layer** | conv1, relu, pool, conv2, relu, pool, fc×3 (9 launches). One thread per output element, and every weight read comes from global memory. | Grid mapping, and why launch overhead dominates: expect **tens of µs of launches for well under 1 ms of math**. Measure with events around each launch. |
| **B. Fused layers** | conv+ReLU+pool as one kernel (each thread computes a 2×2 window and writes the max). FC+ReLU fused. That's 5 launches. | Fusion removes launches *and* intermediate round trips through DRAM. |
| **C. Shared-memory tiling** | Stage the input tile and the filter in `__shared__`. | Barriers (the emulator's phase split), reuse, bank conflicts. Compare with `notes.md`'s occupancy reasoning. |
| **D. Whole network in one block** | One launch. The activations (at most 6×28×28 floats, or 18.8 KB) live in shared memory. The weights (241 KB) stream through L2 (256 KB!). | The Nano-specific answer for batch 1: one launch, one SM, and nothing touches DRAM except weights and input. |
| **E. Batching** (optional) | One block per digit when a frame contains several digits. | When the GPU finally wins over the CPU. |

Also measure the **CPU reference on the Nano's A57**. Seeing whether it beats step A is a result worth recording.

**Baseline:** `trtexec --onnx=lenet.onnx --fp16` on the Nano. Try to match it with step D.

**Weight upload:** upload once at startup (`cuMemAlloc` + `cuMemcpyHtoD`). Memory is unified, so
another option is `cuMemHostAlloc` with mapped memory, which is a zero-copy experiment worth one row
in the benchmark table.

---

## 5. Preprocessing: the part that decides accuracy

PLAN.md is right that this matters more than the network. Phase 2's `threshold_ink` already produces
the mask and bounding box. Phase 3 adds a kernel (also emulated on the Mac) that:

1. Crops the bounding box and scales it so the longer side is 20 px, keeping the aspect ratio (MNIST's rule).
2. Places it in a 28×28 frame so the **center of mass** lands at (14, 14).
3. Normalizes it to [-1, 1], white digit on black, **exactly** as `train.py` did.

Phase 2 can already save frames: `build/phase2 -n 300 -d captures/frame` writes `_y.pgm` and
`_mask.pgm`. Pull them back with `scripts/pull.sh captures` and use them as **real camera test inputs
on the Mac**. Preprocessing can then be developed and tuned against real frames without the Nano
attached. Hand-label a few dozen frames (a text file of `name digit` lines) to get a camera-accuracy
number alongside MNIST test accuracy.

---

## 6. Layout and build

```
phase3/
  PLAN.md               this file
  Makefile              `make` on the Mac builds ref + emu; on the Nano it also builds CUDA + PTX
  train/                Mac only: train.py, export.py, requirements.txt (torch, torchvision)
  data/                 (gitignored) MNIST idx files, weights.bin, golden.bin, lenet.onnx
  src/
    lenet.h             backend interface, layer shape constants
    weights.c/.h        weights.bin loader, shared by every backend
    mnist.c/.h          idx and golden.bin readers
    lenet_ref.c         plain loops
    emu.c/.h            KIdx, emu_launch, phase split, EMU_ORDER, emu_mem_*
    lenet_emu.c         the kernel bodies driven by emu.c
    lenet_cuda.c        the kernel bodies driven by cuLaunchKernel (Nano only)
  kernels/
    lenet_kernels.h     per-thread bodies, compiled as both C and CUDA
    lenet.cu            thin __global__ wrappers → build/lenet.ptx
    preprocess.cu       crop / scale / center-of-mass / normalize
  tools/
    lenet_test.c        accuracy on the test set + per-layer diff against golden.bin
    lbench.c            per-step timing on the Nano
```

The Makefile detects the platform with `uname`. On Darwin it uses Apple clang and builds only
`lenet_ref`, `lenet_emu`, and `lenet_test`. On the Nano it uses clang-12, as Phase 2 does, and adds
the CUDA targets. The `.c` files are the same on both machines.

Data and weight files are **generated artifacts**. Keep them out of git and regenerate them with
`train/export.py`. `sync.sh` carries them to the Nano like any other file in the tree.

---

## 7. Milestones

1. `train.py` reaches at least 98.8% test accuracy. `export.py` writes `weights.bin`, `golden.bin`, and `lenet.onnx`.
2. `lenet_test --backend ref` matches golden per layer, and its test accuracy equals PyTorch's (Mac).
3. `lenet_test --backend emu` step A matches ref bit-for-bit or within tolerance, in all `EMU_ORDER` modes (Mac).
4. The same two commands pass on the Nano (aarch64 Linux, clang-12).
5. `lenet_test --backend cuda` step A matches emu. `lbench` records the launch-overhead lesson.
6. Steps B through D, each emulated first. Compare against `trtexec`.
7. Preprocessing kernel: accuracy on the pulled-back camera frames (Mac), then live in the Phase 2
   pipeline with results printed per frame. UDP output is Phase 4.
