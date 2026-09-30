# Choosing a kernel launch for a 640×360 frame on the Jetson Nano

These notes derive the launch configuration for `threshold_ink` from first principles, starting from
the hardware's limits and ending at `cuLaunchKernel(..., 20, 45, 1, 32, 8, 1, ...)`. Every number
marked *measured* was produced on this Nano (Tegra X1, MAXN, GPU locked at 921.6 MHz) on 2026-09-30.
Rerun them with `build/kbench` (see [../tools/kbench.md](../tools/kbench.md)). Expect ±20% between
runs, so compare configurations within one run.

The job: for each pixel of a 640×360 luma plane (row pitch 768 bytes), write a mask byte and
accumulate an ink count and bounding box.

---

## 1. What the hardware is

### Ask the GPU, don't trust a spec sheet

`cuDeviceGetAttribute` reports the limits the driver actually enforces. On this Nano (*measured*):

| Attribute | Value | What it limits |
|---|---|---|
| name, compute capability | Tegra X1, **5.3** (Maxwell) | Which rules below apply |
| `MULTIPROCESSOR_COUNT` | **1** | Only one SM does all the work |
| `WARP_SIZE` | 32 | Threads that execute in lockstep |
| `MAX_THREADS_PER_BLOCK` | 1024 | Upper bound for `bx*by*bz` |
| `MAX_THREADS_PER_MULTIPROCESSOR` | **2048** | Threads *resident* on the SM at once |
| `MAX_REGISTERS_PER_MULTIPROCESSOR` | 65536 | Shared by all resident threads |
| `MAX_REGISTERS_PER_BLOCK` | 32768 | |
| `MAX_SHARED_MEMORY_PER_MULTIPROCESSOR` | 64 KB | Shared by all resident blocks |
| `MAX_SHARED_MEMORY_PER_BLOCK` | 48 KB | |
| `L2_CACHE_SIZE` | 256 KB | The only cache global memory goes through by default on sm_5x |
| `CLOCK_RATE` | 921,600 kHz | GPU core clock |
| `GLOBAL_MEMORY_BUS_WIDTH` | 64 bits | LPDDR4, shared with the CPU |
| `INTEGRATED` | 1 | CPU and GPU share the same DRAM |

(`MEMORY_CLOCK_RATE` reports 12,750 kHz, which is meaningless on Tegra. The Nano's LPDDR4 runs at
1600 MHz, double data rate, over 64 bits: about **25.6 GB/s peak**, shared with the CPU, ISP, VIC and NVENC.)

One more limit comes from NVIDIA's compute-capability table rather than an attribute (CUDA 10.2
doesn't expose it): **at most 32 resident blocks per SM** on 5.x.

### Cores, warps, and "resident" threads

Three different numbers get mixed up. Keep them apart:

- **128 CUDA cores.** The SM is split into 4 partitions. Each has a **warp scheduler** and 32 lanes
  (cores). Every cycle, each scheduler picks one *ready* warp and issues its next instruction across
  its 32 lanes. So at most **4 warps = 128 threads execute an instruction in any one cycle**.
- **2048 resident threads = 64 resident warps.** "Resident" means the warp has been placed on the SM:
  its registers are allocated and its program counter exists, so the scheduler *could* pick it. The
  limit comes from the hardware's size. Each partition tracks up to 16 warps (4 × 16 = 64 warp slots),
  and the register file is sized for that. It's a fixed per-generation number from the
  compute-capability table, not something you configure.
- **Why hold 16× more threads than can run?** To **hide latency**. A global memory load takes
  hundreds of cycles. A warp that issues one stalls, and the scheduler switches to another ready warp
  at no cost, because every resident warp already has its own registers. With too few resident warps,
  all of them end up waiting on memory at the same time, and the cores sit idle.
- **Occupancy** = resident warps ÷ 64. It's a means to an end (enough ready warps to hide latency),
  not a score to maximize.

With only one SM, the entire grid goes through this one set of 64 warp slots, in **waves**.

---

## 2. Map the work to threads

**Start with one thread per output pixel.** The simplest mapping: a 640×360 plane needs 230,400
threads. Each reads 1 byte and writes 1 byte. (Section 7 revisits this, because it turns out to be too
little work per thread.)

Use a **2D** grid, because the data is 2D:
```
x = blockIdx.x * blockDim.x + threadIdx.x     // column
y = blockIdx.y * blockDim.y + threadIdx.y     // row
```

---

## 3. Choose the block shape

### Rule 1: `bx` is a multiple of 32, and 32 is the natural choice

Warps are formed from consecutive thread IDs, **x-fastest**: `tid = threadIdx.y * bx + threadIdx.x`.
With `bx = 32`, one warp = one row segment of 32 consecutive pixels = **32 consecutive bytes** in
memory. The memory system serves a warp's load in **32-byte sectors**, so one sector covers the warp's
whole request. That's **coalescing**.

What happens with other shapes (*measured*, whole kernel, one ink square):

| Block | One warp covers | Sectors per warp load | Time |
|---|---|---|---|
| 32×8 | 32 px of 1 row | 1 | **236 µs** |
| 16×16 | 16 px × 2 rows | 2 | 231 µs |
| 8×8 | 8 px × 4 rows | 4 | 277 µs |
| 8×32 | 8 px × 4 rows | 4 | 271 µs |
| **1×256** | 1 px × 32 rows | **32** | **901 µs** |

A column-shaped block is almost **4× slower**: every lane hits a different row, so a single warp
instruction turns into 32 separate memory transactions. 16×16 costs the same as 32×8 here, because at
1 byte per thread memory transactions aren't the bottleneck yet (see section 7). But `bx = 32` is the
shape that stays correct as you make the kernel faster.

### Rule 2: 128–256 threads per block

The block size decides how many blocks fit on the SM (`2048 / threads_per_block`, capped at 32 blocks):

| Threads/block | Blocks resident | Warps resident | Occupancy | Example (*measured*) |
|---|---|---|---|---|
| 32 | 32 (block cap) | 32 | **50%** | 32×1: **349 µs** |
| 64 | 32 | 64 | 100% | 8×8: 277 µs |
| 128 | 16 | 64 | 100% | 32×4: 229 µs |
| 256 | 8 | 64 | 100% | 32×8: 236 µs |
| 512 | 4 | 64 | 100% | 32×16: 251 µs |
| 1024 | 2 | 64 | 100% | 32×32: 294 µs |

- **Too small (32):** you hit the 32-blocks-per-SM limit before the 2048-thread limit, so the SM runs
  half full.
- **Too big (1024):** occupancy is still 100%, but only 2 blocks are resident. This kernel has
  `__syncthreads()` barriers. When a 32-warp block waits at a barrier, half the SM's warps wait with
  it, and there's only one other block to switch to. Big blocks also leave a larger partial final wave.
- **128–256** keeps full occupancy with 8–16 independent blocks to switch between. Both sizes divide
  2048 exactly, so no warp slots go unused.

### Rule 3: choose `by` so the grid tiles the frame exactly

`bx = 32` and 256 threads gives `by = 8`. Check that it divides the image:

- 640 / 32 = **20** blocks across, exactly
- 360 / 8 = **45** blocks down, exactly

So **900 blocks, 230,400 threads, zero idle threads**. Compare the shapes that don't divide evenly:

| Block | Grid | Threads launched | Wasted | Time |
|---|---|---|---|---|
| 32×8 | 20 × 45 | 230,400 | 0% | 236 µs |
| 16×16 | 40 × 23 | 235,520 | 2% | 231 µs |
| 256×1 | 3 × 360 | 276,480 | **17%** | **277 µs** |

640 isn't a multiple of 256, so each row gets 3 blocks (768 threads), and 128 of them do nothing
except run the bounds check.

For other sizes: `by` ∈ {4, 8} with `bx = 32` divides any height that's a multiple of 8. When nothing
divides evenly, round up (`(h + by - 1) / by`) and keep the `x < width && y < height` check. That's
why the check is in the kernel even though 640×360 doesn't need it.

---

## 4. Check what the kernel itself uses

Block shape isn't the only limit on occupancy. Registers and shared memory can cap it too. Ask the
driver about the compiled kernel (`cuFuncGetAttribute`, *measured* for `threshold_ink`):

| Resource | Per thread / block | Limit that could bind |
|---|---|---|
| Registers | **9** per thread | 65536 / (9 × 2048) → fine. Trouble starts around **32 regs/thread** (65536 / 2048). |
| Static shared memory | **20 B** per block (the five `__shared__ unsigned`) | 64 KB / 20 B → fine |
| Local memory | **0 B** | Nonzero would mean register spills to DRAM, which is slow |

With 256 threads per block, the binding limit is **threads**: 2048 / 256 = **8 blocks**, i.e.
100% occupancy. `cuOccupancyMaxActiveBlocksPerMultiprocessor(&n, fn, 256, 0)` returns **8**, which
agrees.

Recheck this whenever the kernel grows. LeNet's convolution kernels will use more registers and real
shared-memory tiles, and one of those two will become the binding limit instead of threads.

---

## 5. The launch

```c
const unsigned bx = 32, by = 8;                          // warp-wide rows, 256 threads
cuLaunchKernel(fn,
               (640 + bx - 1) / bx, (360 + by - 1) / by, 1,   // grid: 20 × 45 × 1 = 900 blocks
               bx, by, 1,                                     // block: 32 × 8 × 1 = 256 threads
               0,        // dynamic shared memory: none (the kernel's __shared__ sizes are fixed)
               NULL,     // default stream: the following cuMemcpyDtoH waits for this kernel
               args, NULL);
```

How it runs on the single SM:
- 8 blocks (64 warps) resident at a time.
- 900 / 8 = **112.5 waves**. As soon as one block finishes, the next takes its slot, so waves overlap
  rather than running in lock-step.
- Each cycle, the 4 schedulers issue from whichever of the 64 warps are ready.

---

## 6. Inside the kernel: why shared memory before global atomics

Every ink pixel updates five values (count, min_x, min_y, max_x, max_y). Doing that directly on global
memory means thousands of threads doing atomics on **the same 5 addresses**, all serialized in L2.

So the kernel reduces **per block first**:
1. Thread 0 initializes 5 `__shared__` counters; `__syncthreads()`.
2. Ink threads do atomics on shared memory. Shared memory is on the SM, so these are much cheaper and
   contend only within 256 threads.
3. `__syncthreads()`; thread 0 does **5 global atomics for the whole block** (skipped if the block
   saw no ink).

That's at most 900 × 5 global atomics per frame, instead of up to 230,400 × 5.

*Measured* cost: the same kernel on an all-white frame (no ink, no atomics) takes **184 µs**, versus
**236 µs** with a 29,651-pixel ink square (13% of the frame). About 50 µs is atomics, even with the
shared-memory step.

---

## 7. Is it fast? Compare against the floor

Always compare against what the memory system allows:

- Bytes moved per frame: 230 KB read + 230 KB written ≈ **460 KB**
- At the 25.6 GB/s peak: **~18 µs**. More realistically (60–75% of peak, shared with the rest of
  the SoC): ~25–30 µs.

The kernel takes 236 µs, **about 10× above the floor**. Where the time goes (*measured* with
stripped-down kernels, same 900-block grid):

| Kernel | Time | What it shows |
|---|---|---|
| empty body | **28 µs** | Launching and scheduling 900 blocks costs this much before any work |
| threshold only, 1 px/thread | **138 µs** | 1 byte per thread is too little work: overhead per thread and per load dominates |
| threshold only, **4 px/thread** (`uchar4`), 225 blocks | **79 µs** | Same work in 4-byte loads and stores: 1.75× faster |
| `threshold_ink` without ink | 184 µs | + shared-memory setup, two barriers, stat writes |
| `threshold_ink` with ink | 236 µs | + atomics |

The conclusion: 32×8 is the right **shape**, but one byte per thread is the wrong **granularity**.
In order of expected payoff:

1. **4 pixels per thread** with `uchar4` loads and stores, using a 32×8 block over a 160-wide
   (640/4) grid: 5 × 45 = 225 blocks. The 4-byte loads also need the row pitch and base address to be
   multiples of 4, which the 768 pitch satisfies.
2. **Warp-level reduction** before the shared atomics. `__ballot_sync` gives the warp's ink count
   with a single `__popc`, and `__shfl_down_sync` finds the warp's min and max. That leaves one shared
   atomic per warp instead of one per ink pixel.
3. **Fewer, bigger blocks** (a grid-stride loop, where each thread processes several rows). This cuts
   the 28 µs of block scheduling.

### The kernel isn't the whole "kernel" stage

In `build/phase2`, the "kernel" stage measures about **780 µs**, but in isolation the kernel takes
about **245 µs** of GPU time. `tools/kbench` also times the host-side part (synchronous stats reset,
launch, synchronous readback) and finds only about **63 µs** on top of the kernel: **~308 µs in
total**. So host overhead doesn't explain the gap. The remaining ~470 µs appears only in the real
pipeline. Likely causes, not yet measured separately: the kernel running on **NVMM memory mapped
through EGL** (possibly with different caching) instead of `cuMemAlloc` memory, and **contention**
from the ISP, VIC, NVENC and CPU threads sharing DRAM. Time the kernel with
`cuEventRecord`/`cuEventElapsedTime` inside `gpu_threshold()` to separate the two.

---

## 8. The recipe, for any frame size

1. **Query the device** (section 1). Note the SM count, 2048 threads per SM, the 32-block cap,
   registers, and shared memory.
2. **One thread per output element to start.** Use a 2D grid for 2D data.
3. **`bx = 32`** (a full warp along the contiguous dimension) → coalesced rows.
4. **128–256 threads per block** → `by = 4` or `8`. Choose the one that divides the height; otherwise
   round the grid up and keep the bounds check.
5. **Grid = ceil(width / bx) × ceil(height / by).**
6. **Check the kernel's resources** (registers, shared, local) and compute occupancy with
   `cuOccupancyMaxActiveBlocksPerMultiprocessor`. With 1 SM, confirm there are many waves (here 112),
   so the final partial wave doesn't matter.
7. **Time it with CUDA events** and compare against bytes ÷ bandwidth. If you're far above the floor,
   give each thread more work (vector loads, several pixels) before adjusting the block shape.
8. **Re-measure after every change.** The 16×16 vs 32×8 result shows that the textbook reason
   (coalescing) isn't always what's limiting you.
