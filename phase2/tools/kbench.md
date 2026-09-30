# kbench: benchmarking the threshold kernel without the camera

`build/kbench` measures `threshold_ink` (the same `build/threshold.ptx` that `phase2` loads) on
synthetic frames, and reports the GPU limits and kernel resources that explain the numbers.
The reasoning behind the launch configuration is in [../kernels/notes.md](../kernels/notes.md);
this file covers how the tool is built and what it does and doesn't test.

```
cd phase2 && make tools        # build/kbench + build/probe.ptx (+ threshold.ptx)
build/kbench                   # full run at 640x360, about 10 s
build/kbench -b 32x8,32x4      # only these block shapes
build/kbench -s 1280x720       # another frame size (pitch defaults to width rounded up to 256)
build/kbench -S blocking       # context scheduling mode for the host-path section
build/kbench -r 1000           # more reps for steadier medians
```

From the Mac: `scripts/remote.sh make -C phase2 tools`, then
`scripts/remote.sh phase2/build/kbench`.

## Design

### Files

| File | Role |
|---|---|
| `tools/kbench.c` | Host program, C11 and the CUDA driver API only. No GStreamer, EGL or camera. |
| `tools/probe.cu` | Three stripped-down kernels (`empty_k`, `thresh1_k`, `thresh4_k`), compiled to `build/probe.ptx` |
| `kernels/threshold.cu` | The real kernel, **not a copy**. kbench loads the same `build/threshold.ptx` as `phase2`. |

kbench finds both PTX files next to its own executable, like `phase2` does. Build flags come from the
same Makefile variables (`CLANG`, `CUFLAGS`), so the benchmarked PTX is byte-for-byte the one the
pipeline runs.

### Known input, checked output

Every measurement runs on one of two synthetic frames uploaded once with `cuMemAlloc`:

- **ink frame:** luma 200 everywhere, with a dark (20) rectangle over the middle third horizontally
  and middle half vertically. At 640×360 that's 213×180 = 38,340 px, **16.6%** of the frame.
- **blank frame:** luma 200 everywhere, so no pixel passes the threshold and **no atomics run**.

The frames use a real **pitch** (768 bytes for 640 pixels, matching `nvvidconv`'s NVMM output), so
the kernel's `y * pitch + x` indexing is exercised exactly as in the pipeline.

Because the right answer is known, every configuration in the sweep reads back its `InkStats` and
compares it with the expected count and bounding box (`check` column: `ok/ok` for the ink and blank
frames). A fast wrong answer shows up as `FAIL`.

### Timing method

- **GPU time** (`time_gpu`): `cuEventRecord` right before and right after `cuLaunchKernel` in the
  same stream, then `cuEventSynchronize` and `cuEventElapsedTime`. The stats reset
  (`cuMemcpyHtoD`) happens *before* the first event, so it isn't counted. This measures the kernel
  as the GPU ran it, including block scheduling, and nothing on the CPU side.
- **CPU wall time** (`time_host_path`): `clock_gettime` around exactly what `gpu_threshold()` does
  per frame: synchronous reset, launch, synchronous readback.
- **Warm-up:** 10 unrecorded reps before each measurement. They absorb the one-time costs (PTX JIT
  on first use, cache warm-up) so these don't skew the medians.
- **Statistics:** the **median** is the headline number, because a single scheduling hiccup can't
  move it. The **minimum** is in parentheses, as "the best this configuration can do". Averages aren't
  reported, because on a shared SoC the outliers are interference, not the kernel's cost.

### Output sections

1. **device:** `cuDeviceGetAttribute` for the limits that decide occupancy (SMs, resident threads,
   registers, shared memory, L2, clock, bus width). The 32-blocks-per-SM limit is printed from
   the compute-capability table, because CUDA 10.2 can't query it.
2. **kernel:** `cuFuncGetAttribute` on the compiled `threshold_ink`: registers per thread, static
   shared memory, and local memory. Nonzero local memory means register spills and is flagged.
3. **frame:** size, pitch, ink box, and the **floor**: bytes moved (luma read + mask write) divided by
   the Nano's 25.6 GB/s LPDDR4 peak. The driver's `MEMORY_CLOCK_RATE` is wrong on Tegra, so the peak
   is a constant in the code (`PEAK_GBPS`).
4. **block shape sweep:** for each shape:

   | Column | Meaning |
   |---|---|
   | `thr` | threads per block |
   | `grid`, `blocks` | grid size (rounded up) and total blocks |
   | `idle%` | launched threads that fail the bounds check, i.e. wasted |
   | `blk/SM` | `cuOccupancyMaxActiveBlocksPerMultiprocessor`: resident blocks on the one SM |
   | `occ` | resident threads ÷ 2048 |
   | `ink us`, `blank us` | median (min) GPU time on each frame; the difference is the atomics |
   | `GB/s` | effective bandwidth on the ink frame (bytes ÷ median) |
   | `check` | correctness on ink/blank |

5. **probes:** same 32×8 grid, blank frame, with parts of the work removed:
   - `empty_k`: an empty body, i.e. launch plus scheduling 900 blocks
   - `thresh1_k`: threshold and mask only, one pixel per thread (no shared memory, barriers or stats)
   - `thresh4_k`: the same with `uchar4`, four pixels per thread (needs width and pitch divisible by 4)
6. **host path:** CPU wall time of reset + launch + readback, compared with the kernel's GPU time. The
   difference is the cost of the synchronous copies, the launch call and waking the CPU thread,
   under the chosen `-S` scheduling mode.

## The space it tests

| Dimension | Values | How |
|---|---|---|
| Block shape | 12 shapes by default: warp-wide (32×N), square, row-only (N×1), column (1×256), tiny (8×8, 32×1), huge (32×32) | `-b` |
| Frame size and pitch | default 640×360 / 768 | `-s`, `-p` |
| Ink density | 0% (blank) and 16.6% (ink box), both on every shape | fixed |
| Work per thread | 1 px (`threshold_ink`, `thresh1_k`) and 4 px (`thresh4_k`) | fixed in the probes |
| Kernel content | full kernel vs threshold-only vs empty | probes |
| Context scheduling | auto / spin / yield / blocking-sync | `-S` (affects the host-path section) |

The shapes are chosen to isolate one effect each: coalescing (32×8 vs 8×8 vs 1×256), the block-count
limit (32×1 → 50% occupancy), barrier granularity (32×32), and wasted threads (256×1, 16.7% idle).

## What it does not test

These gaps are deliberate. Know them before trusting a number for the real pipeline:

- **Camera memory.** Frames here come from `cuMemAlloc`. In `phase2`, the luma is an NVMM surface
  mapped through EGL, which may have different caching and mapping attributes. The kernel may well run
  slower on it.
- **Contention.** Nothing else is running. In the pipeline, the ISP, VIC, NVENC and CPU threads share
  the same DRAM with the kernel, and the encoder competes for memory bandwidth.
- **Realistic images.** One solid rectangle is the best case for the shared-memory reduction (ink
  clustered in few blocks). Scattered ink (noise, texture, a dim room) spreads atomics across every
  block.
- **Interop costs.** EGLImage creation and CUDA registration (`gpu_map_fd`) aren't measured here.
  `phase2` reports those as "map".

The gap between kbench and `phase2` is itself a measurement. On 2026-09-30, kbench's CPU wall time
for the host path was **~308 µs** (245 µs kernel + 63 µs overhead), while `phase2`'s "kernel"
stage for the same work was **~780 µs**. The ~470 µs difference comes from the points above,
most likely the NVMM mapping and contention. The next step is timing the kernel with events *inside*
`phase2`.

## Reference run (2026-09-30, 640×360, sched auto)

```
32x8      256     20x45    900   0.0%      8  100%   244.5 ( 239.2)   190.4 ( 185.3)   1.88  ok/ok
32x4      128     20x90   1800   0.0%     16  100%   235.4 ( 230.9)   183.9 ( 180.1)   1.96  ok/ok
32x1       32    20x360   7200   0.0%     32   50%   363.7 ( 358.5)   308.8 ( 304.6)   1.27  ok/ok
1x256     256     640x2   1280  29.7%      8  100%   917.7 ( 903.9)   919.6 ( 906.5)   0.50  ok/ok

empty_k (launch + scheduling)       900 blocks    49.2 us
thresh1_k (1 px/thread, no stats)   900 blocks   119.4 us   3.86 GB/s
thresh4_k (4 px/thread, uchar4)     225 blocks    57.8 us   7.97 GB/s

CPU wall 308.4 us = GPU kernel 245.1 us + 63.3 us of copies, launch and waiting
```

Numbers move by roughly ±20% between runs and after reboots, for example the empty kernel measured
28 µs in an earlier ad-hoc test and 49 µs here. Compare configurations **within one run**, not
across runs.
