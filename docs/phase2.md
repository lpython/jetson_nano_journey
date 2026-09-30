# Phase 2 walkthrough: a C program that owns the pipeline

`phase2/` is PLAN.md Phase 2. A C program builds the GStreamer pipeline itself, pulls each frame from
an `appsink`, hands it to CUDA **without copying it**, runs a kernel on it, and gives it back. It
prints what that costs once a second.

## Build and run

On the Nano (or from the Mac with `scripts/remote.sh`):

```
cd phase2 && make                     # clang-12 for the C code and for the kernel
build/phase2                          # inference branch only
build/phase2 -h 192.168.1.233         # also stream the preview to the Mac on :5000
build/phase2 -n 300 -d captures/frame # stop after 300 frames; save frame 120 as PGM images
build/phase2 -w 30                    # hold every buffer 30 ms longer (simulated slow inference)
```

Other options: `-s 640x360` sets the inference size and `-t 100` sets the ink threshold. `build/phase2 -?` lists them all.

Output, one line per second:

```
camera  60 fps | processed  60 fps, dropped   0 | age   0.7 ms (max   0.9) | map 0.36 (0.50) | kernel 0.79 (0.86) | held  1.34 (1.51) ms | ink 151616 px [0,0]-[639,359]
```

| Field | Meaning |
|---|---|
| camera | Frames entering the tee (counted by a pad probe on the camera thread) |
| processed / dropped | Frames the C code handled, and the difference. Drops happen in the leaky queue and in the appsink. |
| age | How old the frame is when `appsink` hands it over: pipeline clock now minus the buffer's PTS. See the caveat below. |
| map | dmabuf fd → EGLImage → CUDA pointer |
| kernel | Threshold kernel plus reading back its 20-byte result |
| held | From pulling the buffer to releasing it. This is how long an NVMM buffer is out of its pool. |
| ink | Pixels darker than the threshold, and their bounding box |

## The pipeline

```
nvarguscamerasrc → 1280x720 NV12 NVMM @60 → tee
   ├─ queue(4) → nvv4l2h264enc → h264parse → rtph264pay → udpsink :5000      (only with -h)
   └─ queue(1, leaky) → nvvidconv bl-output=false → 640x360 NV12 NVMM
        → appsink max-buffers=1 drop=true sync=false
```

This is `tee_demo.py` `good` mode with the end replaced. The `identity sleep-time` fake is gone.
The time your C code spends between pulling a sample and releasing it *is* the inference cost now.
`appsink max-buffers=1 drop=true` works like a second leaky depth-1 queue: if the program is busy,
the appsink keeps only the newest frame.

`nvvidconv` here runs on the VIC and does two jobs. It scales 1280x720 → 640x360, and it hands the
C code a buffer from **its own** pool instead of the camera's. So even if the C code holds a buffer
too long, the camera's 8 buffers are safe. At worst `nvvidconv` blocks and the leaky queue in front
of it drops frames.

## Zero-copy: from a GStreamer buffer to a CUDA pointer

`process_sample()` in `src/main.c` and `gpu_map_fd()` in `src/gpu.c`:

1. **`gst_buffer_map`** on an NVMM buffer doesn't give you pixels. `map.data` points to an NvBuffer
   *handle*.
2. **`ExtractFdFromNvBuffer`** (nvbuf_utils) gets the handle's **dmabuf file descriptor**. dmabuf is
   the Linux kernel's way of sharing one block of memory between devices.
3. **`NvEGLImageFromFd`** wraps that memory as an EGLImage, the object CUDA's interop API accepts.
   It needs an initialized EGL display, which works over SSH with no X session.
4. **`cuGraphicsEGLRegisterImage`** + **`cuGraphicsResourceGetMappedEglFrame`** return a `CUeglFrame`.
   For pitch-linear NV12, `frame.pPitch[0]` is a device pointer to the Y plane and `frame.pitch`
   is its row stride in bytes.
5. Launch the kernel on that pointer. On the Nano, CPU and GPU share the same DRAM, so nothing was
   copied. The ISP wrote those bytes and the GPU reads them in place.
6. Undo steps 4 → 3 → 1 (`gpu_unmap`, `gst_buffer_unmap`) and drop the sample. The buffer returns to
   `nvvidconv`'s pool.

Cost measured on the Nano: about 0.36 ms for steps 2–4, per frame.

### Why `bl-output=false`

`nvvidconv` outputs NV12 NVMM as **block-linear** by default: pixels are stored in tiles, which suits
the hardware blocks. CUDA sees block-linear frames as `CUarray`s (`frameType` 0,
`CU_EGL_FRAME_TYPE_ARRAY`), readable only through texture or surface objects. **Pitch-linear** is
ordinary row-major memory (row *y* starts at `base + y * pitch`), so a kernel can index it directly.
The first run failed with `expected a pitch-linear frame, got frameType=0` until this was set.

## The kernel: `kernels/threshold.cu`

One thread per Y-plane pixel. The Y plane of NV12 is already grayscale, so there's no color
conversion.

- `luma < thr` → ink (dark ink on white paper). Writes a mask in MNIST polarity: white digit on black.
- Blocks are 32×8 threads with x varying fastest. Each warp reads 32 consecutive bytes of one row,
  which is a **coalesced** memory access.
- Ink count and bounding box: each block first combines results with atomics on **shared memory**,
  then does one atomic per value on global memory. Doing every pixel's atomic in global memory would
  serialize thousands of threads on five addresses.

**Built by clang, loaded by the driver.** `clang-12 -x cuda --cuda-device-only -S` compiles the kernel
to **PTX**, NVIDIA's portable GPU assembly. At startup, `cuModuleLoad` JIT-compiles it for sm_53 (and
caches the result in `~/.nv/ComputeCache`), and `cuLaunchKernel` runs it. The host program links only
`libcuda` (the driver API, a C API), with no nvcc, no CUDA runtime and no C++. That's the path PLAN.md
Phase 6 uses from Rust.

clang-12 prints `Unknown CUDA version 10.2, assuming 10.1`. It's harmless for this code, and the
Makefile silences it.

## Experiments

**Slow inference (`-w 30`).** Each buffer is held about 31.5 ms, so the program handles about 32 fps
out of 60. Measured on the Nano: `processed 32 fps, dropped 29`, **age about 9 ms, max 17 ms**. Drops
went up, latency didn't, and the camera and preview stayed at 60. That's the `good` mode lesson,
now in C.

**Look at what the GPU saw (`-d`).** It writes `_y.pgm` (the luma plane, copied from the GPU after
the kernel) and `_mask.pgm`. If the image is skewed or torn, the pitch is wrong. Pull the files with
`scripts/pull.sh phase2/captures/`; Preview on macOS doesn't open PGM, but most image tools do.

**Tune the threshold.** With a dim room most of the scene counts as ink (about 150k of 230k pixels at
`-t 100`). Point the camera at white paper with a thick marker digit and lower `-t` until only
the digit is left. That's the input Phase 3's preprocessing starts from.

**Try to starve the camera.** Remove `nvvidconv` so the appsink receives camera buffers directly,
then hold them (`-w`). With `max-buffers=1` it takes more than that to starve the pool. Think about
what else holds camera buffers, and try it.

## Caveats

- **`age` isn't glass-to-glass.** `nvarguscamerasrc` stamps the PTS when it outputs the buffer, after
  sensor exposure and the ISP. So about 0.7 ms is "time since the camera element released the frame".
  True capture time needs the Argus sensor timestamp (the Phase 5 shim) or the stopwatch photo test
  from PLAN.md.
- **Registration every frame.** Steps 3–4 run for every frame. The pool reuses a handful of buffers,
  so caching the CUDA mapping per dmabuf fd would cut the 0.36 ms. It's worth measuring, but it adds
  lifetime bugs to watch for.
- **All CUDA calls stay on the main thread**, the one that created the context. Pulling samples in
  the main loop (instead of the `new-sample` callback on the streaming thread) keeps it that way.
