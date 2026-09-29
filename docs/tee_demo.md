# tee_demo.py walkthrough

`scripts/tee_demo.py` splits one camera feed into two branches: a smooth video preview and a
deliberately slow "inference" branch. It then shows what happens to the slow branch under three
queue policies. This page explains the pipeline piece by piece, focusing on `SLOW_TAIL`, the part
that stands in for inference.

To see the exact pipeline for any mode, without a camera or GStreamer installed:

```
python3 scripts/tee_demo.py --print bad
```

## Five GStreamer ideas you need first

**1. A pipeline is a chain of elements joined by `!`.** Each element does one job (capture,
convert, encode, send). Data flows left to right in chunks called **buffers**. For video, one
buffer is one frame.

**2. Caps say what flows between two elements.** A bare string like
`video/x-raw(memory:NVMM),format=NV12` is a *caps filter*. It isn't an element that does work; it
forces the two elements on either side to agree on that exact format. `(memory:NVMM)` means the
frame lives in NVIDIA's hardware buffer memory (DMA buffers the ISP, VIC, NVENC and GPU can all
reach). Without it, the frame is in ordinary CPU memory ("system memory").

**3. `tee` splits one stream into several branches.** `tee name=t` gives the tee a name. Each later
`t. !` starts a new branch from it. The tee doesn't copy frames. It hands every branch a
*reference* to the same buffer, and the buffer returns to its pool only when every branch has let
go of it. That detail is what makes `starve` mode fail.

**4. A `queue` is a thread boundary.** GStreamer moves a buffer down the chain by function calls on
a *streaming thread*. Without queues, the whole pipeline runs on the camera's thread, so if any
element is slow, the camera waits. A `queue` stores buffers and starts a new thread for everything
after it. That makes a queue the only place where frames can pile up or be dropped, so choosing its
size and leak policy is choosing the backpressure policy.

**5. Sinks can wait for a clock.** Every buffer carries a timestamp (PTS). By default a sink holds
each buffer until the pipeline clock reaches that time, which is right for playing a file at normal
speed. For a live, low-latency stream you want "send it the moment it arrives", which is `sync=false`.

## The whole pipeline (`bad` mode)

```
nvarguscamerasrc                                     camera + ISP, 1280x720 NV12 @ 60 fps, NVMM
  └─ tee name=t
       ├─ PREVIEW   queue(4) → nvv4l2h264enc → h264parse → rtph264pay → udpsink :5000
       │
       └─ inference branch
            TO_SYSMEM   nvvidconv → 320x240 I420 (CPU memory) → clockoverlay
            queue iq    ← the queue under test
            SLOW_TAIL   identity(sleep) → nvvidconv → NVMM NV12 → nvv4l2h264enc
                        → h264parse → rtph264pay → udpsink :5001
```

The threads: the camera thread runs up to the tee and into both queues. The preview queue's thread
runs the preview encoder. The `iq` queue's thread runs all of `SLOW_TAIL`. So everything in
`SLOW_TAIL` shares one thread, which is why one sleeping element in it slows the whole tail.

## SLOW_TAIL, element by element

```
identity sleep-time={slow_us} !
nvvidconv ! video/x-raw(memory:NVMM),format=NV12 !
nvv4l2h264enc maxperf-enable=1 insert-sps-pps=true idrinterval=30 bitrate=1000000 !
h264parse ! rtph264pay config-interval=1 pt=96 !
udpsink host={host} port=5001 sync=false
```

`SLOW_TAIL` has two jobs:

1. **Be slow on purpose.** The `identity` element is the inference stand-in.
2. **Make the result visible on the Mac.** Everything after `identity` exists only so you can
   *watch* the slow branch in a second window. The real design won't encode video here.

### `identity sleep-time=50000`: the fake inference

`identity` passes every buffer through unchanged. Normally it's a debugging tap. With `sleep-time`
set, it calls `g_usleep()` for that many **microseconds** on every buffer before passing it on.
50000 µs = 50 ms.

The sleep blocks the `iq` queue's streaming thread, which is exactly what real inference does. The
thread can't take the next frame out of the queue until it's done with the current one.

### `nvvidconv ! video/x-raw(memory:NVMM),format=NV12`: back into hardware memory

`nvvidconv` is NVIDIA's converter. It runs on the **VIC**, a dedicated scaling and color-conversion
block, not on the CPU or GPU. The caps filter after it asks for two things at once:

- **I420 → NV12.** Both are YUV 4:2:0 formats that store color the same way. I420 keeps U and V in
  separate planes; NV12 interleaves them into one plane. The encoder wants NV12.
- **System memory → NVMM.** The hardware encoder reads only NVMM buffers, so the frame gets copied
  back into hardware memory.

The round trip (NVMM → CPU memory in `TO_SYSMEM`, then back to NVMM here) exists only because
`clockoverlay` draws text with the CPU and needs a CPU-readable frame. The real pipeline will stay in
NVMM the whole way and hand the buffer to CUDA with zero copies (PLAN.md Phase 2).

### `nvv4l2h264enc ...`: hardware H.264 encoder (NVENC)

| Argument | Meaning |
|---|---|
| `maxperf-enable=1` | Run NVENC at maximum clock. Lower latency per frame, a little more power. |
| `insert-sps-pps=true` | Put the SPS and PPS headers (the decoder's "how to read this stream" setup) in front of every IDR frame. |
| `idrinterval=30` | Emit an IDR frame every 30 frames. |
| `bitrate=1000000` | Target 1 Mbit/s. The preview branch uses 4 Mbit/s for 1280x720; 320x240 needs far less. |

An **IDR frame** is a complete, self-contained picture. Most other frames are **P-frames**, which
store only the change since the previous frame. A decoder can't start until it sees SPS/PPS plus an
IDR. `insert-sps-pps` and `idrinterval` together decide how quickly a viewer that joins late (or
loses packets) can start showing a picture.

### `h264parse`: tidy the bitstream

It splits the encoder output into clean NAL units (H.264's basic packets) and fills in stream
metadata so the next element knows the boundaries. It changes no pixels.

### `rtph264pay config-interval=1 pt=96`: wrap in RTP

RTP is the standard header for real-time media over UDP. It adds a sequence number, timestamp and
stream ID (SSRC) to each packet. The payloader also splits NAL units larger than the network MTU
(about 1400 bytes) across several packets.

| Argument | Meaning |
|---|---|
| `config-interval=1` | Also re-send SPS/PPS inside RTP every 1 second, so a receiver started late can decode. |
| `pt=96` | RTP payload type. 96–127 are "dynamic" numbers with no fixed meaning; the receiver's caps (`payload=96`) must match. |

### `udpsink host=... port=5001 sync=false`: send it

Each RTP packet becomes one UDP datagram to the Mac, on port 5001 so it doesn't mix with the preview
on 5000. The Mac's kernel sorts packets to the right receiver by port. `sync=false` means send
immediately, without waiting on timestamps (idea 5 above).

## How `slow_us` creates latency

The camera delivers a frame every 16.7 ms (60 fps). The tail can finish one frame every 50 ms
(20 fps). Every second, 60 frames arrive and 20 leave, so **40 frames a second have to go somewhere**.
Only the `iq` queue can absorb them, and the mode decides what it does with them:

| Mode | `iq` queue | Where the extra 40/s go | Effect |
|---|---|---|---|
| `good` | `max-size-buffers=1 leaky=downstream` | Dropped. When a new frame arrives, the old one waiting in the queue is thrown away. | The tail always gets the newest frame. About 20 fps, but only ever about 17 ms old. |
| `bad` | unbounded (all limits `0`) | Kept, waiting in line. | Every frame waits behind all the earlier ones. Measured: 4.4 s behind after 7 s, and growing. |
| `starve` | unbounded, placed *before* `TO_SYSMEM` | Kept, and each one still holds a camera NVMM buffer. | The camera has only 8 buffers. Once the queue holds all 8, the camera has nowhere to write and stalls. The **preview stalls too**. |

In `bad` mode the lateness is easy to calculate: after *t* seconds, 40*t* frames are waiting, and
each needs 50 ms, so the newest frame waits about 2*t* seconds. Frame rate on port 5001 still looks
like a steady 20 fps. Throughput looks healthy while latency grows without limit. That's why PLAN.md
says latency, not frame rate, is the number to watch.

Why `bad` and `starve` differ: in `bad`, `TO_SYSMEM` runs *before* the queue, so each queued frame is
a small CPU-memory copy and the camera's NVMM buffer is released right away. In `starve`, the queue
sits straight after the tee, so it holds the camera's own buffers (idea 3).

`good` also puts its queue straight after the tee, so it holds camera buffers too, but at most one.
Add the one frame `nvvidconv` is converting, and the up to 4 in the preview queue, and the camera
still has spare buffers out of its 8. The bound is only half of it, though. The other half is
`leaky=downstream`. A full queue normally **blocks** the thread pushing into it, and here that's the
camera's thread. A bounded but non-leaky depth-1 queue would fill on the first slow frame, block the
tee, and throttle the camera (and the preview) to the tail's 20 fps. That's the same symptom as
`starve`, reached without running out of buffers. Leaky means a full queue drops a frame instead of
blocking, so the camera thread never waits:

| Queue after the tee | Camera buffers held | Camera thread blocks? | Result |
|---|---|---|---|
| unbounded (`starve`) | grows until all 8 are used | no, until the pool runs dry | camera stalls |
| depth 1, not leaky | at most 1 | **yes**, whenever full | camera throttled to 20 fps |
| depth 1, leaky (`good`) | at most 1 | never | camera runs at 60, tail skips frames |

Useful settings to try:

- `--slow-ms 10`: faster than the 16.7 ms frame time. Nothing piles up in any mode.
- `--slow-ms 16`: just under the frame time. Stable, but with almost no slack.
- `--slow-ms 100`: `bad` falls behind twice as fast.

## What `slow_us` becomes when real inference arrives

In PLAN.md Phase 2 the branch changes shape:

```
tee → [queue leaky, depth 1] → nvvidconv (VIC crop/scale, stays NVMM) → appsink max-buffers=1 drop=true
                                                                            │
                                                          your C code: EGLImage → CUDA → LeNet
                                                                            │
                                                          small UDP result packet (not video)
```

- **`identity sleep-time` goes away.** The time your code spends inside the appsink callback (map the
  buffer, run CUDA kernels, release it) *is* the inference cost. Before, you picked `slow_us`; now the
  GPU decides it, and you measure it.
- **The queue lesson carries over unchanged.** `appsink max-buffers=1 drop=true` is the same policy
  as `good` mode's leaky depth-1 queue, built into the sink.
- **The encoder half of `SLOW_TAIL` goes away.** Results leave as a few bytes per frame (sequence
  number, capture timestamp, digit, confidence) instead of a video stream.
- **`starve` becomes a real risk.** If your code holds the NVMM buffer while CUDA runs, you're
  holding a camera buffer, the same as `starve` mode. Release it as soon as the frame is copied or
  consumed.

`slow_us` stays useful as a tool:

- **A budget.** At 60 fps you get 16.7 ms per frame. Finding where things break with `--slow-ms`
  tells you how fast inference must be before you write any CUDA.
- **A test harness without a model.** You can build and debug the timestamps, sequence numbers and UDP
  protocol (Phase 4) against a fake stage that takes a known, fixed time.
- **Added stress.** Adding a delay on top of real inference simulates a slower model or a busier
  board, to check that drops land where you expect and latency stays bounded.

What a sleep does *not* simulate: a sleeping thread uses no CPU, no GPU and no memory bandwidth. Real
inference competes with the ISP, VIC and NVENC for the Nano's shared 4 GB of memory, and its time
varies from frame to frame. Expect real numbers to be somewhat worse than a sleep of the same average
length.
