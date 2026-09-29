This is a great second project, and the Nano is well suited to it, because the lesson you missed last time was mostly about where data lives and how it moves. A few platform facts change the design a lot, so I'll start there, then give you the architecture, the backpressure model, the "without GStreamer" route, and a phased plan that ends in Rust.

## Hardware and software facts that shape everything

**The Nano is stuck on JetPack 4.6.x** (L4T R32.7, Ubuntu 18.04, kernel 4.9, CUDA 10.2, TensorRT 8.2, GStreamer 1.14, GCC 7). Nothing newer is officially supported. The GPU is a 128-core Maxwell with compute capability 5.3. It handles FP16 well but has no tensor cores, so INT8 buys you little.

**Memory is unified.** The CPU and GPU share the same 4 GB of LPDDR4. When you worry that "moving from Argus to CUDA costs performance," you're picturing a discrete GPU with a PCIe copy. On the Nano, the camera output lands in an NVMM buffer (a dmabuf), and you can map that same memory into CUDA through EGLImage interop with zero copies. Done correctly, handing the frame to CUDA costs a few API calls, not a memcpy. Your old pipeline's real problem was the round trip through the SD card. The new rule is that a frame never leaves DRAM and ideally never gets copied.

**You have several dedicated hardware blocks, and the skill is keeping each one busy.** The ISP (reached through Argus) turns raw sensor Bayer data into clean NV12. The VIC does scaling, cropping, and color conversion without touching the GPU; `nvvidconv` uses it. NVENC does H.264/H.265 encoding. The GPU is free for your own kernels. A well-built pipeline treats these like stations on an assembly line.

**NV12's Y plane is already grayscale.** MNIST is grayscale, so you can ignore the chroma plane and read luminance directly. Your preprocessing gets that step for free.

**Some APIs are C and some are C++, and this matters for both "C only" and the Rust endgame.** Here is how they split:

- Plain C: GStreamer (GObject-based), the CUDA driver API (`libcuda`), `nvbuf_utils` in JetPack 4, EGL, V4L2 (ioctls), and sockets.
- C++ only: libargus and TensorRT.

I'd build the project to depend only on the C-surface APIs. That keeps your C phase honest, and it makes the Rust port a matter of `bindgen` against C headers instead of wrapping C++ classes.

**The kernel is 4.9, which limits what networking features exist.** io_uring (5.1+) and UDP GSO (4.18+) aren't available. Your tools are epoll, `sendmmsg`/`recvmmsg`, socket buffer tuning, and careful packetization. That is the same epoll model libuv sits on, so the reactor internals you've been studying carry straight over.

**Recent VS Code Remote-SSH servers require glibc 2.28 or newer, and Ubuntu 18.04 ships 2.27.** Plan your remote workflow around that; details are in Phase 0.

## Target architecture

```
IMX219 ──► ISP (Argus) ──► NVMM NV12 1280x720@60
                               │
                              tee
             ┌─────────────────┴──────────────────┐
     queue (small, non-leaky)          queue leaky=downstream, depth 1
             │                                     │
        NVENC H.264                    VIC crop/scale (nvvidconv)
             │                                     │
        RTP ──► UDP ──► PC viewer        EGLImage ──► CUDA (zero-copy)
                                                   │
                                         preprocess: threshold, bbox,
                                         center-of-mass, resize to 28x28
                                                   │
                                         LeNet inference (your kernels)
                                                   │
                                         results ──► UDP ──► PC
                                         (seq no., capture timestamp, digit, confidence)
```

This design has two outputs with different requirements. The video preview has to be smooth, so its branch is allowed to push back a little. The inference branch has to be fresh, so it drops aggressively and always works on the newest frame. Keeping those two policies separate is most of the design.

## Backpressure and frame dropping

This is the core of what you want to learn, so here is the model I'd hold in your head.

**A live source can't be slowed down.** A camera produces a frame every 16.7 ms whether or not anyone is ready for it. Backpressure only works when the producer can wait, like a file reader or a network download. When backpressure reaches a live source, one of two bad things happens. Either the source drops frames in a place you didn't choose, or queues fill up and latency grows without bound while throughput looks fine. In a live system, latency is the metric that matters, and unbounded queues hide latency.

**Drop early, drop the oldest, and keep queues shallow.** For the inference path, use a queue with a depth of 1 that discards the old frame when a new one arrives (`leaky=downstream`). Then when the model is slow, it skips frames but always works on the current one. That is the behavior you want for "what digit is showing right now."

**Backpressure is still correct in some places.** Between the encoder and the payloader, a small bounded queue smooths out jitter, and dropping there would corrupt the H.264 stream (P-frames depend on earlier frames). The principle: use backpressure where losing data breaks correctness, and drop where losing data only costs freshness. When you do drop encoded video, drop whole GOPs or request a new keyframe; don't drop arbitrary packets.

**Buffer pools are a hidden form of backpressure.** `nvarguscamerasrc` has a fixed pool of NVMM buffers. If your CUDA code holds a buffer too long, or an appsink hoards buffers, the camera runs out and stalls, and the stall shows up as a camera problem. You'll likely hit this on purpose at least once. It's one of the most useful lessons on the platform.

**Measure every stage.** Stamp every frame with its sensor capture timestamp (Argus provides one, and it flows through as buffer metadata and PTS). Record a timestamp at each stage boundary, and carry the capture timestamp and a sequence number in your result packets. Then the PC can compute end-to-end latency and see drops as gaps in the sequence numbers. Use Nsight Systems (it supports Jetson targets and can connect from your PC) for timelines, and `tegrastats` to see which hardware blocks are saturated. For glass-to-glass latency, point the camera at a stopwatch on a screen next to the receiver window and photograph both.

## Phased plan

**Phase 0 — Baseline and workflow.** Flash JetPack 4.6.x. Set `sudo nvpmodel -m 0` (10 W mode) and run `sudo jetson_clocks` while benchmarking, and make sure it has a fan or you'll measure thermal throttling instead of your code. Set up SSH keys.

For development, I'd edit on the Ubuntu laptop or the Mac and build natively on the Nano. Cross-compiling CUDA for aarch64 from macOS isn't worth the pain; from the Ubuntu laptop it's doable later. For the editor, you have three options: an older pinned VS Code release, VS Code on your machine with an `rsync`-on-save plus `ssh make` loop, or Neovim/Helix over SSH. Use `gdbserver` or `cuda-gdb` for debugging and Nsight Systems for profiling.

**Phase 1 — GStreamer from the command line.** Get video to your PC and learn caps, NVMM memory, and queues before writing any code.

On the Nano:
```
gst-launch-1.0 nvarguscamerasrc ! \
  'video/x-raw(memory:NVMM),width=1280,height=720,framerate=60/1,format=NV12' ! \
  nvv4l2h264enc maxperf-enable=1 insert-sps-pps=true idrinterval=30 bitrate=4000000 ! \
  h264parse ! rtph264pay config-interval=1 pt=96 ! \
  udpsink host=<PC_IP> port=5000 sync=false
```
On the PC (works on the Ubuntu laptop, or on the Mac with Homebrew GStreamer, where you can swap in `vtdec` for hardware decode):
```
gst-launch-1.0 udpsrc port=5000 \
  caps='application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000' ! \
  rtpjitterbuffer latency=0 ! rtph264depay ! avdec_h264 ! videoconvert ! \
  autovideosink sync=false
```
Then add the `tee` and the leaky inference branch. Try deliberately breaking it: make a queue unbounded and slow the sink, and watch latency climb.

Past the single command above, drive pipelines from Python (`gi` + `Gst.parse_launch`, system Python 3.8 on the Nano) rather than bash. `scripts/tee_demo.py good|bad|starve` runs the tee experiment and prints the inference queue depth each second: `good` holds at 1 frame, `bad` grows about 40 frames/s (4.4 s behind after 7 s), and `starve` pins at 8 frames when the camera's NVMM pool runs dry.

**Phase 2 — A C program that owns the pipeline.** Build the pipeline in C and end the inference branch in an `appsink` configured with `max-buffers=1 drop=true sync=false`. For each buffer:

1. Get the dmabuf fd with `ExtractFdFromNvBuffer`.
2. Wrap it with `NvEGLImageFromFd`.
3. Register it with `cuGraphicsEGLRegisterImage`.
4. Get device pointers from `cuGraphicsResourceGetMappedEglFrame`.

Your first kernel can be something simple like thresholding the Y plane. The `nvivafilter` element and its sample in the Jetson sources show the same interop, and they're worth reading as a reference. Release the buffer back quickly; this is where the pool-starvation lesson happens.

**Phase 3 — Inference.** My strong recommendation is to hand-write LeNet-5 inference in CUDA C: convolution, pooling, ReLU, fully connected layers, and softmax, with weights loaded from a flat binary file. At 28×28 input the network is tiny, you'll learn more about memory coalescing, shared memory, and occupancy than any tutorial will teach you, and it avoids TensorRT's C++-only API, which keeps the Rust port clean.

For a performance baseline, run the same model through `trtexec` using an ONNX MNIST model from the ONNX model zoo, and try to beat or match it. To get weights without Python, you can use a pretrained ONNX model and extract the tensors, or train in Rust with `burn` or `candle` on the Mac.

Preprocessing matters more than the network here. MNIST digits are white on black, fit inside a 20×20 box, and are centered by center of mass in a 28×28 frame. Camera digits need that same normalization, or accuracy will fall apart. Use handwritten digits on paper to start; seven-segment or printed fonts are a different distribution from MNIST.

**Phase 4 — Networking done properly.** Replace the ad hoc result path with your own small binary protocol over UDP: a fixed header with sequence number, capture timestamp, and inference timestamp, followed by the payload. Build an epoll loop on both ends and use `sendmmsg` for batching. Tune `SO_SNDBUF`/`SO_RCVBUF`, and measure drops and jitter on the receiver. Optionally, drop `rtph264pay` and write your own RTP packetizer for the H.264 NAL units. It's a very instructive exercise in MTU-sized framing.

**Phase 5 — Remove GStreamer, still in C.** Each piece has a lower-level equivalent. For encoding, drive NVENC directly through the V4L2 memory-to-memory interface (DMABUF buffers in, bitstream out); this is what `nvv4l2h264enc` does internally, and it's all ioctls in C. For capture there are two options. You can wrap Argus in a thin `extern "C"` C++ shim of about 200 lines, keeping the ISP. Or you can capture raw Bayer through V4L2 and write your own debayer, white balance, and gamma in CUDA. The second is harder and the image will look worse than the ISP's, but it removes all C++ from the project and is an excellent CUDA exercise. The Jetson Multimedia API samples (`/usr/src/jetson_multimedia_api`) are the reference for both options.

**Phase 6 — Rust.** There are two stages. First, use `gstreamer-rs` to rebuild the Phase 2 app, then write your preprocessing as a custom GStreamer element in Rust using the patterns in `gst-plugins-rs`. That covers your "Rust loaded into GStreamer" option, and it's well supported. Second, go fully native by porting Phase 5. Use `bindgen` against `libcuda`, `nvbuf_utils`, EGL, and the V4L2 headers, and use `mio` or raw epoll for networking.

**The one place "no C at all" gets hard is the GPU kernels themselves.** You have three options. You can compile your kernels to PTX once and load them from Rust through the driver API (`cuModuleLoadData`), which is the pragmatic answer. You can write kernels in Rust using the `nvptx64-nvidia-cuda` target or the Rust-CUDA project; both are real but rough, especially against CUDA 10.2. Or, if you pick the raw-Bayer V4L2 path, the whole host side is Rust talking to kernel interfaces. Also check which CUDA versions any crate like `cudarc` supports before relying on it, since 10.2 is old.

## Why this ordering works

Each phase replaces exactly one layer of abstraction with the thing underneath it, and you keep a working end-to-end system the whole time. Your measurement setup (timestamps, sequence numbers, Nsight) carries across every phase, so you'll know whether each rewrite made things faster or just different. By the end you'll have touched every hardware block on the board, which covers what the Nano has to teach.

Which camera are you using, a CSI module like the IMX219 or a USB webcam? That decides whether Argus is even in the picture for capture.