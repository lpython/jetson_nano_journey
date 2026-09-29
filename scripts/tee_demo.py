#!/usr/bin/python3
"""PLAN.md Phase 1: tee the camera into a smooth preview + a slow "inference" branch.

Both branches stream to the Mac so you can watch them side by side:
  port 5000 = preview (1280x720@60)
  port 5001 = inference branch (320x240, capture clock burned in)

Modes (what the inference queue does):
  good   - depth 1, leaky=downstream. Slow branch skips frames but stays fresh.
  bad    - unbounded, holding system-memory copies. Preview stays smooth,
           inference falls further behind every second.
  starve - unbounded, holding NVMM buffers straight from the camera.
           The camera's buffer pool runs dry and the preview stalls too.

Prints the inference queue's depth once a second. Ctrl-C sends EOS (clean
camera release); a second Ctrl-C quits immediately.

Targets the Nano's system Python 3.8 (/usr/bin/python3) with gi; no conda.
"""
import argparse
import signal
import sys

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

CAMERA = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM),width=1280,height=720,framerate=60/1,format=NV12 ! "
    "tee name=t"
)

# Small bounded queue that blocks rather than drops: dropping would break H.264.
PREVIEW = (
    "t. ! queue max-size-buffers=4 max-size-bytes=0 max-size-time=0 ! "
    "nvv4l2h264enc maxperf-enable=1 insert-sps-pps=true idrinterval=30 bitrate=4000000 ! "
    "h264parse ! rtph264pay config-interval=1 pt=96 ! "
    "udpsink host={host} port=5000 sync=false"
)

QUEUES = {
    "good": "queue name=iq leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0",
    "bad": "queue name=iq max-size-buffers=0 max-size-bytes=0 max-size-time=0",
    "starve": "queue name=iq max-size-buffers=0 max-size-bytes=0 max-size-time=0",
}

# Downscale to system memory and stamp the wall clock. Placed before the queue
# in "bad" mode so the overlay shows capture time, not processing time.
TO_SYSMEM = (
    "nvvidconv ! video/x-raw,format=I420,width=320,height=240 ! "
    'clockoverlay time-format="%H:%M:%S" font-desc="Sans 24"'
)

# Stand-in for inference: sleep per frame, then re-encode so the Mac can see it.
SLOW_TAIL = (
    "identity sleep-time={slow_us} ! "
    "nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
    "nvv4l2h264enc maxperf-enable=1 insert-sps-pps=true idrinterval=30 bitrate=1000000 ! "
    "h264parse ! rtph264pay config-interval=1 pt=96 ! "
    "udpsink host={host} port=5001 sync=false"
)


def build_description(mode, host, slow_us):
    queue = QUEUES[mode]
    if mode == "bad":
        infer = "t. ! {} ! {} ! {}".format(TO_SYSMEM, queue, SLOW_TAIL)
    else:
        infer = "t. ! {} ! {} ! {}".format(queue, TO_SYSMEM, SLOW_TAIL)
    return " ".join([CAMERA, PREVIEW, infer]).format(host=host, slow_us=slow_us)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", default="bad", choices=sorted(QUEUES))
    ap.add_argument("host", nargs="?", default="192.168.1.233", help="Mac IP")
    ap.add_argument("--slow-ms", type=float, default=50.0, help="fake inference cost per frame")
    ap.add_argument("--print", action="store_true", help="print the pipeline and exit")
    args = ap.parse_args()

    desc = build_description(args.mode, args.host, int(args.slow_ms * 1000))
    if args.print:
        print(desc.replace(" ! ", " !\n  ").replace(" t. ", "\n t. "))
        return 0

    Gst.init(None)
    pipeline = Gst.parse_launch(desc)
    iq = pipeline.get_by_name("iq")
    loop = GLib.MainLoop()
    stopping = [False]

    def on_message(_bus, msg):
        if msg.type == Gst.MessageType.EOS:
            loop.quit()
        elif msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            print("ERROR from {}: {}\n{}".format(msg.src.get_name(), err.message, dbg), file=sys.stderr)
            loop.quit()

    def on_sigint():
        if stopping[0]:
            loop.quit()
        else:
            stopping[0] = True
            print("\nEOS sent; draining (Ctrl-C again to force quit)")
            pipeline.send_event(Gst.Event.new_eos())
        return GLib.SOURCE_CONTINUE

    def report():
        n = iq.get_property("current-level-buffers")
        ms = iq.get_property("current-level-time") / Gst.MSECOND
        print("inference queue: {:4d} frames  {:7.0f} ms behind".format(n, ms), flush=True)
        return GLib.SOURCE_CONTINUE

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message)
    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGINT, on_sigint)
    GLib.timeout_add_seconds(1, report)

    print("mode={} host={} slow={}ms  (Ctrl-C to stop)".format(args.mode, args.host, args.slow_ms))
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
    return 0


if __name__ == "__main__":
    sys.exit(main())
