#!/usr/bin/python3
"""Camera stack diagnostic for the Jetson Nano, checked bottom-up.

  platform   L4T release, power mode, temperatures
  users      processes currently holding the camera
  daemon     nvargus-daemon (libargus clients all go through it) + its recent errors
  v4l2 raw   raw Bayer straight from sensor -> CSI -> VI, bypassing ISP and Argus
  argus      nvarguscamerasrc through GStreamer (daemon + ISP)
  encoder    NVENC on a test pattern (no camera involved)

Reading the result: raw OK but argus FAIL => the daemon is wedged, use --reset.
Raw FAIL => sensor, ribbon cable or kernel driver: power off, reseat the cable, reboot.

--reset stops processes holding the camera (after asking) and restarts nvargus-daemon.
That's the deepest reset available without a reboot; the VI/CSI driver is built into the kernel.

Targets the Nano's system Python 3.8 (/usr/bin/python3) with gi; no conda.
"""
import argparse
import ctypes
import glob
import os
import re
import signal
import subprocess
import sys
import tempfile
import time

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

DEVICE = "/dev/video0"
RAW_W, RAW_H = 1280, 720  # IMX219 60 fps mode
CAMERA_PATTERN = re.compile(r"nvarguscamerasrc|argus_|v4l2-ctl|tee_demo|gst_upload")
TEMP_WARN_C = 70.0
# Every *successful* Argus session logs this block twice: most likely the probe of the Nano's
# second (empty) CSI port ("Failed ImagerGUID 1"). Not a problem; real failures show in the argus test.
BENIGN_DAEMON = re.compile(r"ModuleNotPresent|NvPcl|ImagerGUID|Sensor could not be opened|"
                           r"Acquiring SCF Camera device source|CameraDriver\.cpp|CaptureService\.cpp")
LIBC = ctypes.CDLL(None)

results = []


def report(layer, status, msg):
    results.append((layer, status))
    print("[{:4}] {:9} {}".format(status, layer, msg), flush=True)


def run(cmd, timeout=10):
    """Run a command, return (returncode, combined output). Never raises."""
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=timeout)
        return p.returncode, p.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return -1, str(e)


# ---------------------------------------------------------------- platform

def check_platform():
    try:
        with open("/etc/nv_tegra_release") as f:
            report("platform", "OK", f.readline().strip().lstrip("# ")[:60])
    except OSError:
        report("platform", "WARN", "no /etc/nv_tegra_release")

    rc, out = run(["sudo", "-n", "nvpmodel", "-q"])
    mode = next((l for l in out.splitlines() if "Power Mode" in l), out[:60])
    report("platform", "OK" if "MAXN" in mode else "WARN", mode)

    hot = []
    temps = []
    for zone in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        try:
            name = open(zone + "/type").read().strip()
            c = int(open(zone + "/temp").read()) / 1000.0
        except (OSError, ValueError):
            continue
        temps.append("{} {:.0f}C".format(name, c))
        if c >= TEMP_WARN_C and name != "PMIC-Die":
            hot.append(name)
    report("platform", "WARN" if hot else "OK", ", ".join(temps))


# ---------------------------------------------------------------- users

def camera_users():
    """PIDs whose command line looks like a camera client (excluding us)."""
    me = os.getpid()
    users = {}
    for d in glob.glob("/proc/[0-9]*"):
        pid = int(d[6:])
        if pid == me:
            continue
        try:
            cmd = open(d + "/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if CAMERA_PATTERN.search(cmd) and "camdiag" not in cmd:
            users[pid] = cmd.strip()[:100]
    # The daemon itself opens the device during any Argus session; that's expected, not a user.
    rc, daemon_pid = run(["systemctl", "show", "nvargus-daemon", "-p", "MainPID", "--value"])
    rc, out = run(["sudo", "-n", "fuser", DEVICE])
    for tok in out.split():
        if tok.isdigit() and int(tok) not in (me, int(daemon_pid or 0)):
            users.setdefault(int(tok), "(holds {} open)".format(DEVICE))
    return users


def check_users():
    users = camera_users()
    if not users:
        report("users", "OK", "nothing holding the camera")
    for pid, cmd in users.items():
        report("users", "WARN", "pid {} {}".format(pid, cmd))
    return users


# ---------------------------------------------------------------- daemon

def check_daemon():
    rc, state = run(["systemctl", "is-active", "nvargus-daemon"])
    if state != "active":
        report("daemon", "FAIL", "nvargus-daemon is {}".format(state or "unknown"))
        return
    rc, props = run(["systemctl", "show", "nvargus-daemon", "-p", "ActiveEnterTimestamp", "-p", "NRestarts"])
    report("daemon", "OK", "active; " + props.replace("\n", "; "))

    # Only this run of the daemon: errors from before the last restart don't matter.
    rc, inv = run(["systemctl", "show", "nvargus-daemon", "-p", "InvocationID", "--value"])
    rc, log = run(["journalctl", "--no-pager", "-o", "cat", "_SYSTEMD_INVOCATION_ID=" + inv])
    errors = [l for l in log.splitlines() if re.search(r"error|fail|timeout", l, re.I)]
    real = [l for l in errors if not BENIGN_DAEMON.search(l)]
    if real:
        report("daemon", "WARN", "{} error lines since daemon start, last: {}".format(len(real), real[-1][:90]))
    else:
        report("daemon", "OK", "no errors since daemon start ({} known sensor-probe lines ignored)".format(
            len(errors)))


# ---------------------------------------------------------------- v4l2 raw

def check_v4l2_raw(frames=120):
    if not os.path.exists(DEVICE):
        report("v4l2 raw", "FAIL", "{} missing: sensor not probed (cable? device tree?)".format(DEVICE))
        return False
    try:
        name = open("/sys/class/video4linux/{}/name".format(os.path.basename(DEVICE))).read().strip()
    except OSError:
        name = "?"
    cmd = ["v4l2-ctl", "-d", DEVICE,
           "--set-fmt-video=width={},height={},pixelformat=RG10".format(RAW_W, RAW_H),
           "--set-ctrl=bypass_mode=0", "--stream-mmap", "--stream-count={}".format(frames),
           "--stream-to=/dev/null"]
    t0 = time.monotonic()
    rc, out = run(cmd, timeout=15)
    dt = time.monotonic() - t0
    if rc != 0:
        report("v4l2 raw", "FAIL", "{}: v4l2-ctl rc={} {}".format(name, rc, out[-120:]))
        return False
    fps = frames / dt
    status = "OK" if fps > 45 else "WARN"
    report("v4l2 raw", status, "{}: {} Bayer frames {}x{} in {:.2f}s (~{:.0f} fps incl. startup)".format(
        name, frames, RAW_W, RAW_H, dt, fps))
    return True


# ---------------------------------------------------------------- gstreamer

class QuietOutput:
    """Send fds 1 and 2 to a temp file while NVIDIA libraries chatter; count GStreamer CRITICALs.

    GST_ARGUS lines are printf'd into libc's stdout buffer, so flush that too before
    restoring, or they'd appear at exit."""

    def __enter__(self):
        sys.stdout.flush()
        sys.stderr.flush()
        self.tmp = tempfile.TemporaryFile(mode="w+")
        self.saved = [os.dup(1), os.dup(2)]
        os.dup2(self.tmp.fileno(), 1)
        os.dup2(self.tmp.fileno(), 2)
        return self

    def __exit__(self, *exc):
        sys.stdout.flush()
        sys.stderr.flush()
        LIBC.fflush(None)
        for fd, saved in ((1, self.saved[0]), (2, self.saved[1])):
            os.dup2(saved, fd)
            os.close(saved)
        self.tmp.seek(0)
        self.criticals = sum("CRITICAL" in l for l in self.tmp)
        self.tmp.close()


def run_pipeline(desc, frames, timeout_s):
    """Run a pipeline ending in `fakesink name=sink` until EOS. Returns (ok, first_frame_s, fps, error)."""
    with QuietOutput() as q:
        result = _run_pipeline(desc, frames, timeout_s)
    return result + (q.criticals,)


def _run_pipeline(desc, frames, timeout_s):
    pipeline = Gst.parse_launch(desc)
    sink = pipeline.get_by_name("sink")
    stamps = []

    def probe(_pad, _info):
        stamps.append(time.monotonic())
        return Gst.PadProbeReturn.OK

    sink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, probe)
    t0 = time.monotonic()
    pipeline.set_state(Gst.State.PLAYING)
    msg = pipeline.get_bus().timed_pop_filtered(
        int(timeout_s * Gst.SECOND), Gst.MessageType.EOS | Gst.MessageType.ERROR)
    pipeline.set_state(Gst.State.NULL)

    if msg is None:
        return False, None, None, "timed out after {}s with {} frames".format(timeout_s, len(stamps))
    if msg.type == Gst.MessageType.ERROR:
        err, dbg = msg.parse_error()
        return False, None, None, "{}: {}".format(msg.src.get_name(), err.message)
    first = stamps[0] - t0 if stamps else None
    fps = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
    return len(stamps) >= frames, first, fps, None


def check_elements():
    missing = [e for e in ("nvarguscamerasrc", "nvvidconv", "nvv4l2h264enc")
               if Gst.ElementFactory.find(e) is None]
    if missing:
        rc, out = run(["gst-inspect-1.0", "-b"])
        report("gst", "FAIL", "missing {}; blacklisted: {}".format(", ".join(missing), out[-120:]))
        return False
    report("gst", "OK", "nvarguscamerasrc, nvvidconv, nvv4l2h264enc present")
    return True


def check_argus(frames=120):
    desc = ("nvarguscamerasrc num-buffers={} ! "
            "video/x-raw(memory:NVMM),width=1280,height=720,framerate=60/1,format=NV12 ! "
            "fakesink name=sink sync=false").format(frames)
    ok, first, fps, err, crit = run_pipeline(desc, frames, timeout_s=20)
    if err:
        report("argus", "FAIL", err)
        return False
    report("argus", "OK" if ok and fps > 55 else "WARN",
           "{} frames, first after {:.2f}s, steady {:.1f} fps, {} CRITICAL".format(frames, first, fps, crit))
    return ok


def check_encoder(frames=60):
    desc = ("videotestsrc num-buffers={} ! video/x-raw,width=1280,height=720,framerate=60/1 ! "
            "nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! nvv4l2h264enc ! "
            "fakesink name=sink sync=false").format(frames)
    ok, first, fps, err, crit = run_pipeline(desc, frames, timeout_s=15)
    if err:
        report("encoder", "FAIL", err)
    else:
        # nvv4l2h264enc on L4T 32.x logs one gst_buffer_resize_range CRITICAL per frame; harmless.
        report("encoder", "OK" if ok else "WARN",
               "{} frames encoded, {:.0f} fps ({} known-harmless CRITICALs)".format(frames, fps, crit))


# ---------------------------------------------------------------- reset

def reset(assume_yes):
    users = camera_users()
    if users:
        print("\nProcesses holding the camera:")
        for pid, cmd in users.items():
            print("  {:6} {}".format(pid, cmd))
        if not assume_yes and input("Stop them (SIGINT, then SIGTERM)? [y/N] ").strip().lower() != "y":
            print("Left them running; restarting the daemon anyway.")
        else:
            for pid in users:
                try:
                    os.kill(pid, signal.SIGINT)  # lets gst-launch -e / tee_demo send EOS
                except OSError:
                    pass
            time.sleep(3)
            for pid in users:
                if os.path.exists("/proc/{}".format(pid)):
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except OSError:
                        pass
    print("Restarting nvargus-daemon ...")
    rc, out = run(["sudo", "-n", "systemctl", "restart", "nvargus-daemon"], timeout=30)
    if rc != 0:
        print("restart failed ({}); run: sudo systemctl restart nvargus-daemon".format(out))
        return
    time.sleep(2)
    print("Done. Re-checking:\n")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reset", action="store_true", help="stop camera users and restart nvargus-daemon first")
    ap.add_argument("--yes", action="store_true", help="don't ask before stopping processes")
    ap.add_argument("--no-capture", action="store_true", help="skip the capture tests (safe while streaming)")
    args = ap.parse_args()

    Gst.init(None)
    if args.reset:
        reset(args.yes)

    check_platform()
    users = check_users()
    check_daemon()
    if args.no_capture:
        print("(capture tests skipped)")
    elif users:
        report("capture", "SKIP", "camera is in use; stop the users above or pass --reset")
    else:
        raw_ok = check_v4l2_raw()
        time.sleep(1)  # let VI release before Argus opens the sensor
        if check_elements():
            argus_ok = check_argus()
            check_encoder()
            if raw_ok and not argus_ok:
                print("\nHint: sensor works but Argus doesn't => daemon problem. Try --reset.")
        if not raw_ok:
            print("\nHint: raw capture failed => below Argus. Power off, reseat the ribbon cable, reboot.")

    worst = "FAIL" if any(s == "FAIL" for _, s in results) else "WARN" if any(s == "WARN" for _, s in results) else "OK"
    print("\noverall: {}".format(worst))
    return 1 if worst == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
