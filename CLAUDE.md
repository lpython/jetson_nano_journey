# jetson_nano_journey

Low-latency camera → CUDA inference (MNIST/LeNet) → UDP pipeline on a Jetson Nano,
progressing from GStreamer CLI → C → no GStreamer → Rust. The full phased plan is in
[PLAN.md](PLAN.md). VNC setup is in [vnc.md](vnc.md).

## Two machines, one source of truth

| | Mac (local) | Jetson Nano (remote) |
|---|---|---|
| Path | `~/Repso/jetson_nano_journey` | `~/jetson_nano_journey` (`/home/jetson/jetson_nano_journey`) |
| Role | Edit, git, Claude Code | Build, run, profile |
| Git | **The repo.** All commits happen here. | **Not a repo.** A disposable mirror. |

- Edit locally and push to the Nano with `scripts/sync.sh` (a one-way rsync with `--delete`).
  Never edit files on the Nano directly, because the next sync overwrites them.
- Build and run on the Nano over SSH: `scripts/remote.sh make` (runs in the remote project dir).
- Build outputs stay on the Nano. Pull back only what's worth keeping (logs, captures, benchmark
  results) with `scripts/pull.sh <remote-subpath>`, then commit it locally.
- Don't run Claude Code, VS Code Server, or other heavy tools on the Nano. It has 4 GB of RAM
  shared with the GPU.

## Reaching the Nano

- User `jetson`, hostname `nano`, currently at `192.168.1.209`. The IP has changed before (it was .211),
  so it's DHCP-assigned and may move again.
- Key: `~/.ssh/jetson`. The default key is rejected.
- Scripts use the SSH alias in `$JETSON` (default `nano`). They expect a `Host nano` entry in
  `~/.ssh/config` with `HostName`, `User jetson`, `IdentityFile ~/.ssh/jetson`, and `IdentitiesOnly yes`.
  The older `jetbot` alias points to the stale .211 address.
- Claude Code runs SSH from the Mac. LAN access needs the sandbox disabled for that command.
  Use `-o BatchMode=yes` because there's no TTY for password prompts.
- `jetson` has passwordless sudo (`NOPASSWD: ALL`), so `sudo -n` works from scripts over SSH.
  Still ask before doing anything with system-wide effects (reboot, apt, nvpmodel changes).
- Camera trouble (like `Failed to create CaptureSession`): run `scripts/camdiag.py` first, and
  `scripts/camdiag.py --reset` to stop camera users and restart `nvargus-daemon`.
- VNC: x11vnc on port **5990** (see vnc.md). Zed remote mode is also set up as a fallback editor.

## Actual platform (differs from PLAN.md)

PLAN.md assumes stock JetPack 4.6 on Ubuntu 18.04. This board runs a **Ubuntu 20.04** image on top of L4T:

- L4T R32.6.1 (JetPack 4.6), kernel 4.9.253-tegra, aarch64, 4 cores, 4 GB
- Ubuntu 20.04.6, **glibc 2.31**, **GCC 8.4**, **GStreamer 1.16.3**
- CUDA 10.2 (`/usr/local/cuda/bin/nvcc`, not on PATH by default), Maxwell sm_53
- Camera: **IMX219 over CSI** on `/dev/video0`, so the Argus/`nvarguscamerasrc` path applies
- Power mode is MAXN. The fan mode isn't set, so check thermals before benchmarking.
- Kernel 4.9 limits still apply: no io_uring and no UDP GSO. Use epoll, `sendmmsg`, and `recvmmsg`.

## Conventions

- The main code is C, built only against C-surface APIs (GStreamer, CUDA driver API, nvbuf_utils,
  EGL, V4L2, sockets). Keep C++ (Argus, TensorRT) behind thin `extern "C"` shims so the Rust
  port can use `bindgen`.
- Build natively on the Nano with **clang-12** (installed via apt): C11 for host code, and CUDA kernels
  compiled by clang to PTX (`--cuda-gpu-arch=sm_53 --cuda-device-only -S`), loaded at runtime with
  `cuModuleLoad`. Host code links only `libcuda` (the driver API), with no nvcc and no CUDA runtime. See `phase2/Makefile`.
  Don't cross-compile from macOS.
- NVMM buffers going to CUDA must be pitch-linear: `nvvidconv bl-output=false` (it defaults to block-linear).
- Keep generated files, binaries, and large captures out of git (see `.gitignore`).

## Layout

- `scripts/`: workflow and diagnostic tools (`sync.sh`, `remote.sh`, `camdiag.py`, `tee_demo.py`)
- `phase2/`: C pipeline + appsink → zero-copy CUDA. `make` there; walkthrough in `docs/phase2.md`
  - `phase2/kernels/notes.md`: deriving launch parameters for the Nano's single Maxwell SM
  - `phase2/tools/kbench`: kernel benchmark on synthetic frames (`make tools`; design in `tools/kbench.md`)
- `docs/`: newcomer-level walkthroughs of each piece

## Scripting: bash only for the basics

- **Bash is for thin wrappers only**: a fixed command with at most an argument or two and defaults
  (like `sync.sh` and `start_gstream_upload.sh`). No branching on modes, no building strings, and no `eval`.
- **Anything with logic uses Python**, and that includes GStreamer experiments. Build pipelines with
  `Gst.parse_launch` through `gi` rather than generating shell. This also lets a script read element
  properties at runtime, such as queue depth. Example: `scripts/tee_demo.py`.
- Python on the Nano means system `/usr/bin/python3` (**3.8**) with stdlib plus `gi` only. Don't use conda, pip
  installs, or 3.9+ syntax (no `match`, `dict | dict`, or `list[str]` hints). Use `#!/usr/bin/python3`.
- Don't use Nushell here because it isn't installed on the Nano.
