# Assessment: the 2022 battery-display reader

A read-only review of the earlier project on the Micron2TB drive
(`SoftDev/DL/Vision/opencv-course-master (2)/`): the `old/` scripts, the `battery_find/` package
(5 commits), `battery_masking_simple/`, and the Rust port `battery-mon-rs/`. I didn't see
`nv_jetson_battery_status/` (referenced in `moto_lcd.py`), so any SD-card clip recording that
lived there isn't covered.

**The problem it was solving.** It read the battery percentage off a motorcycle dash LCD
(`Resources/test_5.jpg`: "80%" in **seven-segment** digits next to a battery icon). The camera was
the Nano's CSI camera, and iPhone videos (`IMG_1794.MOV`) were used for development.

**How far it got.** It found the battery icon in stills and in the recorded video. The digit-reading
stage (`battery_find/sub_image_sequencing.py`) only ever ran on one still photo
(`cv2.imread(image_src)`, line 45), had templates for digits 0–3 only, and has a bug that combines the
two digits wrongly (section 3). So the project never produced a correct end-to-end reading from live
video. Everything below explains why, and what to carry into the digit project.

---

## TL;DR

1. **You found the right lesson yourself.** GStreamer should own capture, conversion, scaling and
   dropping. Your code should attach at one point (an `appsink`) and only ever see the newest frame,
   already in the format you need. You had actually hand-built a leaky depth-1 queue (the capture
   thread in `camera.py`), but it sat after two wasted color conversions and had no frame identity.
2. **The algorithm did too much work per frame, in the wrong space.** It ran brute-force multi-scale
   template matching (50 scales × 2 methods, every frame) on hand-thresholded images, with a
   hard-coded rotation. The display is a flat, rigid object with fixed internal geometry. Find it
   once, **rectify** it into a fixed canonical view, and then every digit sits at a constant pixel
   position.
3. **Seven-segment digits don't need template matching or a neural network.** Sample the 7 segment
   positions (on/off) and look the pattern up in a 128-entry table. It takes microseconds and is exact.
4. **Tracking was the right idea, done with the wrong structure.** "Search near the previous find" is
   correct. It needed an explicit SEARCH → TRACK → LOST state machine owned by one object. Instead it
   used module globals, mixed two match metrics, and ran a lock-on threshold that could only drift
   downward.
5. **Several bugs would have blocked it even if the design were fine** (section 3), including a typo'd
   geometry constant and digits being *added* instead of concatenated.
6. **Missing process:** no labeled test frames, no accuracy number, no per-stage timing, and
   versioning by copying files (`copy`, `copy 2`, `copy_3`). You were judging by watching `imshow`
   windows.

## What was good

This is worth saying, because some of it carries forward directly:

- **Recorded video used as a replay source.** That's the right instinct; it just needed labels.
- **Template masks.** Ignoring the battery's interior fill was the correct way to match a shape whose
  contents change.
- **Measuring the display's internal geometry** (`coords.txt`: digit heights and widths as ratios of the
  outer box). This becomes exact coordinates once the display is rectified.
- **Using the value's domain**: only check "1" for the hundreds digit (`check_one_only`).
- **A debug tool for picking thresholds** (`histogram_levels`).
- **Moving from loose scripts to a package with git**, and adding `Point`/`Rectangle` types.

---

## 1. The data path: where frames lived and how they moved

### What the code did

From `battery_find/camera.py:62`:

```
nvarguscamerasrc 3280x2464 NV12 @ 21 fps   ← slowest sensor mode (full res)
  → nvvidconv → 1000x1000 BGRx             ← 4:3 squashed into 1:1: aspect distorted
  → videoconvert → BGR                     ← CPU color conversion, every frame
  → appsink                                ← default: no max-buffers, no drop
  → Python thread: self.value = image      ← hand-rolled "latest frame" slot
  → main loop: cvtColor(BGR→GRAY)          ← CPU again, throwing the color away
  → threshold → 100 template matches → imshow/waitKey
```

Problems, in order of cost:

- **Two color conversions whose result is discarded.** The camera produces NV12, whose Y plane is
  already grayscale. The code converted to BGRx (VIC), then BGR (CPU), then back to gray (CPU).
  Asking `nvvidconv` for `GRAY8` (confirm with `gst-inspect-1.0 nvvidconv`), or reading the Y plane,
  removes both CPU passes.
- **Aspect distortion.** Scaling 3280×2464 (4:3) to 1000×1000 squashes the image horizontally. That
  alone changes the battery's shape, forcing a wide scale search and weakening matches.
- **The slowest sensor mode.** Full resolution caps the IMX219 at 21 fps (see `camdiag.py`'s mode list),
  and the image was shrunk right away anyway. 1640×1232 @ 30 would have given the same detail at the
  output size.
- **The hand-rolled leaky queue.** `Camera._capture_frames` overwrites `self.value` with the newest
  frame. That's a depth-1 leaky queue, and the instinct was right. But it had no frame number, so the
  main loop couldn't tell a new frame from the one it just processed. It spun on repeats, and latency
  wasn't measurable. The Python thread also competes with the processing loop for the GIL.
- **The display paced the loop.** `cv2.waitKey(100)` in `sub_image_sequencing.py:172` caps processing at
  10 fps by itself, and `imshow` calls sit inside the processing functions.

### Should this happen "under GStreamer"? Can a sink be a Python script?

**Partly.** GStreamer should own everything up to "here is the newest frame, in the size and format I
need". Your algorithm is ordinary code attached at the end. There are four ways to attach Python:

| Option | What it is | Use it for |
|---|---|---|
| **`appsink` via `gi`** | Your callback receives each buffer (`new-sample` signal) | The default. Same pattern as `tee_demo.py` |
| OpenCV `VideoCapture(pipeline, CAP_GSTREAMER)` | OpenCV drives an `appsink` for you | What the old code did; fine if you set the appsink properties |
| `appsrc` | Your code pushes buffers *into* a pipeline | Sending annotated frames or results back out (to an encoder or the network) |
| A GStreamer element written in Python | Subclass `GstBase.BaseTransform` (gst-python) | Possible, but slow per-pixel work in Python hurts on the Nano; better in Rust later (PLAN.md Phase 6) |

The minimal fix for the old `camera.py` was a different pipeline string:

```
nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1640,height=1232,framerate=30/1 !
nvvidconv ! video/x-raw,format=GRAY8,width=820,height=616 !
appsink max-buffers=1 drop=true sync=false
```

That gives you correct aspect, grayscale done on the VIC, and GStreamer dropping stale frames (the
`good` mode from [tee_demo.md](tee_demo.md)). Pulling samples directly also gives you each buffer's
PTS (presentation timestamp), which serves as a frame ID and capture time.

One more thing worth knowing: **DeepStream** is NVIDIA's GStreamer framework for exactly this
detect-then-track pattern (`nvinfer` for TensorRT models, `nvtracker` for IOU/KLT/NvDCF tracking), and
DeepStream 6.0 supports the Nano on JetPack 4.6. It's heavy on configuration and hides what this
project is trying to learn, but it's a good reference for how the pieces are divided.

**Replay vs live mode.** For a recorded video, swap the source
(`uridecodebin uri=file:///…/IMG_1794.MOV ! videoconvert ! video/x-raw,format=GRAY8 ! appsink`) and keep
everything downstream the same. For **evaluation**, set `drop=false` and process every frame, so runs
are deterministic and comparable. For **live** use, drop. The old code needed three separate scripts
(`find_battery.py`, `_in_movie`, `_from_camera`) for three sources. With GStreamer, the source is one
element in the pipeline string.

---

## 2. The algorithm

### 2.1 Brute-force multi-scale template matching on every frame

`process_image` resizes the whole frame to 50 scales, and at each scale runs `matchTemplate` with
`TM_SQDIFF` *and* `TM_CCORR_NORMED` (the older version) using a mask. That's around 100 full-frame
correlations per frame, repeated even when the display hasn't moved. The Nano's CPU can't do that at
video rate. `Subdivision` then multiplies it by the number of sliding windows.

Template matching is a sound tool when **scale and rotation are known** and you're searching a small
area. It's a poor tool for "find this object at unknown scale, rotation and perspective in a large
image". The code patched each missing invariance by hand: scale with a 50-step pyramid, rotation with
a hard-coded `rotation = -13`, and perspective not at all.

### 2.2 Hand-tuned global thresholds before matching

The thresholds differ between files: 45, 95, 140, 155. Each was picked per image with
`histogram_levels`. A motorcycle dash sees full sun, shade and night backlight, so a fixed global
threshold is wrong most of the time. Thresholding *before* matching also throws away the gradient
information that makes matching reliable.

Better options:
- Match on grayscale with **`TM_CCOEFF_NORMED`**. It subtracts the mean, so it tolerates brightness and
  contrast changes.
- Where you need binary images, use **Otsu** or **adaptive** thresholds inside a small, known region.

A note on the method switch: `TM_CCORR_NORMED` doesn't subtract the mean, so it favors bright areas.
That's part of why the scores needed a moving lock-on threshold.

### 2.3 The display is a rigid plane, so rectify it

The key idea the project was missing: the LCD is flat and never changes shape. The camera sees it
through a **homography**, a 3×3 transform covering position, scale, rotation and perspective all at
once. Find 4 corresponding points (the display corners, or feature matches against a reference photo)
and `cv2.warpPerspective` maps every frame into a fixed canonical view, say 400×200. In that view:

- `coords.txt`'s ratios become constant pixel boxes. No `OUTER_BOX_*_FACTOR` arithmetic on each frame.
- Digit size is always the same, so no scale search.
- Rotation is already removed, so no hard-coded -13°.

If the camera is **rigidly mounted** on the bike, you can calibrate the homography once (click 4
corners, or detect them once) and store it. Then only check for drift from vibration
(section 5).

### 2.4 Reading seven-segment digits by template matching

`scan_digit` matched each digit crop against per-digit templates, with only 0–3 loaded
(`moto_lcd.py:34`). For a seven-segment display this is the hard way around. In the rectified view:

```
 aaa        each digit box has 7 fixed sample regions (a–g);
f   b       a segment is ON if its region is mostly dark;
 ggg        the 7 on/off bits index a lookup table:
e   c         (1,1,1,1,1,1,0) → 0,  (0,1,1,0,0,0,0) → 1, …
 ddd        patterns not in the table → "unreadable", not a guess
```

It's deterministic, costs microseconds, and has a built-in validity check. This is the classic
approach; there's a well-known PyImageSearch tutorial on it (the same site `imutils`/`rotate_bound`
came from).

### 2.5 A threshold that could only drift downward

`sub_image_sequencing.py:187–193` lowers `lock_on_thresh` whenever the best match is below it. With
the object out of view, the threshold keeps falling until *something* matches, and the code then
locks onto noise. Use **fixed hysteresis** instead: enter TRACK above a high threshold, leave it below
a lower one, and never adjust either at run time.

---

## 3. Bugs that would have blocked it

All paths are relative to the old project root.

| Where | Bug | Effect |
|---|---|---|
| `battery_find/sub_image_sequencing.py:28` vs `coords.txt:16` | `OUTER_BOX_WIDTH_FACTOR = 2.594`; the measured value is 495/171 = **2.894** | Outer box, and so every digit box, placed wrongly |
| `battery_find/sub_image_sequencing.py:165` | `text_to_digit(tens) + text_to_digit(ones)` adds two ints (8 + 0 = **8**, not 80), then stores the int in `number: str` | Wrong reading even when both digits are right |
| `battery_find/moto_lcd.py:34` | Digit templates for 0–3 only | 4–9 can never be read |
| `battery_find/moto_lcd.py:46` | `digit_to_text` raises using an undefined `name` | `NameError` instead of the intended message |
| `battery_find/process_image.py:53–57, 155–170` | After "switched to ccorr_norm", the tracker compares a `TM_SQDIFF` score (lower is better) to a `TM_CCORR_NORMED` score (higher is better) | The "better search" / "accept" logic is meaningless |
| `battery_find/process_image.py:55, 169` | `process_image(...)[0]` now returns the namedtuple's first field (a float), then code unpacks it as a 7-tuple | Crash on the no-previous or fallback path |
| `old/process_image.py:74`, `battery_find/process_image.py:92` | `clipped_end_y += abs(clipped_end_y)` should be `abs(clipped_start_y)` | Search box doubles in height near the top edge |
| `old/process_image.py:175–176`, `battery_find/process_image.py:193–194` | "Width" difference is computed from y coordinates (copy of the height line) | Width is never compared |
| `battery_find/process_image.py:247–252` | When no match is found: `while True: cv2.waitKey(1000)` | Program hangs forever |
| `find_battery_in_movie.py:33–41` | `ret` from `cap.read()` is never checked | Crashes on `None` at end of video |
| `find_battery_in_movie.py:51–53` | `cap.stop()` doesn't exist on `VideoCapture`; `time.sleep(50)` | Crash on quit; 50 s wait |
| `sub_image_sequencing.py:68–70` | Debug mode skips the first 25 search windows | Debug and production behave differently |
| `sub_image_sequencing.py:92–95` | `consecutive_successful_finds` incremented twice per loop | Lock-on after about 8 frames, not 15 |
| `Rectangle.py:65` | `area` returns the perimeter (the docstring admits it) | Latent; unused so far |

**The Rust port** (`battery-mon-rs/src/main.rs`) reimplemented a single-scale version of the same
matching, on one still image, with no commits. Porting an algorithm that doesn't work yet to a faster
language doesn't fix it. Get it right in Python on replayed video first, then port.

---

## 4. How the code should have been structured

### What went wrong structurally

- **Versioning by copying files**: `copy`, `copy 2`, `copy_3`, `_old`. That's what git branches and
  commits are for, and later you did switch.
- **Duplicated helpers**: `rotate_bound` and `histogram_levels` appear in about 8 files.
- **Tracker state as module globals** (`previous_found_min_sqdiff`, `previous_found_max_ccorr`). You
  can't have two trackers, can't reset one, and can't test it.
- **Display inside logic**: `cv2.imshow('cropped', …)` sits inside `process_image_use_prev`, so
  processing can't run headless on the Nano.
- **One script per input source**, each repeating the setup.
- **No ground truth**: success meant "the rectangle looked right in the window".

### Structure to use instead

Separate stages. Each one is a pure function, or a small class that owns its own state:

```
source ──► locate ──► rectify ──► read ──► smooth ──► output
(GStreamer) (tracker)  (warp)   (7-seg)  (vote)    (UDP / log)
```

```python
class Frame(NamedTuple):          # made at the appsink; never mutated
    seq: int                      # counter, or buffer offset
    pts_ns: int                   # capture time from the buffer
    gray: np.ndarray

class Reading(NamedTuple):        # the only thing that leaves the pipeline
    seq: int
    pts_ns: int
    value: Optional[int]          # None = unreadable this frame
    confidence: float
    state: str                    # SEARCH / TRACK / LOST

class DisplayLocator:             # owns tracker state; no globals
    def update(self, frame: Frame) -> Optional[np.ndarray]: ...   # homography or None

def rectify(gray, H) -> np.ndarray: ...                  # pure
def read_digits(canonical) -> Tuple[Optional[int], float]: ...  # pure, 7-seg lookup

class ValueSmoother:              # owns reading history
    def update(self, seq, value, conf) -> Reading: ...
```

Then add the two things that would have changed everything:

1. **A labeled replay set.** Take ~100 frames from the MOV files and label them in a CSV
   (`frame_index,expected_value`). The same pipeline runs on them in evaluation mode (no drops) and
   prints an accuracy percentage. Now every change is a number, not an impression.
2. **Per-stage timing.** Record a timestamp at each stage boundary for every frame (the same idea as
   PLAN.md's measurement section). You'd have seen immediately that locating took 95% of the time.

Keep display in a separate debug sink (or a `tee` branch) that can be switched off. Never put it in
the logic.

---

## 5. Tracking from frame to frame

The core pattern, whatever you use for the matching:

```
         found (score ≥ HIGH)                 score ≥ LOW
SEARCH ─────────────────────► TRACK ◄──────────────────┐
  ▲       (expensive, rare)     │  (cheap, local)      │
  │                             └──────────────────────┘
  │  N consecutive misses             score < LOW
  └──────────────────────────── LOST ◄──────────
```

- **SEARCH** runs the expensive global method, only when there's no lock.
- **TRACK** searches a small window around the last position (your `process_image_use_prev` idea) or
  refines the last homography. It's cheap enough to run every frame.
- **LOST** tolerates a few bad frames (glare, a hand in the way) before paying for SEARCH again.
- The thresholds are **fixed** and HIGH > LOW (hysteresis), so the state doesn't flicker.

Track at two levels: **where** the display is (geometry), and **what** it says (the value).

### Traditional methods

| Method | What it does | Fit here |
|---|---|---|
| **Fixed-mount calibration** | Measure the homography once, store it | Best if the camera is rigidly mounted. Zero per-frame search |
| **ECC alignment** (`cv2.findTransformECC`) | Refines a known transform to sub-pixel accuracy against a reference view | Correcting vibration drift from a stored calibration; cheap per frame |
| **Features + RANSAC** (ORB/AKAZE, `findHomography`) | Matches keypoints to a reference photo of the dash; recovers a full homography | The right SEARCH step for a handheld or moving camera; handles scale, rotation and perspective in one go |
| **Optical flow** (KLT, `calcOpticalFlowPyrLK`) | Follows corner points from frame to frame | A cheap TRACK step between feature matches |
| **Template in a small window** (`TM_CCOEFF_NORMED`) | What you did, restricted to a search window | Acceptable TRACK step once scale and rotation are known (after rectifying) |
| **Kalman / alpha-beta filter** | Smooths the tracked corners' positions and predicts the next ones | Steadies the rectification and narrows the search window |

For the **value**, battery percentage changes slowly and predictably:
- Report the **majority over the last N readings**, not the latest frame.
- **Reject implausible jumps.** 80 → 30 within a second is a misread, not a discharge.
- Treat "unreadable" as a real outcome (`value=None`). Never guess.

### Learned and LLM methods

| Method | What it does | Fit and caveats |
|---|---|---|
| **Small object detector** (SSD-MobileNet or small YOLO on TensorRT; jetson-inference or DeepStream) | Finds the display/icon box in each frame | Replaces the SEARCH step. Needs a few hundred labeled boxes (from your MOVs). Runs on the Nano |
| **Detector + tracker** (SORT, ByteTrack, DeepStream `nvtracker`) | Associates detections across frames using motion and IoU | The learned version of the state machine. Usually overkill for a single rigid display |
| **Small CNN digit classifier** (LeNet-sized) | Classifies each rectified digit crop | Works well, but a 7-segment lookup is simpler and exact. **Synthetic training data** (rendered segments plus noise, blur and perspective) is very effective for 7-seg |
| **OCR engines** (Tesseract, PaddleOCR, EasyOCR) | General text reading | Stock Tesseract handles seven-segment fonts poorly (community-trained 7-seg models exist). PaddleOCR/EasyOCR are heavy for the Nano |
| **Vision-language models** (Claude, GPT-4o-class, or small open VLMs such as Florence-2, Moondream, Qwen2-VL-2B) | Read "80%" off a photo zero-shot, with no training | **Not per-frame.** API calls take seconds and need a network connection. Running locally is impractical on this Nano's software stack (CUDA 10.2, old PyTorch). But the battery value changes slowly, so an occasional check (every 30–60 s) is viable |

**The best use of an LLM here is offline, not in the frame loop:**
- **Auto-label your replay set.** Send the frames from the MOVs to a VLM, then spot-check the labels.
- **Verify the fast path.** Every so often, compare the local reading with a VLM reading and log
  disagreements.
- **Generate and review code and tests**, which is how this repo is being built.

The fast local path stays deterministic and on-device. The LLM supplies ground truth and
spot-checks.

---

## 6. Back to this project: digits from a camera stream

The current plan (PLAN.md) reads handwritten digits with LeNet, which isn't the old seven-segment
problem. The structure and the failure modes carry over directly, though.

### Don't try these again

- Converting color formats you then throw away. Stay in NV12/Y in NVMM; the Y plane *is* the gray
  image (PLAN.md Phase 2).
- Unbounded or hand-rolled frame hand-off. Use `appsink max-buffers=1 drop=true` and the leaky queue
  from `tee_demo.py good`.
- Scanning every scale and position on every frame. Locate cheaply (threshold, contours, bounding boxes
  on the Y plane in CUDA), then run the classifier only on the candidates.
- Global thresholds tuned on one image. Use Otsu or adaptive thresholds, computed per candidate region.
- Thresholds that adjust themselves at run time. Use fixed hysteresis.
- Tracker state in globals, and `imshow` inside processing. Stages own their state; debug output goes to
  its own branch.
- Judging by eye. Build a labeled replay set and an accuracy number before tuning anything.
- Porting to C, CUDA or Rust before the Python or command-line version reads correctly. PLAN.md's
  order (GStreamer CLI → C → …) already guards against this. Keep a Python reference implementation of
  preprocessing and LeNet to check each port's outputs against.
- Trusting single frames. Vote across frames, and report "unreadable" rather than guessing.

### What to do instead, mapped to PLAN.md

| Stage | Old project | This project |
|---|---|---|
| Frame hand-off | Python thread + `self.value` | `appsink max-buffers=1 drop=true`, frame ID = buffer PTS (Phase 2) |
| Locate | Multi-scale template search | Threshold + connected components/contours on the Y plane → candidate boxes |
| Normalize | Hard-coded rotation; digit boxes from ratios | MNIST-style: crop to the box, fit in 20×20, center by center of mass in 28×28 (Phase 3) |
| Read | Per-digit templates (0–3 only) | LeNet in CUDA; softmax confidence; below threshold → "unreadable" |
| Track | Global "previous find" | Per-digit **tracks**: associate candidate boxes between frames by IoU or centroid distance (greedy matching is enough for a few digits), add a Kalman filter if motion is fast |
| Smooth | None | Keep a history of recent labels per track → report the majority and its share as confidence; drop a track after N frames unseen |
| Output | `print` | UDP packet with sequence number, capture timestamp, track ID, digit, confidence (Phase 4) |
| Evaluate | Watching `imshow` | Labeled clip + accuracy + per-stage latency, rerun after every change |

A digit's **track ID** is the frame-to-frame identity the old project never had. It turns a stream of
independent guesses into "the digit at this position has read 7 for the last 12 frames at 0.94". That
second kind of output is the one you can act on.
