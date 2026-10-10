# kinematic
A Python computer vision application that turns webcam hand gestures into real-time desktop mouse controls, clicks, and scrolling.

## Status

**Early prototype — do not leave it running unattended.** It drives the real mouse, clicks and scrolls
as soon as it detects a hand: there is no dry-run mode, no pause key and no opt-in gate, and gesture
thresholds are absolute rather than scaled to hand size.

Per-phase implementation status and the nine-phase roadmap (foundation/safety, capture, geometry,
gesture state machine, cursor, clicks, scroll, pause/runtime, calibration) are documented in
[`AGENTS.md`](AGENTS.md) — see §2 "Phase status" for what is actually implemented today.

## Features
- Cursor control from the wrist/palm (finger articulation cannot move the pointer)
- Click debounce using pinch detection
- Right-click gesture support
- Scroll gestures based on finger movement
- OpenCV + MediaPipe pipeline with smoothing to reduce jitter

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Run

```bash
python3 -m kinematic.main
```

Or run the entry point directly from the project root:

```bash
python3 kinematic/main.py
```

**The app starts paused.** It opens the default webcam, shows the preview and analyses frames, but sends
nothing to the mouse until you press `SPACE` (or `p`). After that it moves the pointer with your index
finger, left-clicks on a thumb–index pinch and right-clicks on a thumb–middle pinch. `ESC` or `q` quits.
The preview shows `PAUSED` / `ACTIVE`, the live loop rate and the detected hand skeleton in its top-left
corner.

```bash
python3 -m kinematic.main --active      # start controlling the mouse immediately
python3 -m kinematic.main --dry-run     # run the whole pipeline, emit no mouse events at all
python3 -m kinematic.main --no-preview  # no window (quit with Ctrl-C)
```

Nothing is emitted while paused, and `--dry-run` swaps the real output adapter for a recording one, so
that mode needs no Accessibility permission and is safe to run anywhere.

## Smoothing and reach

Two knobs decide how the pointer *feels*:

- **Smoothing** (`smoothing_mode`, default `one_euro`): the 1-euro filter adapts its cutoff to hand speed —
  heavy smoothing when your hand is still, almost none when it moves. It replaced a fixed-factor EMA after
  a seeded comparison (`tools/compare_smoothing.py`, 16 seeds) showed it with 25% less jitter, a 2-frame
  instead of 6-frame step response, and 151 px instead of 529 px of lag during fast movement.
  `--smoothing-mode ema` restores the old filter if you prefer its feel.
- **Reach** (`frame_margin`, default 0.15, absolute mode only): the inner 70% of the camera frame maps onto
  the whole screen, so the screen corners are reachable without moving your hand to the edge of the frame.
  `screen_margin` keeps a margin free at the screen edges instead.

## Pointing

The cursor is driven by your **wrist/palm**, not your fingertip, and by default it moves *relative* to
your hand — like a mouse rather than a laser pointer:

- **Move your hand** → the cursor moves. Deliberate hand travel of about 20% of the frame's width crosses
  the screen once.
- **Curl or spread your fingers, pinch, or rotate your wrist** → the cursor does not move at all. Fingers
  only ever produce gestures (pinch = click, index-finger sweep while the hand holds still = scroll).
- **Move your hand back to a comfortable position** → the cursor moves back too, so there is no "lift and
  reposition" clutch yet. If you would rather point at a spot and have the cursor stay there, use
  position mapping:

```bash
python3 -m kinematic.main --pointer-mode absolute      # cursor maps to where your hand is in the frame
python3 -m kinematic.main --pointer-reference wrist    # the wrist joint alone (even more stable than the palm)
python3 -m kinematic.main --pointer-reference index    # the old fingertip behaviour (fingers drag the cursor)
python3 -m kinematic.main --smoothing-mode ema         # the fixed-factor filter instead of the 1-euro filter
```

Which hand point is used, how fast the pointer moves and how much hand jitter is ignored are all fields on
`HandMouseConfig` (`pointer_reference`, `pointer_sensitivity`, `pointer_deadzone`,
`relative_scale_by_hand`). `pointer_deadzone` is worth tuning for your own hands and camera:
`python3 tools/measure_stability.py --seconds 15` reports the per-frame wrist/palm jitter in your setup
next to the configured deadzone (it only reads landmarks — it cannot move your mouse).

## Performance

The control loop now runs at the camera's own frame rate. A ~30 fps webcam caps it: this FaceTime
camera negotiates 30 fps whatever resolution is requested, and `read()` blocks for the frame interval
either way, so 30 fps is the ceiling and no setting raises it.

Measured on macOS, same machine, same stages, with a hand in frame:

| | before | after |
| --- | --- | --- |
| processed frames per second | 5.8 fps | **28.8 fps** |
| per-frame budget | 173 ms | **34 ms** |

What made the difference:

- **`pyautogui.PAUSE = 0`** — the library default of `0.1` is slept *before every public call*, so
  each cursor move, click and scroll cost ~113 ms instead of ~11 ms. This was the single biggest cost.
- **Threaded capture** (`kinematic/capture.py`) — the loop no longer waits for `read()` before it can
  start working, and it never processes a frame that was queued while it was busy.
- **`model_complexity = 0`** (MediaPipe lite) — ~1.5 ms/frame cheaper. MediaPipe's cost does not depend
  on the frame size at all; it resizes to its own 256x256 internally.
- **Preview at `preview_scale`** (0.5) instead of full frame, and an ~11 ms `moveTo` skipped when it
  cannot shift the cursor by `min_move_pixels`.

The preview draws the live loop rate in its top-left corner. To measure the loop rate without moving your
mouse, run `python3 tools/measure_loop.py` (add `--synthetic-hand --active` to exercise the cursor path
with no hand in frame — it intercepts every mouse call and reports the count). Everything above is a field
on `HandMouseConfig`, so it can be tuned without touching the loop:

| Field | Default | Effect |
| --- | --- | --- |
| `mouse_pause` | `0.0` | seconds pyautogui sleeps per mouse call (library default is 0.1) |
| `min_move_pixels` | `0.5` | skip a cursor move smaller than this many pixels |
| `model_complexity` | `0` | 0 = lite/faster, 1 = full/slower hand model |
| `preview_scale` | `0.5` | preview size as a fraction of the frame (1.0 = full size) |
| `show_preview` | `True` | `False` drops the ~14 ms display cost (then quit with Ctrl-C, not `q`) |
| `show_fps` | `True` | draw the loop rate on the preview |
| `smoothing` | `0.35` | EMA factor; raise it for a snappier pointer, lower it for less jitter |
| `capture_width` / `capture_height` / `capture_fps` / `capture_buffer_size` | 1280 / 720 / 30 / 1 | what is requested from the camera |

## Requirements

- macOS (tested) or Linux, with a webcam.
- Python 3.9 — verified with the system `python3` (3.9.6) and `opencv-python 5.0.0`,
  `mediapipe 0.10.14`, `numpy 2.0.2`, `pyautogui 0.9.54`.
- `mediapipe` is pinned to `0.10.14` on purpose: the code uses the legacy `mp.solutions.hands` API,
  which was removed in 0.10.30+ and 1.x, so a newer release installs cleanly and then breaks the
  capture pipeline at import.
- PyAutoGUI needs **Accessibility** permission and OpenCV needs **Camera** permission — see below.

## Permissions (macOS)

| Permission | Needed for | Where |
| --- | --- | --- |
| Camera | OpenCV / MediaPipe reading the webcam | System Settings → Privacy & Security → Camera |
| Accessibility | PyAutoGUI moving the cursor and clicking | System Settings → Privacy & Security → Accessibility |

Both must be granted to the **host process** that launches Python (Terminal, VS Code, …), not to Python
itself. Without the camera grant, `cv2.VideoCapture(0)` fails with
`OpenCV: not authorized to capture video`; without the Accessibility grant, the cursor simply does not
move.

## Tests

```bash
python3 -m unittest discover -s tests
```

Run from the repository root (the tests import `kinematic.gesture`). The four tests cover the pure
gesture helpers in `kinematic/gesture.py`; the capture/control loop in `main.py` is not covered yet.

## Project context

[`AGENTS.md`](AGENTS.md) is the working brief for contributors and AI coding agents: verified
environment facts, the safety contract, the nine-phase roadmap with per-phase prompts and status, the
architectural decisions, and gotchas (including why `mediapipe` is pinned).
