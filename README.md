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
- Cursor control via index-finger tracking
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

The app opens the default webcam and uses the live hand landmarks to drive the mouse pointer. Press `q` in the video window to quit.

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
