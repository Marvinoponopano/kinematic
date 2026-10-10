# kinematic
A Python computer vision application that turns webcam hand gestures into real-time desktop mouse controls, clicks, and scrolling.

## Status

**Usable, with the safety layer in place.** It starts **paused** and emits nothing until you press
`SPACE`; `--dry-run` runs the whole pipeline with a recording adapter and touches nothing on your
desktop. Gesture thresholds are relative to hand size, gestures go through a tested state machine with
hysteresis, hold times and cooldowns, and the cursor follows the wrist/palm so finger articulation
cannot drag it.

Per-phase implementation status and the nine-phase roadmap (foundation/safety, capture, geometry,
gesture state machine, cursor, clicks, scroll, pause/runtime, calibration) are documented in
[`AGENTS.md`](AGENTS.md) — see §2 "Phase status" for what is actually implemented today. Phases 1–4
are complete.

The pointer path is tuned by measurement rather than by feel: **256 tests**, and
`python3 tools/measure_smoothness.py` reports how often the cursor holds still, how large each step is
and how far it trails your hand — no camera or mouse required.

## Features
- Cursor control from the wrist/palm (finger articulation cannot move the pointer)
- Left click on a thumb–index pinch, right click on a thumb–middle pinch
- Two-finger scroll with a dead zone, sensitivity and click suppression
- Hysteresis, hold times, cooldowns and release-to-re-arm, so a gesture fires exactly once
- Paused startup, a keyboard pause toggle, `--dry-run`, and a mock output adapter for tests
- OpenCV + MediaPipe pipeline with One Euro smoothing to reduce jitter

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
nothing to the mouse until you press `SPACE` (or `p`). After that it moves the pointer with your hand,
left-clicks on a thumb–index pinch, right-clicks on a thumb–middle pinch and scrolls with a two-finger
pinch. `ESC` or `q` quits. The preview shows `PAUSED` / `ACTIVE`, the live loop rate, the current
gesture state (`neutral`, `left_pinch`, `tracking_lost`, …), the sensitivity dial's value and the
detected hand skeleton, all in its top-left corner.

```bash
python3 -m kinematic.main --active      # start controlling the mouse immediately
python3 -m kinematic.main --dry-run     # run the whole pipeline, emit no mouse events at all
python3 -m kinematic.main --no-preview  # no window (quit with Ctrl-C)
```

Nothing is emitted while paused, and `--dry-run` swaps the real output adapter for a recording one, so
that mode needs no Accessibility permission and is safe to run anywhere.

## Sensitivity — the dial

How far the cursor moves for a given hand movement is adjustable **while the app runs**, from the
preview window's keyboard. No restart, no config file, no UI:

| Key | Effect |
| --- | --- |
| `]` or `=` or `+` | one notch **faster** |
| `[` or `-` or `_` | one notch **slower** |
| `0` | back to the value you started with |

The current setting is drawn on the preview — `sens  1.00 [#---------]  ] faster  [ slower  0 reset` —
as the value plus a log-scale bar (each notch is ×1.15, so the bar moves evenly wherever you are). The
range is 1.0 to 60, and it clamps at the ends rather than wrapping.

**Why 1.0 is the bottom: the screen has to stay reachable.** The gain means *screen widths of cursor
travel per frame width of hand travel*, so below 1.0 a full sweep of the hand cannot cover the screen,
and a strip along each edge becomes unreachable. That is the band you may have noticed after turning the
sensitivity down — it is arithmetic, not a feel. The dial therefore stops at 1.0
(`sensitivity_min` = `sensitivity_reach_floor`), a starting `--sensitivity` below it is clamped with a
note in the banner, and if you override the floor deliberately the preview appends
`! edges unreachable` instead of leaving you guessing.

```bash
python3 -m kinematic.main --sensitivity 3          # start faster than the default 1.0
python3 -m kinematic.main --sensitivity-step 1.3   # coarser dial: 30% per notch
```

In the default absolute mode, 1.0 maps the inner 70% of the frame onto the whole screen, so the screen
corners are reachable by moving your hand to that region's corners; the dial scales that reach. In
relative mode the gain scales hand displacement, and reaching both edges needs a little more than 1.0
(measured ~1.5 in a single sweep) because the return sweep drags the cursor back. Either way, turning the
dial never moves the cursor by itself and never makes it jump, and the value you end up at is printed
when the app exits (pass `--sensitivity` to start a later session at the same place).

To check reachability for yourself:
`python3 tools/measure_loop.py --synthetic-hand --active --sweep-reach 30 --reach-dwell 12` walks a
synthetic hand to each corner, *stops* there for 12 frames (the way you do when you point at a corner)
and reports whether all four screen edges were reached — separately from the few px a smooth filter
trails while the hand is still travelling.

## Gestures

Gestures are recognised by a pure, timestamp-driven state machine (`kinematic/recognizer.py`): one pose
means one thing, and a pose must persist before it acts.

| Pose | What happens |
| --- | --- |
| thumb and index fingertip together | **left click** |
| thumb and middle fingertip together (index away) | **right click** |
| thumb touching both index and middle tips | **scroll** — up/down from the two-finger midpoint |
| no hand in the frame | tracking lost: nothing is emitted, and a pinch held through the loss must be released and re-pinched |
| any pose while paused | nothing, including a pinch that was already held when you resumed |

The rules, so a gesture cannot fire twice or fire by accident:

- A pose must **hold for 0.08 s** before it registers — a passing pinch does nothing.
- Two clicks are never closer than **0.18 s** (`click_cooldown_seconds`), and the pinch must be
  **released** above the release threshold before it can fire again. A click inside the cooldown is
  dropped, not deferred.
- **Hysteresis**: a pinch engages below 0.40 of your hand's size and only releases above 0.56, so a hand
  resting near the threshold cannot chatter.
- Thresholds are **relative to hand size** (`hand_scale`, wrist → middle knuckle), so the same pose works
  near and far from the camera.
- The scroll pose **outranks the clicks**, so pointing at a scroll pose cannot also click; scroll
  movement under the dead zone is ignored and accumulates rather than being thrown away.

Every number is a field on `HandMouseConfig`: `pinch_ratio`, `pinch_release_ratio`, `pinch_hold_seconds`,
`click_cooldown_seconds`, `scroll_deadzone_ratio`, `scroll_gain`. To see what a pose does before you
trust it with your mouse:

```bash
python3 tools/gesture_trace.py                       # every scripted pose, transition by transition
python3 tools/gesture_trace.py --pinch-ratio 0.35    # try a tighter threshold
python3 -m kinematic.main --gesture-engine legacy    # the pre-Phase-4 inline pinch/scroll behaviour
```

`--gesture-engine legacy` keeps the previous gesture map (right click = thumb–index *and*
index–middle pinched; scroll = index-finger sweep while the hand holds still) for comparison or if you
preferred it. It has no hold time or hysteresis.

## Smoothing and reach

Two knobs decide how the pointer *feels*:

- **Smoothing** (`smoothing_mode`, default `one_euro`, `one_euro_min_cutoff` 4.0, `one_euro_beta` 0.0): a
  4 Hz low-pass on the cursor position, which is what turns noisy, quantised landmarks into a smooth
  path. `--smoothing-mode ema` uses the older fixed-factor filter instead.
  - The tuning is measured, not tasteful: see `tools/measure_smoothness.py` (no camera, no mouse needed),
    which reports how often the cursor *holds* still, how big each emitted step is, and how far it trails
    your hand. The defaults put a still hand's step at ~2.6 px and free of held frames on a slow sweep.
  - `one_euro_beta` is 0 on purpose. One Euro's speed term is defined in pixel units, and the cursor is
    fed screen pixels: the old `beta=8` saturated the cutoff (alpha 0.96 even at 0.5 px/frame) and
    **switched the filter off entirely**, which is a large part of why the pointer used to feel stuttery.
    Raising `beta` above ~0.1 brings that behaviour back.
  - Heavier smoothing (lower `min_cutoff`, e.g. 1-2) is calmer at rest but trails your hand by 20-40 px on
    slow movement; lighter (8+) is more responsive and shakier. `tools/measure_smoothness.py --min-cutoff
    N` prints the trade-off for any setting.
  - A tween on each `pyautogui` move is **not** an option: it only interpolates above 0.1 s, and that
    blocks the loop for ~171 ms per move (5 frame intervals) — see AGENTS.md gotcha 8.12.
- **Reach** (`frame_margin`, default 0.15, absolute mode only): the inner 70% of the camera frame maps onto
  the whole screen, so the screen corners are reachable without moving your hand to the edge of the frame.
  `screen_margin` keeps a margin free at the screen edges instead.
- **`deadzone`** (default **0.0**, absolute mode): set it above 0 and the cursor *holds still* until your
  hand moves that far (in frame units — 0.02 is ~50 px of screen), then jumps. That hold-then-jump is
  exactly the "snapping to a grid" feel, so it is off; jitter is the filter's job instead. It is a radius,
  not a per-axis box.

## Pointing

The cursor is driven by your **wrist/palm**, not your fingertip, and by default it follows the hand's
**position in the frame** — the inner 70% of the frame covers the whole screen, so every screen edge and
corner is directly reachable:

- **Move your hand** → the cursor goes to the matching spot on the screen, corner to corner.
- **Curl or spread your fingers, pinch, or rotate your wrist** → the cursor does not move at all. Fingers
  only ever produce gestures (pinch = click, two-finger pinch = scroll).
- **Move your hand to a comfortable position** → the cursor goes there too. There is no clutch in either
  mode, so if you would rather have the cursor move *by displacement* like a mouse (where you can lift
  and reposition, and the return stroke drags the cursor back), use:

```bash
python3 -m kinematic.main --pointer-mode relative      # cursor moves by hand displacement (mouse-like)
python3 -m kinematic.main --pointer-reference wrist    # the wrist joint alone (even more stable than the palm)
python3 -m kinematic.main --pointer-reference index    # the old fingertip behaviour (fingers drag the cursor)
python3 -m kinematic.main --smoothing-mode ema         # the fixed-factor filter instead of the 1-euro filter
```

Which hand point is used, how far the reach extends and how much hand jitter is ignored are all fields on
`HandMouseConfig` (`pointer_reference`, `pointer_sensitivity`, `frame_margin`, `deadzone`,
`pointer_deadzone`, `relative_scale_by_hand`). `pointer_deadzone` is worth tuning for your own hands and
camera: `python3 tools/measure_stability.py --seconds 15` reports the per-frame wrist/palm jitter in your
setup next to the configured deadzone (it only reads landmarks — it cannot move your mouse). How far the
cursor travels per unit of hand movement is on the dial — see [Sensitivity](#sensitivity--the-dial)
above, or set `pointer_sensitivity` directly.

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
| `smoothing` | `0.35` | EMA factor, used only with `--smoothing-mode ema`; raise it for a snappier pointer, lower it for less jitter |
| `one_euro_min_cutoff` | `4.0` | the cursor filter's cutoff (Hz): lower is smoother and trails more |
| `one_euro_beta` | `0.0` | One Euro speed term, in pixel units — above ~0.1 it saturates the cutoff and disables smoothing |
| `pointer_sensitivity` | `1.0` | the dial's gain: cursor travel per unit of hand travel (the floor is 1.0) |
| `sensitivity_step` | `1.15` | how much one notch of the dial changes it, as a multiplier |
| `sensitivity_min` / `sensitivity_max` | `1.0` / `60.0` | the ends of the dial; the floor keeps the screen reachable |
| `sensitivity_reach_floor` | `1.0` | the gain below which the screen edges cannot be reached |
| `sensitivity_keys_up` / `_down` / `_reset` | `= + ]` / `- _ [` / `0` | which keys turn the dial |
| `show_sensitivity` | `True` | draw the dial's value and bar on the preview |
| `pointer_mode` | `"absolute"` | `absolute` maps the hand's position to the screen; `relative` moves by displacement |
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

Run from the repository root (the tests import `kinematic.gesture`). **256 tests**, none of which can
touch your mouse, camera or display: they inject a recording output adapter, fake landmarks and fake
timestamps. They cover the pure gesture geometry, the capture/control loop, the Phase 1 safety layer
(paused startup, no output while paused, adapters, keys), the landmark conversion and preview drawing,
the geometry/mapping/smoothing components, the pointer reference (finger curl must not move the cursor),
and the gesture state machine — relative thresholds, hysteresis, hold time, cooldown and re-arm, finger
distinction, scroll priority, tracking loss and pause.

## Project context

[`AGENTS.md`](AGENTS.md) is the working brief for contributors and AI coding agents: verified
environment facts, the safety contract, the nine-phase roadmap with per-phase prompts and status, the
architectural decisions, and gotchas (including why `mediapipe` is pinned).
