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
[`AGENTS.md`](AGENTS.md) — see §2 "Phase status" for what is actually implemented today. **All nine
phases are implemented.** The parts that need real hardware — a camera negotiating frames, a hand in
front of it, the OS obeying a cursor move — are verified by the manual checklist near the end of this
file rather than in the suite, because the suite must never touch your desktop.

The pointer path is tuned by measurement rather than by feel: **487 tests**, and
`python3 tools/measure_smoothness.py` reports how often the cursor holds still, how large each step is
and how far it trails your hand — no camera or mouse required.

Before the first run, ask the app what it can see:

```bash
python3 -m kinematic.main --diagnose      # camera, screen, MediaPipe, permissions: one report, no window
```

## Features
- Cursor control from the wrist/palm (finger articulation cannot move the pointer)
- Left click on a thumb–index pinch, right click on a thumb–middle pinch — **held for as long as you
  hold the pinch, so you can drag** (a quick pinch-and-release is an ordinary click)
- Two-finger scroll — up/down **and sideways** — with a per-axis dead zone, gains, direction switches
  and click suppression
- Hysteresis, hold times, cooldowns and release-to-re-arm, so a gesture fires exactly once
- A last-action readout on the preview, so you can see what was clicked without watching your desktop
- Optional caps on cursor moves and scroll events (`--move-max-hz`, `--scroll-max-hz`, off by default);
  a held-back scroll is accumulated, never dropped
- Optional **gesture pause**: hold an open hand for a second to pause or resume (`--gesture-pause`)
- Paused startup, a keyboard pause toggle, `--dry-run`, and a mock output adapter for tests
- `--diagnose` (camera, screen, MediaPipe, Accessibility), `--log-level` / `--log-file` for a JSON-lines
  session log, and a `kinematic` console script once installed
- OpenCV + MediaPipe pipeline with One Euro smoothing to reduce jitter

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate     # optional but recommended
python3 -m pip install -r requirements.txt             # or: python3 -m pip install -e .
```

`pip install -e .` (or `pip install .`) additionally gives you the `kinematic` command, which is the
same entry point as `python3 -m kinematic.main`. Both invocation styles are supported on purpose. The
only hard pin is `mediapipe==0.10.14` — 0.10.30 and later removed the `mp.solutions.hands` API this
pipeline uses, so a newer release installs cleanly and then cannot find a hand tracker.

Then check the machine and the permissions before anything touches the mouse:

```bash
python3 -m kinematic.main --diagnose                   # all checks; opens the camera once
python3 -m kinematic.main --diagnose --no-camera-probe # skip the only check that turns on the light
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
gesture state (`neutral`, `left_pinch`, `tracking_lost`, …), the sensitivity dial's value, the last
click/scroll it emitted, and the detected hand skeleton, all in its top-left corner.

```bash
python3 -m kinematic.main --active      # start controlling the mouse immediately
python3 -m kinematic.main --dry-run     # run the whole pipeline, emit no mouse events at all
python3 -m kinematic.main --no-preview  # no window (quit with Ctrl-C)
```

Nothing is emitted while paused, and `--dry-run` swaps the real output adapter for a recording one, so
that mode needs no Accessibility permission and is safe to run anywhere.

## Preview orientation

The preview shows the footage **mirrored by default** — the way a webcam selfie looks — with the hand
skeleton drawn on the hand:

```bash
python3 -m kinematic.main --preview-flip horizontal   # the default: mirror, the view the pointer follows
python3 -m kinematic.main --preview-flip none         # exactly what the camera delivers
python3 -m kinematic.main --preview-flip vertical     # camera mounted upside down
python3 -m kinematic.main --preview-flip both         # upside down and mirrored
```

Each mode is named by its flip of the *captured* frame. Two reasons `horizontal` is the default: it is
the view the pointer math is built on (MediaPipe analyses the mirrored frame, so in this view your hand
and the cursor move the same way — move right on screen, cursor goes right), and it is where the
skeleton lines up with your hand.

That last part was a bug until it was fixed: the preview used to draw the *captured* frame while the
landmarks come from the *mirrored* one, so the skeleton was drawn reflected beside your hand instead of
on it. The flip mode now decides both halves together, which is why every mode keeps them aligned. The
preview is display-only: it never changes where the cursor goes.

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
| thumb and index fingertip together | **left button** — press on the pinch, release when you open it (a quick pinch-and-release is an ordinary left click; **hold the pinch to drag**) |
| thumb and middle fingertip together (index away) | **right button** — same press/release behaviour |
| thumb touching both index and middle tips | **scroll** — up/down *and* sideways from the two-finger midpoint |
| no hand in the frame | tracking lost: nothing is emitted, and a pinch held through the loss must be released and re-pinched |
| any pose while paused | nothing, including a pinch that was already held when you resumed |

The rules, so a gesture cannot fire twice or fire by accident:

- A pose must **hold for 0.08 s** before it registers — a passing pinch does nothing.
- Two clicks are never closer than **0.18 s** (`click_cooldown_seconds`), and the pinch must be
  **released** above the release threshold before it can fire again. A click inside the cooldown is
  dropped, not deferred.
- The button stays **down** for as long as the pinch is held, and comes **up** the moment it opens —
  including if you pause, the hand leaves the frame, or a tracking loss happens mid-drag. The hold time
  and cooldown rules decide when a *press* is allowed to fire; they never delay a release, so a button
  cannot be left stuck down.
- **Hysteresis**: a pinch engages below 0.40 of your hand's size and only releases above 0.56, so a hand
  resting near the threshold cannot chatter.
- Thresholds are **relative to hand size** (`hand_scale`, wrist → middle knuckle), so the same pose works
  near and far from the camera.
- The scroll pose **outranks the clicks**, so pointing at a scroll pose cannot also click; scroll
  movement under the dead zone is ignored and accumulates rather than being thrown away.

Every number is a field on `HandMouseConfig`: `pinch_ratio`, `pinch_release_ratio`, `pinch_hold_seconds`,
`click_cooldown_seconds`, `scroll_deadzone_ratio`, `scroll_gain`, `scroll_horizontal_gain`,
`scroll_invert_vertical`, `scroll_invert_horizontal`, `scroll_max_hz`. To see what a pose does before you
trust it with your mouse:

```bash
python3 tools/gesture_trace.py                       # every scripted pose, transition by transition
python3 tools/gesture_trace.py --pinch-ratio 0.35    # try a tighter threshold
python3 -m kinematic.main --gesture-engine legacy    # the pre-Phase-4 inline pinch/scroll behaviour
```

`--gesture-engine legacy` keeps the previous gesture map (right click = thumb–index *and*
index–middle pinched; scroll = index-finger sweep while the hand holds still) for comparison or if you
preferred it. It has no hold time or hysteresis, so it can fire one *or two* clicks per tap. Its pinch
test is no longer a second copy of the geometry: both engines call `gesture.is_pinch()`, with the
legacy path asking for the `l1` metric that reproduces its old behaviour exactly.

### Seeing what it clicks

The preview draws the last action it emitted, how long ago, and how many have fired this session:

```
last: left click  0.3s ago  (4 this session)
```

It appears once the first click or scroll has fired, and it counts *emitted* actions — so in
`--dry-run` it is the readout for "what would have happened" with nothing reaching your desktop. That is
what the [click verification procedure](#verifying-clicks-and-scrolling-5-minutes) below uses.

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

## Two-finger scrolling

Touch your thumb to both the index and middle fingertips, hold for a moment, then move your hand:
the page follows the two-finger **midpoint**, in both directions.

| Hand movement | What is sent |
| --- | --- |
| down / up | vertical wheel scroll (down = page down, the default) |
| left / right | horizontal wheel scroll (right = scroll right) |
| diagonally | both, each on its own axis |

The rules, all of them fields on `HandMouseConfig`:

- **Per-axis dead zone** (`scroll_deadzone_ratio`, 0.05 hand widths): movement under it on an axis is
  ignored and **accumulates** until it is worth sending. It is per axis on purpose — a hand scrolling
  down with a little sideways wobble scrolls vertically only, instead of also dragging the view sideways.
- **Gains** (`scroll_gain` 900 vertical, `scroll_horizontal_gain` 900 horizontal): wheel units per hand
  width, and the two axes are independent — sideways movement often deserves a slower setting than
  vertical. `scroll_horizontal_gain = 0` (`--no-horizontal-scroll`) turns sideways scrolling off
  entirely, which is exactly how the app behaved before this was wired up.
- **Direction switches** (`scroll_invert_vertical`, `scroll_invert_horizontal`): flip either axis.
  Which way macOS moves the *content* for a positive horizontal wheel value comes from Quartz rather
  than from anything this project can measure without scrolling your screen, so if sideways scrolling
  feels backwards, flip `--invert-scroll-horizontal` (one field, nothing else changes).
- **Event rate cap** (`scroll_max_hz`, default 0 = off): caps how many wheel events are sent per second.
  A held-back event is **accumulated, not dropped**, so the event rate falls while the distance scrolled
  stays what your hand asked for; the remainder is flushed when you open your hand. This is only worth
  setting if your machine is slow (see [Cursor move rate limit](#cursor-move-rate-limit-off-by-default));
  the same reasoning as there applies.
- **No clicks while scrolling**, ever — the pose outranks both pinches.
- Leaving the pose, pausing, or losing tracking **ends** the scroll and re-anchors the reference, so
  bringing your hand back cannot jump the page.

Measured with a controlled sweep — a synthetic hand walking a straight line at a known amplitude, wheel
calls intercepted (`tools/measure_loop.py --synthetic-hand --active --synthetic-scroll <axis> --frames 150`):

| Sweep | vertical | horizontal | verdict |
| --- | --- | --- | --- |
| `vertical` | 45 events, −174 units | 0 events | hand down → page down, nothing sideways |
| `horizontal` | 0 events | 45 events, +174 units | hand right → +174, nothing vertical |
| `diagonal` | 45 events, −174 units | 45 events, +174 units | both axes, correct signs |
| `diagonal --scroll-max-hz 10` | 32 events, −174 units | 32 events, +174 units | same distance, fewer events |

(A hand travelling 0.20 frame widths at gain 900 asks for 180 units; the 174 that arrive are the sweep's
own dead-zone remainder, not a loss — the same run emitted 135 units before the rounding bug in
[AGENTS.md](AGENTS.md) §8.16 was fixed.)

```bash
python3 -m kinematic.main --no-horizontal-scroll      # vertical scrolling only
python3 -m kinematic.main --invert-scroll-horizontal  # if sideways scrolling feels backwards
python3 -m kinematic.main --scroll-horizontal-gain 400  # slower sideways movement
python3 -m kinematic.main --scroll-max-hz 10          # batch wheel events (distance preserved)
```

## Cursor move rate limit (off by default)

`move_max_hz` caps how many cursor moves are sent per second. It is **0 (uncapped) at the shipped
settings, and that is measured rather than assumed**: the loop is camera-bound at ~30 fps and one
`pyautogui.moveTo` costs ~11 ms, so exactly one move per frame already happens and the output path is
about a third of the frame budget. Capping below the frame rate is the *hold-then-jump* the smoothness
work removed — fewer, larger steps:

| `--move-max-hz` | moves / 90-frame sweep | mean step | held frames |
| --- | --- | --- | --- |
| off (default) | 90 | 6.2 px | 0 |
| 30 (the loop rate) | 90 | 6.2 px | 0 |
| 20 | 45 | 12.4 px | 45 |
| 10 | 30 | 18.7 px | 60 |
| 5 | 15 | 36.7 px | 75 |

(`python3 tools/measure_smoothness.py --rate-sweep` prints that table — seeded hand, no camera. Note the
effective rate quantises to the frame rate: a 20 Hz cap on a 30 fps loop is really 15 Hz, which is why
20 and 15 give identical rows in the raw sweep.)

What it is genuinely *for* is a slow output adapter. With pyautogui's own `PAUSE = 0.1` (every move costs
~113 ms, longer than the 33 ms frame interval) the loop blocks on the mouse and collapses; a cap under
that rate gives the loop its time back, at the price of the step size above:

| Command | moves | loop rate |
| --- | --- | --- |
| `tools/measure_loop.py --synthetic-hand --active --frames 150` | 150 | **29.7 fps** |
| the same with `--move-max-hz 10` | 47 | 30.0 fps (nothing to fix) |
| `--mouse-pause 0.1` (a 100 ms adapter), no cap | 120 | 6.8 fps |
| `--mouse-pause 0.1 --move-max-hz 5` | 32 | **17.4 fps** |
| `--mouse-pause 0.1 --move-max-hz 3` | 15 | **22.9 fps** |

Two properties worth knowing before you turn it on: the limiter **never queues** — a frame inside the
interval is superseded by the next one that is due, so a burst of frames cannot send a backlog of stale
positions — and it applies to **cursor moves only**, never to clicks or scrolls.

```bash
python3 -m kinematic.main --move-max-hz 10     # cap cursor moves at 10 per second
```

## Gesture pause (off by default)

`--gesture-pause` lets a *pose* change the pause state, so you do not have to reach for the keyboard
mid-gesture. Hold an **open hand** — all four fingers straight, no pinch engaged — for
`gesture_pause_hold_seconds` (1.2 s by default) and it toggles, exactly like `SPACE`:

```bash
python3 -m kinematic.main --gesture-pause                       # hold an open hand to pause/resume
python3 -m kinematic.main --gesture-pause --gesture-pause-hold 2
```

- **One toggle per hold.** After it fires, the pose is disarmed until you take your hand out of it, so a
  long hold cannot flip the state back and forth.
- **It works while paused**, which is what lets the same pose resume — the only thing besides the
  keyboard that does.
- **Nothing is emitted by it**: a gesture pause never clicks or scrolls, and it is not a mouse action, so
  `--dry-run` shows it changing the HUD's `PAUSED`/`ACTIVE` and nothing else.
- **Off by default on purpose.** A pose that can freeze the pointer has to be a deliberate choice: the
  hand passes through open shapes constantly on its way to a pinch, and a brief stretch must not stop the
  mouse. The hold time is what keeps that safe; 1.2 s is long enough that only an intentional hold acts.

If you press `SPACE` while already holding the pose, the in-flight hold is cancelled rather than allowed
to toggle straight back a moment later.

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
| `preview_flip` | `"horizontal"` | which way the preview footage is flipped for display: `horizontal` (mirror, the default), `none` (exactly what the camera delivers), `vertical`, `both` |
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
| `move_max_hz` | `0.0` | cap on cursor moves per second; 0 = uncapped, one move per frame (see above) |
| `scroll_gain` | `900.0` | vertical scroll units per hand width (hand down = page down) |
| `scroll_horizontal_gain` | `900.0` | horizontal scroll units per hand width; **0 disables sideways scrolling** |
| `scroll_invert_vertical` / `scroll_invert_horizontal` | `False` | flip either scroll axis |
| `scroll_deadzone_ratio` | `0.05` | per-axis dead zone, in hand widths (sub-threshold movement accumulates) |
| `scroll_max_hz` | `0.0` | cap on scroll events per second; 0 = uncapped; held-back amounts accumulate |
| `show_last_action` | `True` | draw the last click/scroll, its age and its session count on the preview |
| `gesture_pause_enabled` | `False` | hold an open hand to pause/resume (`--gesture-pause`); see the section above |
| `gesture_pause_fingers` | `4` | how many of the four fingers must be straight for that pose |
| `gesture_pause_hold_seconds` | `1.2` | how long the open hand must be held before it toggles |
| `log_level` | `"warning"` | JSON-lines log level (`--log-level`); the pipeline speaks at info/debug, so the default prints nothing |
| `log_file` | `""` | where the log goes: a path appends to it, empty means stderr |
| `capture_width` / `capture_height` / `capture_fps` / `capture_buffer_size` | 1280 / 720 / 30 / 1 | what is requested from the camera |

## Diagnostics

`--diagnose` answers "why does nothing move?" without opening a window or touching the pointer: it
checks the platform and interpreter, that MediaPipe imports *and* runs a frame through the legacy
`solutions.hands` API, which output adapter this run would use, the screen size, the macOS
Accessibility grant, the effective configuration, and — unless `--no-camera-probe` — that the camera
opens and delivers a frame:

```bash
python3 -m kinematic.main --diagnose
python3 -m kinematic.main --diagnose --no-camera-probe    # skips the only hardware check
```

```
camera         FAIL  could not open index 0: OpenCV: not authorized to capture video
               -> macOS has not granted Camera access to this process. Camera access: System Settings
                  -> Privacy & Security -> Camera -> enable your Terminal/IDE.
```

The exit status is 0 when every check passes, so it can gate a script. A permission problem and a
camera that simply is not there get *different* hints on purpose. The Accessibility check is a
read-only question to the OS (`AXIsProcessTrusted`), never a test move of the pointer, and the camera
probe reports what it saw — resolution, how long the first frame took, the mean brightness (`very dark`
gets its own hint) and whether a hand was in view.

The app also makes this check at startup when it is driving the real pointer, so a missing
Accessibility grant is reported as a sentence rather than as "the cursor does not move".

## Logging

A session can write a structured log, one JSON object per line, for diffing, plotting or filing a bug
report. Nothing is written unless you ask: the pipeline speaks at `info`/`debug` and the default level is
`warning`, so an ordinary run prints its banner and nothing else.

```bash
python3 -m kinematic.main --log-level info                    # to stderr
python3 -m kinematic.main --log-level info --log-file run.jsonl
```

```json
{"time": 1760272471.502, "level": "info", "logger": "kinematic.runtime", "event": "click", "kind": "left", "action": "left click"}
{"time": 1760272473.115, "level": "info", "logger": "kinematic.runtime", "event": "pause", "source": "key", "frames": 61}
{"time": 1760272475.802, "level": "error", "logger": "kinematic.runtime", "event": "loop_error", "error": "RuntimeError('Unable to read frames from the webcam.')", "frames": 92}
```

The event names are the API: `session_start`, `loop_start`, `loop_stop`, `loop_error`, `pause`,
`resume`, `tracking_lost`, `tracking_restored`, `click`, `scroll`, `window_closed`, `summary` (the
session's frames, emitted calls, pause counters and tracking losses) and — at `debug` only — `frame`
(one heartbeat a second, not one per frame) and `move`. That last split is deliberate: the loop is
camera-bound and logging must not become the second thing it waits on, so per-frame lines are dropped at
any level below `debug` and the per-call guard is an attribute check.

## Profiling the loop

`tools/profile_loop.py` prints where a frame's milliseconds actually go — with every mouse call
intercepted and replayed at the current position, so nothing on the desktop moves:

```bash
python3 tools/profile_loop.py --active                      # real camera, real MediaPipe
python3 tools/profile_loop.py --synthetic-hand --active     # no camera either
python3 tools/profile_loop.py --active --preview             # adds the preview's own cost
python3 tools/profile_loop.py --active --mouse-pause 0.1     # what a slow adapter costs
python3 tools/profile_loop.py --synthetic-hand --cprofile 20 # top 20 cumulative functions
```

Measured on this machine (FaceTime camera at 1280x720, 90 frames, unpaused, no preview — **one run**;
MediaPipe alone varies by ±30% between runs, so treat the split between MediaPipe and `output` as
indicative and the frame total as the number to compare like with like):

| stage | calls/frame | mean ms | share | what it is |
| --- | --- | --- | --- | --- |
| capture | 1.72 | 0.003 | 0.0% | `read()` handing back the newest frame (the waiting is on the capture thread) |
| flip | 1.00 | 0.217 | 1.4% | the mirror `cv2.flip` |
| rgb | 1.00 | 0.177 | 1.2% | BGR→RGB for MediaPipe |
| mediapipe | 1.00 | 6.901 | 45.8% | hand detection |
| output | 0.66 | 11.664 | 50.8% | the macOS cursor-move event (11.7 ms each) |
| logic | — | 0.108 | 0.7% | landmarks, mapping, filter, gestures, bookkeeping |
| **frame** | 1.00 | **15.055** | 100% | the whole per-frame pipeline |

with **29.58 processed frames/second** and a **33.8 ms frame interval** — so this loop is *camera-bound*
(the interval is 2.2× the work) and the frame arrival rate, not the pipeline, sets the speed. Adding
`--preview` costs about **3.4 ms more per iteration** here (≈1.7 ms drawing at `preview_scale 0.5` plus
≈1.7 ms in `imshow`, with the window's first frame dominating the mean — the medians are much lower).
Two optimisations were measured and **rejected** in the same pass, which is the point of having the
numbers:

- **Feeding MediaPipe a smaller frame saves nothing.** `hands.process` costs the same at 1280x720 (8.04
  ms) as at 640x360 (7.83), 480x270 (7.86) or 320x240 (7.80) — it resizes to its own 256x256 internally.
  Pre-resizing would add work, not remove it.
- **Bypassing pyautogui's own API saves nothing either.** `pyautogui.moveTo` (11.36 ms) and its
  platform hook `_pyautogui_osx._moveTo` (11.12 ms) are the same event: the cost is macOS's, not
  pyautogui's Python, so reaching into a private hook would buy 2% and couple the adapter to the OS
  backend.

The slow-adapter case, for comparison (`--active --mouse-pause 0.1`, which is pyautogui's *own*
default): `output` goes to 108.6 ms per move, the loop drops to **8.47 fps** and the verdict flips to
*work-bound* — that is the case the `--move-max-hz` cap exists for (see
[Cursor move rate limit](#cursor-move-rate-limit-off-by-default)).

## Verifying clicks and scrolling (5 minutes)

Two halves each: a scripted check that cannot touch your desktop, and a manual one that uses the
preview's last-action line as the readout.

**Scripted — the mouse calls are intercepted and counted.** A synthetic hand taps five times in five
seconds (or walks a straight scroll sweep), and the harness replaces every `moveTo` / `click` / `scroll` /
`hscroll` with a counter:

```bash
python3 tools/measure_loop.py --synthetic-hand --active --synthetic-pinch left --frames 150
python3 tools/measure_loop.py --synthetic-hand --active --synthetic-pinch right --frames 150
python3 tools/measure_loop.py --synthetic-hand --active --synthetic-pinch two_finger --frames 150
python3 tools/measure_loop.py --synthetic-hand --active --synthetic-scroll vertical --frames 150
python3 tools/measure_loop.py --synthetic-hand --active --synthetic-scroll horizontal --frames 150
python3 tools/measure_loop.py --synthetic-hand --active --synthetic-scroll diagonal --frames 150
```

| Run | clicks | scrolls | what the tool prints |
| --- | --- | --- | --- |
| left pinch | 5 | 0 | `pass - one click per deliberate tap (5 clicks for 5 taps)` |
| right pinch | 5 | 0 | the same, with `right click` as the last action |
| two-finger | 0 | 5 | `pass - the scroll pose scrolls and never clicks` |
| scroll sweeps | 0 | both axes | `pass - hand down -> … vertical, no sideways events` (etc.) |

(With `--gesture-engine legacy` the same taps give 6 clicks for 5 taps, and the two-finger pose clicks
instead of scrolling: that is the documented pre-Phase-4 behaviour, not a failure. With
`--invert-scroll-vertical` the scroll verdict deliberately prints `CHECK`, because the check expects the
documented direction.)

**Manual — dry run first.** In `--dry-run` the preview tells you what *would* have been sent while
nothing reaches your desktop, which is the safe way to learn the poses:

```bash
python3 -m kinematic.main --dry-run
```
1. Press `SPACE` (or `p`) to leave `PAUSED`.
2. **Clicks.** Pinch thumb + index and hold ~0.2 s, then open your hand. The preview line
   `last: left click  0.3s ago  (1 this session)` appears, and the count goes up **once per pinch** —
   not once per frame. Repeat with thumb + middle (`right click`).
3. **Scrolling.** Touch the thumb to both index and middle tips, hold, then move the hand **down** and
   **up**: the page should follow the hand (hand down = page down). Then move the hand **left/right**
   without moving it up/down: sideways scrolling, with no vertical movement. A long document or a wide
   page is the easiest place to see both axes.
4. Check the negative cases: a pinch you keep held never clicks twice; the two-finger pose never clicks,
   however you move; opening your hand stops the scroll immediately; taking your hand out of frame and
   back does not jump the page; pinching or scrolling while `PAUSED` does nothing, and a pose you were
   holding does not fire when you resume.
5. **If the direction feels wrong**, that is one field per axis: `--invert-scroll-vertical` (default is
   hand down = page down) and `--invert-scroll-horizontal`. If sideways scrolling is unwelcome, turn it
   off with `--no-horizontal-scroll`. If it scrolls too fast, lower the gain it applies to —
   `--scroll-horizontal-gain 400` for sideways, `scroll_gain` for vertical.

**Manual — for real.** Park the pointer over something harmless first (a text editor or a Finder
window) and keep `SPACE` under your finger:

```bash
python3 -m kinematic.main --active
```
The same checks apply, with the HUD counting them, plus the one that matters most: **one click per
deliberate pinch**, and a scroll that follows the hand. `SPACE` pauses, `ESC`/`q` quits, and `Ctrl-C` in
the terminal always works.

## Final macOS verification checklist

Everything below is what "it works on this machine" means, end to end. The suite covers the logic (487
tests, no camera, no mouse); this list covers the parts that need the hardware and the OS.

**Fresh setup**

- [ ] Python 3.9+ available; `python3 -m pip install -r requirements.txt` (or `pip install -e .`) succeeds.
- [ ] `mediapipe` is 0.10.14 (or ≤ 0.10.21): `python3 -c "import mediapipe as mp; print(mp.__version__)"`.
- [ ] `python3 -m kinematic.main --diagnose --no-camera-probe` reports `ok` for platform, mediapipe,
      output, screen and configuration.

**Permissions**

- [ ] Camera granted to the host process (Terminal/IDE) — `--diagnose` shows the camera check passing,
      with a resolution and a brightness.
- [ ] Accessibility granted — `--diagnose` shows `accessibility  ok  granted to this process`.
- [ ] Denial is diagnosed, not guessed: revoke Accessibility and confirm the startup line names it and
      `--diagnose` exits non-zero.

**Pipeline (nothing on the desktop moves)**

- [ ] `python3 -m kinematic.main --dry-run` starts **paused** and emits nothing; `SPACE` switches it to
      `ACTIVE` in the HUD; `ESC`/`q` quits; the terminal returns to a prompt with no traceback.
- [ ] `python3 -m unittest discover -s tests` → 487 tests, `OK`.
- [ ] `python3 tools/measure_loop.py --synthetic-hand --active --frames 150` → 5 clicks / 5 taps.
- [ ] `python3 tools/measure_smoothness.py` and `tools/profile_loop.py --synthetic-hand --active` run
      with no camera attached.

**Gestures (dry run first, then live over a harmless window)**

- [ ] Left click: thumb + index pinch held ~0.2 s → exactly one `left click` per deliberate pinch.
- [ ] Right click: thumb + middle pinch → one `right click`.
- [ ] Scroll: two fingers down/up → page follows (hand down = page down); left/right → sideways; the pose
      never clicks. Flip `--invert-scroll-vertical` / `--invert-scroll-horizontal` if a direction feels
      backwards; `--no-horizontal-scroll` turns sideways off.
- [ ] Reach: every screen edge and corner is reachable with the hand *stopped* there; the sensitivity
      dial (`]` / `[` / `0`) is drawn and turning it does not move the cursor by itself.
- [ ] Finger curl does not move the cursor: open and close your hand and watch the HUD's cursor position.
- [ ] Pause suppression: pinching or scrolling while `PAUSED` emits nothing, and a pose held across the
      resume does not fire.
- [ ] Tracking: take your hand out of frame and back — the cursor re-anchors (it does not glide or jump
      from stale state) and the HUD reports `tracking_lost` then `neutral`.
- [ ] Movement: press `SPACE` with a hand in frame and confirm the loop rate in the HUD stays ~30 fps.

**The optional gesture pause**

- [ ] `python3 -m kinematic.main --gesture-pause --dry-run` → holding an open hand for 1.2 s toggles
      `PAUSED`/`ACTIVE`; a brief open hand on the way to a pinch does *not*.

**Logging and packaging**

- [ ] `--log-level info` prints one JSON object per line to stderr; `--log-file run.jsonl` writes the
      same lines to a file; `python3 -c "import json;[json.loads(l) for l in open('run.jsonl')]"` parses.
- [ ] `python3 -m pip install -e .` (in a venv) gives a working `kinematic` command.
- [ ] Launch, quit and relaunch five times in a row: the preview closes each time, no camera stays busy,
      and no Python process is left behind.

**Repeatability**

- [ ] Run each scripted harness twice and compare: the numbers should be reproducible (seeded synthetic
      hands, no camera), which is what makes a "before/after" claim checkable.

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

Run from the repository root (the tests import `kinematic.gesture`). **487 tests**, none of which can
touch your mouse, camera or display: they inject a recording output adapter, fake landmarks and fake
timestamps. They cover the pure gesture geometry, the capture/control loop, the Phase 1 safety layer
(paused startup, no output while paused, adapters, keys), the landmark conversion and preview drawing
(including the preview orientation, in `test_preview_orientation.py`), the geometry/mapping/smoothing
components, the pointer reference (finger curl must not move the cursor), the gesture state machine —
relative thresholds, hysteresis, hold time, cooldown and re-arm, finger distinction, scroll priority,
tracking loss and pause — the click path through the recording adapter (exact counts per tap, cooldown,
re-arm, suppression, the last-action readout), the move rate limiter/coalescer
(`test_move_rate_limit.py`), the scroll path on both wheel axes (`test_phase7_scroll.py`), the gesture
pause and the teardown guarantees (`test_phase8_runtime.py`), the diagnostics and the JSON log
(`test_phase9_diagnostics.py`, `test_phase9_logging.py`), and the end-to-end regression suite
(`test_phase9_e2e.py`: denied permissions, unavailable cameras, low light, partial tracking, sustained
gestures, repeated launch/cleanup, and a call-by-call replay of a scripted session).

## Project context

[`AGENTS.md`](AGENTS.md) is the working brief for contributors and AI coding agents: verified
environment facts, the safety contract, the nine-phase roadmap with per-phase prompts and status, the
architectural decisions, and gotchas (including why `mediapipe` is pinned).
