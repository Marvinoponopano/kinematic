# AGENTS.md — Kinematic (webcam hand mouse)

Context file for AI coding agents working in this repo. Read this before touching code.
Last updated: 2026-10-10 (Phases 1–4 of the roadmap complete — see §2 "Phase status" and §5).

> If your tool looks for a different filename (`CLAUDE.md`, `.cursorrules`, `.windsurfrules`,
> `.github/copilot-instructions.md`), copy or symlink this file to that name — the content is
> tool-agnostic.

---

## 1. What this project is

A Python computer-vision app that turns webcam hand gestures into real-time desktop mouse
control: cursor movement from the index finger, pinch to click, a second pinch for right
click, and finger movement for scroll. Pipeline is OpenCV + MediaPipe Hands, with a
smoothing stage to damp jitter, and PyAutoGUI as the mouse backend.

The goal of the roadmap below is to turn the current single-file prototype into layered,
testable components where **gesture logic is pure, hardware-independent, and unit-tested
against synthetic landmarks**, and where real mouse control is an explicit opt-in rather
than the default.

---

## 2. Current state (verified 2026-10-10)

```
kinematic/
├── .gitattributes
├── .gitignore
├── AGENTS.md
├── README.md
├── requirements.txt
├── kinematic/
│   ├── __init__.py        # re-exports HandMouseConfig, is_pinch, is_right_click_gesture, smooth_point
│   ├── capture.py         # ThreadedCamera: background-thread grabber handing out the newest frame
│   ├── config.py          # HandMouseConfig dataclass (behavioural + performance + safety + pointer + smoothing fields)
│   ├── gesture.py         # pure helpers: distances, midpoints, pinch distances, hand scale/span,
│   │                      #   pinch/right-click detection, clamping, mirroring, EMA, pointer helpers
│   ├── landmarks.py       # Landmark / HandLandmarks + the one MediaPipe conversion point
│   ├── mapping.py         # Region + map_point_to_screen (frame region -> screen region)
│   ├── output.py          # MouseOutput adapters: RealMouseOutput / RecordingMouseOutput (mock)
│   ├── preview.py         # landmark skeleton drawing for the debug preview (drawn with cv2, no protobuf)
│   ├── recognizer.py      # Phase 4: the pure gesture state machine (states, relative thresholds,
│   │                      #   hysteresis, hold/cooldown/re-arm, scroll priority, event list)
│   ├── runtime.py         # HandMouseRuntime: pipeline coordinator + the pause gate on all output
│   ├── smoothing.py       # cursor filters: ExponentialMovingAverage, OneEuroFilter (both resettable)
│   └── main.py            # CLI + HandMouseApp facade over the runtime
└── tests/
    ├── test_gesture_logic.py     # 4 cases: pure gesture helpers
    ├── test_capture_pipeline.py  # 9 cases: perf defaults, cursor-move guard, camera error path
    ├── test_phase1_safety.py     # 19 cases: no output while paused, adapters, keys, no-camera paths
    ├── test_phase2_landmarks.py  # 15 cases: protobuf conversion, hand protocol, skeleton drawing
    ├── test_phase3_geometry.py   # 23 cases: distances, midpoints, clamping, region mapping
    ├── test_phase3_smoothing.py  # 21 cases: EMA + 1-euro behaviour, jitter, latency, reset
    ├── test_phase4_recognizer.py # 54 cases: transitions, hysteresis, hold/cooldown/re-arm, priority, scroll
    ├── test_phase4_runtime.py    # 18 cases: the recogniser wired in, the pause gate, the legacy engine
    ├── test_sensitivity_dial.py  # 38 cases: the pointer-gain dial, its keys, its bar, its wiring
    └── test_pointer_reference.py # 21 cases: wrist/palm reference, curl must not move the cursor, scroll guard
tools/
├── compare_smoothing.py         # seeded jitter-vs-lag comparison of the cursor filters
├── gesture_trace.py             # scripted poses through the recogniser: transition table, no camera
├── measure_loop.py              # measure the loop rate with the mouse intercepted (nothing moves)
└── measure_stability.py         # per-frame wrist/palm jitter vs pointer_deadzone (no output adapter at all)
```

Git: published as a private repo, `origin` = `github.com/Marvinoponopano/kinematic`, branch `main`.
Everything above is committed except the changes in the working tree at the time you read this; run
`git status` before assuming either way.

### Verified working commands (run from the repo root)

```bash
python3 kinematic/main.py                   # starts PAUSED: SPACE controls the mouse, ESC/q quits
python3 -m kinematic.main                   # same, module invocation (see gotcha 8.1)
python3 -m kinematic.main --active          # start controlling the mouse straight away
python3 -m kinematic.main --dry-run --no-preview   # whole pipeline, zero output, headless-safe
python3 -m kinematic.main --pointer-reference wrist    # use the wrist joint instead of the palm
python3 -m kinematic.main --pointer-mode absolute      # old position-mapped pointing
python3 -m kinematic.main --smoothing-mode ema         # the pre-Phase-3 fixed-factor filter
python3 -m kinematic.main --gesture-engine legacy      # the pre-Phase-4 inline pinch/scroll tests
python3 -m kinematic.main --sensitivity 8              # start the pointer gain somewhere else (dial: ] / [ / 0)
python3 -m unittest discover -s tests       # 222 tests, all pass
python3 tools/compare_smoothing.py          # seeded jitter vs lag table for the cursor filters
python3 tools/gesture_trace.py              # every scripted pose through the recogniser, transition by transition
python3 tools/measure_loop.py --synthetic-hand --active --fingers-only   # the finger-curl regression demo
python3 tools/measure_loop.py --synthetic-hand --active --synthetic-pinch two_finger   # the scroll pose in the real loop
python3 tools/measure_loop.py --synthetic-hand --active --dial-notches 6  # turn the dial and measure the travel change
python3 tools/measure_stability.py --seconds 15          # tune pointer_deadzone against your own hand
```

### Verified interpreter / dependency facts

| Thing | Value |
| --- | --- |
| Interpreter on PATH | `/usr/bin/python3` → **Python 3.9.6** (CommandLineTools) |
| User site-packages | `~/Library/Python/3.9/lib/python/site-packages` (all deps live here, no venv) |
| numpy | 2.0.2 |
| opencv-python | 5.0.0 |
| mediapipe | **0.10.14** |
| pyautogui | 0.9.54 |
| Screen used in tests | 1710 x 1112 (from `pyautogui.size()`) |
| Camera | FaceTime camera, negotiates 1280x720 @ 30 fps; **30 fps is a hard ceiling** (see §2 Performance) |
| Loop rate | 28.8 fps with a hand in frame, 30 fps idle (was 5.8 fps before the performance pass) |

Also present but not used by this project: Homebrew `python3.13` / `python3.14`, `~/.local/bin/python3.11`.

### Known gaps in the current prototype (read from the code, not yet fixed)

- ~~Real mouse control is live on startup~~ — **fixed in Phase 1.** Every side effect now goes through
  `HandMouseRuntime` behind the `can_emit` gate, the app starts paused, and `--dry-run` swaps in a
  recording adapter so the pipeline can run with no desktop side effects at all.
- ~~The cursor followed the index fingertip~~ — **fixed 2026-10-10.** Fingers curling or spreading
  dragged the cursor around while the hand itself was still. The cursor now follows the wrist/palm
  (see §2 "Pointer reference"), so finger articulation cannot move it.
- ~~**No scroll mode.**~~ **Replaced in Phase 4** — `runtime.handle_scroll()` (index-finger
  movement, gain hardcoded `-900`) still exists as the legacy path, but the default engine scrolls
  from the two-finger pose instead, with a dead zone, a configurable gain and click suppression. See
  §2 "Gestures".
- ~~**Duplicated pinch logic.**~~ **Resolved in Phase 4 for the default engine.** The recogniser
  computes pinch from `gesture.pinch_distances()` divided by hand scale; the inline L1 test
  (`abs(dx)+abs(dy) < click_threshold`) remains only in the legacy engine, and
  `gesture.is_pinch()`/`is_right_click_gesture()` are still used by nothing but their own tests.
- ~~**Thresholds are absolute landmark distances**~~ — **fixed in Phase 4.** `pinch_ratio` and its
  release ratio are multiples of the hand's scale, so a pinch engages at the same physical pose near
  and far from the camera (tested: the same pose on a 0.10-scale and a 0.20-scale hand behaves
  identically). The legacy fields `click_threshold`/`right_click_threshold`/`scroll_threshold` are
  still in the config for that path.
- ~~The absolute mapping was not calibrated~~ — **fixed in Phase 3.** `pointer_scale × clamp` (which
  skewed the reachable area off-centre) is replaced by `mapping.map_point_to_screen()`: the inner 70% of
  the frame maps onto the whole screen, with configurable frame and screen margins.
- ~~Tracking loss left the smoothing state behind~~ — **fixed in Phase 3.** The pointer anchor re-anchors,
  the gesture flags reset, and now `cursor_filter.reset()` clears the filter too, so a reacquired hand
  neither jumps nor glides.
- **The loop is covered only through fakes.** `runtime.run()` is tested with a fake camera and fake
  landmarks; MediaPipe's own output, the real camera path and the click/scroll branches are not
  asserted in the suite (they are exercised manually).
- **No packaging** (no `pyproject.toml`) and no logging.
- **One duplicated helper:** `gesture.is_pinch()` is exported and tested but unused by the pipeline,
  which still computes pinch inline with an L1 metric (Phase 6 owns the switch). `gesture.clamp()` is now
  used — by `clamp_point()` and the pointer target clamp.

### Landmarks and preview (Phase 2, finished 2026-10-10)

MediaPipe's protobuf no longer reaches the pipeline. `capture`/`runtime` convert it exactly once,
at the boundary, in `landmarks.hands_from_results()`:

- `Landmark` (frozen dataclass: x/y/z, normalized) and `HandLandmarks` (21 landmarks + handedness +
  score). `HandLandmarks` supports `len()`, iteration and indexing, so every pure helper in
  `gesture.py` accepts it unchanged — and accepts a test's `SimpleNamespace` list too.
- The conversion **copies** (a test mutates the "proto" afterwards to prove it), and an incomplete
  detection is flagged `is_complete=False`, which the runtime ignores rather than letting a 3-landmark
  hand raise `IndexError` deep in the geometry.
- `preview.draw_hands()` draws the 21-point skeleton with plain `cv2` line/circle calls — no
  `drawing_utils`, no protobuf — so the overlay is testable by asserting pixels on a black frame, and
  an incomplete hand draws nothing. `show_landmarks` (default on) switches it off.

Nothing else in the package imports protobuf-shaped objects or MediaPipe drawing utils now.

### Smoothing and screen mapping (Phase 3, finished 2026-10-10)

Two new pure modules, both importable without a camera:

- `smoothing.py`: `ExponentialMovingAverage` (the previous inline EMA, kept bit-for-bit) and
  `OneEuroFilter` (Casiez et al.): the cutoff opens with hand speed, so a still hand is smoothed hard
  and a moving hand is followed closely. Both are timestamp-driven and resettable; `filter_from_config`
  is the only place `smoothing_mode` is interpreted.
- `mapping.py`: `Region` + `map_point_to_screen()`. Absolute mode now maps the **inner 70% of the frame**
  (`frame_margin=0.15`) onto the whole screen, so the screen corners are reachable without moving your
  hand to the frame edge, where landmarks get unreliable. `pointer_scale` fell from 1.8 to 1.0 because
  the region mapping replaced its job — the old `pointer_scale × clamp` skewed the reachable area
  off-centre, which was the documented calibration gap.

**The default filter was chosen from measurements**, not taste. `tools/compare_smoothing.py` (16 seeds,
300 frames each, jitter amplitude ±0.01, 30 fps, results in normalized landmark units, lag converted to
pixels at 1710 px and `pointer_sensitivity` 5.0):

| Filter | Jitter (stdev) | 90% step | Lag @0.25/s | Lag @1.0/s |
| --- | --- | --- | --- | --- |
| `ema` 0.35 (previous default) | 0.00268 | 6 frames | 132 px | **529 px** |
| `ema` 0.50 | 0.00336 | 4 frames | 71 px | 285 px |
| `one_euro` min 1.0, beta 0.0 | 0.00179 | 13 frames | 340 px | 1360 px |
| `one_euro` min 1.0, beta 2.0 | 0.00185 | 2 frames | 227 px | 454 px |
| **`one_euro` min 1.0, beta 8.0 (new default)** | **0.00201** | **2 frames** | **113 px** | **151 px** |
| `one_euro` min 2.0, beta 6.0 | 0.00254 | 2 frames | 97 px | 170 px |

Only `min=1.0 beta=8.0` and `min=2.0 beta=6.0` are Pareto-better than the old EMA (less jitter *and*
no more lag at either speed); beta=8 wins on jitter and on the fast-motion case that matters most in
relative mode (151 px vs 529 px of trailing). Hence `smoothing_mode="one_euro"`, `one_euro_beta=8.0`.
`--smoothing-mode ema` restores the previous feel for anyone who preferred it.

Tracking loss now resets the filter (`cursor_filter.reset()`), which is the phase's last requirement:
after a lost-and-reacquired hand the next sample passes straight through, so the cursor neither glides
from stale state nor keeps drifting on a hand that is no longer moving.

### Pointer reference (fixed 2026-10-10)

The cursor used to be mapped from the **index fingertip**, so any finger articulation dragged it across
the screen even when the hand did not move. It is now driven by the **wrist/palm**, which barely moves
when the fingers do, and in `relative` mode it moves only by *hand displacement* — so finger curl, wrist
rotation and changing distance from the camera all leave the cursor where it is.

| Config field | Default | Meaning |
| --- | --- | --- |
| `pointer_mode` | `"relative"` | `relative`: the cursor is shifted by how far the hand moved (like a mouse). `absolute`: the cursor maps to the hand's position in the frame |
| `pointer_reference` | `"palm"` | `palm` = mean of wrist + the four knuckles; `wrist` = the wrist joint alone (most stable); `index` = the old fingertip behaviour |
| `pointer_sensitivity` | `5.0` | relative gain: screen widths of cursor travel per frame width of hand travel (5.0 ⇒ crossing the screen takes ~20% of the frame width of hand movement) |
| `pointer_deadzone` | `0.003` | hand displacement below this is treated as jitter and held back (it accumulates, so slow movement is not lost) |
| `relative_scale_by_hand` | `False` | divide displacement by apparent hand size, so the feel does not change as you move nearer/further from the camera |
| `scroll_requires_still_hand` | `True` | index-finger scrolling only acts while the hand holds still, so pointing cannot scroll the page |

Measured with `tools/measure_loop.py` (real camera, real MediaPipe, mouse calls intercepted), synthetic
hand parked with only the index finger moving — the reported bug's input:

| Reference | Cursor moves per 120 frames |
| --- | --- |
| `palm` (default) | **0** |
| `wrist` | **0** |
| `index` (what the app used to do) | **105** |

And with the whole hand sweeping, the `palm` reference produces 95 moves — the cursor follows deliberate
hand movement.

Known trade-offs, stated plainly:

- **Relative mode drifts and has no clutch.** Moving the hand back to a comfortable position moves the
  cursor back with it. `pointer_mode="absolute"` avoids that but needs the frame-to-screen mapping
  calibrated (still a Phase 3/9 job), and re-anchors nothing.
- **`pointer_deadzone` is a guess until measured against a real hand.** `tools/measure_stability.py`
  reports the per-frame jitter of the wrist/palm in your setup next to the configured deadzone; tune the
  field so it sits above that jitter and below deliberate movement.
- **MediaPipe gives no separate "hand moved" signal.** The displacement used for the scroll guard is the
  same measured wrist/palm delta, so a slow deliberate translation below the deadzone still reads as
  "still".

### Performance (measured 2026-10-10)

Profiled on this machine before optimising. The headline finding: **`pyautogui`'s default `PAUSE = 0.1`
is slept before every public call**, so each `moveTo`/`click`/`scroll` cost **113 ms instead of 11 ms**.

| Stage | Measured cost | Note |
| --- | --- | --- |
| `cap.read()` | 33.5 ms | the camera's frame interval — identical at 1920x1080, 1280x720, 640x480, 320x240, and with 60 fps requested |
| `pyautogui.moveTo` | **113 ms** with `PAUSE=0.1`, **11.5 ms** with `PAUSE=0` | 100 ms of it is sleep |
| `pyautogui.click` / `scroll` | 12.7 ms / 11.3 ms | with `PAUSE=0` |
| `cv2.imshow` + `waitKey` | 20.7 ms full 1080p, 13.9 ms at 0.5 scale | display cost, not vision |
| MediaPipe `hands.process` | 9.6 ms (complexity 1), 8.1 ms (complexity 0) | **independent of frame size** — it resizes to 256x256 internally |
| `flip` + `cvtColor` | 0.3 ms at 1080p | negligible |

Result (processed frames per second, measured inside the loop with the same stages either side):

| Scenario | Before | After |
| --- | --- | --- |
| hand in frame | **5.77 fps** (173 ms/frame) | **28.80 fps** (34 ms/frame) |
| idle, no hand | 18.72 fps | 29.98 fps |

The camera's 30 fps is now the limit; no setting raises it (that is a hardware fact, not a code one).
Changes, all config-driven on `HandMouseConfig`: `mouse_pause`, `min_move_pixels`, `model_complexity`,
`preview_scale`, plus threaded capture in `capture.py`.

Measuring this correctly matters: timing the loop from outside is wrong — macOS window teardown adds
~1.5 s, which made a 30 fps loop look like 22 fps. Take frame timestamps **inside** the loop:
`tools/measure_loop.py` does exactly that and intercepts the mouse calls so nothing on the desktop moves
(`python3 tools/measure_loop.py`, add `--synthetic-hand --active` to measure the cursor path without
putting a hand in front of the camera).

### Gestures: the Phase 4 recogniser (2026-10-10)

`recognizer.py` is the only gesture logic the default pipeline uses. It is pure — no camera, no
MediaPipe, no mouse, no clock of its own: `update(Observation(landmarks, timestamp))` returns a
`GestureUpdate(state, events)`, and `runtime.apply_gestures()` is the single place that turns an event
into an adapter call (still behind `can_emit`).

**The gesture map, and what changed.** Poses are resolved by explicit priority
(`GESTURE_PRIORITY` = scroll → left → right), so a pose that could mean two things never does:

| Pose | State | Event | Before Phase 4 |
| --- | --- | --- | --- |
| thumb–index pinch, middle finger away | `LEFT_PINCH` | `left_click` | same finger, but an absolute L1 threshold and no hold time |
| thumb–middle pinch, index away | `RIGHT_PINCH` | `right_click` | **changed** — the old right click was thumb–index *and* index–middle pinched |
| thumb touches both index and middle | `TWO_FINGER_SCROLL` | `scroll_begin` / `scroll` / `scroll_end` | **changed** — the old scroll was index-finger movement with a still-hand guard, and this pose used to fire a right click |
| no hand in frame | `TRACKING_LOST` | `tracking_lost` / `tracking_restored` | new — the runtime used to reset silently |
| paused | `PAUSED` | `paused` / `resumed` | new — the state machine models the pause key |

**The rules** (each has its own tests):

- **Relative thresholds** — `pinch_ratio` (0.40) and `release_ratio` (1.4, so release above 0.56) are
  multiples of `hand_scale`; the same pose on a small and a large hand gives the same verdict.
- **Hysteresis, per pair** — engage below 0.40, release only above 0.56, so a hand resting at the
  threshold cannot chatter (100 alternating frames → exactly one click).
- **Hold time** — `pinch_hold_seconds` (0.08 s) must elapse before a pose registers; the clock starts
  on the pose's own first frame and a flicker restarts it.
- **Cooldown and re-arm** — two clicks are never closer than `click_cooldown_seconds` (0.18 s), and a
  pair must be released above the release threshold before it can fire again. A click inside the
  cooldown is dropped, not deferred.
- **Scroll reference** — the two-finger midpoint is the reference, reset on entry (so movement during
  the hold cannot jump the page); movement below `scroll_deadzone_ratio` × hand scale is ignored and
  accumulates; `scroll_gain` (900) converts hand widths into scroll units (negative = page down).
- **No gesture switch without neutral** (`require_neutral_between`) — going straight from one pinch to
  another is ignored; release first. The same rule is why a pinch held across a resume or a tracking
  loss is *consumed* rather than queued: those pairs come back disarmed and must be pinched again.

**Evidence** (real camera, mouse calls intercepted and counted, 150 frames ≈ 5 s, synthetic hand
tapping 5 times in the clip):

| Engine | Synthetic pose | move | click | scroll |
| --- | --- | --- | --- | --- |
| `recognizer` (default) | left pinch | 121 | **5** | 0 |
| `recognizer` | right pinch | 121 | **5** | 0 |
| `recognizer` | two-finger | 121 | **0** | **5** |
| `legacy` | left pinch | 121 | 6 | 0 |
| `legacy` | two-finger | 121 | 6 | **0** |

Exactly one click per deliberate tap under the recogniser; the legacy path fires one *or two* clicks
per tap (left and right can both trigger on the same pose) and never scrolls from the two-finger pose.
With `--fingers-only` — the finger-curl input that used to drag the cursor — the recogniser engine
still reports **0 moves** over 120 frames, so the pointer fix is unaffected.

`python3 tools/gesture_trace.py` prints these transitions without a camera: use it to see what a pose
does and to sanity-check the thresholds against your own hand (`--pinch-ratio`, `--hold`, `--cooldown`).

### The sensitivity dial (added 2026-10-10)

The pointer gain can be changed **while the app is running**, from the preview window's keyboard — no
restart, no config editing, no UI:

| Key | Effect |
| --- | --- |
| `]` `=` `+` | one notch faster (`pointer_sensitivity × sensitivity_step`) |
| `[` `-` `_` | one notch slower (`÷ sensitivity_step`) |
| `0` | back to the value the app started with |

The dial is drawn on the preview as a value plus a log-scale bar, e.g. `sens  5.00 [#####-----]  ]
faster  [ slower  0 reset`, and the startup banner repeats it. All the fields are on
`HandMouseConfig`: `sensitivity_step` (1.15 per notch — multiplicative, so one turn feels the same at
either end), `sensitivity_min` / `sensitivity_max` (0.5 → 60, clamped rather than wrapped), the three
key tuples, and `show_sensitivity` for the overlay.

Programmatic equivalents, for scripts and for anyone driving the app without the window:
`runtime.sensitivity` (read), `runtime.set_sensitivity(value)` (clamped), `runtime.adjust_sensitivity(n)`
and `runtime.reset_sensitivity()`. `runtime.summary()` reports the value the session *ended* at, the
value it started at, and how many times the dial moved.

Why it is safe to turn mid-gesture: the gain multiplies each frame's *hand displacement* before it is
added to the cursor position (`pointer_target()`), so changing it cannot rescale movement that is
already on screen — the cursor does not jump, and turning the dial emits nothing.

**Evidence** — `tools/measure_loop.py` turns the dial through the real key handler halfway through a run
with the mouse intercepted, then reports the mean cursor travel before and after (160 frames each,
~30 fps):

| Command | Sensitivity | Mean cursor travel |
| --- | --- | --- |
| `--dial-notches 6` | 5.000 → **11.565** (2.31×) | 35.2 px → **72.8 px** (2.07×) |
| `--dial-notches -6` | 5.000 → **2.162** (0.43×) | 35.2 px → **15.4 px** (0.44×) |
| `--sensitivity 20 --dial-notches -10` | 20.000 → **4.944** (0.25×) | 106.1 px → **35.2 px** (0.33×) |

The travel figures track the gain closely but not exactly: the synthetic hand translates
sinusoidally, so the frames before and after the turn cover different parts of its sweep (and moves
under `min_move_pixels` are skipped). The exact relationship — travel = displacement × gain × screen
width — is pinned by tests in `tests/test_sensitivity_dial.py` (`test_travel_matches_the_documented_units`,
`test_double_the_gain_doubles_the_travel`, `test_turning_the_dial_mid_motion_does_not_rescale_what_is_already_on_screen`).

### Phase status — which phases are actually implemented (audited 2026-10-10)

**Phases 1, 2, 3 and 4 are complete (5/5 each).** Before Phase 1 this repo was a prototype that drove
the real mouse from startup with none of the safety layer in place; that is fixed — see §5 for what
each phase landed. Phases 5–9 are not implemented in the shape the plan asks for, though the
prototype's subject matter (clicks, scroll) exists in some form; read each phase's own `Status:` line
below before starting work.

*How this was audited:* every non-`.git` file was listed and read; the source was grep'd for each
phase's marker concepts (`pause`, `mock`, `record`, `adapter`, `dry`, `Enum`, `state`, `hyster`,
`putText`, `fps`, `try:`, `finally`, `logging`, `calibrat`); the deliverable bullets were checked one by
one against the package modules; and `python3 -m unittest discover -s tests` was run. The "0 hits" and
per-phase counts below are from that pre-Phase-1 audit; the Phase 1 row and the Tests row have been
updated since.

| Phase | Status | Exists | Missing |
| --- | --- | --- | --- |
| 1 — Foundation & safety | ✅ **5/5 (2026-10-10)** | Modules split: `capture.py`, `gesture.py`, `output.py` (adapters), `runtime.py` (coordinator), `main.py` (CLI); `requirements.txt`; 222-test suite; **paused startup** with SPACE toggle and ESC/q quit; **`RecordingMouseOutput` mock adapter** used by tests and `--dry-run`; macOS permission docs in README | — |
| 2 — Capture pipeline | ✅ **5/5 (2026-10-10)** | `capture.py`'s threaded `ThreadedCamera` (`isOpened()` error, buffer size, device-error reporting); `cv2.flip` mirroring, BGR→RGB, `mp.solutions.hands`, invalid-frame error, exception-safe teardown (the runtime's `try/finally`); **`landmarks.py`** (Landmark/HandLandmarks, the single protobuf conversion point, incomplete detections flagged); **preview overlay** of the loop rate *and* the 21-point skeleton (`preview.py`, drawn with cv2, `show_landmarks`) | — |
| 3 — Geometry/mapping/smoothing | ✅ **5/5 (2026-10-10)** | `gesture.py`: distances, `midpoint`/`midpoints`, `pinch_distances`, `hand_scale` + `hand_span`, `clamp`/`clamp_point`, `mirror_point`, EMA; `mapping.py`: `Region` + `map_point_to_screen` (frame region → screen region, clamping, mirroring); `smoothing.py`: EMA + 1-euro filters with `reset()`, wired into the runtime with a tracking-loss reset; 44 deterministic tests for distances, clamping, smoothing, latency and reset | Midpoints are available but not yet *used* by a gesture; One Euro parameters are tuned for this camera rather than for everyone |
| 4 — Gesture state machine | ✅ **5/5 (2026-10-10)** | `recognizer.py`: `HandState` (neutral / left pinch / right pinch / two-finger scroll / paused / tracking-lost), `EventKind` + `GestureEvent`, `GestureThresholds`/`Observation`/`GestureUpdate`, `GESTURE_PRIORITY`; thumb–index vs thumb–middle pinch detection from `pinch_distances()` ÷ `hand_scale`; per-pair hysteresis, hold time, cooldown, re-arm (`reset(disarm=True)`), scroll-priority resolution, "no switch without neutral", scroll reference/dead-zone/gain; wired into the runtime via `apply_gestures()` behind `can_emit`, with the paused/tracking states driven by the pause key and the frame loop; 72 new tests (54 recogniser + 18 runtime) | The thresholds are reasoned defaults, not measured against a real hand (Phase 9 owns calibration); horizontal scroll is computed and reported (`GestureEvent.secondary`) but not yet emitted |
| 5 — Cursor movement | 🟡 **3/5 + 1 partial** | Palm/wrist-relative pointer pipeline (see §2 "Pointer reference") with smoothing, clamp and `min_move_pixels`; **dry-run and real adapters** (`output.py`); **opt-in gate** (starts paused, `--dry-run`); paused/bounds suppression; re-anchor on reacquire; the explicit *tracking-lost* state Phase 4 defined (`HandState.TRACKING_LOST`, emitted once per loss) | A real rate limit/coalescer and the absolute mapping's calibration |
| 6 — Left & right clicks | 🟡 **3/5 + 1 partial** | Left click, right click (thumb–middle), shared cooldown, release-then-re-arm, **scroll precedence** and last-action suppression — all in the recogniser, whose 72 tests assert exact click counts; **suppressed while paused** (Phase 1's `can_emit` gate and the recogniser's own `PAUSED` state, including a pinch held across a resume); the legacy inline path (L1 pinch test, `is_right_click_gesture`) is still reachable via `--gesture-engine legacy` | Last-action status on the preview (hovering a click's target is not tracked); the legacy path still re-implements the pinch inline |
| 7 — Two-finger scroll | 🟡 **3/5 + 1 partial** | Two-finger (index+middle) pose with the midpoint as reference, reference reset on entry, dead zone, configurable `scroll_gain` (900) and direction (negative = page down), **click suppression while scrolling** (the pose scrolls instead of clicking); horizontal component computed and reported | Emitting the horizontal component, direction inversion as a config field, and rate limiting; the *legacy* index-finger scroll path is unchanged (and still non-conforming) |
| 8 — Pause/runtime/feedback | 🟡 **4/5 + 1 partial** | `runtime.py` coordinator; configurable keyboard toggle (`toggle_pause_keys`) with paused startup; PAUSED/ACTIVE **and the live gesture state** on the preview (`f"… fps  {gesture_state}"`); the recogniser's `PAUSED`/`TRACKING_LOST` states, kept in step by `_sync_gesture_pause()` and the no-hand branch; exception-safe teardown (`try/finally`) | The optional *gesture* pause, and tests for toggle idempotence, recovery and window-close cleanup |
| 9 — Calibration/hardening | ⛔ **0/5 + 1 partial** | `HandMouseConfig` with the tunables this phase asks for — camera, margins, smoothing, pinch thresholds, cooldown, hotkeys (`toggle_pause_keys`/`quit_keys`/the dial's key tuples) and sensitivity, now adjustable at **runtime** via the dial (`set_sensitivity`/`adjust_sensitivity`, `]`/`[`/`0`, `--sensitivity`, `--sensitivity-step`) | Mirroring as a config field, diagnostics, logging (0 hits), profiling, packaging (`pyproject.toml`), final setup docs, and the settings-do-not-persist gap: the dial's value is per-session (pass `--sensitivity` to pin it) |
| Tests | 222 passing | gesture helpers; config defaults; cursor-move guard; camera error path; the Phase 1 safety suite; the pointer-reference suite (curl must not move the cursor, gain/clamp/deadzone maths, re-anchor, scroll guard — now pinned to `gesture_engine="legacy"`); the landmark suite (protobuf conversion, incomplete hands, skeleton pixels); the Phase 3 geometry suite (distances, midpoints, clamping, region mapping); the Phase 3 smoothing suite (EMA + 1-euro behaviour, jitter, latency, tracking-loss reset); the Phase 4 recogniser suite (54 cases) and runtime suite (18 cases); the sensitivity-dial suite (38 cases: clamping, multiplicative notches, the keys, the bar, the CLI flags, cursor travel scaling) | MediaPipe's own detections, the real camera path, and the legacy click/scroll branches beyond the pinned cases |

**What this means for the next session:**

- **Phases 1–4 are done.** Capture, geometry, mapping, smoothing, the safety layer and the gesture
  state machine are in place and tested, so every remaining phase can be built on a pipeline that
  cannot hurt the desktop and does not need a camera in the test suite.
- **Phase 5 is the next real step**: it is now the only phase whose deliverable list still has a
  *missing* item that nothing else covers — a real rate limit/coalescer — plus the absolute mapping's
  calibration. The tracking-lost suppression it was waiting on exists (`HandState.TRACKING_LOST`).
- **Retrofit, don't rewrite.** Every phase so far added a module and switched the runtime over while
  leaving the previous path reachable (EMA next to One Euro, index next to palm, absolute next to
  relative, `legacy` next to `recognizer`). Keep doing that: it is what makes the changes reviewable
  and revertible, and it is why the 5 pre-existing gesture tests could be pinned instead of deleted.
- **Phase 4 changed the default behaviour on purpose** (the gesture map in §2 "Gestures": right click
  is now thumb–middle, the two-finger pose scrolls instead of right-clicking). The old behaviour is one
  flag away (`--gesture-engine legacy`) if the user prefers it — but do not flip the default back
  without being asked.
- **Phases 7–9 still own the rough edges:** the horizontal scroll component is computed but not
  emitted, `scroll_gain`'s direction has no config field, there is no logging or packaging, and the
  recogniser's thresholds are reasoned defaults rather than measured against a real hand.

---

## 3. Working agreement (how this user wants it done)

1. **Strictly additive.** "Only add the parts that are missing, do not edit already existing code if
   you do not need to." Never rewrite or reorganize working code to make room for a new feature —
   add a new module/function/branch beside it, and keep existing behaviour intact. If an existing
   behaviour genuinely must change, say so explicitly and get agreement first.
2. **Work in staged phase briefs** (Goal / Requirements / Testing), one phase at a time, and stop at
   the end of a phase for review. Do not run ahead into the next phase's changes.
3. **Every number in a report comes from a scripted run.** For anything stochastic or timing-related,
   report aggregates over many seeded runs (not a single run) and save results machine-readably with
   the seed and full config recorded alongside. Never present a hand-computed or illustrative number
   as a result.
4. **Say plainly what the platform cannot do.** When a phase names a real product/standard as the
   target, implement the referenced behaviour faithfully and state which parts the underlying engine
   genuinely cannot match, instead of approximating silently.
5. **Never fabricate output.** If a dependency, camera, or permission blocks a run, report the blocker
   with the actual error text rather than a plausible-looking result.
6. **Do not commit or push** unless asked. Do not touch `~/Library/Python/3.9/site-packages` or
   install global packages; use a project venv if a new dependency is unavoidable.
7. **Match existing style**: type hints everywhere, `from __future__` not needed on 3.9, stdlib
   `dataclasses` for config, `unittest` (not pytest) for tests, 4-space indent, ~100 col lines.

---

## 4. Safety contract (non-negotiable for this project)

Moving a user's real mouse can click destructive things. Every phase must preserve all of these:

- **Default to no real output.** Tests, examples and default startup use a mock/recording output
  adapter. Real cursor/keyboard output is opt-in via an explicit flag or constructor argument.
- **Tests must never move the cursor, click, or scroll.** Any test that could reach PyAutoGUI is a
  bug — inject the recording adapter instead.
- **Start paused (or in dry-run) with a documented, keyboard-driven pause/resume and an emergency
  quit** that works even mid-gesture. Phase 8 requires this to be verified explicitly.
- **`pyautogui.FAILSAFE`**: the current code disables it; that is only acceptable for the real-output
  path, and the phase that owns it must state the recovery story (what the user presses to stop it).
- **Suppress output when paused, when tracking is lost, or when the gesture is ambiguous.**
  Never emit a click from a sustained pinch, and never emit clicks while scrolling.
- **macOS permissions are runtime requirements**: Camera access for OpenCV/MediaPipe, and
  Accessibility (System Settings → Privacy & Security → Accessibility) for the Terminal/IDE process
  before PyAutoGUI can control the pointer. Document both; diagnose denial in Phase 9.

### How each requirement is enforced today (Phase 1, implemented 2026-10-10)

| Requirement | Mechanism |
| --- | --- |
| default to no real output | `HandMouseRuntime` starts paused; `update_cursor`/`handle_clicks`/`handle_scroll` all check `can_emit` and the runtime is the only code that touches a `MouseOutput`. `--dry-run` (or `dry_run=True`) swaps the adapter to `RecordingMouseOutput` |
| tests never reach the machine | `tests/test_phase1_safety.py` injects `RecordingMouseOutput`, a fake camera and fake hands; the suite needs no camera, no display and no permissions |
| keyboard pause/resume + emergency quit | SPACE or `p` toggles (`config.toggle_pause_keys`), ESC or `q` quits (`config.quit_keys`), Ctrl-C always works from the terminal. The preview window keeps keyboard focus while the pointer is being driven |
| suppress output while paused / untracked | paused frames are still analysed and previewed, but nothing is emitted; a pinch already held when the user resumes does **not** click (the rising edge belonged to the paused period) |
| recovery story for `FAILSAFE` | `FAILSAFE` stays off by default; the documented stop is SPACE to pause, ESC/q to quit, Ctrl-C as the always-available fallback. `RealMouseOutput(failsafe=True)` re-enables pyautogui's corner abort for anyone who wants it |

---

## 5. Roadmap

Nine phases. Each phase lists deliverables, the verification bar, and a ready-to-paste prompt, and each
heading carries a `Status:` line taken from the audited table in §2 — read it before starting.
**Phase 1 is complete**; Phases 2–9 are not, and the prototype already covers slices of 2/3/5/6. Real
desktop output is no longer live by default: it waits for the SPACE key, per the Phase 1 safety layer.

### Phase 1 — Project foundation and safety

**Status: ✅ 5/5 complete (2026-10-10).** All five deliverables landed. Modules: `capture.py` (camera),
`gesture.py` (pure geometry/smoothing), `output.py` (mouse adapters), `runtime.py` (coordinator),
`main.py` (CLI + facade). Safety: the app starts paused with a keyboard toggle and quit keys, and
`RecordingMouseOutput` is the mock adapter that tests and `--dry-run` use. The README documents the
macOS Camera and Accessibility permissions.

**Verification evidence** (reproduce with `python3 tools/measure_loop.py`, which intercepts the mouse calls
and counts them, plus the subprocess check below):

- 32 tests pass, none of which can reach pyautogui: they inject `RecordingMouseOutput`, a fake camera
  and fake landmarks.
- Real app, real camera, real MediaPipe, mouse calls counted: **0** calls over 45 frames while paused
  and **0** with no hand in frame after resuming.
- Real loop with a synthetic hand: **0** calls over 60 frames while paused, then 36 moves + 4 clicks
  over 100 frames after pressing SPACE.
- Real CLI (`python3 kinematic/main.py --dry-run --no-preview`) run as a subprocess for 5 s: exit 0,
  `frames=110 output_calls=0 loop_fps=30.28`, recorded `{move: 0, click: 0, scroll: 0}`.

- Create the Python environment, dependency manifest, package structure, and test setup.
- Define camera, gesture, smoothing, runtime, and mouse-output modules.
- Start paused by default with an emergency pause/exit mechanism.
- Add a mock output adapter so tests never move the real cursor.
- Document macOS Camera and Accessibility permissions.

**Verification:** create the environment, run tests, and confirm no-camera/mock startup produces no
mouse events.

**Prompt:**
> Implement Phase 1 of the webcam hand mouse roadmap. Set up the Python project structure, dependency
> manifest, test configuration, README setup and safety contract described in the plan. Do not add
> real mouse-control behavior yet; include a mock output adapter and a no-op startup path, then run
> the focused tests.

**Behaviour change this phase introduced (do not "fix" it back):** the app used to seize the cursor the
moment it started. It now starts **paused** — the preview reads
`PAUSED - press SPACE to control the mouse` — and emits nothing until SPACE (or `p`) is pressed.
`--active` restores start-immediately, `--dry-run` selects the recording adapter, and `--no-preview`
runs without a window. This is the one place where Phase 1 deliberately changed existing behaviour at
runtime; the old code path is still reachable, just not the default.

### Phase 2 — Webcam and MediaPipe pipeline

**Status: ✅ 5/5 complete (2026-10-10).** All five deliverables landed. Capture is a threaded grabber in
`capture.py` with device-error reporting and a buffer size of 1; the runtime owns teardown (`try/finally`);
mirroring, RGB conversion, missing-camera and invalid-frame handling are all in place; the preview overlays
the loop rate *and* the 21-point landmark skeleton; and the internal representation (`landmarks.py`) means
no protobuf-shaped object reaches the pipeline.

Evidence: 15 new tests cover the conversion (including that it copies rather than aliases, and that an
incomplete hand is ignored instead of raising), the `HandLandmarks` protocol, and the skeleton pixels on a
black frame. Live: `python3 tools/measure_loop.py --synthetic-hand --active --preview` runs the overlay in
the real loop at ~26.8 fps versus 29.8 fps without the preview window — the window costs ~3 fps, the
drawing itself is in the noise.

- Implement OpenCV camera capture and cleanup.
- Add MediaPipe hand detection and landmark extraction.
- Handle RGB conversion, mirroring, missing cameras, and invalid frames.
- Add a debug preview with landmarks and FPS.
- Define an internal landmark representation.

**Verification:** run the webcam preview manually and confirm landmarks, mirroring, FPS, and shutdown
work without mouse events.

**Prompt:**
> Implement Phase 2 of the webcam hand mouse roadmap. Add the MediaPipe/OpenCV capture and landmark
> pipeline with a debug preview only, including resource cleanup and no-camera handling. Keep all
> desktop control disabled and run the phase tests.

### Phase 3 — Geometry, mapping, and smoothing

**Status: ✅ 5/5 complete (2026-10-10).** All five deliverables landed, and the phase's own requirement —
"write deterministic tests for distances, clamping, smoothing, tracking-loss reset, and latency-sensitive
behaviour; do not connect them to real mouse output" — is met: 44 new tests, none of which can reach the
mouse (the recording adapter and the `can_emit` gate are unchanged).

- Bullet 1 (distances, pinch distances, midpoints, hand scale): `point_distance`, `pinch_distances`,
  `midpoint`, `midpoints`, `hand_scale`, `hand_span`.
- Bullet 2 (normalize index-finger coordinates): `normalize_screen_point`, `mirror_point`, and
  `HandLandmarks.point()` for landmark-space coordinates.
- Bullet 3 (map to a configurable screen control region): `mapping.Region` +
  `map_point_to_screen(frame_region, screen_region)`, wired into absolute mode with `frame_margin=0.15`.
- Bullet 4 (EMA or One Euro): both, in `smoothing.py`, with the default chosen from the measured table in
  §2 (One Euro, min_cutoff 1.0, beta 8.0 beat the previous EMA on jitter, step response and lag).
- Bullet 5 (reset filters on tracking loss): `cursor_filter.reset()` in the runtime's no-hand branch, with
  two tests (no glide after reacquire; a still hand does not keep drifting).

- Add landmark distances, pinch distances, midpoints, and hand-scale calculations.
- Normalize index-finger coordinates.
- Map webcam coordinates to a configurable screen control region.
- Add exponential moving average or One Euro smoothing.
- Reset filters when tracking is lost.

**Verification:** use synthetic landmark sequences to test clamping, noise reduction, tracking-loss
reset, responsiveness, and coordinate mapping.

**Prompt:**
> Implement Phase 3 of the webcam hand mouse roadmap. Add pure, hardware-independent geometry,
> coordinate normalization, screen-region mapping, and configurable jitter smoothing components.
> Write deterministic tests for distances, clamping, smoothing, tracking-loss reset, and
> latency-sensitive behavior; do not connect them to real mouse output.

### Phase 4 — Gesture recognition and state machine

**Status: ✅ 5/5 complete (2026-10-10).** `kinematic/recognizer.py` holds the whole phase: `HandState`
(neutral, left pinch, right pinch, two-finger scroll, paused, tracking lost), `GestureEvent`/`EventKind`,
`GestureThresholds`, `Observation`/`GestureUpdate` and `GESTURE_PRIORITY`. It is pure — no cv2, no
MediaPipe, no mouse, no wall clock (a test asserts the module never mentions them) — and the runtime
feeds it from `apply_gestures()` and translates its events, still behind `can_emit`. The phase's own
verification bar is met: deterministic transition tests with fake timestamps for sustained pinches,
repeated pinches, ambiguous input, tracking loss and simultaneous gestures.

- Bullet 1 (the six states): `HandState`, driven by the pause key (`set_paused`) and the frame loop.
- Bullet 2 (thumb/index and thumb/middle pinches): `pinch_distances()` ÷ `hand_scale()`, per pair.
- Bullet 3 (relative thresholds): `pinch_ratio` 0.40 with `release_ratio` 1.4 — a test feeds the same
  pose on a 0.10-scale and a 0.20-scale hand and asserts the same outcome.
- Bullet 4 (hysteresis, cooldowns, hold times, release/re-arm): per-pair engage/release bands, the
  `hold_seconds` clock, the `cooldown_seconds` window, and `_armed` per pair so a pose that was already
  held (across a resume or a tracking loss) is consumed instead of queued.
- Bullet 5 (scroll priority): `GESTURE_PRIORITY` puts the two-finger pose first, and the two-finger
  pose emits `scroll_begin`/`scroll`/`scroll_end` and no click — verified in the real loop (5 scrolls,
  0 clicks per clip) as well as in the tests.

**Behaviour change this phase introduced (deliberate — do not revert without being asked):** the
gesture map changed. A right click is now a thumb–middle pinch (it used to require thumb–index *and*
index–middle), and the two-finger pose now scrolls (it used to fire a right click). Both are stated in
§2 "Gestures" along with the evidence; `--gesture-engine legacy` restores the previous behaviour and
keeps its own 5 pinned regression tests green.

**Evidence:** 72 new tests (54 recogniser cases + 18 runtime cases), 184 in total at that point (222
after the sensitivity dial landed); `python3
tools/gesture_trace.py` prints every scripted transition (hold, cooldown, chatter, scroll, tracking
loss, pause, right pinch) without a camera; and `python3 tools/measure_loop.py --synthetic-hand
--active --frames 150` with the mouse intercepted counted exactly one click per deliberate tap for the
pinch poses, 5 scrolls and 0 clicks for the two-finger pose, and 0 moves for the finger-curl input.

- Define neutral, left-pinch, right-pinch, two-finger-scroll, paused, and tracking-lost states.
- Detect thumb/index and thumb/middle pinches.
- Use relative hand-scale thresholds instead of raw pixel thresholds.
- Add hysteresis, cooldowns, hold times, and release/re-arm behavior.
- Give scroll gestures priority over click gestures.

**Verification:** add deterministic transition tests using fake timestamps for sustained pinches,
repeated pinches, ambiguous input, tracking loss, and simultaneous gestures.

**Prompt:**
> Implement Phase 4 of the webcam hand mouse roadmap. Build a pure gesture recognizer and
> timestamp-driven state machine for left pinch, right pinch, simultaneous two-finger scroll, neutral,
> tracking loss, and pause states. Use relative thresholds, hysteresis, cooldown/debounce rules,
> explicit gesture priority, and comprehensive deterministic transition tests. Do not emit real mouse
> events.

### Phase 5 — Cursor movement

**Status: 🟡 3/5 (updated 2026-10-10).** The adapter split landed with Phase 1 (`RealMouseOutput` /
`RecordingMouseOutput`), cursor output is opt-in (the app starts paused; `--dry-run` emits nothing at
all), moves smaller than `min_move_pixels` are skipped, and the tracking-lost suppression the phase was
waiting on exists now (`HandState.TRACKING_LOST`, emitted once per loss, with the filter and anchor
reset). What remains: a genuine rate limit/coalescer, and the absolute mapping's calibration.

- Map filtered coordinates to the active screen.
- Add dry-run and real cursor adapters.
- Suppress movement while paused, untracked, or outside valid bounds.
- Rate-limit or coalesce movement events if needed.
- Keep real output opt-in.

**Verification:** test screen corners, center mapping, mirroring, bounds, pause suppression, and
tracking loss with the recording adapter. Then perform a cautious manual cursor test.

**Prompt:**
> Implement Phase 5 of the webcam hand mouse roadmap. Connect the filtered coordinate pipeline to a
> screen mapper and guarded cursor adapter, keeping dry-run recording as the default. Add tests for
> screen mapping, mirror orientation, bounds, pause/tracking-loss suppression, and movement
> throttling, then provide a cautious manual verification path for real cursor movement.

### Phase 6 — Left and right clicks

**Status: 🟡 3/5 + 1 partial (updated 2026-10-10, after Phase 4).** All five bullets now hold for the
default engine — left/right clicks are one-shot events from the recogniser, a sustained pinch clicks
once, release-and-re-arm is enforced (including the `_armed` flag that survives a resume or a tracking
loss), and clicks are suppressed while scrolling, paused and untracked. What is still missing is
last-action status on the preview, and the *legacy* engine still computes its pinch inline with an L1
test instead of `gesture.is_pinch()`.

- Convert gesture transitions into one-shot left/right click events.
- Prevent repeated clicks during sustained pinches.
- Require release and re-entry before another click.
- Suppress clicks during scrolling, pausing, and tracking loss.
- Display optional last-action status.

**Verification:** use the recording adapter to confirm exact click counts, cooldown behavior, gesture
distinction, and suppression rules.

**Prompt:**
> Implement Phase 6 of the webcam hand mouse roadmap. Wire debounced left- and right-click transition
> events into the output adapter with cooldown, release/re-arm, scroll precedence, pause suppression,
> and tracking-loss suppression. Extend recording-adapter tests and give me a minimal manual click
> verification procedure.

### Phase 7 — Two-finger scrolling

**Status: 🟡 3/5 + 1 partial (updated 2026-10-10, after Phase 4).** The two-finger gesture now exists and
is the default: the index/middle **midpoint** is the reference, it resets on scroll entry, the dead zone
and `scroll_gain` are config fields, direction is documented (hand down = page down), and the pose is
suppressed as a click (scroll outranks the clicks by construction). **Horizontal movement is not yet
implemented end to end**: the recogniser computes and reports it (`GestureEvent.secondary`) but the
runtime only emits the vertical amount. Direction inversion as a config field and rate limiting are also
missing, and the legacy index-finger path is unchanged (still non-conforming).

- Use the index/middle pinch midpoint or another documented reference point.
- Support vertical and horizontal movement.
- Add dead zones, sensitivity, direction inversion, and rate limiting.
- Reset the reference point when entering scroll mode.
- Prevent scroll gestures from producing clicks.

**Verification:** test vertical, horizontal, diagonal, stationary, reversed, and tracking-loss
sequences with recorded scroll events.

**Prompt:**
> Implement Phase 7 of the webcam hand mouse roadmap. Add two-finger pinch scrolling with a resettable
> motion reference, vertical/horizontal direction handling, dead zone, sensitivity, rate limiting, and
> click suppression. Add deterministic gesture-to-scroll tests using the recording adapter, then
> provide a controlled manual scroll test.

### Phase 8 — Pause, runtime, and feedback

**Status: 🟡 4/5 + 1 partial (updated 2026-10-10, after Phase 4).** Phase 1 built the coordinator and the
pause path: a configurable keyboard toggle, paused startup, PAUSED/ACTIVE feedback on the preview, and
cleanup in a `finally` block that also runs when the loop raises. Phase 4 added the status content the
phase asked for — the live gesture state (`neutral`, `left_pinch`, `two_finger_scroll`,
`tracking_lost`, `paused`) is drawn next to the loop rate, and `_sync_gesture_pause()` keeps the state
machine honest when the pause key is used. Remaining: the optional *gesture* pause, and explicit tests
for toggle idempotence, recovery and window-close teardown.

- Add a configurable keyboard toggle, with startup paused.
- Optionally add a gesture pause only after the keyboard control is reliable.
- Build the runtime coordinator connecting capture, landmarks, smoothing, gestures, and output.
- Add paused, active, and tracking-lost status.
- Guarantee cleanup on normal exit, window close, and exceptions.

**Verification:** test fake frames, fake timestamps, fake keyboard events, recording output, toggle
idempotence, immediate suppression, recovery, and cleanup.

**Prompt:**
> Implement Phase 8 of the webcam hand mouse roadmap. Add the paused-by-default toggle, runtime
> coordinator, status feedback, exception-safe cleanup, and integration tests with fake inputs and a
> recording output adapter. Keep real desktop control explicitly opt-in and verify the emergency pause
> path.

### Phase 9 — Calibration and hardening

**Status: ⛔ 0/5 + 1 partial.** Only the config surface exists, and it is short of this phase's list
(no mirroring, margins, sensitivity or hotkey fields). No diagnostics, logging, profiling or packaging.

- Add configuration for camera, mirroring, control margins, smoothing, pinch thresholds, cooldown,
  sensitivity, and hotkeys.
- Add diagnostics for camera access, MediaPipe startup, screen size, and macOS Accessibility
  permissions.
- Add structured logging and performance metrics.
- Profile the frame loop and reduce unnecessary work.
- Add packaging and final setup documentation.

**Verification:** test fresh setup, denied permissions, unavailable cameras, low light, partial
tracking, sustained gestures, repeated launch/cleanup, and full regression checks.

**Prompt:**
> Implement Phase 9 of the webcam hand mouse roadmap. Add configuration/calibration, diagnostics,
> logging, performance hardening, packaging/setup documentation, and end-to-end regression tests for
> the hand-mouse pipeline. Preserve dry-run defaults and document the final macOS verification
> checklist.

---

## 6. Architectural decisions (already agreed — do not relitigate)

- Use **PyAutoGUI behind an adapter** initially; keep **pynput replaceable** as the backend.
- Keep **gesture recognition pure and timestamp-driven** — no camera, no wall-clock `time.time()` in
  the logic layer; take a timestamp argument so tests can use fake clocks.
- Use **synthetic landmarks and fake clocks** for deterministic tests. Never require a real webcam in
  the test suite.
- **Start paused** and require explicit opt-in for real mouse output.
- Use a **keyboard hotkey as the primary pause mechanism**; gesture-pause is optional and comes later.
- Treat **macOS Camera and Accessibility permissions** as documented runtime requirements, and
  diagnose them in-app rather than failing with a bare OpenCV warning.
- **`pyautogui` may be imported by exactly one module: `output.py`.** Everywhere else goes through a
  `MouseOutput`, which is what keeps dry runs, tests and the paused path honest.
- **Only `runtime.py` may call an output adapter**, and only after checking `can_emit`.
- **The cursor reference is the wrist/palm, never the fingertip.** Fingertips are the most
  articulation-dependent points on the hand; anything a finger does must be able to affect *gestures*
  (clicks, scroll) but never the pointer position. See §2 "Pointer reference" for the modes and gains.
- **MediaPipe's protobuf is converted once, at the boundary** (`landmarks.hands_from_results()`). Nothing
  outside `capture.py`/`runtime.py` may see a MediaPipe object, and pure modules must keep accepting any
  landmark sequence (a `HandLandmarks` or a test's `SimpleNamespace` list).
- **Filters are pure and timestamp-driven**: they take the timestamp as an argument, own no clock, hold
  resettable state, and never touch the mouse. A filter that cannot be reset is a filter that will glide
  after a tracking loss.
- **Gesture logic is one pure module plus one translation point.** `recognizer.py` decides (states,
  priority, hysteresis, hold/cooldown, re-arm) and never sees a mouse; `runtime.apply_gestures()` is the
  only place that turns a `GestureEvent` into an adapter call, and it checks `can_emit` first. Adding a
  gesture means adding an event kind and a state — not touching the pointer path.
- **One pose means one thing.** Poses are resolved by an explicit `GESTURE_PRIORITY` order (scroll,
  then left, then right) rather than by whichever comparison happens to run first, so the old "this
  pose clicks *and* scrolls" overlap cannot come back.
- **Every gesture rule is stated as a number in the config** (`pinch_ratio`, `pinch_release_ratio`,
  `pinch_hold_seconds`, `click_cooldown_seconds`, `scroll_deadzone_ratio`, `scroll_gain`) and enforced in
  the pure module, so a rule can be changed and regression-tested without a camera.
- **Behaviour changes ship with a switch, not a delete.** `gesture_engine` selects `recognizer`
  (default) or `legacy`; the old path keeps its tests. Same discipline as EMA/One-Euro, palm/index and
  absolute/relative.
- **Live settings are config fields, not a second source of truth.** The sensitivity dial writes
  `config.pointer_sensitivity` (clamped) through `runtime.set_sensitivity()`, so the config, the
  preview readout, the CLI flag and the summary always agree; the keys are just a keyboard front end for
  that one setter. Anything tunable at runtime should follow this shape: one field, one setter, one
  readout.

---

## 7. Pre-report checklist

Before reporting a phase complete:

- [ ] `python3 -m unittest discover -s tests` passes (all old + new tests).
- [ ] `python3 kinematic/main.py` and `python3 -m kinematic.main` both still start.
- [ ] No test or default configuration can move the real cursor, click or scroll.
- [ ] New behaviour is additive; existing working behaviour is unchanged.
- [ ] Any measured number is backed by a recorded run (seed + config stored), not a guess.
- [ ] The §2 phase status table and this phase's own `Status:` line are updated to match what you
      actually built (they are the next session's starting point).
- [ ] Anything the platform cannot do is stated plainly, not papered over.

---

## 8. Gotchas (each one cost real debugging time — keep them fixed)

**8.1 Relative imports when running `kinematic/main.py` directly.**
`main.py` starts with a guard (lines 1–12) that, when `__package__` is empty (script invocation),
puts the project root on `sys.path` and sets `__package__ = "kinematic"`. Without it, Python raises
`ImportError: attempted relative import with no known parent package`. **Do not delete or "clean up"
this block** — both invocation styles are supported on purpose. It has already been reverted once in this
workspace, by the parallel GitHub Copilot worktree (see 8.8). Do **not** double-fix it either: that
copy solves the same problem differently (an `if __package__:` import fallback).

**8.2 Pin mediapipe; the legacy solutions API is gone in newer versions.**
`main.py` uses `mp.solutions.hands`, the legacy Solutions API. Verified by installing each version and
checking `hasattr(mp, 'solutions')`: **`0.10.14` and `0.10.21` still have it; `0.10.30`, `0.10.35` and
`1.1.0` do not** (0.10.21 is the newest release below 0.10.30, so it is the last one with the legacy
API). `requirements.txt` **used to say** `mediapipe>=0.10.14`, which resolves to 1.1.0 on a fresh install
and breaks the app the moment the capture pipeline is imported; it is now pinned. If you ever need to
change it, either keep a pin (`mediapipe==0.10.14`, or `>=0.10.14,<=0.10.21`) or migrate to the Tasks API
(`HandLandmarker`) — whichever you do, state it explicitly in the phase report. **Resolved on `main`:**
`requirements.txt` pins `mediapipe==0.10.14`, matching both the interpreter's installed version and the
Copilot branch's choice.

**8.3 `PYTHONPATH` poisoning in agent shells.**
An agent shell can have `PYTHONPATH` pointing at another interpreter's site-packages (observed:
a Python 3.14 venv). The result is a confusing `numpy._core._multiarray_umath` / "compiled module
files ... incompatible with cpython-39" failure that looks like a broken install but is not. Check
`echo $PYTHONPATH` and run with `env -u PYTHONPATH python3 ...` when in doubt.

**8.4 Camera access needs a permission grant.**
`cv2.VideoCapture(0)` fails with `OpenCV: not authorized to capture video` + `camera failed to
properly initialize!` until the *host process* (Terminal / IDE / agent) is allowed in
System Settings → Privacy & Security → Camera. The code then raises
`RuntimeError: Could not open camera index 0.` — a permission problem, not a code bug. Verify capture
separately from gesture logic. **Current status on this machine (2026-10-10): granted** —
`cv2.VideoCapture(0)` opens and returns 1920x1080 frames, so the app genuinely runs now. See 8.9 for what
that implies about smoke tests.

**8.5 Accessibility permission for PyAutoGUI.**
PyAutoGUI's mouse calls are silently ignored on macOS until the host process is granted Accessibility.
Any "the cursor doesn't move" report should be triaged against this before touching the mapping code.

**8.6 Tests are `unittest`, run from the repo root, and `tests/` has no `__init__.py`.**
`python3 -m unittest discover -s tests` needs the repo root as cwd so that `from kinematic.gesture
import ...` resolves.

**8.7 Untracked code.**
`kinematic/`, `tests/` and `requirements.txt` were not in git and there was no `.gitignore`; both are
fixed now (`.gitignore` covers `__pycache__/`, `*.py[cod]`, `.venv/`, `.DS_Store`). Still do not assume
`git stash`/`git checkout` protects uncommitted work, and do not commit or push unless asked.

**8.8 Two divergent copies of this project exist, and only one is tracked.**
The checkout you are in (`main`, commit `b500351`) tracks only `.gitattributes` and `README.md` — the
source is untracked here. In parallel there is a **GitHub Copilot worktree branch**
`copilot/worktree-2026-10-10T17-15-28` (worktree dir
`/Users/marvinma/Desktop/VSCode/kinematic.worktrees/copilot-worktree-2026-10-10T17-15-28`, 2 commits on
top of `main`: `479ea8d` "Add core functionality for hand-tracking mouse control", `d15af45` "Refactor
import statements for dynamic package handling and update mediapipe version constraint"). File-by-file
(diffed with `git diff` / `git show branch:path`):

- `kinematic/__init__.py`, `config.py`, `gesture.py`, `tests/test_gesture_logic.py` — **byte-identical**
  to this checkout.
- `kinematic/main.py` — **differs**: that branch fixes the script-import problem with an
  `if __package__:` fallback that imports `kinematic.config` / `kinematic.gesture` absolutely; this
  checkout uses the `__package__ = "kinematic"` guard from 8.1. Both work. Pick one deliberately and
  never merge the two mechanisms.
- `requirements.txt` — was the only other divergence (the branch pinned `mediapipe==0.10.14` while this
  checkout said `>=0.10.14`); **`main` now pins `0.10.14` too** (8.2), so the file matches the branch.
- `README.md` — **differs**: this checkout has the extra "run the entry point directly" paragraph.

Consequences: Copilot may write into this directory again (it already did once, silently reformatting
`main.py` at 13:28 and dropping an edit), editing a file here does not change it there, and untracked
files here are not protected by git. Before starting a phase, state which copy you are working in. The
branch commits are the only tracked record of the prototype, so if you consolidate the two, do it
deliberately rather than by overwriting either side.

**8.9 Never smoke-test this app by launching it.**
With the camera permission granted (8.4), `python3 kinematic/main.py` and `python3 -m kinematic.main`
open the webcam, pop a window and immediately start moving the user's **real** cursor — a "does it still
start?" check is enough to hijack their mouse. Use one of these instead:

```bash
python3 -c "import runpy; runpy.run_path('kinematic/main.py', run_name='probe')"   # imports all, never calls main()
python3 -m unittest discover -s tests
```

Observed 2026-10-10: a clone-verification run of `python3 -m kinematic.main` really did run the live loop
on the user's machine and then exited 0 with no `q` keypress and no traceback — the exit path is not
explained by the code (the only `break` is on `ord("q")`). Treat that as an open question, and as another
argument for Phase 1's explicit pause/quit and Phase 8's exception-safe teardown.

**8.10 Do not write timing tests with a step equal to the threshold.** `0.12 - 0.04` is
`0.07999999999999999` in binary floating point, so a recogniser test that advances 0.04 s per frame and
asserts the 0.08 s hold time has elapsed lands one frame late and looks like a bug in the recogniser.
The Phase 4 tests step 0.05 s (three frames = 0.10 s, comfortably past 0.08) and use exact binary
fractions (0.5625, 0.625) for the boundary assertion itself. The same trap applies to
`tools/gesture_trace.py`'s scripted sequences: give each pose enough frames, or it silently never
commits.

**8.11 A state machine that can re-enter its own transition gets stuck.** The recogniser's first
version left `HandState.TRACKING_LOST` only when something else committed, so a reacquired hand stayed
"lost" and the restore branch fired every frame — resetting the pending pose before it could ever hold,
which made *every* gesture dead after a tracking loss. Three bugs of this shape were found by the
Phase 4 tests, all in the same class: (1) the state must leave `TRACKING_LOST` explicitly on the
restore frame; (2) a pair disarmed by a restore/tracking-loss must re-arm when it is *observed*
disengaged, not only on a release transition, or a hand held open can never pinch again; (3) the
per-pair `_armed` flag is what makes "a pinch held across a resume is consumed" true, and it has to be
set on the fire path *and* on a suppressed click. When adding a transition, ask what clears it — if the
answer is "another gesture", it is wrong.
