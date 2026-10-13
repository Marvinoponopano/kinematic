# AGENTS.md — Kinematic (webcam hand mouse)

Context file for AI coding agents working in this repo. Read this before touching code.
Last updated: 2026-10-11 (all nine phases implemented; Phases 8 and 9 finished today — see §2 "Phase status" and §5).

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
├── pyproject.toml         # Phase 9: packaging (setuptools, the `kinematic` console script, the mediapipe pin)
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
│   ├── ratelimit.py       # Phase 5/7: MoveRateLimiter (supersede) + ScrollRateLimiter (accumulate)
│   ├── log.py             # Phase 9: JSON-lines logging (configure/get_logger/log_event), silent by default
│   ├── diagnostics.py     # Phase 9: checks for camera, screen, MediaPipe, Accessibility, configuration
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
    ├── test_phase6_clicks.py     # 32 cases: exact click counts per tap, cooldown/re-arm, suppression,
    │                            #   the last-action readout, the shared is_pinch() helper
    ├── test_move_rate_limit.py   # 25 cases: the limiter itself, its wiring, the measured trade-off
    ├── test_phase7_scroll.py     # 42 cases: per-axis dead zone, signs/gains, inversion, both wheel
    │                            #   axes through the adapter, rate cap fidelity, the sweep harness
    ├── test_preview_orientation.py # 18 cases: the preview flip modes and the skeleton landing on the hand
    ├── test_sensitivity_dial.py  # 38 cases: the pointer-gain dial, its keys, its bar, its wiring
    ├── test_pointer_modes.py     # 21 cases: absolute is the default, every edge reachable, the reach floor
    ├── test_pointer_reference.py # 21 cases: wrist/palm reference, curl must not move the cursor, scroll guard
    ├── test_phase8_runtime.py    # 31 cases: the open-hand gesture pause, toggle idempotence, recovery
    │                             #   after a failed run, cleanup on exit / window close / exception
    ├── test_phase9_diagnostics.py # 31 cases: camera/permission/screen/MediaPipe checks, the report,
    │                             #   and the --diagnose CLI path (exit status, --no-camera-probe)
    ├── test_phase9_logging.py    # 18 cases: JSON-lines format, level handling, file logging, no
    │                             #   per-frame lines at info, the runtime's own events
    └── test_phase9_e2e.py        # 19 cases: fresh setup, denied permissions, unavailable camera, low light,
                                  #   partial tracking, sustained gestures, repeated launch/cleanup, and a
                                  #   call-by-call replay of a scripted session
tools/
├── compare_smoothing.py         # seeded jitter-vs-lag comparison of the cursor filters
├── gesture_trace.py             # scripted poses through the recogniser: transition table, no camera
├── measure_loop.py              # measure the loop rate with the mouse intercepted (nothing moves),
│                                #   count clicks per synthetic tap, --move-max-hz / --mouse-pause
├── measure_stability.py         # per-frame wrist/palm jitter vs pointer_deadzone (no output adapter at all)
├── measure_smoothness.py        # held frames / step size / trailing of the emitted cursor (no camera, no mouse),
│                                #   plus --rate-sweep for the move_max_hz trade-off
└── profile_loop.py              # Phase 9: per-stage ms of a frame (capture/flip/rgb/mediapipe/output/preview),
                                 #   --cprofile, --mouse-pause; mouse calls intercepted, nothing moves
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
python3 -m kinematic.main --pointer-mode relative      # the old displacement-based scheme (edges need a sweep)
python3 -m unittest discover -s tests       # 256 tests, all pass
python3 tools/compare_smoothing.py          # seeded jitter vs lag table for the cursor filters
python3 tools/gesture_trace.py              # every scripted pose through the recogniser, transition by transition
python3 tools/measure_loop.py --synthetic-hand --active --fingers-only   # the finger-curl regression demo
python3 tools/measure_loop.py --synthetic-hand --active --synthetic-pinch two_finger   # the scroll pose in the real loop
python3 tools/measure_loop.py --synthetic-hand --active --dial-notches 6  # turn the dial and measure the travel change
python3 tools/measure_loop.py --synthetic-hand --active --sweep-reach 30 --reach-dwell 12  # do all four edges get reached?
python3 tools/measure_stability.py --seconds 15          # tune pointer_deadzone against your own hand
python3 tools/measure_smoothness.py                      # held frames / step size / trailing, no camera at all
python3 tools/measure_smoothness.py --sweep               # the deadzone x filter grid this tuning came from
python3 tools/measure_smoothness.py --min-cutoff 2 --one-euro-beta 8   # feel the trade-offs yourself
python3 tools/measure_smoothness.py --rate-sweep        # what a move_max_hz cap costs (held frames, step size)
python3 tools/measure_loop.py --synthetic-hand --active --frames 150   # clicks per synthetic tap, counted
python3 tools/measure_loop.py --synthetic-hand --active --mouse-pause 0.1 --move-max-hz 3  # the slow-adapter case
python3 tools/measure_loop.py --synthetic-hand --active --synthetic-scroll vertical --frames 150   # scroll direction
python3 tools/measure_loop.py --synthetic-hand --active --synthetic-scroll diagonal --scroll-max-hz 4  # the cap, accumulating
python3 -m kinematic.main --preview-flip none          # preview exactly as the camera delivers it (default: horizontal)
python3 -m kinematic.main --diagnose                   # camera/screen/MediaPipe/Accessibility report, then exit
python3 -m kinematic.main --diagnose --no-camera-probe  # same, without turning the camera light on
python3 -m kinematic.main --gesture-pause              # hold an open hand for 1.2 s to pause/resume
python3 -m kinematic.main --log-level info --log-file run.jsonl   # structured session log (JSON lines)
python3 tools/profile_loop.py --active                 # where a frame's milliseconds go (mouse intercepted)
python3 tools/profile_loop.py --synthetic-hand --active --cprofile 20   # no camera, plus a cProfile dump
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
- **One duplicated helper, resolved in Phase 6:** `gesture.is_pinch()` used to be exported and tested but
  unused by the pipeline, which computed its pinch inline. The default engine works from hand-scale
  relative ratios, and the *legacy* engine now calls `is_pinch(..., metric="l1")` — the one helper, with
  the prototype's metric as an explicit argument rather than a second copy of the arithmetic.

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

**Preview orientation (fixed 2026-10-11).** `process_frame` flips the captured frame before detection
(`cv2.flip(frame, 1)`), so every landmark the preview draws is in *mirrored* coordinates — but
`draw_preview` was handed the captured frame, which drew the skeleton reflected beside the hand, and the
footage itself was the raw camera view while the pointer follows the mirrored one. `preview.py` now owns
both halves of the rule: `display_frame(frame, mode)` picks the orientation shown and
`landmark_flip(mode)` the mirror the landmarks need, with `HandMouseConfig.preview_flip`
(`--preview-flip`, default `"horizontal"` = the webcam mirror). All four modes keep the skeleton on the
hand; the readout text is drawn *after* the flip so it is never mirrored; the 18 cases in
`tests/test_preview_orientation.py` pin each mode with a marker-block hand at a known asymmetric world
position (skeleton centroid within ~3 px of the marker, and >100 px off if the display flip is removed).

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
| `one_euro` min 1.0, beta 8.0 (then the default) | 0.00201 | 2 frames | 113 px | 151 px |
| `one_euro` min 2.0, beta 6.0 | 0.00254 | 2 frames | 97 px | 170 px |

> **Superseded for the default mode (2026-10-10).** That table's filter input is in *landmark* units,
> which is not what the runtime feeds it: the runtime smooths **screen pixel** positions, and One Euro's
> `beta` is a pixel-unit constant. In pixel units `beta=8.0` saturates the cutoff and the filter stops
> filtering (measured: alpha 0.96 even at 0.5 px/frame), so `one_euro_beta` is now **0.0** and
> `one_euro_min_cutoff` is **4.0** — see "Smoothness: the stutter, found and fixed" below, which has the
> pixel-unit measurements. The old numbers still describe relative mode's filter *input* correctly.

Tracking loss now resets the filter (`cursor_filter.reset()`), which is the phase's last requirement:
after a lost-and-reacquired hand the next sample passes straight through, so the cursor neither glides
from stale state nor keeps drifting on a hand that is no longer moving.

### Smoothness: the stutter, found and fixed (2026-10-10)

Reported as "the mouse feels very stuttery, like it is snapping to a grid". Two independent causes, both
in the pointer path, and both measured before and after with `tools/measure_smoothness.py` (a seeded
synthetic hand, no camera, no mouse; `still` / `slow` / `fast` are straight palm paths of 90 frames at
30 fps with ±0.002 landmark jitter, the same jitter `tools/measure_stability.py` reports for a real hand):

| Configuration | Path | moves / held | step mean | step max | step CV | trailing |
| --- | --- | --- | --- | --- | --- | --- |
| **before** (`deadzone=0.02`, filter `beta=8` — effectively off) | still | 1 / 89 | 0.0 | 0.0 | 0.00 | 1.3 px |
| | slow | **10 / 80** | 64.6 | 77.2 | 0.10 | **480.5 px** |
| | fast | 19 / 71 | 64.6 | 77.2 | 0.10 | 648.5 px |
| **after** (`deadzone=0.0`, filter `min_cutoff=4.0 beta=0.0`) | still | 88 / 2 | **2.6** | 7.1 | 0.65 | 2.0 px |
| | slow | **90 / 0** | 6.2 | 12.1 | 0.49 | 7.1 px |
| | fast | 90 / 0 | 12.4 | 18.2 | 0.25 | 14.4 px |

1. **The absolute-mode deadzone froze the reference.** `deadzone` is in *frame* units, so the inherited
   0.02 held the palm reference until the hand had moved ~49 px of screen and then jumped: on a slow
   sweep, 80 of 90 frames held the cursor still and each update was a ~65 px jump. That is the grid.
   `deadzone=0.0` is now the default, a non-zero value is a **radius** rather than a per-axis box (a box
   pins slow *diagonal* movement to the grid), and jitter is left to the filter, which smooths
   continuously instead of in steps. Non-zero is still available and measured — `0.02` now costs 20 held
   frames, 31 px jumps, CV 1.04 and **+15.5 px of trailing** on the slow path (measured 2026-10-11;
   the earlier "+39 px" came from sampling the ideal path by move index, which mixed the sample spacing
   into the number — `measure_smoothness.py` now samples it on the frame each move was emitted on).
2. **The One Euro filter was effectively switched off.** `beta` is a *pixel-unit* parameter (the paper's
   default is 0.007 for pixel coordinates), but the filter is fed screen-pixel positions whose per-frame
   deltas are 1-30 — so `beta=8.0` gave `alpha≈0.96` even at 0.5 px/frame, passing landmark noise and
   quantisation straight to the cursor. Measured on a still hand: **7.2 px mean step with beta=8 vs 2.6 px
   with beta=0** (see `tools/measure_smoothness.py --one-euro-beta`). The tuning is now a plain 4 Hz
   low-pass (`min_cutoff=4.0`, `beta=0.0`), chosen from the grid over `min_cutoff` × `beta`: 4 Hz keeps
   the trailing at ~7 px slow / ~14 px fast while cutting the still-hand shimmer by ~3×. Lower cutoffs
   (1-2 Hz) are smoother still but trail 21-42 px; higher ones (8 Hz) come back to ~6 px of shimmer.
3. **pyautogui's own tween is not an option here** (investigated, rejected, and now documented so nobody
   re-tries it): it only tweens above `MINIMUM_DURATION = 0.1 s`, and a 0.11 s tween measured **171 ms
   per move** (5.2 camera frame intervals) with its sleep floored at `MINIMUM_SLEEP = 0.05 s`. Below
   0.1 s it teleports regardless (0.2-0.7 ms). So there is no usable per-frame tween; the smoothness has
   to come from the filter and the reference. `MouseEvent.duration` records what each move was asked for
   (always 0) so that stays explicit.

**Reach is unaffected where it matters.** The `--sweep-reach` harness now travels to each corner, holds
it for `--reach-dwell` frames (12, ~0.4 s — what a user does at an edge) and reports the two phases
separately, because a lagging filter cannot hit an extreme *while the hand is still moving*:

| Mode | Phase | Cursor span | Worst shortfall | Verdict |
| --- | --- | --- | --- | --- |
| `absolute` (default) | hand still moving | x 26.0…1710.0, y 0.0…1112.0 | 26.0 px | the filter's lag while travelling |
| `absolute` (default) | hand held at edge | x 0.9…1710.0, y 0.0…1112.0 | **0.9 px** | **all four edges reached** |
| `relative` | hand held at edge | — | — | never reached: a hand that stops moves a relative cursor nowhere |

The tolerance is `min_move_pixels + 0.5` rather than an arbitrary half pixel, because the move guard
deliberately stops emitting inside that band; at 0.9 px the cursor is on the edge as far as the pointer
is concerned. `python3 tools/measure_loop.py --synthetic-hand --active --sweep-reach 30 --reach-dwell 12`
reproduces the table. The absolute-mode tuning also leaves the loop rate alone: 29.93 fps measured
internally on the real camera with the mouse intercepted, the same as before the change.

**Also verified after the change:** the finger-curl drift fix still holds — with the synthetic hand
parked and only the fingers moving, the same **11** moves arrive over 120 frames and over 240 frames
(they are the one-time settle of the cursor onto the hand's mapped position, which the smoother filter
now spreads over ~11 frames instead of one jump), then the guard emits nothing.

### Pointer reference and control scheme (2026-10-10)

The cursor used to be mapped from the **index fingertip**, so any finger articulation dragged it across
the screen even when the hand did not move. It is now driven by the **wrist/palm**, which barely moves
when the fingers do, in either control scheme.

**The default scheme is `absolute`** (changed 2026-10-10): the cursor maps to the hand's *position* in
the frame, with the inner 70% of the frame covering the whole screen. The reason is reachability —
absolute pointing puts every screen edge and corner directly in reach, whereas relative pointing can
only get there by accumulating travel, which is what makes a low gain feel like it "cannot reach the
edge" (see "Reach and the sensitivity floor" below for the measurement).

| Config field | Default | Meaning |
| --- | --- | --- |
| `pointer_mode` | `"absolute"` | `absolute`: the cursor maps to the hand's position in the frame — every edge reachable. `relative`: the cursor is shifted by how far the hand moved (like a mouse); the edges need accumulated travel |
| `pointer_reference` | `"palm"` | `palm` = mean of wrist + the four knuckles; `wrist` = the wrist joint alone (most stable); `index` = the old fingertip behaviour |
| `pointer_sensitivity` | `1.0` | the dial's gain: screen widths of cursor travel per frame width of hand travel. In absolute mode it scales the mapped reach about the screen centre; in relative mode it multiplies the displacement. 1.0 is the neutral gain and the floor |
| `pointer_deadzone` | `0.003` | hand displacement below this is treated as jitter and held back (it accumulates, so slow movement is not lost) — relative mode |
| `deadzone` | `0.0` | absolute mode: the palm reference is held until the hand moves this far (a **radius**, in frame units — `0.02` is ~49 px of screen). **0 since 2026-10-10**: a non-zero value is the hold-then-jump grid the user reported — see "Smoothness: the stutter, found and fixed" |
| `one_euro_min_cutoff` / `one_euro_beta` | `4.0` / `0.0` | the cursor filter, in **screen-pixel** units: a 4 Hz low-pass, no speed term (`beta>0` saturates the cutoff at pixel speeds and stops filtering) |
| `relative_scale_by_hand` | `False` | divide displacement by apparent hand size, so the feel does not change as you move nearer/further from the camera |
| `scroll_requires_still_hand` | `True` | (legacy engine) index-finger scrolling only acts while the hand holds still |

The fingertip-fix evidence is unchanged — with a parked hand and only the index finger moving, the palm
and wrist references move the cursor **0** times over 120 frames while `index` moves it **105** times —
and the absolute-mode test suite asserts the same property for the new default
(`test_finger_curl_does_not_move_the_cursor_in_absolute_mode`).

Known trade-offs, stated plainly:

- **Absolute mode follows the hand's position, so moving your hand to a comfortable spot moves the
  cursor with it.** There is no clutch in either mode: `relative` drifts back when you return your hand,
  `absolute` jumps to wherever the hand is (including when the hand is reacquired after a tracking loss,
  which is inherent to absolute pointing rather than a bug).
- **`pointer_deadzone` is a guess until measured against a real hand.** `tools/measure_stability.py`
  reports the per-frame jitter of the wrist/palm in your setup next to the configured deadzone; tune the
  field so it sits above that jitter and below deliberate movement.
- **Smoothing is fixed at startup and does not scale with the dial.** The filter's cutoff is constant, so
  turning `pointer_sensitivity` up multiplies the landmark noise that reaches the screen — at gain 5 a
  still hand's per-frame step is ~5× the ~2.6 px measured at gain 1.0. A gain-aware cutoff (or a
  screen-pixel move guard that scales with the gain) is the fix; until then, high gains are visibly
  shakier than the default.
- **MediaPipe gives no separate "hand moved" signal.** The displacement used by the legacy scroll guard
  is the same measured wrist/palm delta, so a slow deliberate translation below the deadzone still reads
  as "still".

### Reach and the sensitivity floor (added 2026-10-10)

Reported problem: turning the sensitivity down far enough left a band at the screen edges the cursor
could not enter. The cause is arithmetic. `pointer_sensitivity` is *screen widths of cursor travel per
frame width of hand travel*, so a gain below 1.0 cannot cover the screen:

- **absolute mode**: the gain zooms the mapped region about the screen centre
  (`runtime.apply_reach_gain()`), so below 1.0 the region shrinks inside the screen and a strip at every
  edge is unreachable — measured below.
- **relative mode**: each sweep of the hand moves the cursor `0.70 × gain` screen widths, and the return
  sweep drags it back, so a low gain leaves an accumulating shortfall at the edges.

So the floor is not a taste knob. `sensitivity_min` (and `sensitivity_reach_floor`, which it defaults
to) is **1.0**, the gain at which one frame width of hand travel covers one screen width, and the dial
cannot be turned below it. `runtime.reach` reports the current reach and `runtime.reach_warning` says so
in words if the floor is overridden by hand; the HUD appends `! edges unreachable` in that case.

**Measured** with `python3 tools/measure_loop.py --synthetic-hand --active --sweep-reach 60` (real
camera, real MediaPipe, mouse intercepted; the synthetic palm is walked corner to corner across the
frame region, so the cursor must reach all four edges):

| Mode | Gain | Cursor x range | Cursor y range | Verdict |
| --- | --- | --- | --- | --- |
| `absolute` (default) | 1.0 | 0.1 … 1710.0 | 0.1 … 1112.0 | **all four edges reached** |
| `absolute` | 3.0 | 0.1 … 1710.0 | 0.1 … 1112.0 | **all four edges reached** |
| `absolute` | 8.0 | 0.1 … 1710.0 | 0.1 … 1112.0 | **all four edges reached** |
| `relative` | 1.0 (the floor) | 511.5 … 1710.0 | 316.9 … 1096.8 | **edges NOT reached** |
| `relative` | 1.5 – 5.0 | 0.1 … 1710.0 | 0.1 … 1112.0 | all four edges reached |

(Screen is the real 1710 × 1112 of this machine; the 0.1 is the cursor filter landing a hair inside the
edge.) So the floor guarantees the reach in the default absolute mode, and in relative mode it removes
the worst case but does not by itself guarantee a single sweep spans the screen — relative mode needs
about 1.5 for this sweep, which is another reason absolute is the default.

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

The dial is drawn on the preview as a value plus a log-scale bar, e.g. `sens  1.00 [#---------]  ]
faster  [ slower  0 reset`, and the startup banner repeats it. All the fields are on
`HandMouseConfig`: `sensitivity_step` (1.15 per notch — multiplicative, so one turn feels the same at
either end), `sensitivity_min` / `sensitivity_max` (1.0 → 60 — the floor is the reach requirement, see
"Reach and the sensitivity floor"; the bar never draws empty because the floor is still a usable gain),
`sensitivity_reach_floor` (what the floor means), the three key tuples, and `show_sensitivity` for the
overlay. If the floor is overridden by hand, the HUD appends `! edges unreachable` and the banner prints
the warning, rather than leaving the user to wonder why the pointer stops short of an edge.

Programmatic equivalents, for scripts and for anyone driving the app without the window:
`runtime.sensitivity` (read), `runtime.set_sensitivity(value)` (clamped), `runtime.adjust_sensitivity(n)`,
`runtime.reset_sensitivity()`, `runtime.reach` and `runtime.reach_warning`. `runtime.summary()` reports
the mode, the reach, the value the session *ended* at, the value it started at, and how many times the
dial moved.

Why it is safe to turn mid-gesture: the gain multiplies each frame's *hand displacement* (relative) or
zooms the mapped point about the screen centre (absolute), so changing it cannot rescale movement that is
already on screen — the cursor does not jump, and turning the dial emits nothing.

**Evidence** — `tools/measure_loop.py` turns the dial through the real key handler halfway through a
relative-mode run with the mouse intercepted, then reports the mean cursor travel before and after (160
frames each, ~30 fps; `--pointer-mode relative` because that is where travel-per-displacement is the
thing being measured):

| Command | Sensitivity | Mean cursor travel |
| --- | --- | --- |
| `--pointer-mode relative --dial-notches 6` | 1.000 → **2.313** (2.31×) | 7.0 px → **16.4 px** (2.34×) |
| `--pointer-mode relative --sensitivity 20 --dial-notches -10` | 20.000 → **4.944** (0.25×) | 116.9 px → **35.2 px** (0.30×) |
| `--sweep-reach 60 --dial-notches 6` (absolute) | 1.000 → **2.313** | reach stays **all four screen edges reached** |

The travel figures track the gain closely but not exactly: the synthetic hand translates
sinusoidally, so the frames before and after the turn cover different parts of its sweep (and moves
under `min_move_pixels` are skipped). The exact relationship — travel = displacement × gain × screen
width — is pinned by tests in `tests/test_sensitivity_dial.py`
(`test_travel_matches_the_documented_units`, `test_double_the_gain_doubles_the_travel`,
`test_turning_the_dial_mid_motion_does_not_rescale_what_is_already_on_screen`).

### Clicks through the recording adapter (Phase 6, finished 2026-10-11)

Phase 6's own verification bar — "use the recording adapter to confirm exact click counts, cooldown
behaviour, gesture distinction, and suppression rules" — is met by `tests/test_phase6_clicks.py` (32
cases) and by the loop harness. What the phase added on top of Phase 4's recogniser:

* **Last-action status** (`show_last_action`, on by default). The runtime records every click/scroll it
  *emits* (`last_action`, `action_counts`, and `summary()["actions"]`) and the preview draws
  `last: left click  0.3s ago  (4 this session)` on the line below the sensitivity dial. It appears once
  the first action has fired, so a fresh session draws nothing extra (which is also why the older
  preview tests still pass unchanged). Because it counts *emitted* actions, it works under `--dry-run` as
  the readout for "what would have been sent" — that is what the manual click verification procedure in
  the README is built on.
* **The duplicated pinch test is gone.** `runtime.handle_clicks()` (the legacy engine) used to compute
  `abs(dx) + abs(dy) < click_threshold` inline; it now calls `gesture.is_pinch(..., metric="l1")`. The
  metric is an explicit argument (`PINCH_METRICS = ("euclidean", "l1")`) instead of an accident of where
  the test was written, so the legacy path's feel is byte-identical (a test compares the helper against
  the old expression over a grid of tip offsets) and the recogniser keeps its own relative, hand-scale
  thresholds — a test asserts the recogniser never calls the absolute helper at all.
* The runtime also sets `last_action_at` on the recogniser path, which is the field the legacy debounce
  reads; the two engines never share a runtime, so this is bookkeeping only.

**Evidence — the real loop, real camera, real MediaPipe, mouse calls intercepted and counted**
(`--synthetic-hand --active --synthetic-pinch <pose> --frames 150`; the synthetic hand taps for 6 frames
of every 30, so 5 deliberate taps):

| Engine | Pose | moves | clicks | scrolls | Tool verdict |
| --- | --- | --- | --- | --- | --- |
| `recognizer` (default) | left pinch | 150 | **5** | 0 | `pass - one click per deliberate tap (5 clicks for 5 taps)` |
| `recognizer` | right pinch | 150 | **5** | 0 | the same, last action `right click` |
| `recognizer` | two-finger | 150 | **0** | **5** | `pass - the scroll pose scrolls and never clicks` |
| `legacy` | left pinch | 150 | 6 | 0 | one *or two* clicks per tap: its left and right tests can both fire |
| `legacy` | two-finger | 150 | 6 | **0** | documented pre-Phase-4 behaviour: it clicks instead of scrolling |

Exact one-to-one counts under the recogniser; the legacy column is the reason it is not the default. The
verdict line is printed by `tools/measure_loop.py` itself (taps offered vs clicks recorded), so the
procedure is checkable at a glance rather than by eye.

### Cursor move rate limit (Phase 5's last bullet, finished 2026-10-11)

`kinematic/ratelimit.py` holds `MoveRateLimiter`: a pure, timestamp-driven cap on how often a cursor move
may be emitted. `move_max_hz` is the config field (CLI `--move-max-hz`), and it is **0 — off — by
default**, because that is what the measurements say:

| `move_max_hz` | moves / 90 frames | mean step | held frames | trailing |
| --- | --- | --- | --- | --- |
| 0 (default) | 90 | 6.2 px | 0 | 7.1 px |
| 30 (the loop rate) | 90 | 6.2 px | 0 | 7.1 px |
| 20 / 15 | 45 | 12.4 px | 45 | 7.2 px |
| 10 | 30 | 18.7 px | 60 | 7.2 px |
| 5 | 15 | 36.7 px | 75 | 7.2 px |

(`tools/measure_smoothness.py --rate-sweep`, seeded hand, slow path, no camera. Note 20 and 15 are the
same row: on a 30 fps loop the effective rate quantises to divisors of the frame rate, so a 20 Hz cap is
really 15 Hz.)

Two conclusions from it, both pinned by `tests/test_move_rate_limit.py` (25 cases):

1. **At the default settings it is not needed.** The loop is camera-bound at ~30 fps and a move costs
   ~11 ms, so one move per frame already happens; a cap at the frame rate is *identical* to uncapped
   (same 90 moves, same mean step) and anything lower just makes the steps bigger — the hold-then-jump
   the smoothness work removed. This is why the answer to "if needed" is "off", not "always on".
2. **It is needed when the output adapter is slow.** With `--mouse-pause 0.1` (pyautogui's own default,
   ~113 ms per move, longer than the 33 ms frame interval) the loop blocks on the mouse and collapses:

   | Command (real camera, mouse intercepted) | moves | loop rate |
   | --- | --- | --- |
   | `--frames 150` | 150 | **29.7 fps** |
   | `--frames 150 --move-max-hz 10` | 47 | 30.0 fps (nothing to fix) |
   | `--frames 120 --mouse-pause 0.1` | 120 | 6.8 fps |
   | `--frames 120 --mouse-pause 0.1 --move-max-hz 5` | 32 | **17.4 fps** |
   | `--frames 120 --mouse-pause 0.1 --move-max-hz 3` | 15 | **22.9 fps** |

   Note the 8 Hz row that is *missing* from the table: a cap **above** the rate the adapter can sustain
   changes nothing (measured 6.76 fps with `--mouse-pause 0.1 --move-max-hz 8` versus 6.78 fps uncapped),
   because the loop is already slower than the cap. A rate limit is only useful below the achieved rate.

Two properties the class is built around, each with a test: **nothing is queued** (a frame inside the
interval emits nothing and is superseded by the next due frame — a 20-frame burst produces one move, not
a backlog) and **no clock of its own** (timestamps are arguments, like the filters and the recogniser).
It runs *after* the `min_move_pixels` guard in `_move_towards()`, applies to cursor moves only (never to
clicks or scrolls), and is reset with the cursor filter when tracking is lost so a short dropout cannot
hold back the reacquired hand's first move. `summary()` reports `move_max_hz` and `moves_suppressed`.

### Two-finger scrolling on both axes (Phase 7, finished 2026-10-11)

The two-finger pose already existed (Phase 4); Phase 7 finished the path it feeds. What changed:

1. **The horizontal component is emitted.** `GestureEvent.secondary` used to be computed and dropped;
   `runtime.emit_scroll()` now puts it on `output.hscroll()` while the vertical amount goes to
   `output.scroll()`. The two are separate adapter methods because they reach different wheel axes in
   the OS event, and `MouseEvent.axis` records which one a call used — `counts()` still counts both
   under `"scroll"`, so every earlier `counts() == {...}` assertion still holds.
   `scroll_horizontal_gain = 0` (CLI `--no-horizontal-scroll`) restores the pre-Phase-7 behaviour
   exactly, which is the switch for anyone who does not want sideways scrolling.
2. **The dead zone became per axis.** Before, one axis over the zone meant both were sent; with
   horizontal scrolling live that would put every sideways wobble on the screen as sideways scrolling.
   Now the axis under the zone is dropped and the other still goes (a test pins a vertical scroll with
   a sub-dead-zone horizontal drift producing zero `hscroll` calls). Both axes under the zone still
   hold the reference and accumulate, as before.
3. **Per-axis direction inversion** (`scroll_invert_vertical`, `scroll_invert_horizontal`) and a
   separate horizontal gain, applied in `GestureRecognizer._axis_amount()` — the pure module is where
   a hand movement becomes a scroll amount, so the switches live there rather than in the adapter.
   The gains must be non-negative (a `ValueError` otherwise): the sign is the inversion field's job.
4. **The event rate can be capped** (`scroll_max_hz`), and the cap **accumulates** instead of dropping:
   scroll amounts are deltas, so a held frame is added to the pending total and sent on the next due
   frame (see `ScrollRateLimiter` and the decision in §6). At the end of the gesture the remainder is
   flushed (`scroll_end`), so capping the rate never leaves distance unscrolled.

**A real bug this phase found and fixed:** every scroll amount used to be truncated to an `int` *per
event*. With a slow hand each frame asks for a fraction of a wheel unit, and on a measured sweep that
rounding ate **25% of the distance** (135 units emitted of the 180 the hand travelled). Amounts now go
into the accumulator as floats and the sub-unit remainder carries over. That also fixed the pre-existing
vertical path, which had the same silent loss.

**Evidence — real loop, real camera, real MediaPipe, wheel calls intercepted and counted**
(`--synthetic-hand --active --synthetic-scroll <axis> --frames 150 --scroll-frames 140`, a straight
sweep from −0.10 to +0.10 frame widths, so the net hand movement is down/right):

| Sweep | vertical events / units | horizontal events / units | Tool verdict |
| --- | --- | --- | --- |
| `vertical` | 45 / **−174** | 0 / 0 | `pass - hand down -> -174 vertical (page down), no sideways events` |
| `horizontal` | 0 / 0 | 45 / **+174** | `pass - hand right -> +174 horizontal (scroll right), no vertical events` |
| `diagonal` | 45 / **−174** | 45 / **+174** | `pass - hand down-right -> -174 vertical, +174 horizontal` |
| `diagonal --scroll-max-hz 10` | 32 / −174 | 32 / +174 | 13 frames held and accumulated — same distance, fewer events |
| `diagonal --scroll-max-hz 4` | 15 / −167 | 15 / +167 | 30 frames held; the last interval was still pending when the run ended |
| `diagonal --no-horizontal-scroll` | 45 / −174 | **0 / 0** | the pre-Phase-7 switch |
| `vertical --invert-scroll-vertical` | 45 / **+174** | 0 / 0 | the tool's verdict prints `CHECK` — correct: it expects the default direction, and this run deliberately inverted it |

0 clicks in every run (the scroll pose never clicks), and the loop rate is unchanged at ~30 fps.

**What this project genuinely cannot verify here, stated plainly:** which way *content* moves for a
positive macOS horizontal wheel value. pyautogui's `hscroll` (whose docstring says horizontal scrolling
is "currently just Linux" — stale: `_pyautogui_osx._hscroll` posts a 2-axis `CGEventCreateScrollWheelEvent`)
takes the sign from Quartz, and finding out what an app does with it would mean scrolling the user's
screen. So the convention is documented (hand right → positive) and `scroll_invert_horizontal` is the
one-field fix; the README's manual test says to check it and flip that field if it feels backwards.

### Pinch press-and-hold: hold to drag (fixed 2026-10-11)

Reported: "when I pinch using my pointer finger, I can't seem to hold down left click. It activates
left click, then immediately releases." The cause: `runtime._apply_gesture_events()` turned a
`LEFT_CLICK`/`RIGHT_CLICK` event into `output.click("left"/"right")`, and `click()` is pyautogui's
`mouseDown()` + `mouseUp()` back to back — there was never a code path that kept the button down while
the pinch was held, so holding the pose longer than an instant did nothing extra and dragging was
impossible.

**Fixed by splitting the click into a press and a release**, deliberately tied to different things:

- **The press is still debounced exactly as before** — `LEFT_CLICK`/`RIGHT_CLICK` fire once, gated by
  `pinch_ratio`/hysteresis/`hold_seconds`/`cooldown_seconds`/`require_neutral_between`, unchanged. The
  runtime now calls `output.press(button)` (pyautogui's bare `mouseDown`) instead of `click()`.
- **The release is tied to the raw engagement flag, not to the state machine.** A new
  `GestureRecognizer._check_releases()` runs right after `_observe_hand()` every frame and fires
  `LEFT_RELEASE`/`RIGHT_RELEASE` the instant `_engaged[pair]` goes false for a pair that is
  `_pressed[pair]`, deliberately bypassing hold time, cooldown and "no gesture switch without neutral" —
  those rules debounce when a *press* may re-arm, and tying a release to them risks a stuck button (e.g.
  the thumb crossing from the index tip straight to the middle tip inside one frame would otherwise be
  blocked by `require_neutral_between`, leaving the left button "held" by the state machine's bookkeeping
  after the index pinch was physically already gone).
- **Pause and tracking loss also force a release.** `GestureRecognizer.pending_releases()` reports what is
  currently pressed without mutating anything; `set_paused()` and the tracking-lost branch of `update()`
  call it *before* `reset()` (which clears `_engaged`/`_pressed`, dropping the information a release is
  owed from) and splice the result into the events they return.
- **The runtime never lets a release be silently dropped.** `_apply_gesture_events()` handles
  `LEFT_RELEASE`/`RIGHT_RELEASE` *before* the `can_emit` gate — the same placement as `PAUSE_TOGGLE`,
  because a release is cleanup for a button already down, not new input-driven output; the case that
  matters is a pause landing on the same frame the pinch releases. `_sync_gesture_pause()` used to discard
  `set_paused()`'s return value outright (harmless while those events were only `PAUSED`/`RESUMED`
  bookkeeping); it now flushes any release through the same `_flush_release()` helper
  `_apply_gesture_events` uses, so pausing mid-drag — SPACE key, API, or the gesture pause — releases the
  real button instead of leaving it down.
- **Recorded as a click, so nothing else in the suite changed.** `RecordingMouseOutput.press()` appends a
  `"click"` kind event rather than a new kind, so every existing `counts()["click"]`/`.clicks` assertion
  in the whole suite still means what it meant before. `release()` appends a new `"release"` kind,
  deliberately left out of `counts()` (which only ever reported move/click/scroll) so no existing
  exact-dict assertion changed. `last_action`/`action_counts` are untouched by a release, for the same
  reason.

**Net effect**: a quick pinch-and-release reaches the adapter as a press immediately followed by a
release — on the real desktop, indistinguishable from the old click. A held pinch keeps the button down
for as long as the pose is held, which is what makes click-and-drag possible. Verified both at the
recogniser level (hold for 30 synthetic frames → zero extra events, release → exactly one `left_release`)
and through the real output adapter via `runtime.process_landmarks()` —
`tests/test_phase6_clicks.py`'s `PressAndHoldThroughTheAdapterTests` (hold, tap, right pinch, the pause
path) and `tests/test_phase4_recognizer.py`'s `PressAndHoldTests` (the pure state machine: pausing
mid-hold, tracking loss mid-hold, cooldown never blocking a release, no re-press while held).

### The gesture pause (Phase 8, finished 2026-10-11)

The phase's fifth bullet is conditional — "add a gesture pause only after the keyboard control is
reliable" — and it landed as an **opt-in** pose in the recogniser, because Phase 8's own ordering was
the right one: the keyboard pause had already been proven by four phases of tests.

* **The pose**: an *open hand* — `pause_gesture_fingers` (4) of the four fingers straight, and no pinch
  pair engaged. `gesture.extended_fingers()` is the coarse test behind it: a finger counts as straight
  when its tip is farther from the wrist than the joint below it. No click pose can look like it (both
  pinches have the index or the middle curled).
* **Hold-gated**: `gesture_pause_hold_seconds` (1.2 s) must elapse. The hold is what makes it safe: the
  hand passes *through* open shapes constantly on its way to a pinch, and a brief stretch must not stop
  the mouse.
* **One toggle per hold**, released to re-arm (`PAUSE_TOGGLE` event, `_pause_armed`), exactly like the
  click pairs. A hold in flight when a pause/resume arrives from elsewhere (the SPACE key, the API) is
  cancelled *and* disarmed, or the same continuing hold would toggle straight back a second later.
* **It works while paused**, which is the only way the same pose can resume: `update()` evaluates the
  pause pose *before* its `PAUSED` early-return, and the runtime handles `PAUSE_TOGGLE` *before* the
  `can_emit` gate — because that event is what moves the gate. It reaches no mouse call, so a gesture
  pause emits nothing.
* **Off by default** (`gesture_pause_enabled = False`, `--gesture-pause`): a pose that can freeze the
  pointer has to be a deliberate choice.

Evidence: `tests/test_phase8_runtime.py` (31 cases: the hold, the flicker restart, the re-arm, the
engaged-pinch refusal, the configurable finger count, working while paused, a hold across a transition,
and the runtime-side toggles with nothing emitted), plus Phase 8's own missing tests — toggle
idempotence (`pause()`/`resume()`/`toggle_pause()` twice, from the API and from the keyboard),
recovery after a failed run, cleanup on a normal exit, on a window close and on an exception, and three
launch/close cycles leaving no capture thread behind.

**Window close is now a real exit** (it was Phase 8's third cleanup case with no code behind it): the
loop asks `cv2.getWindowProperty(window, WND_PROP_VISIBLE)` after each `imshow` and treats "not visible
anymore" as a quit, so clicking the close box stops the analysis and releases the camera instead of
leaving the loop running unseen. A backend that cannot answer the question is treated as *open* — a
missed close is a papercut, a spurious exit is a bug.

### Diagnostics, logging and profiling (Phase 9, finished 2026-10-11)

Three new pieces, all additive:

1. **`diagnostics.py` + `--diagnose`.** Checks the platform/interpreter, that MediaPipe imports *and*
   processes a frame through the legacy `mp.solutions.hands`, which adapter this run would use, the
   screen size, the macOS Accessibility grant, the effective configuration, and (unless
   `--no-camera-probe`) that the camera opens and delivers a frame — reporting resolution, how long the
   first frame took, mean brightness (a `very dark` hint below 30/255) and whether a hand was in view.
   It touches nothing: the Accessibility answer is `AXIsProcessTrusted()` through ctypes (a read-only
   question — **no test move of the pointer, no prompt**), and the camera probe releases the device
   immediately. A permission denial and a missing device get *different* hints. Exit status is 0 only
   when every check passed, so a script can gate on it. Every check takes an injectable probe
   (`opener=`, `hands=`, `trusted=`) which is what makes the denied cases testable at all.
   The app also runs the Accessibility check at startup when the adapter is real, so a missing grant is
   a sentence rather than "the cursor does not move" (gotcha 8.5).
2. **`log.py` + `--log-level`/`--log-file`.** One JSON object per line, named events, fields merged in
   (`event` is the API: `session_start`, `loop_start`/`loop_stop`/`loop_error`, `pause`/`resume` with
   `source`, `tracking_lost`/`tracking_restored`, `click`, `scroll`, `window_closed`, `summary`, and at
   debug only `frame`/`move`). Silent unless asked: the default level is `warning` and the pipeline
   speaks at info/debug; the library adds a `NullHandler` at import (gotcha 8.19), `configure()` replaces
   handlers rather than stacking them, and `logger.propagate = False` keeps a host app's root config
   from duplicating the lines. **Logging must not become the second thing the camera-bound loop waits
   on**: the per-frame heartbeat is periodic (one a second) and every call goes through
   `isEnabledFor`, so a quiet level costs an attribute check, and the test suite asserts that 90 frames
   at `info` produce *no* per-frame lines.
3. **`tools/profile_loop.py` + the measurement below.** Per-stage milliseconds for one frame (capture,
   flip, BGR→RGB, MediaPipe, output, preview), mean/median/p95 and share, with the mouse intercepted and
   replayed at the current position so nothing moves; `--cprofile N` adds the top-N cumulative
   functions. This is the tool that turns "should we optimise X?" into a number.

| stage | calls/frame | mean ms | share |
| --- | --- | --- | --- |
| capture (`read()`) | 1.72 | 0.003 | 0.0% |
| flip | 1.00 | 0.217 | 1.4% |
| rgb | 1.00 | 0.177 | 1.2% |
| MediaPipe | 1.00 | 6.901 | 45.8% |
| output (one move event) | 0.66 | 11.664 | 50.8% |
| logic (landmarks, mapping, filter, gestures) | — | 0.108 | 0.7% |
| **frame** | 1.00 | **15.055** | 100% |

Measured 2026-10-11 with the real camera at 1280x720, real MediaPipe, unpaused, no preview, 90 frames:
**29.58 processed fps** at a **33.8 ms frame interval**, so the loop is *camera-bound* (interval = 2.2×
the work). With `--mouse-pause 0.1` (pyautogui's own default) `output` is 108.6 ms per move, the loop
falls to **8.47 fps** and the verdict flips to *work-bound* — the case `move_max_hz` exists for.

**Two optimisations were measured and rejected in the same pass**, which is the point of having the
table:

| candidate | measurement | verdict |
| --- | --- | --- |
| feed MediaPipe a smaller frame (or capture smaller) | `hands.process` = 8.04 ms at 1280x720, 7.83 at 640x360, 7.86 at 480x270, 7.80 at 320x240 | **rejected** — the cost is flat in input size (it resizes to its own 256x256 internally), so pre-resizing adds work rather than removing it |
| bypass pyautogui's public API and call `_pyautogui_osx._moveTo` | 11.36 ms (`pyautogui.moveTo`) vs 11.12 ms (the platform hook) | **rejected** — the 11 ms is macOS's event cost, not pyautogui's Python; a 2% saving is not worth coupling the adapter to a private OS backend |

### Packaging (Phase 9, finished 2026-10-11)

`pyproject.toml` declares the project (setuptools, `requires-python >=3.9`, the four runtime
dependencies with `mediapipe==0.10.14` pinned for the reason in gotcha 8.2) and one console script:

```toml
[project.scripts]
kinematic = "kinematic.main:main"
```

Both invocation styles stay supported on purpose (`python3 -m kinematic.main` and the `kinematic`
command resolve to the same `main`, and that is why the `__package__` guard at the top of `main.py`
must stay — gotcha 8.1). Verified by building the wheel rather than by installing it into the running
interpreter (the project has no venv by design):

```
~/.local/bin/python3.11 -m pip wheel . --no-deps --no-build-isolation -w <scratch>
→ kinematic-0.9.0-py3-none-any.whl (70 KB), entry_points.txt: kinematic = kinematic.main:main
```

The build is done with a *different* interpreter (3.11 + setuptools 84) because the project's 3.9
interpreter has setuptools 58, which predates PEP 621 `[project]` metadata — so the metadata is
validated without touching the 3.9 environment at all. `tests/test_phase9_e2e.py` also parses
`pyproject.toml` and asserts the console-script target, the mediapipe pin and the packaged module list,
so a broken manifest fails the suite. `build/`, `dist/` and `*.egg-info/` are ignored so a wheel build
cannot dirty the repo.

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
| 1 — Foundation & safety | ✅ **5/5 (2026-10-10)** | Modules split: `capture.py`, `gesture.py`, `output.py` (adapters), `runtime.py` (coordinator), `main.py` (CLI); `requirements.txt`; 256-test suite; **paused startup** with SPACE toggle and ESC/q quit; **`RecordingMouseOutput` mock adapter** used by tests and `--dry-run`; macOS permission docs in README | — |
| 2 — Capture pipeline | ✅ **5/5 (2026-10-10)** | `capture.py`'s threaded `ThreadedCamera` (`isOpened()` error, buffer size, device-error reporting); `cv2.flip` mirroring, BGR→RGB, `mp.solutions.hands`, invalid-frame error, exception-safe teardown (the runtime's `try/finally`); **`landmarks.py`** (Landmark/HandLandmarks, the single protobuf conversion point, incomplete detections flagged); **preview overlay** of the loop rate *and* the 21-point skeleton (`preview.py`, drawn with cv2, `show_landmarks`) | — |
| 3 — Geometry/mapping/smoothing | ✅ **5/5 (2026-10-10)** | `gesture.py`: distances, `midpoint`/`midpoints`, `pinch_distances`, `hand_scale` + `hand_span`, `clamp`/`clamp_point`, `mirror_point`, EMA; `mapping.py`: `Region` + `map_point_to_screen` (frame region → screen region, clamping, mirroring); `smoothing.py`: EMA + 1-euro filters with `reset()`, wired into the runtime with a tracking-loss reset; 44 deterministic tests for distances, clamping, smoothing, latency and reset | Midpoints are available but not yet *used* by a gesture; One Euro parameters are tuned for this camera rather than for everyone |
| 4 — Gesture state machine | ✅ **5/5 (2026-10-10)** | `recognizer.py`: `HandState` (neutral / left pinch / right pinch / two-finger scroll / paused / tracking-lost), `EventKind` + `GestureEvent`, `GestureThresholds`/`Observation`/`GestureUpdate`, `GESTURE_PRIORITY`; thumb–index vs thumb–middle pinch detection from `pinch_distances()` ÷ `hand_scale`; per-pair hysteresis, hold time, cooldown, re-arm (`reset(disarm=True)`), scroll-priority resolution, "no switch without neutral", scroll reference/dead-zone/gain; wired into the runtime via `apply_gestures()` behind `can_emit`, with the paused/tracking states driven by the pause key and the frame loop; 72 new tests (54 recogniser + 18 runtime) | The thresholds are reasoned defaults, not measured against a real hand (Phase 9 owns calibration) — **Phase 7 has since emitted the horizontal component** (`GestureEvent.secondary`), which this row listed as missing when it was written |
| 5 — Cursor movement | ✅ **5/5 (2026-10-11)** | **Absolute position mapping is now the default scheme** (`pointer_mode="absolute"`, inner 70% of the frame → whole screen) with the dial scales its reach, so every screen edge and corner is directly reachable — measured, see §2 "Reach and the sensitivity floor"; palm/wrist-relative pipeline still available and tested; smoothing, clamp and `min_move_pixels`; **dry-run and real adapters**; **opt-in gate** (starts paused, `--dry-run`); paused/bounds suppression; the explicit *tracking-lost* state (`HandState.TRACKING_LOST`, emitted once per loss, filter and anchor reset); **`ratelimit.py`'s `MoveRateLimiter`** — the last bullet: a pure, timestamp-driven cap that coalesces (never queues) cursor moves, reported by `summary()["moves_suppressed"]`, off by default because the measurement says it is only needed when the adapter is slow (25 new tests, `--rate-sweep`, `--mouse-pause`) | — |
| 6 — Left & right clicks | ✅ **5/5 (2026-10-11)** | Left click, right click (thumb–middle), shared cooldown, release-then-re-arm, **scroll precedence** — all in the recogniser, whose 72 tests assert exact click counts and whose 32 new recording-adapter tests (`test_phase6_clicks.py`) pin the counts per tap, the cooldown drop, re-arm, pause/tracking-loss suppression and the HUD readout; **suppressed while paused** (Phase 1's `can_emit` gate and the recogniser's own `PAUSED` state, including a pinch held across a resume); **last-action status** on the preview (`show_last_action`, `last_action`, `summary()["actions"]`), which doubles as the readout for the manual click verification procedure in the README; the legacy path is still reachable via `--gesture-engine legacy` and now uses the shared `gesture.is_pinch(..., metric="l1")` instead of a second inline copy of the metric; measured in the real loop: 5 clicks / 5 taps left, 5 right, 0 clicks + 5 scrolls for the two-finger pose | Hovering a click's target is still not tracked (the phase's optional extra); the legacy engine's one-or-two-clicks-per-tap behaviour is unchanged by design |
| 7 — Two-finger scroll | ✅ **5/5 (2026-10-11)** | Two-finger (index+middle) pose with the midpoint as reference, reference reset on entry, **per-axis dead zone**, configurable `scroll_gain` (vertical) and `scroll_horizontal_gain` (horizontal, `0` = the pre-Phase-7 behaviour of dropping the sideways component), `scroll_invert_vertical`/`scroll_invert_horizontal`, **both wheel axes emitted** (`output.scroll` + `output.hscroll`, distinguished by `MouseEvent.axis`), **click suppression while scrolling** (the pose scrolls instead of clicking), and a `scroll_max_hz` cap that **accumulates** held frames instead of dropping them → distance preserved; measured in the real loop: a vertical sweep −174 units with 0 sideways events, horizontal +174 with 0 vertical, diagonal both, 0 clicks; 42 new tests | The *content* direction of a positive macOS horizontal wheel value is inherited from Quartz and is not verifiable without scrolling the user's screen — hence `scroll_invert_horizontal`; the **legacy** index-finger scroll path is unchanged (still non-conforming: it never had hold/hysteresis or the two-finger reference) |
| 8 — Pause/runtime/feedback | 🟡 **4/5 + 1 partial** | `runtime.py` coordinator; configurable keyboard toggle (`toggle_pause_keys`) with paused startup; PAUSED/ACTIVE **and the live gesture state** on the preview (`f"… fps  {gesture_state}"`); the recogniser's `PAUSED`/`TRACKING_LOST` states, kept in step by `_sync_gesture_pause()` and the no-hand branch; exception-safe teardown (`try/finally`) | The optional *gesture* pause, and tests for toggle idempotence, recovery and window-close cleanup |
| 9 — Calibration/hardening | ✅ **5/5 (2026-10-11)** | `HandMouseConfig` with the tunables this phase asks for — camera, margins, smoothing, pinch thresholds, cooldown, hotkeys (`toggle_pause_keys`/`quit_keys`/the dial's key tuples) and sensitivity, now adjustable at **runtime** via the dial (`set_sensitivity`/`adjust_sensitivity`, `]`/`[`/`0`, `--sensitivity`, `--sensitivity-step`); **mirroring is a config field now** (`preview_flip` / `--preview-flip`, 2026-10-11) | **Configuration/calibration** (all of it: camera, mirroring, margins, smoothing, pinch thresholds, cooldown, sensitivity, hotkeys — with the dial adjustable at *runtime*); **`diagnostics.py` + `--diagnose`** (camera, screen, MediaPipe, Accessibility, configuration; read-only; the denial path diagnosed in-app); **`log.py` + `--log-level`/`--log-file`** (JSON lines, named events, silent by default, no per-frame cost); **`tools/profile_loop.py`** + the measured stage table and the two *rejected* optimisations (§2 "Diagnostics, logging and profiling"); **`pyproject.toml`** with the `kinematic` console script, verified by building the wheel; the README's setup section and **final macOS verification checklist**. 99 new tests: `test_phase8_runtime.py` (31), `test_phase9_diagnostics.py` (31), `test_phase9_logging.py` (18), `test_phase9_e2e.py` (19) | One honest gap and one deliberate one: **the dial's value is per-session** (no settings file — pass `--sensitivity` to pin it), and the hardware-dependent half of the checklist is manual on purpose, because the suite must never touch the desktop. Also still open from earlier phases: no CI, and the legacy engine's non-conforming scroll path |
| Tests | 487 passing | gesture helpers; config defaults; cursor-move guard; camera error path; the Phase 1 safety suite; the pointer-reference suite; the landmark suite; the Phase 3 geometry and smoothing suites; the Phase 4 recogniser suite (54 cases + 8 press-and-hold cases) and runtime suite (18 cases); the Phase 6 click suite (32 cases + 6 press-and-hold-through-the-adapter cases); the move-rate-limit suite (25 cases); the Phase 7 scroll suite (42 cases: per-axis dead zone, both axes through the recording adapter, sign/gain/inversion maths, the accumulating rate cap, the sweep harness's verdict); the preview-orientation suite (18 cases: all four flip modes, and the skeleton landing on the hand in each); the sensitivity-dial suite (38 cases); the pointer-modes suite (21 cases); the smoothness suite (15 cases); the Phase 8 suite (31 cases: the gesture pause and the teardown guarantees); the Phase 9 diagnostics suite (31 cases, incl. the `--diagnose` CLI path) and logging suite (18 cases); the Phase 9 end-to-end suite (19 cases: denied permissions, unavailable camera, low light, partial tracking, sustained gestures, repeated launch, a call-by-call scripted session) | MediaPipe's own detections, the real camera path, the legacy click/scroll branches beyond the pinned cases, the on-screen *direction* of a horizontal wheel event, and the hardware half of the README's final checklist (a camera negotiating frames, a real hand, the OS obeying a move) — all manual by design |

**What this means for the next session:**

- **Phases 1–7 are done.** Capture, geometry, mapping, smoothing, the safety layer, the gesture
  state machine, the click path and both scroll axes are in place and tested, so every remaining phase
  can be built on a pipeline that cannot hurt the desktop and does not need a camera in the test suite.
- **The control scheme is absolute position by default** (changed 2026-10-10) with the sensitivity dial
  scaling its reach and floored at gain 1.0 so the screen stays reachable; `--pointer-mode relative`
  restores the displacement-based scheme. Read §2 "Pointer reference and control scheme" and "Reach and
  the sensitivity floor" before touching either — the reachability requirement is the reason for both.
- **Phase 5, Phase 6 and Phase 7 are closed** (2026-10-11): the rate limiter is in `ratelimit.py` (move
  caps supersede, scroll caps accumulate), the click path has its recording-adapter suite plus the
  last-action readout, and the scroll path emits both wheel axes with per-axis dead zone, inversion and
  an accumulating event cap. Read §2 "Cursor move rate limit", "Clicks through the recording adapter" and
  "Two-finger scrolling on both axes" before touching the move, click or scroll paths.
- **Retrofit, don't rewrite.** Every phase so far added a module and switched the runtime over while
  leaving the previous path reachable (EMA next to One Euro, index next to palm, absolute next to
  relative, `legacy` next to `recognizer`). Keep doing that: it is what makes the changes reviewable
  and revertible, and it is why the 5 pre-existing gesture tests could be pinned instead of deleted.
- **Phase 4 changed the default behaviour on purpose** (the gesture map in §2 "Gestures": right click
  is now thumb–middle, the two-finger pose scrolls instead of right-clicking). The old behaviour is one
  flag away (`--gesture-engine legacy`) if the user prefers it — but do not flip the default back
  without being asked.
- **The pointer path's smoothness is now measured, not guessed** (2026-10-10). `deadzone=0.0` and a
  4 Hz / `beta=0.0` cursor filter replaced a hold-then-jump grid and a filter that was silently off;
  read §2 "Smoothness: the stutter, found and fixed" before touching `deadzone`, `min_move_pixels`,
  `pointer_deadzone`, `one_euro_min_cutoff` or `one_euro_beta`, and re-run
  `python3 tools/measure_smoothness.py` after any change (it needs no camera). The trap that hid the
  bug for a whole phase: **a filter parameter's units are part of its contract** — One Euro's `beta` is
  a pixel-unit constant, so feeding it screen pixels with `beta=8` disabled the filter.
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

**Status: ✅ 5/5 complete (2026-10-11).** All five bullets are done, and the mapping the phase asked to
calibrate is now the **default scheme**: absolute position mapping (the inner 70% of the frame → the
whole screen) with the sensitivity dial scaling its reach, so every screen edge and corner is directly
reachable — measured with `tools/measure_loop.py --sweep-reach` (all four edges at gains 1.0, 3.0 and
8.0). Also in place: the dry-run and real cursor adapters, cursor output behind the `can_emit` gate (the
app starts paused), moves below `min_move_pixels` skipped, bounds clamping, and the explicit
tracking-lost suppression (`HandState.TRACKING_LOST`, with the filter and anchor reset).

The last bullet — "rate-limit or coalesce movement events **if needed**" — landed as
`kinematic/ratelimit.py`'s `MoveRateLimiter` (`move_max_hz`, `--move-max-hz`), and the "if needed" is
answered with a measurement rather than a guess: off by default, because at the shipped settings the loop
is camera-bound and a move costs ~11 ms, so one move per frame already happens and a cap at the frame rate
is bit-identical to uncapped; turning it on is for a *slow adapter* (pyautogui's own `PAUSE = 0.1` drops
the loop to 6.8 fps, and a 3 Hz cap brings it back to 22.9 fps). It never queues — a frame inside the interval
is superseded by the next due one, so a burst cannot send a backlog — and it applies to cursor moves only.
See §2 "Cursor move rate limit" for the tables and `tests/test_move_rate_limit.py` (25 cases) for the
pinned numbers.

- Map filtered coordinates to the active screen.
- Add dry-run and real cursor adapters.
- Suppress movement while paused, untracked, or outside valid bounds.
- Rate-limit or coalesce movement events if needed. → `MoveRateLimiter` in `ratelimit.py`, wired into
  `_move_towards()` after the move guard, reset with the filter on tracking loss, off by default.
- Keep real output opt-in.

**Verification:** test screen corners, center mapping, mirroring, bounds, pause suppression, and
tracking loss with the recording adapter. Then perform a cautious manual cursor test.

**Prompt:**
> Implement Phase 5 of the webcam hand mouse roadmap. Connect the filtered coordinate pipeline to a
> screen mapper and guarded cursor adapter, keeping dry-run recording as the default. Add tests for
> screen mapping, mirror orientation, bounds, pause/tracking-loss suppression, and movement
> throttling, then provide a cautious manual verification path for real cursor movement.

### Phase 6 — Left and right clicks

**Status: ✅ 5/5 complete (2026-10-11).** All five bullets hold: left/right clicks are one-shot events
from the recogniser, a sustained pinch clicks once, release-and-re-arm is enforced (including the
`_armed` flag that survives a resume or a tracking loss), clicks are suppressed while scrolling, paused
and untracked, and the last click/scroll is displayed on the preview (`show_last_action`, with the
session count and the age — the readout the manual verification procedure uses).

This session closed the two gaps the Phase 4 audit left: **last-action status** (the runtime records the
actions it emits — `last_action`, `action_counts`, `summary()["actions"]` — and the HUD draws
`last: left click  0.3s ago  (4 this session)`) and **the duplicated pinch test** (`handle_clicks()` now
calls `gesture.is_pinch(..., metric="l1")` instead of re-writing the L1 arithmetic inline, so there is
one pinch helper and the legacy engine's feel is unchanged — see §2 "Clicks through the recording
adapter").

**Evidence:** `tests/test_phase6_clicks.py` (32 recording-adapter cases: exactly 5 clicks for 5
deliberate taps, 0 clicks for a flicker shorter than the hold time, the cooldown dropping rather than
deferring a click, release-and-re-pinch, scroll precedence with and without a neutral frame, pause and
tracking-loss suppression, the HUD line's pixels and text), plus the real loop with the mouse
intercepted: `--synthetic-pinch left|right` → 5 clicks for 5 taps, `--synthetic-pinch two_finger` → 0
clicks, and the legacy engine's 6 clicks for 5 taps as the documented contrast. (At the time this line
was written `two_finger` also reported 5 scrolls; since Phase 7's per-axis dead zone the tap pattern's
midpoint movement no longer clears the dead zone inside a 6-frame tap, so that variant now reports 0
whole scroll units — it is a click-suppression check, and `--synthetic-scroll` is what covers the
scroll path. Verified identical before and after the press-and-hold change.)

- Convert gesture transitions into one-shot left/right click events.
- Prevent repeated clicks during sustained pinches.
- Require release and re-entry before another click.
- Suppress clicks during scrolling, pausing, and tracking loss.
- Display optional last-action status. → `show_last_action` + `last_action_text()` on the preview,
  `last_action`/`actions()` on the runtime and in `summary()`.

**Verification:** use the recording adapter to confirm exact click counts, cooldown behavior, gesture
distinction, and suppression rules. → `tests/test_phase6_clicks.py`, and the two-part procedure in the
README ("Verifying clicks"): a scripted headless run (`tools/measure_loop.py --synthetic-hand --active
--synthetic-pinch left|right|two_finger --frames 150`, which prints its own pass/CHECK verdict per pose)
followed by a dry run, then a real run with the pointer parked over a harmless window.

**Prompt:**
> Implement Phase 6 of the webcam hand mouse roadmap. Wire debounced left- and right-click transition
> events into the output adapter with cooldown, release/re-arm, scroll precedence, pause suppression,
> and tracking-loss suppression. Extend recording-adapter tests and give me a minimal manual click
> verification procedure.

### Phase 7 — Two-finger scrolling

**Status: ✅ 5/5 complete (2026-10-11).** The two-finger gesture is the default scroll path: the
index/middle **midpoint** is the reference, it resets on scroll entry (and again after a tracking loss,
so a reacquired hand cannot jump the page), the dead zone and `scroll_gain` are config fields, and the
pose is suppressed as a click (scroll outranks the clicks by construction).

This session finished the three missing bullets:

- **Vertical *and* horizontal movement.** `GestureEvent.secondary` is now emitted: the runtime sends it
  to `output.hscroll()` alongside the vertical `output.scroll()`, each recorded with its `MouseEvent.axis`.
  `scroll_horizontal_gain` (0 = off) and `--no-horizontal-scroll` restore the old vertical-only behaviour.
- **Direction inversion.** `scroll_invert_vertical` / `scroll_invert_horizontal`, applied in the pure
  recogniser (`_axis_amount`) alongside a separate horizontal gain; negative gains now raise instead of
  silently inverting.
- **Rate limiting.** `scroll_max_hz` caps scroll events per second with `ScrollRateLimiter`, which
  **accumulates** held frames rather than dropping them (a scroll amount is a delta), and flushes the
  remainder when the gesture ends. Off by default, for the same measured reason as `move_max_hz`.

Also fixed here: per-event `int()` truncation was silently discarding up to a wheel unit per frame — 25%
of the distance on a measured sweep. And the dead zone is now **per axis**, which horizontal scrolling
made necessary (otherwise sideways wobble during a vertical scroll reaches the screen).

- Use the index/middle pinch midpoint or another documented reference point. → the midpoint, re-anchored
  on entry and after a tracking loss.
- Support vertical and horizontal movement. → both axes emitted; separate gains.
- Add dead zones, sensitivity, direction inversion, and rate limiting. → per-axis dead zone,
  `scroll_gain`/`scroll_horizontal_gain`, `scroll_invert_*`, `scroll_max_hz` (accumulating).
- Reset the reference point when entering scroll mode. → on entry, and on reacquire.
- Prevent scroll gestures from producing clicks. → scroll outranks both pinches in `GESTURE_PRIORITY`.

**Verification:** test vertical, horizontal, diagonal, stationary, reversed, and tracking-loss
sequences with recorded scroll events. → `tests/test_phase7_scroll.py` (42 cases: the pure geometry, both
axes through the recording adapter, inversion and zero-gain switches, the accumulating cap, the
truncation bug, and the sweep harness's verdict), plus the controlled harness run in §2 "Two-finger
scrolling on both axes" (real camera, wheel calls intercepted: −174 / +174 units with the expected sign
per axis and 0 events on the other) and the manual procedure in the README.

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

**Status: ✅ 5/5 complete (2026-10-11).** All five bullets landed, each with the evidence the phase
asks for:

- **Configuration for camera, mirroring, control margins, smoothing, pinch thresholds, cooldown,
  sensitivity and hotkeys.** Every one is a field on `HandMouseConfig` with a comment saying what it
  does and where its default came from; the tunables that are numeric defaults are justified by a
  measurement quoted in §2, and the ones that decide *behaviour* (mirroring, sensitivity) are
  adjustable at runtime (`preview_flip`/`--preview-flip`; the sensitivity dial's keys and
  `set_sensitivity`).
- **Diagnostics** (`kinematic/diagnostics.py`, `--diagnose`): platform/interpreter, MediaPipe import
  *and* a frame through the legacy `solutions.hands` API, the output adapter, the screen size, the
  macOS Accessibility grant (a read-only `AXIsProcessTrusted()` question, never a test move), the
  effective configuration, and an optional camera probe reporting resolution, first-frame latency,
  brightness and whether a hand was in view. Denials are diagnosed with their own hints and the exit
  status is 0 only when everything passed. The app also reports a missing Accessibility grant at
  startup when the adapter is real (gotcha 8.5).
- **Structured logging and performance metrics** (`kinematic/log.py`, `--log-level`, `--log-file`): one
  JSON object per line with named events (`loop_start`/`loop_stop`/`loop_error`, `pause`/`resume` with
  their source, `tracking_lost`/`tracking_restored`, `click`, `scroll`, `window_closed`, the exit
  `summary`, and debug-only `frame`/`move`), silent at the default `warning` level, no per-frame cost
  on the hot path, and a `NullHandler` so a library import prints nothing (gotcha 8.19).
- **Profiling and reducing unnecessary work** (`tools/profile_loop.py`): the per-stage breakdown
  (capture / flip / rgb / MediaPipe / output / preview, mean-median-p95 and share), a `--cprofile` mode
  and the verdict line that says whether the loop is camera-bound or work-bound. Measured here: 15.06 ms
  of work in a 33.8 ms frame → 29.58 fps, i.e. **camera-bound**; our own logic is 0.108 ms/frame. The
  two optimisations it was built to test were **measured and rejected** (MediaPipe's cost is flat in
  input size; `_pyautogui_osx._moveTo` saves 0.24 ms of 11.36) — see §2 "Diagnostics, logging and
  profiling", which is also where the slow-adapter case (108.6 ms/move → 8.47 fps) is recorded.
- **Packaging and final setup documentation.** `pyproject.toml` (setuptools, `requires-python >=3.9`,
  the pinned dependencies, the `kinematic` console script) plus the README's setup section, its
  diagnostics/logging/profiling sections, and a **final macOS verification checklist** covering fresh
  setup, both permissions, the dry-run pipeline, every gesture, the gesture pause, logging, packaging
  and repeated launch/cleanup. The wheel is built with a separate interpreter to verify the metadata
  without installing into the 3.9 environment.

**Verification evidence** for the phase's own bar ("fresh setup, denied permissions, unavailable
cameras, low light, partial tracking, sustained gestures, repeated launch/cleanup, full regression
checks"): `tests/test_phase9_e2e.py` (19 cases) walks exactly that list with injected probes — a denied
camera and a missing index get their own check failures, a dark frame is flagged while the pipeline
stays silent, a three-landmark (partial) detection is ignored rather than acted on, 30 taps give 30
clicks with no drift, three launch/close cycles leave no capture thread, and one scripted session is
replayed call by call (both wheel axes, no click after scrolling begins, every move on screen, the
tracking loss noticed). `tests/test_phase9_diagnostics.py` (31) and `tests/test_phase9_logging.py` (18)
cover the checks and the log, including the `--diagnose` CLI exit status and the absence of per-frame
lines at `info`. The remaining half of the checklist needs hardware and stays manual by design — the
suite must never move the user's cursor.

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
- **The controls must be able to reach every corner of the screen.** That is why the default scheme is
  absolute position and why the dial has a floor at gain 1.0 (`sensitivity_min` =
  `sensitivity_reach_floor`): one frame width of hand travel must cover at least one screen width. A
  setting that silently strands a strip of the screen at an edge is a bug, not a preference — so when a
  knob can make the screen unreachable, it clamps, and if it is overridden anyway the runtime says so
  (`reach_warning`, and `! edges unreachable` on the HUD). Reach is judged with the hand *stopped* at
  the edge (`--reach-dwell`, the way a user points at a corner), because a lagging filter is expected to
  trail while the hand is still travelling — that is the smoothness trade, not a reach failure.
- **The cursor reference is never frozen to hide jitter; the filter smooths it.** A deadzone on the
  absolute reference trades a continuous, tiny error for a discontinuous, large one — the hold-then-jump
  that was reported as "snapping to a grid" (measured: 80 of 90 frames held, then a ~65 px jump, and
  +39 px of trailing). `deadzone` therefore defaults to 0 and, when used, is a radius in frame units
  (documented in px: `0.02` ≈ 49 px), never a per-axis box.
- **A numeric parameter's units are part of its contract.** `one_euro_beta` is defined for pixel-scale
  coordinates; the runtime feeds it screen pixels, where any value ≥ 0.1 saturates the cutoff and turns
  the filter off (measured alpha 0.96 at 0.5 px/frame with `beta=8`). When wiring a library's tuned
  constant, either feed it the units it was tuned for or re-derive it — and measure, because this class
  of bug is invisible in the code and described by the user as "it feels bad".

- **A knob that can only make things worse ships off, with the measurement beside it.** `move_max_hz`
  defaults to 0 because the uncapped path is already at the camera's rate; the limiter exists for the
  case the measurement found (a slow adapter) and the cost is documented in numbers (§2 "Cursor move rate
  limit", gotcha 8.15) rather than defended in prose. Same shape as `deadzone`: the value that is right
  for the default configuration is the one the harness measured, not the one that sounds careful.
- **One decided rule, one helper, with its variants as explicit arguments.** Phase 6 ended the second
  copy of the pinch arithmetic by giving `gesture.is_pinch()` a `metric` argument
  (`PINCH_METRICS = ("euclidean", "l1")`) instead of leaving the legacy engine to compute L1 inline: the
  default stays Euclidean, `"l1"` reproduces the prototype exactly, and an unknown metric raises. When a
  compatibility path needs different arithmetic, parameterise the one implementation and prove the
  equivalence — do not fork it.
- **Behaviours that only *report* are the ones that can be on by default.** The last-action line
  (`show_last_action`) draws what the pipeline already did and emits nothing itself, so it is on; the
  rate limit changes output and is off. Ask what a new default costs before making it one.
- **An increment is not a position, so the same cap needs opposite handling.** `MoveRateLimiter`
  *supersedes* a held frame (the newest cursor position wins and nothing is queued); `ScrollRateLimiter`
  *accumulates* it (dropping a scroll amount would lose that much scrolling). Same timing code, opposite
  policy, and the distinction is the reason there are two classes instead of one — see §2 "Two-finger
  scrolling".
- **Rounding per event is a distance bug when the events are small.** Every scroll amount used to be
  truncated to an int before it reached the wheel; with a slow hand each frame asks for a fraction of a
  unit, and 25% of a measured sweep went missing (135 units of 180). Amounts now carry their remainder
  through the accumulator. When a value is quantised for an API, ask what happens to the discarded part
  — if it accumulates anywhere, it must accumulate *here*.

---

## 7. Pre-report checklist

Before reporting a phase complete:

- [ ] `python3 -m unittest discover -s tests` passes (all old + new tests).
- [ ] `python3 kinematic/main.py` and `python3 -m kinematic.main` both still start.
- [ ] No test or default configuration can move the real cursor, click or scroll.
- [ ] New behaviour is additive; existing working behaviour is unchanged.
- [ ] If anything visual changed (preview orientation, overlay, readout), the layer drawn and the data it
      draws are in the same frame of reference — see gotcha 8.17 — and `--preview-flip` was checked in the
      mode the change was about.
- [ ] Any measured number is backed by a recorded run (seed + config stored), not a guess.
- [ ] Anything touching the pointer path (reference, deadzone, filter, gain, mapping, rate limit) is
      checked with `python3 tools/measure_smoothness.py` and, for reach, `python3 tools/measure_loop.py
      --synthetic-hand --active --sweep-reach 30 --reach-dwell 12`; a change to `move_max_hz` or to the
      move path also gets `--rate-sweep`.
- [ ] Anything touching the click/scroll path is checked with `python3 tools/measure_loop.py
      --synthetic-hand --active --synthetic-pinch left|right|two_finger --frames 150`: the left/right
      verdict line must read `pass - one click per deliberate tap`, and the run must not print
      `button state     : WARNING` (that line means presses and releases did not balance, i.e. a real
      button left down). The `two_finger` variant of *this* command is a click-suppression check only —
      it reads `0 clicks` and, since Phase 7's per-axis dead zone, **0 whole scroll units** too, because
      a 6-frame tap never moves the two-finger midpoint past the dead zone. That is expected, not a
      failure: use `--synthetic-scroll` (next line) to check the scroll path itself.
- [ ] A change to the scroll path is checked with `python3 tools/measure_loop.py --synthetic-hand
      --active --synthetic-scroll vertical|horizontal|diagonal --frames 150`, where the verdict line must
      read `pass` with the expected sign on the expected axis (and 0 events on the other one).
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

**8.12 `pyautogui`'s tween cannot be used to smooth a 30 fps cursor — and its `moveTo` silently
ignores small durations.** `_mouseMoveDrag` only tweens when `duration > MINIMUM_DURATION`, and
`MINIMUM_DURATION` is **0.1 s**, with the per-step sleep floored at `MINIMUM_SLEEP = 0.05 s`. So
`pyautogui.moveTo(x, y, duration=0.01)` does *nothing* (measured 0.2-0.7 ms per call, i.e. a teleport),
and passing a value above the threshold blocks the loop for **171 ms per move** (measured, `duration=0.11`
→ 5.2 camera frame intervals; the loop drops to ~6 fps and the cursor lags permanently). There is no
usable per-frame tween on this backend: smoothness has to come from the reference and the filter. Anything
like "just add a short duration to each move" is a dead end — hence `mouse_move_duration` was removed
rather than shipped as a knob with an empty useful range.
*Measure it without moving anything:* patch the platform hook (`pyautogui._pyautogui_osx._moveTo`) to a
no-op and time `moveTo` — `pyautogui.moveTo` resolves the platform move through `platformModule`, so the
stub is effective, and the real cursor never moves.

**8.13 A filter fed the wrong units fails silently and looks like "it feels bad".** The runtime smooths
screen-pixel positions, but One Euro's `beta` is defined for pixel-scale coordinates: with
`one_euro_beta=8.0` the cutoff is `min_cutoff + beta × speed`, which at 30 px/s (1 px/frame) is 244 Hz —
`alpha 0.96`, i.e. no filtering at all. The filter still *runs* and still returns plausible numbers, so
nothing in the code or the tests looked wrong; the symptom was the user reporting a stutter.
`one_euro_beta` is now 0.0 with a 4 Hz cutoff. When wiring any tuned constant, check the units of both
sides and measure the response (`tools/measure_smoothness.py`, `--one-euro-beta`), not just that it runs.

**8.14 A derived metric's sampling rule is part of the measurement.** `measure_smoothness.py` reports how
far the cursor trails the hand; the first version sampled the *ideal* path by the move's index in the move
list. That is right only while moves are one per frame — with held frames (or, later, a rate limit) move
#i happens several frames in, so the tool was comparing the cursor against a point the hand had not
reached yet and over-reported the gap. Corrected to sample on the *frame each move was emitted on*, the
old deadzone's "**+39 px of trailing**" is really **+15.5 px** (same run, same seed). Before quoting a
derived number, check what it is sampled against; a wrong sampling rule produces plausible numbers that
survive code review and tests, exactly like 8.13's unit mismatch.

**8.15 A rate limit only helps below the rate you are already achieving.** The Phase 5 limiter was
motivated by "the adapter can be slow", and the obvious assumption — "cap the moves and the loop gets its
time back" — is only true when the cap is *under* what the loop currently does. Measured with
`--mouse-pause 0.1` (a ~113 ms adapter): 6.78 fps uncapped, **6.76 fps with `--move-max-hz 8`** (no
change: the loop is already slower than the cap, so nothing is suppressed), then 17.4 fps at 5 Hz and
22.9 fps at 3 Hz. The cost side is just as concrete: the same cap triples the mean step size on a slow
sweep (6.2 px → 18.7 px at 10 Hz), which is the hold-then-jump feel the smoothness work removed. Hence
`move_max_hz = 0` by default and a table in §2 instead of an opinion.

**8.16 Quantising a per-event value discards the movement the events are made of.** Scroll amounts were
converted to `int` at each emission. With a hand moving slowly that is well under one wheel unit per
frame (a hand width is 900 units, and a frame of a slow sweep is ~1), so the discarded fraction was the
whole event and the total was 25% short: measured on a controlled sweep, **135 units emitted of the 180
the hand travelled**. Nothing looked wrong — each event was a plausible small number. Amounts now enter
`ScrollRateLimiter` as floats and the remainder carries to the next event (`flush()` at the end of the
gesture sends the last whole units). The same trap applies to any accumulated quantity that gets rounded
before it is summed: round at the boundary, not per sample.

**8.17 An overlay drawn in a different frame of reference than its data looks "reversed".** The preview
composed the *captured* frame while every landmark came from the *mirrored* frame MediaPipe was fed
(`process_frame` flips for detection) — so the skeleton was drawn reflected beside the hand, and the
footage on screen was the raw camera view while the pointer moved according to the mirrored one. Two
symptoms, one cause: an overlay and its data must share one coordinate system. The flip is now one
config field (`preview_flip`) that decides both halves together (`preview.display_frame` for the image,
`preview.landmark_flip` for the landmarks), each of the four modes is tested against a marker-block hand
at a known asymmetric world position, and the readout text is drawn *after* the flip so it is never
mirrored. When something on screen looks backwards, check what frame each layer is in before flipping
individual ones.

**8.18 A test hand's scale constant is part of its geometry, and a "release" pose inside the
hysteresis band silently stops the next click.** A synthetic hand is only as good as the constant you
compute its distances with. In `test_phase9_e2e.py` a hand whose wrist is at (0.50, 0.60) and whose
middle knuckle is at (0.50, 0.45) has a `hand_scale` of **0.15**, and the file's first version said
`SCALE = 0.10` (copied from the phase-8 file, where the geometry had been *designed* to be 0.10). Every
ratio was therefore 1.5× off: the pose meant as "released, 0.70" was really 0.47, which is above the
0.40 engage threshold but **below the 0.56 release threshold** — so the pair stayed engaged, `_armed`
stayed False, and the second tap of a 30-tap script never clicked (a suite that reported `1 != 30`).
The tell was in the numbers, not the code: the ratio printout showed 0.47 where the test intended 0.70.
When a fixture builds geometry by hand, derive the constant from the coordinates or assert it — a
pose-based test is otherwise testing the arithmetic of its own fixture.

**8.19 A library that logs has to install a `NullHandler`, or `logging.lastResort` prints bare event
names.** With no handler anywhere in the chain, Python routes `WARNING` and above through
`logging.lastResort`, a `StreamHandler` at `WARNING` on stderr that prints only the *message* — so this
project's `log_event(logger, "loop_error", ...)` appeared in the middle of a test run as the single word
`loop_error`, with none of its fields and no JSON. Every library that logs should install a
`NullHandler` on its own logger at import (which is what `kinematic/log.py` now does), and an
application should call `configure()` — which replaces handlers, so the `NullHandler` does not swallow
the real output. If a structured log ever shows a bare event name instead of a JSON object, the handler
is missing rather than the formatting being wrong.

**8.20 A single-call "click" cannot represent "hold", and a release must not be gated by the
debounce rules that gate the press.** `pyautogui.click()` (and this project's `MouseOutput.click()`)
is a `mouseDown()` immediately followed by a `mouseUp()`. There is no way to make that call "take
longer" and get a held button out of it — `duration=` exists on `moveTo`, not `click` — so a gesture
meant to support holding *and* dragging needs the down and the up as two separate adapter calls, with
the up tied to the gesture's own release. The second half is the subtle one: if the up is routed
through the same hold-time/cooldown/`require_neutral_between` rules that decide when a down may fire
(the obvious, tidy implementation), then any state-machine quirk that delays or blocks the "leaving the
gesture" transition leaves a **real button stuck down on the user's desktop** — a much worse failure
than the missing feature. Tie the release to the raw per-pair engagement flag and let it bypass the
gates; a press is debounced, a release is not. See §2 "Pinch press-and-hold: hold to drag".

**8.21 A harness that stubs "the calls the code makes" quietly stops stubbing when the code changes.**
`tools/measure_loop.py` and `tools/profile_loop.py` both replace pyautogui's `moveTo`/`click`/`scroll`/
`hscroll` with counters so they can run the real pipeline ("mouse intercepted, nothing moves"). Adding
`mouseDown`/`mouseUp` to the click path did not raise anywhere — those harnesses simply stopped seeing
the clicks (their verdicts read `0 clicks for 5 taps`) **while really pressing the user's mouse button**,
because an unpatched `mouseDown` is a real `mouseDown`. Two lessons: when a stub list enumerates the
methods the adapter can call, enumerate them from the *adapter's interface* (`MouseOutput`), not from the
call sites you happen to know about — and make the harness assert its own coverage (the click verdict
going from `pass` to `0 clicks` was the tell; a `presses != releases` warning is the stuck-button tell).
Stubbing the adapter *class* rather than the third-party module would have made this impossible.

