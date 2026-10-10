# AGENTS.md — Kinematic (webcam hand mouse)

Context file for AI coding agents working in this repo. Read this before touching code.
Last updated: 2026-10-10 (Phases 1–3 of the roadmap complete — see §2 "Phase status" and §5).

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
    └── test_pointer_reference.py # 21 cases: wrist/palm reference, curl must not move the cursor, scroll guard
tools/
├── compare_smoothing.py         # seeded jitter-vs-lag comparison of the cursor filters
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
python3 -m unittest discover -s tests       # 112 tests, all pass
python3 tools/compare_smoothing.py          # seeded jitter vs lag table for the cursor filters
python3 tools/measure_loop.py --synthetic-hand --active --fingers-only   # the finger-curl regression demo
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
- **No scroll mode.** `runtime.handle_scroll()` still drives scrolling from *index-finger* movement
  (gain hardcoded `-900`), but only while the hand itself holds still, so pointing no longer scrolls the
  page. A real two-finger scroll gesture with a dead zone and sensitivity is still Phase 7.
- **Duplicated pinch logic.** `runtime.handle_clicks()` computes pinch as a raw L1 distance
  (`abs(dx)+abs(dy) < click_threshold`) instead of calling the already-tested `gesture.is_pinch()`.
  Note `is_right_click_gesture()` means *thumb–index AND index–middle* both
  pinched, which is stricter than a "right pinch".
- **Thresholds are absolute landmark distances**, not relative to hand scale — they drift with
  distance from the camera. Phase 4 replaces them with hand-scale-relative thresholds.
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

### Phase status — which phases are actually implemented (audited 2026-10-10)

**Phase 1 is complete (5/5). Phases 2 and 3 are complete too (5/5 each).** Before Phase 1 this repo
was a prototype that drove the real mouse from startup with none of the safety layer in place; that is
fixed — see §5 for what each phase landed. Phases 4–9 are not implemented in the shape the plan asks
for, though the prototype's subject matter (clicks, scroll) exists in some form; read each phase's own
`Status:` line below before starting work.

*How this was audited:* every non-`.git` file was listed and read; the source was grep'd for each
phase's marker concepts (`pause`, `mock`, `record`, `adapter`, `dry`, `Enum`, `state`, `hyster`,
`putText`, `fps`, `try:`, `finally`, `logging`, `calibrat`); the deliverable bullets were checked one by
one against the package modules; and `python3 -m unittest discover -s tests` was run. The "0 hits" and
per-phase counts below are from that pre-Phase-1 audit; the Phase 1 row and the Tests row have been
updated since.

| Phase | Status | Exists | Missing |
| --- | --- | --- | --- |
| 1 — Foundation & safety | ✅ **5/5 (2026-10-10)** | Modules split: `capture.py`, `gesture.py`, `output.py` (adapters), `runtime.py` (coordinator), `main.py` (CLI); `requirements.txt`; 112-test suite; **paused startup** with SPACE toggle and ESC/q quit; **`RecordingMouseOutput` mock adapter** used by tests and `--dry-run`; macOS permission docs in README | — |
| 2 — Capture pipeline | ✅ **5/5 (2026-10-10)** | `capture.py`'s threaded `ThreadedCamera` (`isOpened()` error, buffer size, device-error reporting); `cv2.flip` mirroring, BGR→RGB, `mp.solutions.hands`, invalid-frame error, exception-safe teardown (the runtime's `try/finally`); **`landmarks.py`** (Landmark/HandLandmarks, the single protobuf conversion point, incomplete detections flagged); **preview overlay** of the loop rate *and* the 21-point skeleton (`preview.py`, drawn with cv2, `show_landmarks`) | — |
| 3 — Geometry/mapping/smoothing | ✅ **5/5 (2026-10-10)** | `gesture.py`: distances, `midpoint`/`midpoints`, `pinch_distances`, `hand_scale` + `hand_span`, `clamp`/`clamp_point`, `mirror_point`, EMA; `mapping.py`: `Region` + `map_point_to_screen` (frame region → screen region, clamping, mirroring); `smoothing.py`: EMA + 1-euro filters with `reset()`, wired into the runtime with a tracking-loss reset; 44 deterministic tests for distances, clamping, smoothing, latency and reset | Midpoints are available but not yet *used* by a gesture; One Euro parameters are tuned for this camera rather than for everyone |
| 4 — Gesture state machine | ⛔ **0/5** | Only a rising-edge boolean + `debounce_seconds` in `runtime.py` | States/`Enum` (0 hits), hysteresis, hold times, hand-scale-relative thresholds, pure recognizer with injected timestamp, scroll precedence |
| 5 — Cursor movement | 🟡 **3/5 + 1 partial** | Palm/wrist-relative pointer pipeline (see §2 "Pointer reference") with smoothing, clamp and `min_move_pixels`; **dry-run and real adapters** (`output.py`); **opt-in gate** (starts paused, `--dry-run`); paused/bounds suppression; re-anchor on reacquire | A real rate limit/coalescer, an explicit *tracking-lost* state (Phase 4 defines it), and the absolute mapping's calibration |
| 6 — Left & right clicks | 🟡 **3/5 + 1 partial** | Left click on pinch rising edge, right click via `is_right_click_gesture` rising edge, shared cooldown; release-then-re-arm works; **suppressed while paused** (Phase 1's `can_emit` gate, including a pinch held across a resume) | Scroll precedence and last-action status; the pinch test is re-implemented inline with an L1 metric instead of `gesture.is_pinch()` |
| 7 — Two-finger scroll | ⛔ **0/5 + 1 partial** | `pyautogui.scroll()` driven by index-finger delta | Two-finger (index/middle) reference point, horizontal scroll, dead zone, sensitivity config (gain hardcoded `-900`), direction inversion, rate limiting, **click suppression while scrolling** (a pinch while moving both clicks and scrolls) |
| 8 — Pause/runtime/feedback | 🟡 **3/5 + 1 partial** | `runtime.py` coordinator; configurable keyboard toggle (`toggle_pause_keys`) with paused startup; PAUSED/ACTIVE status on the preview; exception-safe teardown (`try/finally`) | A *tracking-lost* status, the optional gesture pause, and tests for toggle idempotence, recovery and window-close cleanup |
| 9 — Calibration/hardening | ⛔ **0/5 + 1 partial** | `HandMouseConfig` with 9 tunables | Mirroring/margins/sensitivity/hotkey knobs, diagnostics, logging (0 hits), profiling, packaging (`pyproject.toml`), final setup docs |
| Tests | 112 passing | gesture helpers; config defaults; cursor-move guard; camera error path; the Phase 1 safety suite; the pointer-reference suite (curl must not move the cursor, gain/clamp/deadzone maths, re-anchor, scroll guard); the landmark suite (protobuf conversion, incomplete hands, skeleton pixels); the Phase 3 geometry suite (distances, midpoints, clamping, region mapping); the Phase 3 smoothing suite (EMA + 1-euro behaviour, jitter, latency, tracking-loss reset) | MediaPipe's own detections, the real camera path, and the click/scroll branches beyond those cases |

**What this means for the next session:**

- **Phases 1–3 are done.** Capture, geometry, mapping, smoothing and the safety layer are all in place and
  tested, so every remaining phase can be built on a pipeline that cannot hurt the desktop and does not
  need a camera in the test suite.
- **Phase 4 is the next real step**, and it is the one that pays off in behaviour: an explicit
  state machine (neutral / left pinch / right pinch / two-finger scroll / paused / tracking-lost) with
  hysteresis, hold times and hand-scale-relative thresholds. `hand_scale`, `hand_span`, `pinch_distances`
  and `midpoints` were added in Phase 3 precisely so Phase 4 can use relative thresholds instead of the
  absolute ones it has now; the gesture logic is still two booleans and a debounce in `runtime.py`.
- **Retrofit, don't rewrite.** Each phase so far added a module and switched the runtime over while leaving
  the previous path reachable (EMA next to One Euro, index reference next to palm, absolute next to
  relative). Keep doing that: it is what makes the changes reviewable and revertible.
- **Phases 6–8 still own the rough edges** listed in §2's known gaps: the click path uses an inline L1
  pinch test instead of `gesture.is_pinch()`, scrolling is still index-finger-driven (now correctly gated
  on a still hand), and there is no status/teardown work beyond what Phase 1 built.

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

**Status: ⛔ 0/5 — nothing exists.** The only gesture state today is two booleans and a timestamp in
`main.py`. This phase is greenfield; keep it pure (no cv2, no camera, no `time.time()`).

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

**Status: 🟡 3/5 (2026-10-10).** The adapter split landed with Phase 1 (`RealMouseOutput` /
`RecordingMouseOutput`), cursor output is opt-in (the app starts paused; `--dry-run` emits nothing at all),
and moves smaller than `min_move_pixels` are skipped rather than issued. What remains: a genuine rate
limit/coalescer, and suppression on an explicit tracking-lost state, which Phase 4 defines.

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

**Status: 🟡 3/5 + 1 partial.** Rising-edge detection, release-then-re-arm, the cooldown and pause
suppression all work (§4 covers the held-pinch-across-resume case). What is missing is scroll precedence
and status feedback; also switch the inline L1 pinch test over to the already-tested `gesture.is_pinch()`.

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

**Status: ⛔ 0/5 + 1 partial, and the partial one is non-conforming.** Scrolling exists but it is driven
by the *pointer* finger's movement with a hardcoded gain, so the present behaviour has to be replaced,
not extended. Treat this as the phase most likely to regress the current UX — verify before switching.

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

**Status: 🟡 3/5 + 1 partial (2026-10-10).** Phase 1 built the coordinator and the pause path: a
configurable keyboard toggle, paused startup, PAUSED/ACTIVE feedback on the preview, and cleanup in a
`finally` block that also runs when the loop raises. Remaining: tracking-lost status, the optional gesture
pause, and explicit tests for toggle idempotence, recovery and window-close teardown.

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
