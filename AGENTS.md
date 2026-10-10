# AGENTS.md — Kinematic (webcam hand mouse)

Context file for AI coding agents working in this repo. Read this before touching code.
Last updated: 2026-10-10 (phase status audit added — see §2 "Phase status").

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
├── README.md
├── requirements.txt
├── kinematic/
│   ├── __init__.py        # re-exports HandMouseConfig, is_pinch, is_right_click_gesture, smooth_point
│   ├── config.py          # HandMouseConfig dataclass (9 fields)
│   ├── gesture.py         # pure helpers: point_distance, smooth_point, clamp, landmark_points,
│   │                      #   is_pinch, is_right_click_gesture, relative_move_delta, normalize_screen_point
│   └── main.py            # HandMouseApp: capture loop + mouse control (monolithic, no seams yet)
└── tests/
    └── test_gesture_logic.py   # 4 unittest cases covering gesture.py only
```

Git: one commit (`b500351 Initial commit`), only `.gitattributes` and `README.md` are tracked —
**`kinematic/`, `tests/`, `requirements.txt` are untracked**, and there is no `.gitignore`.

### Verified working commands (run from the repo root)

```bash
python3 kinematic/main.py            # script invocation (works — see gotcha 8.1)
python3 -m kinematic.main            # module invocation (documented in README)
python3 -m unittest discover -s tests   # 4 tests, all pass
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

Also present but not used by this project: Homebrew `python3.13` / `python3.14`, `~/.local/bin/python3.11`.

### Known gaps in the current prototype (read from the code, not yet fixed)

- **Real mouse control is live on startup.** `HandMouseApp.run()` moves the real cursor, clicks and
  scrolls the moment a hand is seen. No dry-run mode, no pause, no opt-in. This is the biggest
  safety gap and Phase 1/5/8 of the roadmap close it.
- **No scroll mode.** `_handle_scroll()` scrolls from *index-finger* movement alone (gain hardcoded
  `-900`), so moving the pointer scrolls the page. `config.scroll_threshold` is used, but the
  two-finger scroll gesture does not exist yet.
- **Duplicated pinch logic.** `_handle_clicks()` computes pinch as a raw L1 distance
  (`abs(dx)+abs(dy) < click_threshold`) instead of calling the already-imported, unit-tested
  `gesture.is_pinch()`. Note `is_right_click_gesture()` means *thumb–index AND index–middle* both
  pinched, which is stricter than a "right pinch".
- **Thresholds are absolute landmark distances**, not relative to hand scale — they drift with
  distance from the camera. Phase 4 replaces them with hand-scale-relative thresholds.
- **Mapping is not calibrated.** `_update_cursor()` multiplies the normalized point by
  `pointer_scale = 1.8` and then clamps to the screen, so the reachable region is skewed off-centre.
- **State is incomplete on tracking loss.** The `else` branch resets `last_index`, `last_pinch`,
  `last_right_click`, but not the cursor position / smoothing state, so the pointer lurches on reacquire.
- **No test coverage for `main.py`** (the loop, the mapping, the click emission). Only `gesture.py`
  is covered.
- **No packaging** (no `pyproject.toml`), no `.gitignore`, no logging.
- **Dead helper:** `gesture.clamp()` is defined but never called — `main.py` clamps with bare
  `min(max(...))` instead. `gesture.is_pinch()` is imported into `main.py` and never called either.

### Phase status — which phases are actually implemented (audited 2026-10-10)

**No phase is complete.** This repo is a *pre-Phase-1 prototype*: it performs real mouse output on
startup while none of the safety layer that Phase 1 is supposed to establish exists yet. The prototype
touches the later phases' subject matter (capture, smoothing, clicks, scroll) but not in the shape the
phases specify.

*How this was audited:* every non-`.git` file was listed and read; the source was grep'd for each
phase's marker concepts (`pause`, `mock`, `record`, `adapter`, `dry`, `Enum`, `state`, `hyster`,
`putText`, `fps`, `try:`, `finally`, `logging`, `calibrat`); the deliverable bullets below were checked
one by one against `kinematic/main.py`, `kinematic/gesture.py`, `kinematic/config.py` and `tests/`;
and `python3 -m unittest discover -s tests` was run (4 tests, all passing). "0 hits" statements below
mean the concept does not appear anywhere in `kinematic/` or `tests/`.

| Phase | Status | Exists | Missing |
| --- | --- | --- | --- |
| 1 — Foundation & safety | ⛔ **1/5**, safety **0/5** | `kinematic/` package layout, `config.py`/`gesture.py`/`main.py`, `requirements.txt`, runnable `unittest` suite | Camera/gesture/smoothing/**output modules** (everything is in `main.py`), **paused startup (0 hits)**, **emergency pause/exit (0 hits)**, **mock output adapter (0 hits for mock/record/adapter/dry)**, macOS permission docs (README documents none) |
| 2 — Capture pipeline | 🟡 **3/5** | `cv2.VideoCapture` + `isOpened()` error, `cv2.flip` mirroring, BGR→RGB, `mp.solutions.hands`, invalid-frame error | Landmark + FPS overlay (`putText`/`fps` = 0 hits), internal landmark representation (raw MediaPipe objects flow straight through), exception-safe cleanup (`try`/`finally` = 0 hits — both `RuntimeError` paths leak the camera handle) |
| 3 — Geometry/mapping/smoothing | 🟡 **2/5 + 1 partial** | `point_distance`, `smooth_point` (fixed-factor lerp ≈ EMA), `clamp` (uncalled), `landmark_points`, `is_pinch`, `normalize_screen_point`, `relative_move_delta` | Hand-scale calc, midpoints, **configurable screen control region** (`pointer_scale = 1.8` + clamp instead), One Euro smoothing, **filter reset on tracking loss** |
| 4 — Gesture state machine | ⛔ **0/5** | Only a rising-edge boolean + `debounce_seconds` in `main.py` | States/`Enum` (0 hits), hysteresis, hold times, hand-scale-relative thresholds, pure recognizer with injected timestamp, scroll precedence |
| 5 — Cursor movement | 🟡 **1/5** | `pyautogui.moveTo(x, y, duration=0)` on smoothed coordinates | Dry-run/recording adapter, opt-in gate (real output is the only mode), rate limiting/coalescing, paused/untracked suppression (those states don't exist) |
| 6 — Left & right clicks | 🟡 **3/5 + 1 partial** | Left click on pinch rising edge, right click via `is_right_click_gesture` rising edge, shared cooldown; release-then-re-arm works | Scroll precedence, pause suppression, last-action status; pinch is re-implemented inline with an L1 metric — **`is_pinch` is imported and never called** |
| 7 — Two-finger scroll | ⛔ **0/5 + 1 partial** | `pyautogui.scroll()` driven by index-finger delta | Two-finger (index/middle) reference point, horizontal scroll, dead zone, sensitivity config (gain hardcoded `-900`), direction inversion, rate limiting, **click suppression while scrolling** (a pinch while moving both clicks and scrolls) |
| 8 — Pause/runtime/feedback | ⛔ **0/5** | Nothing | Pause toggle (`pause` = 0 hits), runtime coordinator, paused/active/tracking-lost status, exception-safe teardown (cleanup only runs on the `q` path) |
| 9 — Calibration/hardening | ⛔ **0/5 + 1 partial** | `HandMouseConfig` with 9 tunables | Mirroring/margins/sensitivity/hotkey knobs, diagnostics, logging (0 hits), profiling, packaging (`pyproject.toml`), final setup docs |
| Tests | helper-only | 4 passing tests: `is_pinch`, `is_right_click_gesture`, `smooth_point`, `point_distance` | `clamp`, `relative_move_delta`, `normalize_screen_point`, and **all of `main.py`** (loop, mapping, click emission, scroll) |

**What this means for the next session:**

- **Phase 1 is the honest next step, not Phase 2.** Capture, smoothing and clicks mostly exist, but they
  sit on top of a missing safety layer, and Phase 1 is what makes every later phase testable.
- **Phase 1 will have to *remove* live behaviour — this is an approved exception to "strictly additive".**
  Real output must become opt-in behind an adapter; that means changing existing code paths, which rule 3.1
  otherwise forbids. Keep the old path reachable (e.g. a `real_output=True` flag) so nothing is lost, and
  state the exception in the phase report. The safety contract in §4 outranks the additive rule here.
- **Several phases replace rather than extend** (Phase 7's scroll, Phase 4's thresholds, Phase 5's mapping).
  Add the new module, switch the runtime over, and leave the prototype path intact until the switch is
  verified — same additive discipline, just with a deliberate hand-off.

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

---

## 5. Roadmap

Nine phases. Each phase lists deliverables, the verification bar, and a ready-to-paste prompt. Each
phase heading also carries a `Status:` line taken from the audited table in §2 — read it before
starting, because the prototype already covers slices of Phases 2/3/5/6 and none of Phase 1/4/8.
Phases 1–4 are pure/offline and are the highest-value work first; per the plan real desktop output
stays disabled until Phase 5 and is opt-in from there on, but note that the *current prototype breaks
that already* by driving the real cursor the moment it starts.

### Phase 1 — Project foundation and safety

**Status: ⛔ 1/5, safety 0/4 — do not start Phase 2 before this.** What exists is only bullet 1:
the package layout, `requirements.txt` and a runnable test suite. No paused startup, no mock output
adapter, no permission docs; see the §2 table.

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

### Phase 2 — Webcam and MediaPipe pipeline

**Status: 🟡 3/5 (bullets 1–3 done, caveat on cleanup).** Capture/MediaPipe/mirroring all work, but
they live inside `main.py` with the mouse control, there is no `try`/`finally` teardown, and the preview
shows a bare frame. The remaining work is separation + overlay + a landmark type.

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

**Status: 🟡 2/5 + 1 partial.** `gesture.py` already holds most of the pure geometry (bullet 1 partial —
no midpoints or hand scale) and a working fixed-factor EMA (bullet 4). Missing: a configurable screen
control region and the tracking-loss filter reset. Extend `gesture.py`; don't rewrite it.

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

**Status: 🟡 1/5.** Only real output exists (`pyautogui.moveTo`); there is no dry-run adapter, no
opt-in gate, no throttling, and no paused/untracked state to suppress on. This is the phase that has to
convert the prototype's unconditional cursor control into an opt-in adapter.

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

**Status: 🟡 3/5 + 1 partial.** Rising-edge detection, release-then-re-arm and the cooldown all work;
what's missing is scroll precedence, pause suppression, and status feedback. Also switch the inline L1
pinch test over to the already-tested `gesture.is_pinch()`.

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

**Status: ⛔ 0/5 — nothing exists.** No pause toggle anywhere in the code base, no coordinator, and
cleanup only runs when the user presses `q` (an exception in the loop leaves the camera and window open).

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
separately from gesture logic.

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
