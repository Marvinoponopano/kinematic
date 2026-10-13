from dataclasses import dataclass
from typing import Tuple


@dataclass
class HandMouseConfig:
    camera_index: int = 0
    window_name: str = "Kinematic"
    smoothing: float = 0.35
    deadzone: float = 0.0
    #   Absolute mode: from 2026-10-10 this is 0 (no freeze) and it is what made the cursor feel
    #   like it snapped to a grid. A non-zero value holds the palm reference until the hand moves
    #   that far, and since it is in *frame* units it is large on screen — the old 0.02 was ~49 px
    #   horizontally, so the cursor sat still and then jumped. Jitter is handled by the cursor
    #   filter instead (see smoothing_mode), which smooths continuously rather than in steps. Set
    #   it only if you want the freeze; tools/measure_smoothness.py shows exactly what it costs in
    #   held frames, jump size and trailing. It is a radius, not a per-axis box, when non-zero.
    click_threshold: float = 0.06
    right_click_threshold: float = 0.08
    scroll_threshold: float = 0.04
    debounce_seconds: float = 0.18

    # --- performance tuning (added 2026-10-10) ------------------------------
    # Measured effects, macOS + FaceTime camera:
    #   mouse_pause      library default 0.1 adds ~100 ms to EVERY mouse call
    #   model_complexity 0 (lite) is ~1.5 ms/frame cheaper than 1 (full)
    #   preview_scale    0.5 saves ~6 ms/frame over drawing the frame at full size
    #   min_move_pixels  skips an ~11 ms moveTo when it cannot move the cursor visibly
    # capture_width/height do not change the camera's own frame rate (read() blocks
    # for the frame interval at any resolution) but keep the pipeline small.
    capture_width: int = 1280
    capture_height: int = 720
    capture_fps: int = 30
    capture_buffer_size: int = 1
    model_complexity: int = 0
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.7
    mouse_pause: float = 0.0
    min_move_pixels: float = 0.5
    preview_scale: float = 0.5
    preview_flip: str = "horizontal"
    #   Which way the preview footage is flipped for display, named as a flip of the *captured* frame:
    #   "horizontal" (default) shows the mirror a webcam selfie shows; "none" shows exactly what the
    #   camera delivers; "vertical" / "both" are for a camera mounted upside down. It also decides how
    #   the landmark skeleton is mirrored so it lands *on* the hand rather than reflected beside it —
    #   and since MediaPipe analyses the mirrored frame, "horizontal" is the orientation in which the
    #   pointer follows your hand on screen (hand right = cursor right). Change it with --preview-flip.
    show_preview: bool = True
    show_fps: bool = True
    show_landmarks: bool = True

    # --- safety (Phase 1) ---------------------------------------------------
    # The app starts paused: frames are analysed and previewed, but nothing is sent
    # to the cursor until the user presses a toggle key. dry_run swaps the real
    # output adapter for a recording one, so the whole pipeline can run with no
    # desktop side effects at all (tests always do this).
    start_paused: bool = True
    dry_run: bool = False
    toggle_pause_keys: Tuple[str, ...] = (" ", "p")
    quit_keys: Tuple[str, ...] = ("q", "\x1b")

    # --- pointer reference (added 2026-10-10) --------------------------------
    # The cursor used to be driven by the index fingertip, so curling or spreading the
    # fingers dragged it across the screen while the hand itself was still. It is now
    # driven by the wrist/palm, which barely moves when the fingers do.
    pointer_mode: str = "absolute"
    #   "absolute" (default since 2026-10-10): the cursor maps to the hand's *position* in the
    #       frame — the inner 70% of the frame covers the whole screen, so every screen edge and
    #       corner is directly reachable, and the sensitivity dial scales that reach. Moving the
    #       hand to a comfortable spot therefore moves the cursor with it (no clutch either way).
    #   "relative": the cursor follows *hand displacement*, like a mouse. Finger curl still
    #       contributes nothing, but the screen edges can only be reached by accumulating travel
    #       (repeated sweeps), which is what makes a low gain feel like it "cannot reach the edge".
    pointer_reference: str = "palm"
    #   "palm" (default) = mean of wrist + the four knuckles; "wrist" = wrist joint
    #   only (most stable); "index" = the legacy fingertip, which finger curl moves.
    pointer_sensitivity: float = 1.0
    #   The dial's value: screen widths of cursor travel per frame width of hand travel. 1.0 is the
    #   neutral gain — the whole screen is reachable in one frame width of hand travel (absolute
    #   mode maps the inner 70% of the frame onto the screen itself; relative mode needs a full
    #   sweep to cross). It is also the floor: below 1.0 the screen edges are out of reach, so the
    #   dial cannot go lower unless sensitivity_min is deliberately overridden. Raise it to make
    #   the pointer quicker — 5.0, the old relative-mode default, is ~20% of the frame per screen.
    pointer_deadzone: float = 0.003
    #   Hand displacement (normalized frame units) treated as jitter and held back.
    #   It accumulates, so slow movement still moves the cursor instead of being lost.
    relative_scale_by_hand: bool = False
    #   Divide displacement by the hand's apparent size, so the feel does not change
    #   when the hand moves nearer to or further from the camera. Off by default
    #   because it adds landmark noise into the gain.
    reference_hand_width: float = 0.15
    #   The hand size assumed by the above, in normalized frame units.
    scroll_requires_still_hand: bool = True
    #   Index-finger scrolling only acts while the hand itself is holding still; with
    #   palm-relative pointing, translating the hand would otherwise scroll the page
    #   while it moves the cursor.

    # --- the sensitivity dial -------------------------------------------------
    sensitivity_step: float = 1.15
    #   How much one notch of the dial changes pointer_sensitivity: a multiplier, so every
    #   notch feels the same whether you are at 1.0 or 40.0 (a 15% change per press). Use
    #   --sensitivity-step or this field for a finer or coarser dial.
    sensitivity_min: float = 1.0
    #   The slow end of the dial, and it is not a taste decision: one full frame width of hand
    #   travel must cover at least one screen width, or the screen edges cannot be reached —
    #   in absolute mode the mapped region stops covering the whole screen, and in relative mode a
    #   single sweep cannot cross it. Lower it only if you know you want the edge of the screen to
    #   be unreachable; the runtime then warns (reach_warning) instead of failing silently.
    sensitivity_reach_floor: float = 1.0
    #   The gain at which the whole screen becomes unreachable ("reach"), reported by
    #   runtime.reach / runtime.reach_warning. sensitivity_min defaults to it.
    sensitivity_max: float = 60.0
    #   The fast end of the dial: about 1.7% of the frame width crosses the whole screen.
    #   The dial clamps here rather than letting the pointer become uncontrollable.
    sensitivity_keys_up: Tuple[str, ...] = ("=", "+", "]")
    #   Keys that turn the dial up (sensitivity_keys_down turns it down), read from the
    #   preview window's keyboard alongside the pause keys.
    sensitivity_keys_down: Tuple[str, ...] = ("-", "_", "[")
    sensitivity_keys_reset: Tuple[str, ...] = ("0",)
    #   One keypress of the reset restores the startup sensitivity (a detent in the middle
    #   of the dial), so experimenting is always reversible.
    show_sensitivity: bool = True
    #   Draw the dial's current value and position on the preview.

    # --- Phase 3: smoothing and screen mapping ------------------------------
    smoothing_mode: str = "one_euro"
    #   "one_euro" (default) = the 1-euro filter. "ema" uses `smoothing` as a fixed factor.
    one_euro_min_cutoff: float = 4.0
    #   One Euro cutoff (Hz) when slow/still; lower = smoother but more lag. 4.0 was chosen from
    #   tools/measure_smoothness.py against both a still hand and slow/fast sweeps: a still hand's
    #   per-frame cursor step drops from ~7 px (no effective filtering) to ~2.6 px, and a slow sweep
    #   still follows with ~7 px of trailing. Retune there, not by feel.
    one_euro_beta: float = 0.0
    #   How fast the cutoff opens with speed. **0 by design, because beta is in pixel units** and this
    #   filter is fed *screen pixels*: the paper's 0.007 assumes pixel inputs, and our per-frame px
    #   deltas are 1-30, so any beta >= 0.1 saturates the cutoff — measured, beta=8.0 gives alpha 0.96
    #   even at 0.5 px/frame, i.e. the filter was passing landmark noise straight through. That was a
    #   large part of "the cursor feels stuttery". Raise it only if you know you want the speed term;
    #   tools/measure_smoothness.py shows what it costs in jitter.
    one_euro_d_cutoff: float = 1.0
    #   Cutoff (Hz) for the speed estimate itself (only matters when beta > 0).
    frame_margin: float = 0.15
    #   Absolute mode only: the inner part of the camera frame that maps to the screen
    #   (0.15 keeps the middle 70%), so the screen corners are reachable without moving
    #   your hand to the very edge of the frame.
    screen_margin: float = 0.0
    #   Absolute mode only: margin kept free at the screen edges.
    # --- Phase 4: gesture recogniser ----------------------------------------
    gesture_engine: str = "recognizer"
    #   "recognizer": the pure timestamp-driven state machine in recognizer.py — relative
    #   thresholds, per-pair hysteresis, hold/cooldown/re-arm, scroll ahead of clicks.
    #   "legacy": the pre-Phase-4 inline booleans and L1 pinch test, kept reachable so the
    #   previous feel can be compared or restored without a code change.
    pinch_ratio: float = 0.40
    #   A thumb-index / thumb-middle distance that counts as a pinch, as a fraction of the
    #   hand's scale (wrist -> middle knuckle). Relative, so it no longer drifts with distance
    #   from the camera; 0.40 reproduces the old absolute click_threshold at a nominal hand.
    pinch_release_ratio: float = 1.4
    #   Hysteresis: a pinch only releases above pinch_ratio * this, so a hand hovering at the
    #   threshold cannot chatter.
    pinch_hold_seconds: float = 0.08
    #   How long a pose must hold before it registers — about 2.4 frames at 30 fps.
    click_cooldown_seconds: float = 0.18
    #   Minimum spacing between two clicks; a click inside the window is dropped, not deferred.
    scroll_deadzone_ratio: float = 0.05
    #   Two-finger movement below this (in hand widths) is ignored and accumulates. Applied per axis
    #   since 2026-10-11, so a hand scrolling vertically with a little sideways wobble scrolls
    #   vertically only.
    scroll_gain: float = 900.0
    #   Scroll units per hand width of two-finger movement (the old hard-coded 900). Positive hand
    #   movement down = page down; `scroll_invert_vertical` flips it.
    scroll_horizontal_gain: float = 900.0
    #   Horizontal scroll units per hand width (Phase 7). 0.0 turns sideways scrolling off, which is
    #   exactly the pre-Phase-7 behaviour: the horizontal component was computed and reported but
    #   never emitted. See `--no-horizontal-scroll`.
    scroll_invert_vertical: bool = False
    #   Flip the vertical scroll direction (default: hand down = page down).
    scroll_invert_horizontal: bool = False
    #   Flip the horizontal scroll direction. Which way macOS's positive horizontal wheel value moves
    #   the content is inherited from Quartz rather than measured here — scrolling the user's screen
    #   to find out is not something this project does — so this field is the one-field fix if it
    #   feels backwards (see AGENTS.md §2 "Two-finger scrolling").
    scroll_max_hz: float = 0.0
    #   Cap on scroll events per second, 0 = uncapped (the measured default: the loop is camera-bound
    #   and a scroll costs ~11 ms, so one event per frame already happens). When set, held-back frames
    #   are *accumulated* rather than dropped, so the rate falls and the distance scrolled does not.

    pointer_scale: float = 1.0
    #   Extra gain applied after the region mapping (absolute mode only). Was 1.8 with the
    #   old pointer_scale-times-frame mapping, which skewed the reachable area off-centre;
    #   the region mapping replaces that job, so 1.0 is the calibrated value. Only change
    #   it if you want the pointer faster or slower than the frame-to-screen map implies.

    # --- Phase 5: cursor move rate limit / coalescer -------------------------
    move_max_hz: float = 0.0
    #   0 (default) = no limit, and that is the measured status quo rather than a guess: the loop
    #   is camera-bound at ~30 fps and a move costs ~11 ms, so one move per frame already happens
    #   and the output path is about a third of the frame budget. A value > 0 emits at most that
    #   many cursor moves per second and always sends the *newest* position (a frame inside the
    #   interval is superseded, never queued), which is what keeps the loop responsive when the
    #   adapter is slow — e.g. mouse_pause at pyautogui's own 0.1 s default makes every move cost
    #   ~113 ms, longer than the 33 ms frame interval. Lower limits mean fewer, larger steps: the
    #   held-then-jump feel the smoothness work removed. Measure it with
    #   `python3 tools/measure_smoothness.py --rate-sweep` before choosing a value.
    show_last_action: bool = True
    #   Phase 6: draw the last click/scroll the recogniser emitted, how long ago, and how many
    #   have fired this session (`last: left click  1.2s ago  (3 this session)`). It appears on the
    #   line below the sensitivity dial once the first action has fired, and it is the readout the
    #   manual click verification procedure uses — including under --dry-run, where it shows what
    #   *would* have been sent without anything reaching the desktop.

    # --- Phase 8: the optional gesture pause ---------------------------------
    gesture_pause_enabled: bool = False
    #   Off by default. When on, holding an *open hand* (all four fingers straight, no pinch engaged)
    #   for gesture_pause_hold_seconds toggles pause/resume, exactly like SPACE — and it works while
    #   paused too, which is what makes the same pose resume. Off by default because a pose that can
    #   freeze the pointer has to be a deliberate choice: a stretch on the way to a pinch must not
    #   stop the mouse, and the hold time is what keeps that safe.
    gesture_pause_fingers: int = 4
    #   How many of the four fingers (index -> pinky) must be straight for the pause pose.
    gesture_pause_hold_seconds: float = 1.2
    #   How long the open hand must be held before it toggles. Long on purpose: the hand passes
    #   through open shapes constantly while it moves, and only a deliberate hold should act.

    # --- Phase 9: logging and diagnostics ------------------------------------
    log_level: str = "warning"
    #   Level for the structured (JSON-lines) log: debug, info, warning or error. The default is
    #   deliberately quiet — the pipeline logs at info/debug, so nothing is written unless you ask
    #   for it, and the per-frame logging is guarded by isEnabledFor(DEBUG) so a quiet level costs
    #   nothing on the hot path. Use --log-level info to see pause/resume, tracking loss and every
    #   emitted click/scroll, or debug for per-frame detail.
    log_file: str = ""
    #   Where the JSON-lines log goes. Empty (default) logs to stderr; a path appends to that file.
    #   Each line is one JSON object with `time`, `level`, `logger`, `event` and the event's fields,
    #   so a session can be diffed or plotted without parsing prose.
