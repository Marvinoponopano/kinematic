from dataclasses import dataclass
from typing import Tuple


@dataclass
class HandMouseConfig:
    camera_index: int = 0
    window_name: str = "Kinematic"
    smoothing: float = 0.35
    deadzone: float = 0.02
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
    #   "one_euro" (default, changed 2026-10-10) = the 1-euro filter: it beat the previous
    #   EMA on every axis when measured (tools/compare_smoothing.py, 16 seeds): 25% less
    #   jitter, a 2-frame instead of 6-frame step response, and far less lag on fast
    #   movement (151 px vs 529 px at 1.0 landmark-units/s). "ema" restores the old feel.
    one_euro_min_cutoff: float = 1.0
    #   One Euro cutoff (Hz) when the hand is still; lower is smoother (0.5 gives ~25% less
    #   jitter than the default but noticeably more lag on slow movement).
    one_euro_beta: float = 8.0
    #   How fast the cutoff opens up with hand speed. Measured: beta=8 dominates the old
    #   EMA at every speed tested; lower values (1-2) filter harder and trail on slow
    #   movement. The classic 0.007 default assumes pixel units — landmarks are ~1.0 across.
    one_euro_d_cutoff: float = 1.0
    #   Cutoff (Hz) for the speed estimate itself.
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
    #   Two-finger movement below this (in hand widths) is ignored and accumulates.
    scroll_gain: float = 900.0
    #   Scroll units per hand width of two-finger movement (the old hard-coded 900).

    pointer_scale: float = 1.0
    #   Extra gain applied after the region mapping (absolute mode only). Was 1.8 with the
    #   old pointer_scale-times-frame mapping, which skewed the reachable area off-centre;
    #   the region mapping replaces that job, so 1.0 is the calibrated value. Only change
    #   it if you want the pointer faster or slower than the frame-to-screen map implies.
