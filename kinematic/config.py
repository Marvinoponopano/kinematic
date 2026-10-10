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
    pointer_scale: float = 1.8

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
    pointer_mode: str = "relative"
    #   "relative" (default): the cursor follows *hand displacement*, like a mouse.
    #       Finger curl contributes nothing because the palm does not move; rotating
    #       the wrist or moving nearer to the camera does not move the cursor either.
    #   "absolute": the cursor maps to the hand's position in the frame (the previous
    #       behaviour, without the fingertip).
    pointer_reference: str = "palm"
    #   "palm" (default) = mean of wrist + the four knuckles; "wrist" = wrist joint
    #   only (most stable); "index" = the legacy fingertip, which finger curl moves.
    pointer_sensitivity: float = 5.0
    #   relative mode gain: screen widths of cursor travel per frame width of hand
    #   travel. 5.0 means crossing the screen takes ~20% of the frame's width of hand
    #   movement. Raise it to make the pointer quicker.
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
