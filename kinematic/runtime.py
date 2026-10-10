"""Runtime coordinator for the hand mouse.

This module owns the pipeline (capture hand-off -> landmarks -> smoothing -> gesture
handling -> output) and is the **only** place that decides whether output is allowed.
While the runtime is paused — which is the startup default — frames are still
analysed and previewed, but no mouse event is emitted. That single rule is what makes
the paused startup, ``--dry-run`` and the test suite safe.

Phase 1 defines this coordinator with the safety behaviour; later phases extend it
(Phase 5 cursor adapter, Phase 8 status/teardown).
"""

import math
import time
from typing import Any, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from .capture import ThreadedCamera
from .config import HandMouseConfig
from .landmarks import HandLandmarks, hands_from_results
from .preview import draw_hands
from .gesture import (
    clamp,
    clamp_point,
    hand_scale,
    is_right_click_gesture,
    landmark_point,
    needs_cursor_update,
    point_distance,
    pointer_delta,
    pointer_reference_point,
    relative_move_delta,
    INDEX_TIP,
    THUMB_TIP,
)
from .mapping import Region, map_point_to_screen
from .output import MouseOutput, RealMouseOutput, RecordingMouseOutput
from .recognizer import (
    EventKind,
    GestureEvent,
    GestureRecognizer,
    GestureUpdate,
    HandState,
    Observation,
    thresholds_from_config,
)
from .smoothing import CursorFilter, filter_from_config


class HandMouseRuntime:
    """Pipeline coordinator; guards every desktop side effect behind the pause state."""

    def __init__(
        self,
        config: Optional[HandMouseConfig] = None,
        output: Optional[MouseOutput] = None,
        hands: Optional[Any] = None,
        camera: Optional[Any] = None,
    ) -> None:
        self.config = config or HandMouseConfig()
        self.output: MouseOutput = output if output is not None else self._default_output()
        self.hands = hands if hands is not None else self._default_hands()
        self.camera: Optional[Any] = camera

        self.screen_w, self.screen_h = self.output.screen_size()
        self.cursor_x, self.cursor_y = self.output.position()
        self.frame_region = Region.symmetric(self.config.frame_margin)
        self.screen_region = Region.symmetric(self.config.screen_margin)
        self.cursor_filter: CursorFilter = filter_from_config(self.config)
        # Seed the filter with where the cursor already is, so the first update smooths
        # from there instead of snapping — this is what the code did before the filter
        # existed (cursor_x/y started at output.position() and the EMA blended from it).
        self.cursor_filter.update((self.cursor_x, self.cursor_y))
        self.last_index: Optional[Tuple[float, float]] = None
        self.last_pointer_ref: Optional[Tuple[float, float]] = None
        self.last_pinch = False
        self.last_right_click = False
        self.last_action_at = 0.0
        self.current_landmarks: Any = []  # the first hand (HandLandmarks) or [] — kept for the facade
        self.current_hands: List[HandLandmarks] = []
        self.loop_fps = 0.0
        self.frames = 0
        self.emitted = 0
        self.paused = bool(self.config.start_paused)

        # Phase 4: the pure recogniser owns gesture state; the runtime only translates events.
        self.gesture = GestureRecognizer(thresholds_from_config(self.config))
        self.gesture.set_paused(self.paused)
        self.gesture_state: HandState = self.gesture.state
        self.last_gesture_events: Tuple[GestureEvent, ...] = ()

        # The sensitivity dial: `sensitivity_start` is the value the app was launched with, so
        # the reset key always has somewhere to return to.
        self.sensitivity_start: float = float(self.config.pointer_sensitivity)
        self.sensitivity_changes: int = 0

    # -- construction helpers -------------------------------------------------

    def _default_output(self) -> MouseOutput:
        if self.config.dry_run:
            return RecordingMouseOutput()
        return RealMouseOutput(pause=self.config.mouse_pause)

    def _default_hands(self) -> Any:
        return mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            model_complexity=self.config.model_complexity,
        )

    # -- safety / pause -------------------------------------------------------

    @property
    def can_emit(self) -> bool:
        """True when output is allowed; the single gate for desktop side effects."""
        return not self.paused

    @property
    def status(self) -> str:
        return "paused" if self.paused else "active"

    def pause(self) -> bool:
        self.paused = True
        self._sync_gesture_pause()
        return self.paused

    def resume(self) -> bool:
        self.paused = False
        self._sync_gesture_pause()
        return self.paused

    def toggle_pause(self) -> bool:
        self.paused = not self.paused
        self._sync_gesture_pause()
        return self.paused

    def _sync_gesture_pause(self) -> None:
        """Tell the recogniser about a pause/resume so its state machine stays truthful.

        The recogniser drops any gesture in flight here, which is what stops a pinch held
        across a resume from clicking — the rising edge belonged to the paused period.
        """
        self.gesture.set_paused(self.paused)
        self.gesture_state = self.gesture.state

    def handle_key(self, key: int) -> bool:
        """Act on a ``cv2.waitKey()`` code. Returns True when the loop should stop."""
        if key is None or key < 0:
            return False
        char = chr(key & 0xFF)
        if char in self.config.toggle_pause_keys:
            self.toggle_pause()
            return False
        if char in self.config.sensitivity_keys_up:
            self.adjust_sensitivity(1)
            return False
        if char in self.config.sensitivity_keys_down:
            self.adjust_sensitivity(-1)
            return False
        if char in self.config.sensitivity_keys_reset:
            self.reset_sensitivity()
            return False
        return char in self.config.quit_keys

    # -- the sensitivity dial -------------------------------------------------

    @property
    def sensitivity(self) -> float:
        """Current pointer gain: screen widths of cursor travel per frame width of hand travel."""
        return float(self.config.pointer_sensitivity)

    def set_sensitivity(self, value: float) -> float:
        """Set the pointer gain, clamped to the dial's range. Returns the value actually applied.

        Safe to do at any time, including mid-gesture and while paused: the gain is applied to
        each frame's *displacement*, so changing it never rescales movement that is already on
        screen — the cursor does not jump, and nothing is emitted by the change itself.
        """
        clamped = clamp(float(value), float(self.config.sensitivity_min), float(self.config.sensitivity_max))
        self.config.pointer_sensitivity = clamped
        self.sensitivity_changes += 1
        return clamped

    def adjust_sensitivity(self, notches: int = 1) -> float:
        """Turn the dial by *notches* (positive = faster, negative = slower). Returns the new value.

        Steps are multiplicative (``sensitivity_step``, 1.15 by default), so one notch feels the
        same at either end of the dial. Turning past the range stops at the end rather than
        wrapping or becoming uncontrollable.
        """
        if notches == 0:
            return self.sensitivity
        step = float(self.config.sensitivity_step)
        if step <= 0.0:
            return self.sensitivity
        if notches > 0:
            return self.set_sensitivity(self.sensitivity * (step ** notches))
        return self.set_sensitivity(self.sensitivity / (step ** abs(notches)))

    def reset_sensitivity(self) -> float:
        """Return the dial to the value the app started with."""
        return self.set_sensitivity(self.sensitivity_start)

    def sensitivity_bar(self, width: int = 10) -> str:
        """The dial's position as a compact ASCII bar, on a log scale.

        The range spans 0.5 → 60 (a factor of 120), so a linear bar would look nearly empty at
        the default 5.0. Log scale puts the default ~6 cells in, which is what a dial should
        feel like: even turns either side of where you are.
        """
        low = float(self.config.sensitivity_min)
        high = max(float(self.config.sensitivity_max), low * 1.01)
        value = min(max(self.sensitivity, low), high)
        span = math.log(high) - math.log(low)
        position = 0.0 if span <= 0 else (math.log(max(value, low)) - math.log(low)) / span
        filled = int(round(position * width))
        filled = min(max(filled, 0), width)
        return "#" * filled + "-" * (width - filled)

    def sensitivity_text(self) -> str:
        """One line describing the dial: its value, its position, and the keys that turn it."""
        return f"sens {self.sensitivity:5.2f} [{self.sensitivity_bar()}]  ] faster  [ slower  0 reset"

    # -- per-frame pipeline ---------------------------------------------------

    def _move_towards(self, target: Tuple[float, float], now: Optional[float] = None) -> bool:
        """Smooth towards *target* and issue the move if it is worth one."""
        previous = (self.cursor_x, self.cursor_y)
        smoothed = self.cursor_filter.update(target, now)
        self.cursor_x, self.cursor_y = smoothed

        if not needs_cursor_update(previous, smoothed, self.config.min_move_pixels):
            return False

        self.output.move_to(self.cursor_x, self.cursor_y, duration=0)
        self.emitted += 1
        return True

    def update_cursor(self, index_tip: Tuple[float, float], now: Optional[float] = None) -> bool:
        """Absolute mode: map the hand's frame position onto a configurable screen region."""
        if not self.can_emit:
            # Leave the cursor state untouched: on resume it still matches where the
            # OS pointer actually is, so the pointer does not jump.
            return False

        target = map_point_to_screen(
            index_tip,
            (self.screen_w, self.screen_h),
            frame_region=self.frame_region,
            screen_region=self.screen_region,
            clamp=True,
        )
        if self.config.pointer_scale != 1.0:
            # Legacy extra gain, kept for anyone who tuned it; 1.0 is the calibrated value.
            target = (target[0] * self.config.pointer_scale, target[1] * self.config.pointer_scale)
        target = clamp_point(target, self.screen_w, self.screen_h)

        return self._move_towards(target, now=now)

    def hand_is_moving(self, reference: Tuple[float, float]) -> bool:
        """True when the wrist/palm reference shifted more than the jitter deadzone."""
        if self.last_pointer_ref is None:
            return False
        return point_distance(self.last_pointer_ref, reference) >= self.config.pointer_deadzone

    def pointer_target(
        self,
        reference: Tuple[float, float],
        hand_size: Optional[float] = None,
    ) -> Optional[Tuple[float, float]]:
        """Where the cursor should go, from the *displacement* of the hand reference.

        Returns None when there is nothing to do: the first sight of a hand (which only
        anchors the reference, so the cursor never jumps on reacquire), or a
        displacement smaller than the jitter deadzone. The anchor is deliberately left
        alone when the deadzone rejects a frame, so slow movement accumulates instead of
        being thrown away.
        """
        if self.last_pointer_ref is None:
            self.last_pointer_ref = reference
            return None

        delta = pointer_delta(self.last_pointer_ref, reference)
        if self.config.relative_scale_by_hand and hand_size:
            factor = self.config.reference_hand_width / hand_size
            delta = (delta[0] * factor, delta[1] * factor)

        if point_distance((0.0, 0.0), delta) < self.config.pointer_deadzone:
            return None
        self.last_pointer_ref = reference

        target = (
            self.cursor_x + delta[0] * self.config.pointer_sensitivity * self.screen_w,
            self.cursor_y + delta[1] * self.config.pointer_sensitivity * self.screen_h,
        )
        return (clamp(target[0], 0.0, self.screen_w), clamp(target[1], 0.0, self.screen_h))

    def update_cursor_relative(
        self,
        reference: Tuple[float, float],
        hand_size: Optional[float] = None,
        now: Optional[float] = None,
    ) -> bool:
        """Relative pointing: shift the cursor by however far the hand itself moved."""
        target = self.pointer_target(reference, hand_size)
        if target is None or not self.can_emit:
            return False

        return self._move_towards(target, now=now)

    def handle_clicks(
        self,
        index_tip: Tuple[float, float],
        thumb_tip: Tuple[float, float],
        now: Optional[float] = None,
    ) -> bool:
        now = time.monotonic() if now is None else now
        fired = False

        pinch_active = abs(index_tip[0] - thumb_tip[0]) + abs(index_tip[1] - thumb_tip[1]) < self.config.click_threshold
        if pinch_active and not self.last_pinch and (now - self.last_action_at) >= self.config.debounce_seconds:
            if self.can_emit:
                self.output.click("left")
                self.last_action_at = now
                self.emitted += 1
                fired = True

        self.last_pinch = pinch_active

        right_click = is_right_click_gesture(self.current_landmarks, threshold=self.config.right_click_threshold)
        if right_click and not self.last_right_click and (now - self.last_action_at) >= self.config.debounce_seconds:
            if self.can_emit:
                self.output.click("right")
                self.last_action_at = now
                self.emitted += 1
                fired = True
        self.last_right_click = right_click

        return fired

    def handle_scroll(self, index_tip: Tuple[float, float], hand_moving: bool = False) -> bool:
        if self.last_index is None:
            self.last_index = index_tip
            return False

        delta_x, delta_y = relative_move_delta(self.last_index, index_tip)
        scrolled = False
        if abs(delta_y) > self.config.scroll_threshold:
            if self.config.scroll_requires_still_hand and hand_moving:
                # The hand itself is translating: that is pointing, not scrolling. Keep
                # the reference current so the delta does not jump when the hand stops.
                self.last_index = index_tip
                return False
            if self.can_emit:
                scroll_amount = int(delta_y * -900)
                if scroll_amount != 0:
                    self.output.scroll(scroll_amount)
                    self.emitted += 1
                    scrolled = True

        self.last_index = index_tip
        return scrolled

    # -- Phase 4: the gesture state machine -----------------------------------

    def apply_gestures(self, landmarks: Any, now: Optional[float] = None) -> int:
        """Run the pure recogniser for one frame and translate its events into output calls.

        The recogniser decides *what* the hand is doing — states, priority, hysteresis, hold
        and cooldown rules — and holds no reference to the mouse. This method is the only
        place that turns an event into a desktop side effect, and it still obeys ``can_emit``.
        ``landmarks=None`` means tracking was lost this frame.
        """
        now = time.monotonic() if now is None else now
        update = self.gesture.update(Observation(landmarks=landmarks, timestamp=now))
        self.gesture_state = update.state
        self.last_gesture_events = update.events
        return self._apply_gesture_events(update)

    def _apply_gesture_events(self, update: GestureUpdate) -> int:
        emitted = 0
        for event in update.events:
            if not self.can_emit:
                break
            if event.kind is EventKind.LEFT_CLICK:
                self.output.click("left")
            elif event.kind is EventKind.RIGHT_CLICK:
                self.output.click("right")
            elif event.kind is EventKind.SCROLL:
                amount = int(event.value)
                if amount == 0:
                    continue
                self.output.scroll(amount)
            else:
                # scroll_begin/scroll_end, tracking and pause events are state bookkeeping.
                continue
            self.emitted += 1
            emitted += 1
        return emitted

    def process_landmarks(self, landmarks, now: Optional[float] = None) -> int:
        """Drive the cursor/click/scroll path from one frame's landmarks.

        The cursor reference is the wrist/palm, never the fingertip — see the
        ``pointer_*`` config fields for why (a curling finger used to drag the cursor).
        """
        self.current_landmarks = landmarks
        index_tip = landmark_point(landmarks, INDEX_TIP)
        thumb_tip = landmark_point(landmarks, THUMB_TIP)

        reference = pointer_reference_point(landmarks, self.config.pointer_reference)
        hand_size = hand_scale(landmarks)
        # Measured before the cursor update, because that is what moves the anchor.
        hand_moved = self.hand_is_moving(reference)

        emitted = 0
        if self.config.pointer_mode == "absolute":
            anchor = self.last_pointer_ref
            if self.config.deadzone > 0 and anchor is not None:
                dx = abs(reference[0] - anchor[0])
                dy = abs(reference[1] - anchor[1])
                if dx <= self.config.deadzone and dy <= self.config.deadzone:
                    reference = anchor
            self.last_pointer_ref = reference
            emitted += int(self.update_cursor(reference, now=now))
        else:
            emitted += int(self.update_cursor_relative(reference, hand_size, now=now))

        if self.config.gesture_engine == "recognizer":
            # Phase 4: the recogniser owns the gesture path (priority, hysteresis, hold and
            # cooldown rules). The legacy inline booleans stay reachable via
            # config.gesture_engine == "legacy".
            emitted += self.apply_gestures(landmarks, now=now)
        else:
            emitted += int(self.handle_clicks(index_tip, thumb_tip, now=now))
            emitted += int(self.handle_scroll(index_tip, hand_moving=hand_moved))
        return emitted

    def process_frame(self, frame: np.ndarray, now: Optional[float] = None) -> int:
        """Analyse one frame. Returns the number of output calls it produced (0 while paused)."""
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        # MediaPipe's protobuf is converted exactly once, here at the boundary: everything
        # below works on the project-owned types in kinematic.landmarks.
        self.current_hands = hands_from_results(results)
        hand = next((candidate for candidate in self.current_hands if candidate.is_complete), None)

        if hand is not None:
            emitted = self.process_landmarks(hand, now=now)
        else:
            self.current_landmarks = []
            self.last_index = None
            self.last_pointer_ref = None  # re-anchor on reacquire: the cursor must not jump
            self.cursor_filter.reset()  # Phase 3: drop smoothing state so nothing glides in
            self.last_pinch = False
            self.last_right_click = False
            emitted = 0
            if self.config.gesture_engine == "recognizer":
                # The recogniser models tracking loss itself and reports the transition once;
                # the tracking events carry no output, so this stays 0.
                emitted = self.apply_gestures(None, now=now)

        self.frames += 1
        return emitted

    # -- preview --------------------------------------------------------------

    def draw_preview(self, frame: np.ndarray, camera: Optional[Any] = None) -> np.ndarray:
        preview = frame
        if self.config.preview_scale != 1.0:
            preview = cv2.resize(frame, None, fx=self.config.preview_scale, fy=self.config.preview_scale)
        elif self.config.show_fps or self.config.show_landmarks or self.paused:
            preview = frame.copy()

        if self.config.show_landmarks:
            # Drawn on the preview (scaled or not) so the overlay scales with the window.
            draw_hands(preview, self.current_hands)

        y = 22
        if self.config.show_fps:
            size = ""
            width = getattr(camera, "width", 0)
            height = getattr(camera, "height", 0)
            if width and height:
                size = f"  {width}x{height}"
            cv2.putText(
                preview,
                f"{self.loop_fps:4.1f} fps{size}  {self.gesture_state.value}",
                (8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            y += 24

        if self.config.show_sensitivity:
            # The dial: its value, its position on a log bar, and the keys that turn it.
            cv2.putText(
                preview,
                self.sensitivity_text(),
                (8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 220, 0),
                1,
                cv2.LINE_AA,
            )
            y += 24

        if self.paused:
            cv2.putText(
                preview,
                "PAUSED - press SPACE to control the mouse",
                (8, y if (self.config.show_fps or self.config.show_sensitivity) else 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 165, 255),
                1,
                cv2.LINE_AA,
            )
        return preview

    # -- loop -----------------------------------------------------------------

    def run(self, camera: Optional[Any] = None, max_frames: Optional[int] = None) -> None:
        """Capture loop. ``camera`` and ``max_frames`` exist so tests can drive it."""
        cam = camera if camera is not None else self.camera
        owned = cam is None
        if owned:
            cam = ThreadedCamera(
                index=self.config.camera_index,
                width=self.config.capture_width,
                height=self.config.capture_height,
                fps=self.config.capture_fps,
                buffer_size=self.config.capture_buffer_size,
            )

        try:
            if not cam.wait_ready():
                raise RuntimeError("Unable to read frames from the webcam.")

            last_seq = -1
            processed = 0
            fps_at = time.monotonic()

            while True:
                seq, frame = cam.read()

                if getattr(cam, "error", None):
                    raise RuntimeError("Unable to read frames from the webcam.")

                if frame is None or seq == last_seq:
                    # No new frame yet: don't re-run MediaPipe on a duplicate, only
                    # keep the window answering keys.
                    if self.handle_key(cv2.waitKey(1) & 0xFF):
                        break
                    continue
                last_seq = seq

                self.process_frame(frame)

                processed += 1
                now = time.monotonic()
                if now - fps_at >= 0.5:
                    self.loop_fps = processed / (now - fps_at)
                    processed = 0
                    fps_at = now

                if self.config.show_preview:
                    cv2.imshow(self.config.window_name, self.draw_preview(frame, cam))

                if self.handle_key(cv2.waitKey(1) & 0xFF):
                    break
                if max_frames is not None and self.frames >= max_frames:
                    break
        finally:
            if owned:
                cam.release()
            if self.config.show_preview:
                cv2.destroyAllWindows()

    # -- reporting / teardown -------------------------------------------------

    def summary(self) -> dict:
        counts = self.output.counts() if isinstance(self.output, RecordingMouseOutput) else {}
        return {
            "status": self.status,
            "frames": self.frames,
            "emitted": self.emitted,
            "loop_fps": round(self.loop_fps, 2),
            "output": type(self.output).__name__,
            "recorded": counts,
            "sensitivity": round(self.sensitivity, 4),
            "sensitivity_start": round(self.sensitivity_start, 4),
            "sensitivity_changes": self.sensitivity_changes,
        }

    def close(self) -> None:
        close = getattr(self.hands, "close", None)
        if callable(close):
            close()
        if self.camera is not None:
            self.camera.release()
