"""Runtime coordinator for the hand mouse.

This module owns the pipeline (capture hand-off -> landmarks -> smoothing -> gesture
handling -> output) and is the **only** place that decides whether output is allowed.
While the runtime is paused — which is the startup default — frames are still
analysed and previewed, but no mouse event is emitted. That single rule is what makes
the paused startup, ``--dry-run`` and the test suite safe.

Phase 1 defines this coordinator with the safety behaviour; later phases extend it
(Phase 5 cursor adapter, Phase 8 status/teardown).
"""

import time
from typing import Any, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from .capture import ThreadedCamera
from .config import HandMouseConfig
from .gesture import (
    is_right_click_gesture,
    needs_cursor_update,
    normalize_screen_point,
    relative_move_delta,
    smooth_point,
)
from .output import MouseOutput, RealMouseOutput, RecordingMouseOutput


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
        self.last_index: Optional[Tuple[float, float]] = None
        self.last_pinch = False
        self.last_right_click = False
        self.last_action_at = 0.0
        self.current_landmarks: list = []
        self.loop_fps = 0.0
        self.frames = 0
        self.emitted = 0
        self.paused = bool(self.config.start_paused)

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
        return self.paused

    def resume(self) -> bool:
        self.paused = False
        return self.paused

    def toggle_pause(self) -> bool:
        self.paused = not self.paused
        return self.paused

    def handle_key(self, key: int) -> bool:
        """Act on a ``cv2.waitKey()`` code. Returns True when the loop should stop."""
        if key is None or key < 0:
            return False
        char = chr(key & 0xFF)
        if char in self.config.toggle_pause_keys:
            self.toggle_pause()
            return False
        return char in self.config.quit_keys

    # -- per-frame pipeline ---------------------------------------------------

    def update_cursor(self, index_tip: Tuple[float, float], now: Optional[float] = None) -> bool:
        if not self.can_emit:
            # Leave the cursor state untouched: on resume it still matches where the
            # OS pointer actually is, so the pointer does not jump.
            return False

        target = normalize_screen_point(index_tip, self.screen_w, self.screen_h)
        target = (
            target[0] * self.config.pointer_scale,
            target[1] * self.config.pointer_scale,
        )
        target = (
            min(max(target[0], 0), self.screen_w),
            min(max(target[1], 0), self.screen_h),
        )

        previous = (self.cursor_x, self.cursor_y)
        smoothed = smooth_point(previous, target, self.config.smoothing)
        self.cursor_x, self.cursor_y = smoothed

        if not needs_cursor_update(previous, smoothed, self.config.min_move_pixels):
            return False

        self.output.move_to(self.cursor_x, self.cursor_y, duration=0)
        self.emitted += 1
        return True

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

    def handle_scroll(self, index_tip: Tuple[float, float]) -> bool:
        if self.last_index is None:
            self.last_index = index_tip
            return False

        delta_x, delta_y = relative_move_delta(self.last_index, index_tip)
        scrolled = False
        if abs(delta_y) > self.config.scroll_threshold and self.can_emit:
            scroll_amount = int(delta_y * -900)
            if scroll_amount != 0:
                self.output.scroll(scroll_amount)
                self.emitted += 1
                scrolled = True

        self.last_index = index_tip
        return scrolled

    def process_landmarks(self, landmarks, now: Optional[float] = None) -> int:
        """Drive the cursor/click/scroll path from one frame's landmarks."""
        self.current_landmarks = landmarks
        points = [(landmark.x, landmark.y) for landmark in landmarks]
        index_tip = points[8]
        thumb_tip = points[4]

        if self.config.deadzone > 0:
            dx = abs(index_tip[0] - self.last_index[0]) if self.last_index is not None else 0.0
            dy = abs(index_tip[1] - self.last_index[1]) if self.last_index is not None else 0.0
            if dx <= self.config.deadzone and dy <= self.config.deadzone:
                index_tip = self.last_index if self.last_index is not None else index_tip

        emitted = 0
        emitted += int(self.update_cursor(index_tip, now=now))
        emitted += int(self.handle_clicks(index_tip, thumb_tip, now=now))
        emitted += int(self.handle_scroll(index_tip))
        return emitted

    def process_frame(self, frame: np.ndarray, now: Optional[float] = None) -> int:
        """Analyse one frame. Returns the number of output calls it produced (0 while paused)."""
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            emitted = self.process_landmarks(results.multi_hand_landmarks[0].landmark, now=now)
        else:
            self.current_landmarks = []
            self.last_index = None
            self.last_pinch = False
            self.last_right_click = False
            emitted = 0

        self.frames += 1
        return emitted

    # -- preview --------------------------------------------------------------

    def draw_preview(self, frame: np.ndarray, camera: Optional[Any] = None) -> np.ndarray:
        preview = frame
        if self.config.preview_scale != 1.0:
            preview = cv2.resize(frame, None, fx=self.config.preview_scale, fy=self.config.preview_scale)
        elif self.config.show_fps or self.paused:
            preview = frame.copy()

        if self.config.show_fps:
            size = ""
            width = getattr(camera, "width", 0)
            height = getattr(camera, "height", 0)
            if width and height:
                size = f"  {width}x{height}"
            cv2.putText(
                preview,
                f"{self.loop_fps:4.1f} fps{size}",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        if self.paused:
            cv2.putText(
                preview,
                "PAUSED - press SPACE to control the mouse",
                (8, 46 if self.config.show_fps else 22),
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
        }

    def close(self) -> None:
        close = getattr(self.hands, "close", None)
        if callable(close):
            close()
        if self.camera is not None:
            self.camera.release()
