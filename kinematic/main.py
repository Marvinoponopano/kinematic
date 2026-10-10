"""Entry point for the Kinematic hand-tracking mouse controller."""

# Support both invocation styles: as a package module (`python3 -m kinematic.main`)
# and as a plain script (`python3 kinematic/main.py`). A script has no parent
# package, so register the project root on sys.path and give this module a package
# identity, which is what the relative imports below require.
if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "kinematic"

import time
from typing import Optional

import cv2
import mediapipe as mp
import pyautogui

from .config import HandMouseConfig
from .gesture import (
    is_pinch,
    is_right_click_gesture,
    normalize_screen_point,
    relative_move_delta,
    smooth_point,
)


class HandMouseApp:
    def __init__(self, config: Optional[HandMouseConfig] = None) -> None:
        self.config = config or HandMouseConfig()
        pyautogui.FAILSAFE = False
        self.screen_w, self.screen_h = pyautogui.size()
        self.cursor_x = self.screen_w / 2
        self.cursor_y = self.screen_h / 2
        self.last_index: Optional[tuple[float, float]] = None
        self.last_pinch = False
        self.last_right_click = False
        self.last_action_at = 0.0

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1,
        )

    def _update_cursor(self, index_tip: tuple[float, float]) -> None:
        target = normalize_screen_point(index_tip, self.screen_w, self.screen_h)
        target = (
            target[0] * self.config.pointer_scale,
            target[1] * self.config.pointer_scale,
        )
        target = (
            min(max(target[0], 0), self.screen_w),
            min(max(target[1], 0), self.screen_h),
        )

        smoothed = smooth_point((self.cursor_x, self.cursor_y), target, self.config.smoothing)
        self.cursor_x, self.cursor_y = smoothed

        pyautogui.moveTo(self.cursor_x, self.cursor_y, duration=0)

    def _handle_clicks(self, index_tip: tuple[float, float], thumb_tip: tuple[float, float]) -> None:
        pinch_active = abs(index_tip[0] - thumb_tip[0]) + abs(index_tip[1] - thumb_tip[1]) < self.config.click_threshold
        now = time.monotonic()

        if pinch_active and not self.last_pinch and (now - self.last_action_at) >= self.config.debounce_seconds:
            pyautogui.click(button="left")
            self.last_action_at = now

        self.last_pinch = pinch_active

        right_click = is_right_click_gesture(self.current_landmarks, threshold=self.config.right_click_threshold)
        if right_click and not self.last_right_click and (now - self.last_action_at) >= self.config.debounce_seconds:
            pyautogui.click(button="right")
            self.last_action_at = now
        self.last_right_click = right_click

    def _handle_scroll(self, index_tip: tuple[float, float]) -> None:
        if self.last_index is None:
            self.last_index = index_tip
            return

        delta_x, delta_y = relative_move_delta(self.last_index, index_tip)
        if abs(delta_y) > self.config.scroll_threshold:
            scroll_amount = int(delta_y * -900)
            if scroll_amount != 0:
                pyautogui.scroll(scroll_amount)

        self.last_index = index_tip

    def run(self) -> None:
        cap = cv2.VideoCapture(self.config.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self.config.camera_index}.")

        while True:
            success, frame = cap.read()
            if not success:
                raise RuntimeError("Unable to read frames from the webcam.")

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            frame_h, frame_w = frame.shape[:2]
            self.current_landmarks = []

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                self.current_landmarks = hand.landmark
                points = [(landmark.x, landmark.y) for landmark in hand.landmark]
                index_tip = points[8]
                thumb_tip = points[4]

                if self.config.deadzone > 0:
                    dx = abs(index_tip[0] - self.last_index[0]) if self.last_index is not None else 0.0
                    dy = abs(index_tip[1] - self.last_index[1]) if self.last_index is not None else 0.0
                    if dx <= self.config.deadzone and dy <= self.config.deadzone:
                        index_tip = self.last_index if self.last_index is not None else index_tip

                self._update_cursor(index_tip)
                self._handle_clicks(index_tip, thumb_tip)
                self._handle_scroll(index_tip)
            else:
                self.last_index = None
                self.last_pinch = False
                self.last_right_click = False

            cv2.imshow(self.config.window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    app = HandMouseApp()
    app.run()


if __name__ == "__main__":
    main()
