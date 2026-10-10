"""Phase 4 integration tests: the recogniser wired into the runtime.

The recogniser's own rules are covered in ``test_phase4_recognizer.py``; this file checks the
wiring — that the runtime feeds it real frames, translates its events into output adapter calls,
keeps the ``can_emit`` gate in front of them, and still exposes the legacy path. No camera, no
display, no pyautogui: the recording adapter counts what would have happened.
"""

import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from kinematic.config import HandMouseConfig
from kinematic.gesture import INDEX_MCP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, PINKY_MCP, RING_MCP, THUMB_TIP, WRIST
from kinematic.output import RecordingMouseOutput
from kinematic.recognizer import EventKind, HandState
from kinematic.runtime import HandMouseRuntime

SCALE = 0.15  # wrist -> middle knuckle, so ratios below are exact

PINCHED = 0.33  # clearly below the 0.40 engage threshold
OPEN = 0.70  # clearly above the 0.56 release threshold

STEP = 0.05  # frame step for the tests: two steps (0.10 s) clear the 0.08 s hold time
TWO_FINGER: Dict[str, Any] = dict(thumb_index_ratio=PINCHED, thumb_middle_ratio=PINCHED)


def _base_points() -> List[Tuple[float, float]]:
    points = [(0.50, 0.60) for _ in range(21)]
    points[WRIST] = (0.50, 0.60)
    points[INDEX_MCP] = (0.45, 0.46)
    points[MIDDLE_MCP] = (0.50, 0.45)  # exactly SCALE above the wrist
    points[RING_MCP] = (0.55, 0.46)
    points[PINKY_MCP] = (0.59, 0.49)
    points[THUMB_TIP] = (0.40, 0.52)
    return points


def hand_points(
    thumb_index_ratio: float = OPEN,
    thumb_middle_ratio: float = OPEN,
    offset: Tuple[float, float] = (0.0, 0.0),
    scale: float = SCALE,
    index_drop: float = 0.0,
) -> List[Tuple[float, float]]:
    """A plausible 21-landmark hand with exactly the requested pinch ratios.

    ``index_drop`` slides the index fingertip down on its own, without moving the palm — the
    motion the legacy index-finger scroll gesture is built on.
    """
    ox, oy = offset
    points = [(x + ox, y + oy) for x, y in _base_points()]
    thumb = points[THUMB_TIP]
    points[INDEX_TIP] = (thumb[0] + thumb_index_ratio * scale, thumb[1] + index_drop)
    points[MIDDLE_TIP] = (thumb[0], thumb[1] - thumb_middle_ratio * scale)
    return points


def landmarks(points: Sequence[Tuple[float, float]]) -> List[SimpleNamespace]:
    return [SimpleNamespace(x=x, y=y, z=0.0) for x, y in points]


def hand(**kwargs: Any) -> List[SimpleNamespace]:
    return landmarks(hand_points(**kwargs))


def settle(runtime: HandMouseRuntime, points: List[SimpleNamespace], start: float, frames: int = 3) -> float:
    """Feed one pose until its hold time has elapsed; return the last timestamp used."""
    now = start
    for index in range(frames):
        now = start + index * STEP
        runtime.process_landmarks(points, now=now)
    return now


class FakeHands:
    """Stands in for ``mp.solutions.hands.Hands``: replays one fixed hand pose."""

    def __init__(self, points: Sequence[Tuple[float, float]], label: str = "Right") -> None:
        self._points = list(points)
        self._label = label

    def process(self, rgb: Any) -> Any:
        raw = [SimpleNamespace(x=x, y=y, z=-0.02) for x, y in self._points]
        handedness = [SimpleNamespace(classification=[SimpleNamespace(label=self._label, score=0.98)])]
        return SimpleNamespace(
            multi_hand_landmarks=[SimpleNamespace(landmark=raw)],
            multi_handedness=handedness,
        )


class NoHands:
    """A hand detector that never sees anything."""

    def process(self, rgb: Any) -> Any:
        return SimpleNamespace(multi_hand_landmarks=None, multi_handedness=None)


def make_runtime(output: RecordingMouseOutput, **overrides: Any) -> HandMouseRuntime:
    hands = overrides.pop("hands", None)
    config_overrides: Dict[str, Any] = {
        "show_preview": False,
        "start_paused": False,
        "smoothing": 1.0,
    }
    config_overrides.update(overrides)
    return HandMouseRuntime(
        config=HandMouseConfig(**config_overrides),
        output=output,
        hands=hands if hands is not None else NoHands(),
    )


class RecognizerWiringTests(unittest.TestCase):
    """The default engine: events reach the adapter, and only through the pause gate."""

    def setUp(self):
        self.output = RecordingMouseOutput()
        self.runtime = make_runtime(self.output)

    def test_the_default_engine_is_the_recognizer(self):
        self.assertEqual(self.runtime.config.gesture_engine, "recognizer")
        self.assertEqual(self.runtime.gesture_state, HandState.NEUTRAL)

    def test_a_held_pinch_clicks_once(self):
        settle(self.runtime, hand(thumb_index_ratio=PINCHED), 0.0)
        settle(self.runtime, hand(thumb_index_ratio=PINCHED), 0.5)
        self.assertEqual([click.value for click in self.output.clicks], ["left"])
        self.assertEqual(self.runtime.gesture_state, HandState.LEFT_PINCH)

    def test_a_thumb_middle_pinch_clicks_right(self):
        settle(self.runtime, hand(thumb_middle_ratio=PINCHED), 0.0)
        self.assertEqual([click.value for click in self.output.clicks], ["right"])
        self.assertEqual(self.runtime.gesture_state, HandState.RIGHT_PINCH)

    def test_a_two_finger_pinch_scrolls_and_never_clicks(self):
        runtime = make_runtime(self.output)
        settle(runtime, hand(), 0.0)
        settle(runtime, hand(**TWO_FINGER), 0.2)
        self.assertEqual(self.output.clicks, [], "the scroll pose must not click")
        self.assertEqual(runtime.gesture_state, HandState.TWO_FINGER_SCROLL)

        runtime.process_landmarks(hand(offset=(0.0, 0.05), **TWO_FINGER), now=0.5)
        self.assertEqual(len(self.output.scrolls), 1, "moving the two fingers must scroll")
        self.assertLess(self.output.scrolls[0].value, 0, "hand down scrolls down")

    def test_scroll_emits_nothing_while_the_hand_is_still(self):
        runtime = make_runtime(self.output)
        settle(runtime, hand(**TWO_FINGER), 0.0)
        runtime.process_landmarks(hand(**TWO_FINGER), now=0.5)
        self.assertEqual(self.output.scrolls, [])

    def test_nothing_is_emitted_while_paused(self):
        runtime = make_runtime(self.output, start_paused=True)
        self.assertEqual(runtime.gesture_state, HandState.PAUSED)
        for step in range(20):
            runtime.process_landmarks(hand(thumb_index_ratio=PINCHED), now=step * STEP)
        self.assertEqual(self.output.counts(), {"move": 0, "click": 0, "scroll": 0})
        self.assertEqual(runtime.gesture_state, HandState.PAUSED)

    def test_a_pinch_held_across_a_resume_does_not_click(self):
        runtime = make_runtime(self.output, start_paused=True)
        settle(runtime, hand(thumb_index_ratio=PINCHED), 0.0)  # pinched while paused
        self.assertEqual(self.output.clicks, [])

        runtime.resume()
        for step in range(20):  # keep holding well past the hold time
            runtime.process_landmarks(hand(thumb_index_ratio=PINCHED), now=1.0 + step * STEP)
        self.assertEqual(self.output.clicks, [], "the pinch belonged to the paused period")
        self.assertEqual(runtime.gesture_state, HandState.LEFT_PINCH)

    def test_release_and_repinch_after_a_resume_clicks(self):
        runtime = make_runtime(self.output, start_paused=True)
        settle(runtime, hand(thumb_index_ratio=PINCHED), 0.0)
        runtime.resume()
        settle(runtime, hand(thumb_index_ratio=PINCHED), 1.0)  # consumed
        runtime.process_landmarks(hand(), 1.5)  # release

        settle(runtime, hand(thumb_index_ratio=PINCHED), 2.0)
        self.assertEqual([click.value for click in self.output.clicks], ["left"])

    def test_pause_and_resume_keep_the_recogniser_in_step(self):
        runtime = make_runtime(self.output)
        self.assertTrue(runtime.gesture.paused is False)
        runtime.pause()
        self.assertTrue(runtime.gesture.paused)
        self.assertEqual(runtime.gesture_state, HandState.PAUSED)
        runtime.toggle_pause()
        self.assertFalse(runtime.gesture.paused)
        self.assertEqual(runtime.gesture_state, HandState.NEUTRAL)

    def test_the_frame_loop_gets_the_hand_into_the_recogniser(self):
        runtime = make_runtime(self.output, hands=FakeHands(hand_points(thumb_index_ratio=PINCHED)))
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for step in range(6):  # 6 frames at ~30 fps is twice the hold time
            runtime.process_frame(frame, now=step * 0.033)
        self.assertEqual([click.value for click in self.output.clicks], ["left"])
        self.assertEqual(runtime.gesture_state, HandState.LEFT_PINCH)

    def test_tracking_loss_reaches_the_state_machine(self):
        runtime = make_runtime(self.output, hands=NoHands())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        settle(runtime, hand(thumb_index_ratio=PINCHED), 0.0)
        self.assertEqual(runtime.gesture_state, HandState.LEFT_PINCH)

        runtime.process_frame(frame, now=1.0)
        self.assertEqual(runtime.gesture_state, HandState.TRACKING_LOST)
        self.assertIn(EventKind.TRACKING_LOST, [event.kind for event in runtime.last_gesture_events])

    def test_reacquire_requires_a_fresh_pinch(self):
        runtime = make_runtime(self.output)
        settle(runtime, hand(thumb_index_ratio=PINCHED), 0.0)  # one click
        # A lost-tracking frame is the recogniser's empty observation; the frame loop reaches the
        # same call through process_frame() when MediaPipe returns no hand.
        runtime.apply_gestures(None, now=1.0)
        self.assertEqual(runtime.gesture_state, HandState.TRACKING_LOST)

        # The hand comes back already pinched: it must not click.
        settle(runtime, hand(thumb_index_ratio=PINCHED), 1.5)
        self.assertEqual(len(self.output.clicks), 1, "a reacquired pinch must be released first")

        runtime.process_landmarks(hand(), 2.0)  # release
        settle(runtime, hand(thumb_index_ratio=PINCHED), 2.5)
        self.assertEqual(len(self.output.clicks), 2)

    def test_engine_events_are_recorded_for_diagnostics(self):
        runtime = make_runtime(self.output)
        settle(runtime, hand(**TWO_FINGER), 0.0)
        self.assertIn(EventKind.SCROLL_BEGIN, [event.kind for event in runtime.last_gesture_events])


class LegacyEngineIsStillReachableTests(unittest.TestCase):
    """``gesture_engine="legacy"`` must keep working exactly as it did before Phase 4.

    The legacy click path gates on ``debounce_seconds`` measured against ``last_action_at`` (0.0 at
    startup), so these tests start their frames at t=1.0 — the same reason the pre-Phase-4 suite
    used real (large) monotonic timestamps.
    """

    def test_the_legacy_engine_still_clicks_on_a_pinch(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output, gesture_engine="legacy")
        runtime.process_landmarks(hand(), now=1.0)
        runtime.process_landmarks(hand(thumb_index_ratio=PINCHED), now=1.1)
        self.assertEqual([click.value for click in output.clicks], ["left"])

    def test_the_legacy_engine_still_scrolls_from_the_index_finger(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output, gesture_engine="legacy")
        runtime.process_landmarks(hand(), now=1.0)
        runtime.process_landmarks(hand(index_drop=0.10), now=1.1)  # index tip sweeps down
        self.assertGreater(len(output.scrolls), 0, "the pre-Phase-4 scroll path must still work")

    def test_the_two_finger_pose_is_interpreted_differently_by_each_engine(self):
        legacy_output = RecordingMouseOutput()
        legacy = make_runtime(legacy_output, gesture_engine="legacy")
        legacy.process_landmarks(hand(), now=1.0)
        legacy.process_landmarks(hand(**TWO_FINGER), now=1.1)

        modern_output = RecordingMouseOutput()
        modern = make_runtime(modern_output)
        settle(modern, hand(**TWO_FINGER), 0.0)

        self.assertGreater(len(legacy_output.clicks), 0, "legacy: the same pose still clicks")
        self.assertEqual(modern_output.clicks, [], "recognizer: scroll priority, no click")
        self.assertEqual(modern.gesture_state, HandState.TWO_FINGER_SCROLL)


class SummaryTests(unittest.TestCase):
    def test_the_summary_counts_recogniser_events(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output)
        settle(runtime, hand(thumb_index_ratio=PINCHED), 0.0)
        summary = runtime.summary()
        self.assertEqual(summary["emitted"], len(output.events))
        self.assertGreaterEqual(summary["emitted"], 1)

    def test_the_recogniser_never_imports_the_output_layer(self):
        # The whole point of the phase: gesture logic with no path to the mouse.
        import inspect

        import kinematic.recognizer as recognizer_module

        source = inspect.getsource(recognizer_module)
        for forbidden in ("import cv2", "import mediapipe", "pyautogui", "from .output", "import numpy"):
            self.assertNotIn(forbidden, source, f"recognizer.py must not reference {forbidden}")


if __name__ == "__main__":
    unittest.main()
