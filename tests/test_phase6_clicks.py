"""Phase 6: left and right clicks through the recording adapter.

The phase's verification bar is "use the recording adapter to confirm exact click counts, cooldown
behaviour, gesture distinction, and suppression rules", so every assertion below counts real calls
on a ``RecordingMouseOutput``: no camera, no display, no pyautogui, nothing that can reach the
desktop.

Also pinned here: the last-action status line the HUD draws (Phase 6's fifth bullet), and the fact
that the legacy engine now computes its pinch with :func:`kinematic.gesture.is_pinch` — the one
pinch helper — instead of a second inline copy of the metric.
"""

import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple
from unittest import mock

import numpy as np

import kinematic.runtime as runtime_module
from kinematic.config import HandMouseConfig
from kinematic.gesture import (
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    PINKY_MCP,
    RING_MCP,
    THUMB_TIP,
    WRIST,
    PINCH_METRICS,
    is_pinch,
    point_distance,
)
from kinematic.output import RecordingMouseOutput
from kinematic.recognizer import HandState
from kinematic.runtime import HandMouseRuntime

SCALE = 0.15  # wrist -> middle knuckle, so the ratios below are exact
PINCHED = 0.33  # clearly below the 0.40 engage threshold
OPEN = 0.70  # clearly above the 0.56 release threshold
STEP = 0.05  # two steps (0.10 s) clear the 0.08 s hold time, and 0.05 is not a binary fraction trap
FRAME = 1.0 / 30.0  # the camera's frame interval, for the tap-timing tests

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
) -> List[Tuple[float, float]]:
    """A plausible 21-landmark hand with exactly the requested pinch ratios."""
    ox, oy = offset
    points = [(x + ox, y + oy) for x, y in _base_points()]
    thumb = points[THUMB_TIP]
    points[INDEX_TIP] = (thumb[0] + thumb_index_ratio * SCALE, thumb[1])
    points[MIDDLE_TIP] = (thumb[0], thumb[1] - thumb_middle_ratio * SCALE)
    return points


def landmarks(points: Sequence[Tuple[float, float]]) -> List[SimpleNamespace]:
    return [SimpleNamespace(x=x, y=y, z=0.0) for x, y in points]


def hand(**kwargs: Any) -> List[SimpleNamespace]:
    return landmarks(hand_points(**kwargs))


def pose_points(kind: str) -> List[Tuple[float, float]]:
    """The three click/scroll poses, by name."""
    if kind == "left":
        return hand_points(thumb_index_ratio=PINCHED)
    if kind == "right":
        return hand_points(thumb_middle_ratio=PINCHED)
    if kind == "two_finger":
        return hand_points(thumb_index_ratio=PINCHED, thumb_middle_ratio=PINCHED)
    raise ValueError(f"unknown pose {kind!r}")


class NoHands:
    def process(self, rgb: Any) -> Any:
        return SimpleNamespace(multi_hand_landmarks=None, multi_handedness=None)


class TappingHands:
    """Replays tools/measure_loop.py's tap pattern: pinched for 6 frames of every 30."""

    def __init__(self, kind: str = "left", period: int = 30, hold: int = 6) -> None:
        self.kind = kind
        self.period = period
        self.hold = hold
        self.calls = 0

    def process(self, rgb: Any) -> Any:
        pinched = (self.calls % self.period) < self.hold
        self.calls += 1
        points = pose_points(self.kind) if pinched else hand_points()
        raw = [SimpleNamespace(x=x, y=y, z=-0.02) for x, y in points]
        handedness = [SimpleNamespace(classification=[SimpleNamespace(label="Right", score=0.98)])]
        return SimpleNamespace(multi_hand_landmarks=[SimpleNamespace(landmark=raw)], multi_handedness=handedness)


def make_runtime(output: RecordingMouseOutput, **overrides: Any) -> HandMouseRuntime:
    hands = overrides.pop("hands", None)
    config_overrides: Dict[str, Any] = {
        "show_preview": False,
        "start_paused": False,
        "preview_scale": 1.0,
        "smoothing": 1.0,
    }
    config_overrides.update(overrides)
    return HandMouseRuntime(
        config=HandMouseConfig(**config_overrides),
        output=output,
        hands=hands if hands is not None else NoHands(),
    )


def feed(
    runtime: HandMouseRuntime,
    points: Sequence[SimpleNamespace],
    start: float,
    frames: int = 1,
    step: float = STEP,
) -> float:
    """Feed one pose for *frames* frames; returns the timestamp after the last frame."""
    now = start
    for index in range(frames):
        now = start + index * step
        runtime.process_landmarks(points, now=now)
    return now + step


def tap(runtime: HandMouseRuntime, start: float, kind: str = "left", hold: int = 4, release: int = 3) -> float:
    """One deliberate tap: hold a pose long enough to register, then open the hand."""
    now = feed(runtime, landmarks(pose_points(kind)), start, frames=hold)
    return feed(runtime, hand(), now, frames=release)


class ClickCountTests(unittest.TestCase):
    """Exact counts: one click per deliberate tap, and the poses are told apart."""

    def test_five_deliberate_taps_are_exactly_five_clicks(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output)
        now = 0.0
        for _ in range(5):
            now = tap(runtime, now, kind="left")
        self.assertEqual([click.value for click in output.clicks], ["left"] * 5)
        self.assertEqual(output.scrolls, [])
        self.assertEqual(runtime.actions(), {"left click": 5})

    def test_a_right_tap_clicks_right_and_the_pose_is_distinct(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output)
        now = 0.0
        for _ in range(3):
            now = tap(runtime, now, kind="right")
        self.assertEqual([click.value for click in output.clicks], ["right"] * 3)

        # The pose is reported as its own state while it is held (a tap ends with a release).
        feed(runtime, hand(thumb_middle_ratio=PINCHED), 10.0, frames=4)
        self.assertEqual(runtime.gesture_state, HandState.RIGHT_PINCH)
        runtime.process_landmarks(hand(), 11.0)
        self.assertEqual(runtime.gesture_state, HandState.NEUTRAL)

    def test_a_sustained_pinch_is_one_click_not_one_per_frame(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output)
        feed(runtime, hand(thumb_index_ratio=PINCHED), 0.0, frames=60)
        self.assertEqual(len(output.clicks), 1)
        self.assertEqual(runtime.gesture_state, HandState.LEFT_PINCH)

    def test_a_flicker_shorter_than_the_hold_time_never_clicks(self):
        # A pose that comes and goes between frames is a tracking wobble, not a tap: ten single-frame
        # pinches must produce nothing at all.
        output = RecordingMouseOutput()
        runtime = make_runtime(output)
        now = 0.0
        for _ in range(10):
            now = feed(runtime, hand(thumb_index_ratio=PINCHED), now, frames=1)
            now = feed(runtime, hand(), now, frames=3)
        self.assertEqual(output.clicks, [])
        self.assertEqual(runtime.gesture_state, HandState.NEUTRAL)

    def test_the_scroll_pose_scrolls_and_never_clicks(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output)
        feed(runtime, hand(**TWO_FINGER), 0.0, frames=4)
        runtime.process_landmarks(hand(offset=(0.0, 0.05), **TWO_FINGER), now=0.5)
        self.assertEqual(output.clicks, [], "scroll outranks both clicks")
        self.assertEqual(len(output.scrolls), 1)
        self.assertEqual(runtime.gesture_state, HandState.TWO_FINGER_SCROLL)

    def test_the_frame_loop_produces_one_click_per_deliberate_tap(self):
        # The whole pipeline, as the app runs it: MediaPipe-shaped results in, recorded clicks out.
        # 150 frames at 30 fps is 5 taps on the tool's 30-frame period.
        output = RecordingMouseOutput()
        runtime = make_runtime(output, hands=TappingHands("left"))
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for step in range(150):
            runtime.process_frame(frame, now=step * FRAME)
        self.assertEqual(len(output.clicks), 5)
        self.assertTrue(all(click.value == "left" for click in output.clicks))
        self.assertEqual(output.scrolls, [])

    def test_the_frame_loop_clicks_right_for_the_middle_finger_pose(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output, hands=TappingHands("right"))
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for step in range(150):
            runtime.process_frame(frame, now=step * FRAME)
        self.assertEqual([click.value for click in output.clicks], ["right"] * 5)

    def test_the_frame_loop_never_clicks_on_the_scroll_pose(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output, hands=TappingHands("two_finger"))
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for step in range(150):
            runtime.process_frame(frame, now=step * FRAME)
        self.assertEqual(output.clicks, [], "the scroll pose must never click, tap pattern or not")


class CooldownAndReArmTests(unittest.TestCase):
    """Two clicks are never closer than the cooldown, and a pose must be released to fire again."""

    def test_a_pinch_inside_the_cooldown_is_dropped_rather_than_deferred(self):
        # At 30 fps the earliest second pinch after a click: one release frame, then the hold time
        # (0.08 s). That lands ~0.167 s after the click, inside the 0.18 s cooldown — so it must be
        # dropped, and holding it longer must not fire a late click either.
        output = RecordingMouseOutput()
        runtime = make_runtime(output)
        now = 0.0
        now = feed(runtime, hand(thumb_index_ratio=PINCHED), now, frames=4, step=FRAME)  # clicks
        self.assertEqual(len(output.clicks), 1)
        now = feed(runtime, hand(), now, frames=1, step=FRAME)  # release
        now = feed(runtime, hand(thumb_index_ratio=PINCHED), now, frames=10, step=FRAME)  # inside
        self.assertEqual(len(output.clicks), 1, "the cooldown drops the click, it does not queue it")

        now = feed(runtime, hand(), now, frames=2, step=FRAME)  # release again
        feed(runtime, hand(thumb_index_ratio=PINCHED), now, frames=5, step=FRAME)  # past the window
        self.assertEqual(len(output.clicks), 2, "release + re-pinch after the cooldown clicks again")

    def test_release_and_repinch_is_required_for_a_second_click(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output)
        feed(runtime, hand(thumb_index_ratio=PINCHED), 0.0, frames=30)
        self.assertEqual(len(output.clicks), 1, "a held pinch stays one click")

        runtime.process_landmarks(hand(), 2.0)  # release
        feed(runtime, hand(thumb_index_ratio=PINCHED), 2.1, frames=4)
        self.assertEqual(len(output.clicks), 2)

    def test_a_left_pinch_cannot_become_a_right_pinch_without_neutral(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output)
        feed(runtime, hand(thumb_index_ratio=PINCHED), 0.0, frames=4)
        feed(runtime, hand(thumb_middle_ratio=PINCHED), 0.5, frames=8)
        self.assertEqual([click.value for click in output.clicks], ["left"])

        runtime.process_landmarks(hand(), 1.0)  # neutral
        feed(runtime, hand(thumb_middle_ratio=PINCHED), 1.1, frames=4)
        self.assertEqual([click.value for click in output.clicks], ["left", "right"])

    def test_a_pinch_cannot_become_a_scroll_without_neutral(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output)
        feed(runtime, hand(thumb_index_ratio=PINCHED), 0.0, frames=4)
        feed(runtime, hand(**TWO_FINGER), 0.5, frames=8)
        self.assertEqual(output.scrolls, [], "no gesture switch without a neutral frame")
        self.assertEqual(runtime.gesture_state, HandState.LEFT_PINCH)

        runtime.process_landmarks(hand(), 1.0)
        feed(runtime, hand(**TWO_FINGER), 1.1, frames=4)
        runtime.process_landmarks(hand(offset=(0.0, 0.05), **TWO_FINGER), now=1.5)
        self.assertEqual(len(output.scrolls), 1)
        self.assertEqual(len(output.clicks), 1)


class SuppressionTests(unittest.TestCase):
    """Paused and tracking-lost frames emit nothing, including a click that was about to fire."""

    def test_a_pinch_started_while_paused_does_not_click_after_resuming(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output, start_paused=True)
        feed(runtime, hand(thumb_index_ratio=PINCHED), 0.0, frames=6)
        self.assertEqual(output.clicks, [], "nothing may be emitted while paused")
        self.assertEqual(runtime.gesture_state, HandState.PAUSED)

        runtime.resume()
        feed(runtime, hand(thumb_index_ratio=PINCHED), 1.0, frames=20)
        self.assertEqual(output.clicks, [], "the rising edge belonged to the paused period")

        runtime.process_landmarks(hand(), 3.0)  # release
        feed(runtime, hand(thumb_index_ratio=PINCHED), 3.1, frames=4)
        self.assertEqual(len(output.clicks), 1, "a fresh pinch after the resume does click")

    def test_pausing_mid_scroll_ends_it_and_emits_nothing(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output)
        feed(runtime, hand(**TWO_FINGER), 0.0, frames=4)
        runtime.process_landmarks(hand(offset=(0.0, 0.05), **TWO_FINGER), now=0.5)
        self.assertEqual(len(output.scrolls), 1)

        runtime.pause()
        runtime.process_landmarks(hand(offset=(0.0, 0.10), **TWO_FINGER), now=0.6)
        runtime.process_landmarks(hand(offset=(0.0, 0.15), **TWO_FINGER), now=0.7)
        self.assertEqual(len(output.scrolls), 1, "the scroll ended with the pause")
        self.assertEqual(output.clicks, [])

    def test_a_pinch_held_through_a_tracking_loss_does_not_click(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output)
        feed(runtime, hand(thumb_index_ratio=PINCHED), 0.0, frames=2)  # hold time not elapsed
        runtime.apply_gestures(None, now=0.075)  # tracking lost before the pose could register
        self.assertEqual(output.clicks, [], "the loss frame must not click")
        self.assertEqual(runtime.gesture_state, HandState.TRACKING_LOST)

        feed(runtime, hand(thumb_index_ratio=PINCHED), 0.10, frames=20)  # back, still pinched
        self.assertEqual(output.clicks, [], "a reacquired pinch must be released and pinched again")
        self.assertEqual(runtime.gesture_state, HandState.LEFT_PINCH)

        runtime.process_landmarks(hand(), 2.0)
        feed(runtime, hand(thumb_index_ratio=PINCHED), 2.1, frames=4)
        self.assertEqual(len(output.clicks), 1)

    def test_tracking_loss_through_the_frame_loop_emits_nothing(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output, hands=NoHands())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for step in range(30):
            runtime.process_frame(frame, now=step * FRAME)
        self.assertEqual(output.counts(), {"move": 0, "click": 0, "scroll": 0})


class LastActionStatusTests(unittest.TestCase):
    """Phase 6's fifth bullet: the preview says what the last click/scroll was."""

    def preview(self, runtime: HandMouseRuntime) -> np.ndarray:
        return runtime.draw_preview(np.zeros((240, 640, 3), dtype=np.uint8))

    def test_a_fresh_session_has_nothing_to_report(self):
        runtime = make_runtime(RecordingMouseOutput())
        self.assertIsNone(runtime.last_action)
        self.assertEqual(runtime.actions(), {})
        self.assertEqual(runtime.last_action_text(now=0.0), "last: none yet")

    def test_the_text_gives_the_action_its_age_and_the_session_count(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output)
        feed(runtime, hand(thumb_index_ratio=PINCHED), 0.0, frames=3)  # click on the third frame
        text = runtime.last_action_text(now=2.60)
        self.assertIn("left click", text)
        self.assertIn("2.5s ago", text)
        self.assertIn("(1 this session)", text)

    def test_the_counts_accumulate_per_kind(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output)
        now = 0.0
        for kind in ("left", "right", "left"):
            now = tap(runtime, now, kind=kind)
        self.assertEqual(runtime.actions(), {"left click": 2, "right click": 1})
        self.assertEqual(runtime.last_action, "left click")

    def test_a_scroll_counts_as_an_action_too(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output)
        feed(runtime, hand(**TWO_FINGER), 0.0, frames=4)
        runtime.process_landmarks(hand(offset=(0.0, 0.05), **TWO_FINGER), now=0.5)
        self.assertEqual(runtime.actions(), {"scroll": 1})
        self.assertIn("scroll", runtime.last_action_text(now=1.0))

    def test_the_summary_reports_the_last_action_and_the_counts(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output)
        tap(runtime, 0.0, kind="right")
        summary = runtime.summary()
        self.assertEqual(summary["last_action"], "right click")
        self.assertEqual(summary["actions"], {"right click": 1})

    def test_the_hud_line_is_drawn_once_something_has_fired(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output, show_fps=False, show_sensitivity=False, show_landmarks=False)
        self.assertEqual(int(self.preview(runtime).sum()), 0, "nothing to show before a click")

        feed(runtime, hand(thumb_index_ratio=PINCHED), 0.0, frames=3)
        drawn = self.preview(runtime)
        self.assertTrue((drawn[14:24, :, :] != 0).any(), "the last-action line takes the first row")

    def test_the_hud_line_follows_the_dial_when_both_are_on(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output, show_fps=True, show_sensitivity=True, show_landmarks=False)
        feed(runtime, hand(thumb_index_ratio=PINCHED), 0.0, frames=3)
        drawn = self.preview(runtime)
        self.assertTrue((drawn[14:24, :, :] != 0).any(), "fps/gesture line")
        self.assertTrue((drawn[38:48, :, :] != 0).any(), "sensitivity line")
        self.assertTrue((drawn[62:74, :, :] != 0).any(), "last-action line, one row below the dial")

    def test_the_hud_line_can_be_switched_off(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(
            output, show_fps=False, show_sensitivity=False, show_landmarks=False, show_last_action=False
        )
        feed(runtime, hand(thumb_index_ratio=PINCHED), 0.0, frames=3)
        self.assertEqual(runtime.last_action, "left click", "the action still happened")
        self.assertEqual(int(self.preview(runtime).sum()), 0, "it is just not drawn")

    def test_the_legacy_engine_records_its_actions_too(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output, gesture_engine="legacy")
        runtime.process_landmarks(hand(thumb_index_ratio=PINCHED), now=1.0)
        self.assertEqual(runtime.actions(), {"left click": 1})
        self.assertEqual(runtime.last_action_text(now=1.4), "last: left click  0.4s ago  (1 this session)")

    def test_nothing_is_recorded_while_paused(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output, start_paused=True)
        feed(runtime, hand(thumb_index_ratio=PINCHED), 0.0, frames=10)
        self.assertIsNone(runtime.last_action)
        self.assertEqual(runtime.actions(), {})


class LegacyPinchHelperTests(unittest.TestCase):
    """The duplicated pinch test is gone: the legacy engine uses ``gesture.is_pinch``."""

    def test_the_legacy_path_calls_the_shared_helper_with_the_l1_metric(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output, gesture_engine="legacy")
        seen: List[str] = []
        real = runtime_module.is_pinch

        def spy(points, threshold=0.06, metric="euclidean"):
            seen.append(metric)
            return real(points, threshold=threshold, metric=metric)

        with mock.patch.object(runtime_module, "is_pinch", spy):
            runtime.process_landmarks(hand(thumb_index_ratio=PINCHED), now=1.0)
        self.assertTrue(seen, "the legacy click path must go through the shared helper")
        self.assertEqual(set(seen), {"l1"}, "and with the metric the prototype used")

    def test_the_recognizer_does_not_use_the_absolute_pinch_helper(self):
        # The default engine works from hand-scale-relative ratios; it must not call the absolute
        # helper at all, or the two ways of deciding "pinch" would drift apart again.
        output = RecordingMouseOutput()
        runtime = make_runtime(output)
        with mock.patch.object(runtime_module, "is_pinch") as spy:
            feed(runtime, hand(thumb_index_ratio=PINCHED), 0.0, frames=4)
        spy.assert_not_called()
        self.assertEqual(len(output.clicks), 1)

    def test_the_helper_reproduces_the_inline_l1_test_exactly(self):
        threshold = 0.06
        for offset in ((0.001, 0.0), (0.02, 0.02), (0.04, 0.04), (0.05, 0.005), (0.06, 0.0), (0.10, 0.05)):
            points = [(0.50, 0.50) for _ in range(21)]
            points[THUMB_TIP] = (0.40, 0.50)
            points[INDEX_TIP] = (0.40 + offset[0], 0.50 + offset[1])
            expected = abs(offset[0]) + abs(offset[1]) < threshold  # the pre-Phase-4 expression
            self.assertEqual(
                is_pinch(landmarks(points), threshold=threshold, metric="l1"),
                expected,
                f"L1 metric must match the inline test for offset {offset}",
            )

    def test_the_two_metrics_disagree_on_a_diagonal_pinch(self):
        # The reason the metric is an argument rather than an accident: a 45-degree 0.04/0.04 offset
        # is 0.057 Euclidean (a pinch) but 0.08 in L1 (not one).
        points = [(0.50, 0.50) for _ in range(21)]
        points[THUMB_TIP] = (0.40, 0.50)
        points[INDEX_TIP] = (0.44, 0.54)
        self.assertLess(point_distance((0.40, 0.50), (0.44, 0.54)), 0.06)
        self.assertTrue(is_pinch(landmarks(points), threshold=0.06, metric="euclidean"))
        self.assertFalse(is_pinch(landmarks(points), threshold=0.06, metric="l1"))

    def test_an_unknown_metric_is_rejected(self):
        with self.assertRaises(ValueError):
            is_pinch(hand(), threshold=0.06, metric="chebyshev")
        self.assertEqual(PINCH_METRICS, ("euclidean", "l1"))

    def test_the_legacy_threshold_is_still_the_config_field(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output, gesture_engine="legacy", click_threshold=0.10)
        # 0.08 in L1 would not pinch at the default 0.06, but does at the configured 0.10.
        points = [(0.50, 0.50) for _ in range(21)]
        points[THUMB_TIP] = (0.40, 0.50)
        points[INDEX_TIP] = (0.44, 0.54)
        runtime.process_landmarks(landmarks(points), now=1.0)
        self.assertEqual([click.value for click in output.clicks], ["left"])


if __name__ == "__main__":
    unittest.main()
