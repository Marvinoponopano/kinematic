"""Phase 7: two-finger scrolling end to end.

The phase's bar is "test vertical, horizontal, diagonal, stationary, reversed, and tracking-loss
sequences with recorded scroll events", so every assertion below counts calls on a
``RecordingMouseOutput``: the wheel axes are distinguished by ``MouseEvent.axis`` (``scroll`` vs
``hscroll``), no camera is opened and nothing can reach the desktop.

What Phase 7 added to the scroll path:

* the horizontal component is **emitted** (it used to be computed and dropped),
* the dead zone is **per axis**, so sideways wobble during a vertical scroll stays off the screen,
* per-axis **direction inversion** and a separate horizontal gain,
* an event **rate cap** that *accumulates* what it holds back, so capping the rate never loses
  distance (and the sub-unit remainder carries too — truncating each event to an int used to cost
  ~25% of the distance on a measured sweep: 135 units of the 180 the hand asked for).
"""

import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

from kinematic.config import HandMouseConfig
from kinematic.gesture import INDEX_MCP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, PINKY_MCP, THUMB_TIP, WRIST
from kinematic.main import build_config, parse_args
from kinematic.output import MouseEvent, RecordingMouseOutput
from kinematic.ratelimit import ScrollRateLimiter
from kinematic.recognizer import (
    EventKind,
    GestureRecognizer,
    GestureThresholds,
    HandState,
    Observation,
    thresholds_from_config,
)
from kinematic.runtime import HandMouseRuntime

SCALE = 0.15  # wrist -> middle knuckle, so the hand-scale-relative dead zone is exact
PINCHED = 0.33  # below the 0.40 engage ratio
OPEN = 0.70  # above the 0.56 release ratio
STEP = 0.05  # frame step: two steps (0.10 s) clear the 0.08 s hold time
DEADZONE = 0.05 * SCALE  # 0.0075 hand widths

TWO_FINGER: Dict[str, Any] = dict(thumb_index_ratio=PINCHED, thumb_middle_ratio=PINCHED)


def _base_points() -> List[Tuple[float, float]]:
    points = [(0.50, 0.60) for _ in range(21)]
    points[WRIST] = (0.50, 0.60)
    points[INDEX_MCP] = (0.45, 0.46)
    points[MIDDLE_MCP] = (0.50, 0.45)  # exactly SCALE above the wrist
    points[PINKY_MCP] = (0.59, 0.49)
    points[THUMB_TIP] = (0.40, 0.52)
    return points


def hand_points(offset: Tuple[float, float] = (0.0, 0.0), **ratios: float) -> List[Tuple[float, float]]:
    """A two-finger (or other) pose, shifted by *offset* — which moves the scroll reference with it."""
    ox, oy = offset
    points = [(x + ox, y + oy) for x, y in _base_points()]
    thumb = points[THUMB_TIP]
    points[INDEX_TIP] = (thumb[0] + ratios.get("thumb_index_ratio", OPEN) * SCALE, thumb[1])
    points[MIDDLE_TIP] = (thumb[0], thumb[1] - ratios.get("thumb_middle_ratio", OPEN) * SCALE)
    return points


def hand(offset: Tuple[float, float] = (0.0, 0.0), **ratios: float) -> List[SimpleNamespace]:
    return [SimpleNamespace(x=x, y=y, z=0.0) for x, y in hand_points(offset, **ratios)]


class NoHands:
    def process(self, rgb: Any) -> Any:
        return SimpleNamespace(multi_hand_landmarks=None, multi_handedness=None)


def make_runtime(output: RecordingMouseOutput, **overrides: Any) -> HandMouseRuntime:
    config_overrides: Dict[str, Any] = {"show_preview": False, "start_paused": False}
    config_overrides.update(overrides)
    return HandMouseRuntime(config=HandMouseConfig(**config_overrides), output=output, hands=NoHands())


def enter_scroll(runtime: HandMouseRuntime, start: float = 0.0, offset: Tuple[float, float] = (0.0, 0.0)) -> float:
    """Hold the two-finger pose long enough to enter scroll mode; returns the next timestamp.

    The reference is anchored on the frame the gesture registers (the third one here, t = 0.10), so
    movement fed *after* this helper is measured from that pose.
    """
    for index in range(3):
        runtime.process_landmarks(hand(offset, **TWO_FINGER), now=start + index * STEP)
    return start + 3 * STEP


def move(
    runtime: HandMouseRuntime,
    offsets: Sequence[Tuple[float, float]],
    start: float,
    step: float = STEP,
) -> None:
    for index, offset in enumerate(offsets):
        runtime.process_landmarks(hand(offset, **TWO_FINGER), now=start + index * step)


class RecognizerScrollTests(unittest.TestCase):
    """The pure geometry: signs, gains, the per-axis dead zone, inversion."""

    def setUp(self) -> None:
        self.recogniser = GestureRecognizer()

    def enter(self, offset: Tuple[float, float] = (0.0, 0.0), start: float = 0.0) -> float:
        """Hold the two-finger pose until the gesture registers; returns the next timestamp.

        Three frames are needed: the pose must hold for ``hold_seconds`` (0.08 s, so the second frame
        at 0.05 s is too early), and the frame that registers is the one that anchors the scroll
        reference. Movement fed *after* this helper is therefore measured from that pose.
        """
        for index in range(3):
            self.recogniser.update(Observation(hand(offset, **TWO_FINGER), start + index * STEP))
        return start + 3 * STEP

    def scroll_to(self, offset: Tuple[float, float], now: float) -> Any:
        return self.recogniser.update(Observation(hand(offset, **TWO_FINGER), now))

    def test_the_shipped_defaults_scroll_on_both_axes(self):
        thresholds = GestureThresholds()
        self.assertEqual(thresholds.scroll_gain, 900.0)
        self.assertEqual(thresholds.scroll_horizontal_gain, 900.0)
        self.assertFalse(thresholds.invert_vertical)
        self.assertFalse(thresholds.invert_horizontal)

    def test_the_config_fields_reach_the_thresholds(self):
        config = HandMouseConfig(
            scroll_gain=300.0,
            scroll_horizontal_gain=120.0,
            scroll_invert_vertical=True,
            scroll_invert_horizontal=True,
        )
        thresholds = thresholds_from_config(config)
        self.assertEqual(thresholds.scroll_gain, 300.0)
        self.assertEqual(thresholds.scroll_horizontal_gain, 120.0)
        self.assertTrue(thresholds.invert_vertical)
        self.assertTrue(thresholds.invert_horizontal)

    def test_a_negative_gain_is_rejected(self):
        with self.assertRaises(ValueError):
            GestureThresholds(scroll_gain=-900.0)
        with self.assertRaises(ValueError):
            GestureThresholds(scroll_horizontal_gain=-1.0)

    def test_entering_the_pose_anchors_the_reference(self):
        first = self.recogniser.update(Observation(hand((0.0, 0.0), **TWO_FINGER), 0.0))
        second = self.recogniser.update(Observation(hand((0.0, 0.0), **TWO_FINGER), 0.05))
        third = self.recogniser.update(Observation(hand((0.0, 0.0), **TWO_FINGER), 0.10))
        self.assertEqual(first.events, (), "the hold time has not elapsed yet")
        self.assertEqual(second.events, (), "still inside the hold time")
        self.assertEqual([event.kind for event in third.events], [EventKind.SCROLL_BEGIN])
        self.assertEqual(third.state, HandState.TWO_FINGER_SCROLL)

    def test_hand_down_scrolls_down(self):
        now = self.enter()
        event = self.scroll_to((0.0, 0.05), now).events[0]
        self.assertEqual(event.kind, EventKind.SCROLL)
        self.assertAlmostEqual(event.value, -45.0, places=6)
        self.assertEqual(event.secondary, 0.0, "a vertical move must not scroll sideways")

    def test_hand_right_scrolls_right(self):
        now = self.enter()
        event = self.scroll_to((0.05, 0.0), now).events[0]
        self.assertAlmostEqual(event.secondary, 45.0, places=6)
        self.assertEqual(event.value, 0.0, "a horizontal move must not scroll vertically")

    def test_a_diagonal_move_reports_both_axes(self):
        now = self.enter()
        event = self.scroll_to((0.05, 0.05), now).events[0]
        self.assertAlmostEqual(event.value, -45.0, places=6)
        self.assertAlmostEqual(event.secondary, 45.0, places=6)

    def test_the_dead_zone_is_per_axis(self):
        # A hand scrolling down with a wobble in x: the vertical part is sent, the wobble is not.
        now = self.enter()
        vertical = self.scroll_to((0.001, 0.05), now)
        self.assertAlmostEqual(vertical.events[0].value, -45.0, places=6)
        self.assertEqual(vertical.events[0].secondary, 0.0)

        horizontal_recogniser = GestureRecognizer()
        self.recogniser = horizontal_recogniser
        now = self.enter()
        horizontal = self.scroll_to((0.05, 0.001), now)
        self.assertAlmostEqual(horizontal.events[0].secondary, 45.0, places=6)
        self.assertEqual(horizontal.events[0].value, 0.0)

    def test_movement_under_the_dead_zone_on_both_axes_accumulates(self):
        now = self.enter()
        self.assertEqual(self.scroll_to((0.002, 0.002), now).events, ())
        # 0.010 of vertical movement measured from the *held* reference, so nothing is lost.
        update = self.scroll_to((0.002, 0.010), now + STEP)
        self.assertAlmostEqual(update.events[0].value, -0.010 * 900, places=6)

    def test_invert_vertical_flips_only_the_vertical_sign(self):
        self.recogniser = GestureRecognizer(GestureThresholds(invert_vertical=True))
        now = self.enter()
        event = self.scroll_to((0.02, 0.05), now).events[0]
        self.assertAlmostEqual(event.value, 45.0, places=6)
        self.assertAlmostEqual(event.secondary, 0.02 * 900, places=6)

    def test_invert_horizontal_flips_only_the_horizontal_sign(self):
        self.recogniser = GestureRecognizer(GestureThresholds(invert_horizontal=True))
        now = self.enter()
        event = self.scroll_to((0.05, 0.02), now).events[0]
        self.assertAlmostEqual(event.secondary, -45.0, places=6)
        self.assertAlmostEqual(event.value, -0.02 * 900, places=6)

    def test_a_zero_horizontal_gain_is_the_pre_phase_7_behaviour(self):
        self.recogniser = GestureRecognizer(GestureThresholds(scroll_horizontal_gain=0.0))
        now = self.enter()
        event = self.scroll_to((0.05, 0.05), now).events[0]
        self.assertAlmostEqual(event.value, -45.0, places=6)
        self.assertEqual(event.secondary, 0.0)

    def test_the_two_axes_can_have_different_gains(self):
        self.recogniser = GestureRecognizer(GestureThresholds(scroll_gain=100.0, scroll_horizontal_gain=400.0))
        now = self.enter()
        event = self.scroll_to((0.05, 0.05), now).events[0]
        self.assertAlmostEqual(event.value, -5.0, places=6)
        self.assertAlmostEqual(event.secondary, 20.0, places=6)

    def test_a_stationary_two_finger_pose_scrolls_nothing(self):
        now = self.enter()
        self.assertEqual(self.scroll_to((0.0, 0.0), now).events, ())
        self.assertEqual(self.scroll_to((0.0, 0.0), now + STEP).events, ())

    def test_the_reference_resets_when_the_gesture_ends(self):
        # Leave the pose, come back at a completely different place: the re-anchor means the jump
        # between the two positions must not scroll.
        now = self.enter()
        self.scroll_to((0.0, 0.05), now)
        self.recogniser.update(Observation(hand((0.0, 0.0)), now + STEP))  # release: scroll_end
        self.assertEqual(self.recogniser.state, HandState.NEUTRAL)

        # Back in the pose, far from where the hand was: the first frames only re-anchor.
        reanchor = now + 2 * STEP
        self.recogniser.update(Observation(hand((0.30, 0.30), **TWO_FINGER), reanchor))
        update = self.recogniser.update(Observation(hand((0.30, 0.30), **TWO_FINGER), reanchor + STEP))
        self.assertEqual(update.events, (), "the re-anchored reference must not jump")


class RecordingAdapterScrollTests(unittest.TestCase):
    """The runtime: what actually reaches the adapter, per wheel axis."""

    def test_a_vertical_move_emits_one_vertical_scroll(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output)
        now = enter_scroll(runtime)
        runtime.process_landmarks(hand((0.0, 0.05), **TWO_FINGER), now=now)
        self.assertEqual(len(output.vertical_scrolls), 1)
        self.assertEqual(output.horizontal_scrolls, [], "nothing sideways for a vertical move")
        self.assertLess(output.vertical_scrolls[0].value, 0, "hand down = page down")
        self.assertEqual(runtime.gesture_state, HandState.TWO_FINGER_SCROLL)

    def test_a_horizontal_move_emits_one_horizontal_scroll(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output)
        now = enter_scroll(runtime)
        runtime.process_landmarks(hand((0.05, 0.0), **TWO_FINGER), now=now)
        self.assertEqual(len(output.horizontal_scrolls), 1)
        self.assertEqual(output.vertical_scrolls, [], "nothing vertical for a sideways move")
        self.assertGreater(output.horizontal_scrolls[0].value, 0, "hand right = scroll right")
        self.assertEqual(output.horizontal_scrolls[0].axis, "horizontal")

    def test_a_diagonal_move_emits_both_axes(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output)
        now = enter_scroll(runtime)
        runtime.process_landmarks(hand((0.05, 0.05), **TWO_FINGER), now=now)
        self.assertEqual(len(output.vertical_scrolls), 1)
        self.assertEqual(len(output.horizontal_scrolls), 1)
        self.assertEqual(output.counts()["scroll"], 2, "counts() totals both axes")

    def test_sideways_scrolling_can_be_switched_off(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output, scroll_horizontal_gain=0.0)
        now = enter_scroll(runtime)
        runtime.process_landmarks(hand((0.05, 0.05), **TWO_FINGER), now=now)
        self.assertEqual(len(output.vertical_scrolls), 1)
        self.assertEqual(output.horizontal_scrolls, [], "gain 0 = the pre-Phase-7 behaviour")

    def test_the_inversion_fields_reach_the_adapter(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output, scroll_invert_vertical=True, scroll_invert_horizontal=True)
        now = enter_scroll(runtime)
        runtime.process_landmarks(hand((0.05, 0.05), **TWO_FINGER), now=now)
        self.assertGreater(output.vertical_scrolls[0].value, 0)
        self.assertLess(output.horizontal_scrolls[0].value, 0)

    def test_the_scroll_pose_never_clicks(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output)
        now = enter_scroll(runtime)
        move(runtime, [(0.0, 0.01 * (index + 1)) for index in range(10)], now)
        self.assertEqual(output.clicks, [], "scrolling must never click")
        self.assertGreater(len(output.vertical_scrolls), 0)

    def test_a_wobble_during_a_vertical_scroll_emits_no_sideways_events(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output)
        now = enter_scroll(runtime)
        # Straight down, with a sub-dead-zone sideways drift on every frame.
        move(runtime, [(0.001 * (index % 3), 0.01 * (index + 1)) for index in range(12)], now)
        self.assertEqual(output.horizontal_scrolls, [])
        self.assertGreater(len(output.vertical_scrolls), 3)

    def test_a_still_hand_emits_nothing(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output)
        now = enter_scroll(runtime)
        move(runtime, [(0.0, 0.0)] * 10, now)
        self.assertEqual(output.scrolls, [], "a hand that does not move must not scroll")
        self.assertEqual(output.clicks, [])

    def test_a_pinch_before_the_scroll_pose_blocks_the_scroll(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output)
        runtime.process_landmarks(hand((0.0, 0.0), thumb_index_ratio=PINCHED), now=0.0)
        runtime.process_landmarks(hand((0.0, 0.0), thumb_index_ratio=PINCHED), now=0.05)
        runtime.process_landmarks(hand((0.0, 0.0), thumb_index_ratio=PINCHED), now=0.10)
        move(runtime, [(0.1, 0.1)], 0.15)  # straight into the two-finger pose
        self.assertEqual(output.scrolls, [], "no gesture switch without a neutral frame")

    def test_tracking_loss_ends_the_scroll_and_re_anchors(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output)
        now = enter_scroll(runtime)
        runtime.process_landmarks(hand((0.0, 0.05), **TWO_FINGER), now=now)
        self.assertEqual(len(output.vertical_scrolls), 1)

        runtime.apply_gestures(None, now=1.0)  # tracking lost
        self.assertEqual(runtime.gesture_state, HandState.TRACKING_LOST)
        self.assertIn(EventKind.SCROLL_END, [event.kind for event in runtime.last_gesture_events])

        # Back in frame, far from where the hand was: the fresh reference must not scroll the jump.
        back = enter_scroll(runtime, start=1.1, offset=(0.0, -0.20))
        self.assertEqual(len(output.vertical_scrolls), 1, "the reacquire jump must not scroll")

        runtime.process_landmarks(hand((0.0, -0.21), **TWO_FINGER), now=back)  # hand moves up
        self.assertEqual(len(output.vertical_scrolls), 2, "movement after the reanchor does scroll")
        self.assertGreater(output.vertical_scrolls[-1].value, 0, "hand up = page up")

    def test_pausing_mid_scroll_emits_nothing_further(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output)
        now = enter_scroll(runtime)
        runtime.process_landmarks(hand((0.0, 0.05), **TWO_FINGER), now=now)
        self.assertEqual(len(output.vertical_scrolls), 1)

        runtime.pause()
        runtime.process_landmarks(hand((0.0, 0.10), **TWO_FINGER), now=now + STEP)
        self.assertEqual(len(output.vertical_scrolls), 1)
        self.assertEqual(runtime.scroll_limiter.pending, (0.0, 0.0))

    def test_clicks_are_unaffected_by_the_scroll_path(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output, scroll_max_hz=1.0, scroll_horizontal_gain=0.0)
        now = enter_scroll(runtime)
        move(runtime, [(0.0, 0.01 * (index + 1)) for index in range(6)], now)
        later = now + 6 * STEP
        runtime.process_landmarks(hand(), now=later)  # release, so the pinch can register
        runtime.process_landmarks(hand(thumb_index_ratio=PINCHED), now=later + STEP)
        runtime.process_landmarks(hand(thumb_index_ratio=PINCHED), now=later + 2 * STEP)
        runtime.process_landmarks(hand(thumb_index_ratio=PINCHED), now=later + 3 * STEP)
        self.assertEqual([click.value for click in output.clicks], ["left"])

    def test_the_summary_reports_the_scroll_settings(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output, scroll_max_hz=7.0, scroll_horizontal_gain=450.0, scroll_invert_vertical=True)
        summary = runtime.summary()
        self.assertEqual(summary["scroll_max_hz"], 7.0)
        self.assertEqual(summary["scrolls_held"], 0)
        self.assertEqual(summary["scroll_horizontal_gain"], 450.0)
        self.assertTrue(summary["scroll_invert_vertical"])
        self.assertFalse(summary["scroll_invert_horizontal"])


class ScrollFidelityTests(unittest.TestCase):
    """The accumulator: capping the rate must not lose distance, and neither must rounding."""

    def test_the_cap_thins_events_without_losing_distance(self):
        uncapped_out, capped_out = RecordingMouseOutput(), RecordingMouseOutput()
        uncapped, capped = make_runtime(uncapped_out), make_runtime(capped_out, scroll_max_hz=5.0)
        offsets = [(0.0, 0.01 * (index + 1)) for index in range(9)]
        for runtime in (uncapped, capped):
            now = enter_scroll(runtime)
            move(runtime, offsets, now)
            # One frame past the interval, so nothing is left pending in the capped run.
            runtime.process_landmarks(hand((0.0, 0.09), **TWO_FINGER), now=now + 12 * STEP)

        uncapped_total = sum(event.value for event in uncapped_out.vertical_scrolls)
        capped_total = sum(event.value for event in capped_out.vertical_scrolls)
        self.assertLess(len(capped_out.vertical_scrolls), len(uncapped_out.vertical_scrolls))
        self.assertAlmostEqual(capped_total, uncapped_total, delta=1.0, msg="the distance scrolled is preserved")
        self.assertGreater(capped.scroll_limiter.suppressed, 0, "frames really were held back")

    def test_a_burst_of_movement_arrives_as_one_accumulated_event(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output, scroll_max_hz=5.0)
        now = enter_scroll(runtime)
        # 5 frames of movement 10 ms apart (a catching-up loop): all inside the first interval.
        move(runtime, [(0.0, 0.01 * (index + 1)) for index in range(5)], now, step=0.01)
        self.assertEqual(len(output.vertical_scrolls), 1, "one event per interval, not one per frame")

        # Leaving the pose ends the scroll, and the remainder goes out then rather than being lost.
        runtime.process_landmarks(hand(), now=now + 0.02)
        self.assertEqual(len(output.vertical_scrolls), 2)
        self.assertAlmostEqual(
            sum(event.value for event in output.vertical_scrolls),
            -45,
            delta=1.0,
            msg="nothing is lost to the hold bar the sub-unit remainder (which stays pending)",
        )
        self.assertEqual(output.vertical_scrolls[-1].axis, "vertical")

    def test_sub_unit_amounts_are_not_rounded_away(self):
        # With a small gain each frame's amount is a fraction of a wheel unit (a hand width of
        # movement is only 30 units), so per-event `int()` truncation would emit *nothing*. The
        # accumulator carries the remainder, so the total still matches the hand's movement.
        output = RecordingMouseOutput()
        runtime = make_runtime(output, scroll_gain=30.0)
        now = enter_scroll(runtime)
        move(runtime, [(0.0, 0.01 * (index + 1)) for index in range(40)], now)
        total = sum(event.value for event in output.vertical_scrolls)
        self.assertGreater(len(output.vertical_scrolls), 0, "the fractions must eventually add up")
        self.assertAlmostEqual(total, -0.40 * 30.0, delta=1.0)

    def test_the_limiter_keeps_the_sub_unit_remainder(self):
        limiter = ScrollRateLimiter()
        self.assertIsNone(limiter.add(0.4, 0.0, 0.0))
        self.assertEqual(limiter.add(0.7, 0.0, 0.01), (1, 0), "0.4 + 0.7 crosses one unit")
        self.assertAlmostEqual(limiter.pending[0], 0.1, places=9, msg="the remainder stays pending")

    def test_the_limiter_holds_and_releases_whole_units(self):
        limiter = ScrollRateLimiter(1.0)
        self.assertEqual(limiter.add(-9.0, 0.0, 0.0), (-9, 0))
        self.assertIsNone(limiter.add(-3.0, 0.0, 0.5), "inside the interval")
        self.assertEqual(limiter.add(-4.0, 0.0, 1.0), (-7, 0), "the held amount comes out with the next")
        self.assertEqual(limiter.suppressed, 1)

    def test_a_still_hand_does_not_start_the_clock(self):
        limiter = ScrollRateLimiter(1.0)
        self.assertIsNone(limiter.add(0.0, 0.0, 0.0))
        self.assertIsNone(limiter.last_emit_at)
        self.assertIsNone(limiter.add(0.0, 0.0, 0.5))
        self.assertIsNone(limiter.last_emit_at, "nothing was sent, so nothing was scheduled")


class RecordingAdapterAxisTests(unittest.TestCase):
    """The adapter records which wheel axis a call used, without changing counts()."""

    def test_scroll_and_hscroll_are_distinguishable(self):
        output = RecordingMouseOutput()
        output.scroll(-120)
        output.hscroll(45)
        self.assertEqual(output.counts(), {"move": 0, "click": 0, "scroll": 2})
        self.assertEqual([event.axis for event in output.scrolls], ["vertical", "horizontal"])
        self.assertEqual([event.value for event in output.vertical_scrolls], [-120])
        self.assertEqual([event.value for event in output.horizontal_scrolls], [45])
        self.assertEqual(output.scrolls_on("horizontal"), output.horizontal_scrolls)

    def test_a_plain_mouse_event_is_vertical_by_default(self):
        self.assertEqual(MouseEvent("scroll", -3, 0.0).axis, "vertical")


class ScrollCommandLineTests(unittest.TestCase):
    def test_the_scroll_switches_reach_the_config(self):
        config = build_config(
            parse_args(
                [
                    "--scroll-max-hz",
                    "8",
                    "--scroll-horizontal-gain",
                    "250",
                    "--invert-scroll-vertical",
                    "--invert-scroll-horizontal",
                ]
            )
        )
        self.assertEqual(config.scroll_max_hz, 8.0)
        self.assertEqual(config.scroll_horizontal_gain, 250.0)
        self.assertTrue(config.scroll_invert_vertical)
        self.assertTrue(config.scroll_invert_horizontal)

    def test_no_horizontal_scroll_is_the_old_behaviour(self):
        config = build_config(parse_args(["--no-horizontal-scroll"]))
        self.assertEqual(config.scroll_horizontal_gain, 0.0)

    def test_defaults_are_untouched_without_the_flags(self):
        config = build_config(parse_args([]))
        self.assertEqual(config.scroll_gain, 900.0)
        self.assertEqual(config.scroll_horizontal_gain, 900.0)
        self.assertEqual(config.scroll_max_hz, 0.0)
        self.assertFalse(config.scroll_invert_vertical)
        self.assertFalse(config.scroll_invert_horizontal)


def _load_tool() -> Any:
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parent.parent / "tools" / "measure_loop.py"
    spec = importlib.util.spec_from_file_location("measure_loop", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScrollHarnessTests(unittest.TestCase):
    """The sweep harness the manual procedure uses, checked without a camera."""

    def test_the_verdict_accepts_the_documented_signs(self):
        module = _load_tool()
        self.assertTrue(module.scroll_verdict("vertical", -170, 0)[0])
        self.assertTrue(module.scroll_verdict("horizontal", 0, 174)[0])
        self.assertTrue(module.scroll_verdict("diagonal", -174, 174)[0])

    def test_the_verdict_rejects_an_inverted_or_missing_axis(self):
        module = _load_tool()
        self.assertFalse(module.scroll_verdict("vertical", 170, 0)[0], "the hand went down")
        self.assertFalse(module.scroll_verdict("vertical", -170, 12)[0], "unwanted sideways events")
        self.assertFalse(module.scroll_verdict("horizontal", 0, -174)[0])
        self.assertFalse(module.scroll_verdict("diagonal", -174, 0)[0], "the diagonal needs both axes")

    def test_the_sweep_runs_from_minus_to_plus_amplitude(self):
        module = _load_tool()
        hands = module.SyntheticHands(translate=True, scroll_axis="diagonal", scroll_amplitude=0.1, scroll_frames=11)
        self.assertEqual(hands.scroll_offset(1), (-0.1, -0.1))
        self.assertEqual(hands.scroll_offset(11), (0.1, 0.1))
        self.assertEqual(hands.scroll_offset(50), (0.1, 0.1), "and holds once the sweep is done")
        vertical = module.SyntheticHands(scroll_axis="vertical", scroll_frames=11)
        self.assertEqual(vertical.scroll_offset(6), (0.0, 0.0), "one axis only")
        horizontal = module.SyntheticHands(scroll_axis="horizontal", scroll_frames=11)
        self.assertEqual(horizontal.scroll_offset(6), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
