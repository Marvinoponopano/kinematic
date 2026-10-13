"""Phase 4 tests: the pure gesture recogniser and its state machine.

Everything here is deterministic — fake timestamps, synthetic landmarks, no camera, no display
and no mouse: the recogniser cannot reach pyautogui even in principle (it imports nothing from
the output layer). Threshold semantics under test:

* ``pinch_ratio`` 0.40 of the hand's scale engages a pair;
* ``pinch_ratio * release_ratio`` = 0.56 releases it (the hysteresis band sits in between);
* a pose must **hold** for ``hold_seconds`` (0.08 s) before it registers, and the clock starts on
  the pose's own first frame;
* two clicks are never closer than ``cooldown_seconds`` (0.18 s).

That third rule is why almost every test feeds a pose for two or three frames: at 0.04 s per frame
the commit frame is the third one, and ``settle()`` below makes that explicit instead of hiding it
in a hand-counted timestamp.
"""

import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from kinematic.config import HandMouseConfig
from kinematic.gesture import MIDDLE_MCP, MIDDLE_TIP, THUMB_TIP, INDEX_TIP, WRIST
from kinematic.recognizer import (
    EventKind,
    GESTURE_PRIORITY,
    GestureEvent,
    GestureRecognizer,
    GestureThresholds,
    GestureUpdate,
    HandState,
    Observation,
    thresholds_from_config,
)

SCALE = 0.15  # nominal hand scale (wrist -> middle knuckle), in normalized frame units

PINCHED = 0.33  # thumb/finger ratio clearly below the 0.40 engage threshold
BAND = 0.47  # inside the hysteresis band: above engage, below release
OPEN = 0.70  # clearly above the 0.56 release threshold

FRAME = 0.033  # a ~30 fps frame
STEP = 0.05  # settle()'s step: two steps (0.10) clear the 0.08 s hold time with room to spare

TWO_FINGER: Dict[str, Any] = dict(thumb_index_ratio=PINCHED, thumb_middle_ratio=PINCHED)


def hand(
    thumb_index_ratio: float = OPEN,
    thumb_middle_ratio: float = OPEN,
    scale: float = SCALE,
    offset: Tuple[float, float] = (0.0, 0.0),
) -> List[SimpleNamespace]:
    """A 21-landmark synthetic hand whose pinch distances are exactly ``ratio * scale``.

    ``hand_scale()`` measures wrist -> middle knuckle, so the middle knuckle is placed ``scale``
    above the wrist and the index/middle fingertips are placed at the requested distance from the
    thumb tip. Passing ratios (not distances) is the point: every test states its pose in units of
    the hand, which is what the recogniser compares against.
    """
    ox, oy = offset
    points = [(0.40 + ox, 0.90 + oy) for _ in range(21)]
    points[WRIST] = (0.40 + ox, 0.90 + oy)
    points[MIDDLE_MCP] = (0.40 + ox, 0.90 + oy - scale)
    thumb = (0.60 + ox, 0.60 + oy)
    points[THUMB_TIP] = thumb
    points[INDEX_TIP] = (thumb[0] + thumb_index_ratio * scale, thumb[1])
    points[MIDDLE_TIP] = (thumb[0], thumb[1] - thumb_middle_ratio * scale)
    return [SimpleNamespace(x=x, y=y, z=0.0) for x, y in points]


def settle(
    recogniser: GestureRecognizer,
    landmarks: Any,
    start: float,
    frames: int = 3,
    step: float = STEP,
) -> Tuple[GestureUpdate, float]:
    """Feed one pose until its hold time has elapsed; return the last update and its timestamp.

    ``hold_seconds`` is 0.08 and ``step`` is 0.05, so the default three frames always include the
    commit frame with margin (0.10 s of holding). The step deliberately does *not* equal the hold
    time: 0.12 - 0.04 is 0.07999999999999999 in binary floating point, which would make every test
    depend on rounding rather than on the rule. Use ``frames=1`` for "has not held yet" assertions.
    """
    update = recogniser.update(Observation(landmarks, start))
    now = start
    for index in range(1, frames):
        now = start + index * step
        update = recogniser.update(Observation(landmarks, now))
    return update, now


def kinds(events: Sequence[GestureEvent]) -> List[str]:
    return [event.kind.value for event in events]


def clicks(events: Sequence[GestureEvent]) -> List[str]:
    return [kind for kind in kinds(events) if kind in ("left_click", "right_click")]


class ThresholdTests(unittest.TestCase):
    def test_default_thresholds_and_band_ordering(self):
        thresholds = GestureThresholds()
        self.assertLess(thresholds.pinch_ratio, thresholds.pinch_ratio * thresholds.release_ratio)
        self.assertEqual(thresholds.release_ratio, 1.4)
        self.assertEqual(thresholds.hold_seconds, 0.08)
        self.assertEqual(thresholds.cooldown_seconds, 0.18)
        self.assertTrue(thresholds.require_neutral_between)

    def test_the_three_named_poses_sit_where_the_tests_assume(self):
        thresholds = GestureThresholds()
        self.assertLess(PINCHED, thresholds.pinch_ratio, "PINCHED must engage")
        self.assertGreater(BAND, thresholds.pinch_ratio, "BAND must not engage")
        self.assertLess(BAND, thresholds.pinch_ratio * thresholds.release_ratio, "BAND must not release")
        self.assertGreater(OPEN, thresholds.pinch_ratio * thresholds.release_ratio, "OPEN must release")

    def test_threshold_validation(self):
        bad: List[Dict[str, Any]] = [
            {"pinch_ratio": 0.0},
            {"pinch_ratio": -0.1},
            {"release_ratio": 0.9},
            {"hold_seconds": -0.01},
            {"cooldown_seconds": -1.0},
        ]
        for kwargs in bad:
            with self.assertRaises(ValueError, msg=f"{kwargs} should be rejected"):
                GestureThresholds(**kwargs)

    def test_thresholds_come_from_the_config(self):
        config = HandMouseConfig()
        thresholds = thresholds_from_config(config)
        self.assertEqual(thresholds.pinch_ratio, config.pinch_ratio)
        self.assertEqual(thresholds.release_ratio, config.pinch_release_ratio)
        self.assertEqual(thresholds.hold_seconds, config.pinch_hold_seconds)
        self.assertEqual(thresholds.cooldown_seconds, config.click_cooldown_seconds)
        self.assertEqual(thresholds.scroll_deadzone_ratio, config.scroll_deadzone_ratio)
        self.assertEqual(thresholds.scroll_gain, config.scroll_gain)

    def test_default_config_engine_is_the_recogniser(self):
        self.assertEqual(HandMouseConfig().gesture_engine, "recognizer")


class RelativeThresholdTests(unittest.TestCase):
    """The point of the phase: thresholds in hand units, not absolute distances."""

    def test_the_same_pose_pinches_on_a_small_and_a_large_hand(self):
        outcomes: Dict[float, List[str]] = {}
        for scale in (0.10, 0.20):
            recogniser = GestureRecognizer()
            settle(recogniser, hand(scale=scale), 0.0)
            update, _ = settle(recogniser, hand(scale=scale, thumb_index_ratio=PINCHED), 0.2)
            outcomes[scale] = clicks(update.events)
        # The same pose, 3x the distance in frame units, the same outcome.
        self.assertEqual(outcomes[0.10], ["left_click"])
        self.assertEqual(outcomes[0.20], ["left_click"])

    def test_absolute_distance_is_not_the_criterion(self):
        # 0.07 in frame units is a pinch on a large hand (ratio 0.35) and clearly not one on a
        # small hand (ratio 0.70), even though the absolute distance is identical.
        small = GestureRecognizer()
        settle(small, hand(thumb_index_ratio=0.70, scale=0.10), 0.0)
        update, _ = settle(small, hand(thumb_index_ratio=0.70, scale=0.10), 0.2)
        self.assertEqual(update.state, HandState.NEUTRAL)

        large = GestureRecognizer()
        settle(large, hand(thumb_index_ratio=0.70, scale=0.20), 0.0)
        update, _ = settle(large, hand(thumb_index_ratio=0.35, scale=0.20), 0.2)
        self.assertEqual(update.state, HandState.LEFT_PINCH)
        self.assertEqual(clicks(update.events), ["left_click"])

    def test_hand_scale_is_recorded_for_diagnostics(self):
        recogniser = GestureRecognizer()
        recogniser.update(Observation(hand(scale=0.17), 0.0))
        self.assertAlmostEqual(recogniser.last_hand_scale or 0.0, 0.17, places=6)
        self.assertAlmostEqual(recogniser.last_ratios["thumb_index"], OPEN, places=6)


class HoldTimeTests(unittest.TestCase):
    def test_no_click_before_the_hold_time(self):
        recogniser = GestureRecognizer()
        recogniser.update(Observation(hand(thumb_index_ratio=PINCHED), 0.0))  # pose starts here
        for now in (0.01, 0.03, 0.07):
            update = recogniser.update(Observation(hand(thumb_index_ratio=PINCHED), now))
            self.assertEqual(update.events, (), f"nothing may fire at t={now}")
            self.assertEqual(update.state, HandState.NEUTRAL, "the pose is not committed until it holds")

    def test_click_after_the_hold_time(self):
        # Times chosen as exact binary fractions so the assertion is about the rule, not about
        # how 0.08 rounds: 0.0625 s must not commit, 0.125 s must.
        recogniser = GestureRecognizer()
        recogniser.update(Observation(hand(), 0.25))
        recogniser.update(Observation(hand(thumb_index_ratio=PINCHED), 0.5))  # the hold starts here
        too_early = recogniser.update(Observation(hand(thumb_index_ratio=PINCHED), 0.5625))
        self.assertEqual(too_early.events, (), "0.0625 s is less than the 0.08 s hold")
        committed = recogniser.update(Observation(hand(thumb_index_ratio=PINCHED), 0.625))
        self.assertEqual(clicks(committed.events), ["left_click"])
        self.assertEqual(committed.state, HandState.LEFT_PINCH)

    def test_hold_clock_restarts_after_a_flicker(self):
        # Pinch, drop the pose before the hold elapses, pinch again: the click must come a full
        # hold after the *second* pinch, not the first.
        recogniser = GestureRecognizer()
        recogniser.update(Observation(hand(), 0.0))
        recogniser.update(Observation(hand(thumb_index_ratio=PINCHED), 0.03))
        recogniser.update(Observation(hand(), 0.06))
        recogniser.update(Observation(hand(thumb_index_ratio=PINCHED), 0.10))
        early = recogniser.update(Observation(hand(thumb_index_ratio=PINCHED), 0.14))
        self.assertEqual(early.events, (), "the earlier flicker must not count towards the hold")
        late = recogniser.update(Observation(hand(thumb_index_ratio=PINCHED), 0.185))
        self.assertEqual(clicks(late.events), ["left_click"])

    def test_zero_hold_time_clicks_on_the_first_pinched_frame(self):
        recogniser = GestureRecognizer(GestureThresholds(hold_seconds=0.0))
        recogniser.update(Observation(hand(), 0.0))
        update = recogniser.update(Observation(hand(thumb_index_ratio=PINCHED), FRAME))
        self.assertEqual(clicks(update.events), ["left_click"])


class ClickRepeatTests(unittest.TestCase):
    def test_a_sustained_pinch_clicks_once(self):
        recogniser = GestureRecognizer()
        fired = []
        for step in range(40):
            update = recogniser.update(Observation(hand(thumb_index_ratio=PINCHED), step * FRAME))
            fired.extend(clicks(update.events))
        self.assertEqual(len(fired), 1, "holding a pinch is one click, not one per frame")

    def test_release_then_pinch_clicks_again(self):
        recogniser = GestureRecognizer()
        fired: List[str] = []
        for cycle in range(4):
            base = cycle * 0.5
            recogniser.update(Observation(hand(), base))
            update, _ = settle(recogniser, hand(thumb_index_ratio=PINCHED), base + 0.1)
            fired.extend(clicks(update.events))
        self.assertEqual(len(fired), 4, "release + re-pinch must click every time")

    def test_click_inside_the_cooldown_is_dropped_not_deferred(self):
        recogniser = GestureRecognizer()
        recogniser.update(Observation(hand(), 0.0))
        first, first_click_at = settle(recogniser, hand(thumb_index_ratio=PINCHED), 0.10)
        self.assertEqual(clicks(first.events), ["left_click"])

        recogniser.update(Observation(hand(), first_click_at + 0.01))  # release: re-arms the pair
        second, commit_at = settle(recogniser, hand(thumb_index_ratio=PINCHED), first_click_at + 0.02)
        self.assertLess(commit_at - first_click_at, 0.18, "the test must stay inside the cooldown")
        self.assertEqual(second.events, (), "a click inside the cooldown must not fire")
        self.assertEqual(second.state, HandState.LEFT_PINCH, "the pose is consumed, not queued")

    def test_click_after_the_cooldown_fires(self):
        recogniser = GestureRecognizer()
        recogniser.update(Observation(hand(), 0.0))
        first, now = settle(recogniser, hand(thumb_index_ratio=PINCHED), 0.10)
        self.assertEqual(clicks(first.events), ["left_click"])

        recogniser.update(Observation(hand(), now + 0.05))
        later, when = settle(recogniser, hand(thumb_index_ratio=PINCHED), now + 0.25)
        self.assertGreaterEqual(when - now, 0.18)
        self.assertEqual(clicks(later.events), ["left_click"])

    def test_fast_pinching_cannot_exceed_the_cooldown_rate(self):
        # 12 pinch/release cycles of 0.12 s: long enough to hold each time, too fast for the
        # 0.18 s cooldown to allow one click per cycle.
        recogniser = GestureRecognizer()
        cooldown = GestureThresholds().cooldown_seconds
        total = 0.0
        fired: List[str] = []
        for cycle in range(12):
            base = cycle * 0.12
            recogniser.update(Observation(hand(), base))
            update, _ = settle(recogniser, hand(thumb_index_ratio=PINCHED), base + 0.04)
            fired.extend(clicks(update.events))
            total = base + 0.12
        ceiling = int(total / cooldown) + 1
        self.assertLessEqual(len(fired), ceiling, "clicks cannot be closer together than the cooldown")
        self.assertGreaterEqual(len(fired), 1, "and the first one must still land")


class HysteresisTests(unittest.TestCase):
    def test_the_band_does_not_release_an_engaged_pair(self):
        recogniser = GestureRecognizer()
        recogniser.update(Observation(hand(), 0.0))
        settle(recogniser, hand(thumb_index_ratio=PINCHED), 0.1)
        update = recogniser.update(Observation(hand(thumb_index_ratio=BAND), 0.5))
        self.assertEqual(update.state, HandState.LEFT_PINCH, "0.47 is inside the band: stay engaged")
        self.assertEqual(update.events, ())

    def test_oscillating_at_the_threshold_yields_one_click(self):
        recogniser = GestureRecognizer()
        fired: List[str] = []
        for step in range(100):
            ratio = PINCHED if step % 2 == 0 else BAND
            update = recogniser.update(Observation(hand(thumb_index_ratio=ratio), step * 0.02))
            fired.extend(clicks(update.events))
        self.assertEqual(len(fired), 1, "a hand resting at the threshold must not chatter")

    def test_release_needs_the_release_threshold(self):
        recogniser = GestureRecognizer()
        recogniser.update(Observation(hand(), 0.0))
        settle(recogniser, hand(thumb_index_ratio=PINCHED), 0.1)
        in_band = recogniser.update(Observation(hand(thumb_index_ratio=BAND), 0.5))
        self.assertEqual(in_band.state, HandState.LEFT_PINCH)

        released = recogniser.update(Observation(hand(thumb_index_ratio=OPEN), 0.6))
        self.assertEqual(released.state, HandState.NEUTRAL)

    def test_a_pair_in_the_band_cannot_re_arm(self):
        # After a click, moving only into the band leaves the pair engaged, so pinching again must
        # not fire a second click until the release threshold has been passed.
        recogniser = GestureRecognizer()
        recogniser.update(Observation(hand(), 0.0))
        settle(recogniser, hand(thumb_index_ratio=PINCHED), 0.1)
        recogniser.update(Observation(hand(thumb_index_ratio=BAND), 0.5))
        second, _ = settle(recogniser, hand(thumb_index_ratio=PINCHED), 0.9)
        self.assertEqual(second.events, (), "still engaged: the same pinch cannot click twice")

        recogniser.update(Observation(hand(thumb_index_ratio=OPEN), 1.2))  # released
        third, _ = settle(recogniser, hand(thumb_index_ratio=PINCHED), 1.4)
        self.assertEqual(clicks(third.events), ["left_click"])

    def test_armed_and_engaged_are_queryable(self):
        recogniser = GestureRecognizer()
        self.assertTrue(recogniser.armed("thumb_index"))
        recogniser.update(Observation(hand(), 0.0))
        settle(recogniser, hand(thumb_index_ratio=PINCHED), 0.1)
        self.assertTrue(recogniser.engaged("thumb_index"))
        self.assertFalse(recogniser.armed("thumb_index"))


class FingerDistinctionTests(unittest.TestCase):
    def test_thumb_index_is_a_left_click(self):
        recogniser = GestureRecognizer()
        recogniser.update(Observation(hand(), 0.0))
        update, _ = settle(recogniser, hand(thumb_index_ratio=PINCHED, thumb_middle_ratio=OPEN), 0.2)
        self.assertEqual(clicks(update.events), ["left_click"])
        self.assertEqual(update.state, HandState.LEFT_PINCH)

    def test_thumb_middle_is_a_right_click(self):
        recogniser = GestureRecognizer()
        recogniser.update(Observation(hand(), 0.0))
        update, _ = settle(recogniser, hand(thumb_index_ratio=OPEN, thumb_middle_ratio=PINCHED), 0.2)
        self.assertEqual(clicks(update.events), ["right_click"])
        self.assertEqual(update.state, HandState.RIGHT_PINCH)

    def test_the_two_pinches_are_independent(self):
        recogniser = GestureRecognizer()
        recogniser.update(Observation(hand(), 0.0))
        first, now = settle(recogniser, hand(thumb_index_ratio=PINCHED), 0.2)
        recogniser.update(Observation(hand(), now + 0.05))  # release
        second, _ = settle(recogniser, hand(thumb_middle_ratio=PINCHED), now + 0.25)
        self.assertEqual(clicks(first.events), ["left_click"])
        self.assertEqual(clicks(second.events), ["right_click"])


class PriorityTests(unittest.TestCase):
    def test_priority_order_is_declared_and_scroll_is_first(self):
        self.assertEqual(
            GESTURE_PRIORITY,
            (HandState.TWO_FINGER_SCROLL, HandState.LEFT_PINCH, HandState.RIGHT_PINCH),
        )

    def test_two_finger_pinch_scrolls_and_never_clicks(self):
        recogniser = GestureRecognizer()
        recogniser.update(Observation(hand(), 0.0))
        update, _ = settle(recogniser, hand(**TWO_FINGER), 0.2)
        self.assertEqual(update.state, HandState.TWO_FINGER_SCROLL)
        self.assertEqual(kinds(update.events), ["scroll_begin"], "the scroll pose must not click")
        self.assertEqual(clicks(update.events), [])

    def test_scroll_wins_over_a_left_pinch_that_is_already_engaged(self):
        # Pinch the index first (left click), then bring the middle in: the change of gesture needs
        # a neutral frame, so no scroll begins and no second click fires.
        recogniser = GestureRecognizer()
        recogniser.update(Observation(hand(), 0.0))
        settle(recogniser, hand(thumb_index_ratio=PINCHED), 0.1)
        update, _ = settle(recogniser, hand(**TWO_FINGER), 0.5)
        self.assertEqual(update.state, HandState.LEFT_PINCH)
        self.assertEqual(update.events, ())

    def test_neutral_in_between_allows_the_switch(self):
        recogniser = GestureRecognizer()
        recogniser.update(Observation(hand(), 0.0))
        settle(recogniser, hand(thumb_index_ratio=PINCHED), 0.1)
        recogniser.update(Observation(hand(), 0.5))  # neutral frame
        update, _ = settle(recogniser, hand(**TWO_FINGER), 0.7)
        self.assertEqual(update.state, HandState.TWO_FINGER_SCROLL)
        self.assertEqual(kinds(update.events), ["scroll_begin"])

    def test_the_rule_can_be_relaxed(self):
        recogniser = GestureRecognizer(GestureThresholds(require_neutral_between=False))
        recogniser.update(Observation(hand(), 0.0))
        settle(recogniser, hand(thumb_index_ratio=PINCHED), 0.1)
        update, _ = settle(recogniser, hand(**TWO_FINGER), 0.5)
        self.assertEqual(update.state, HandState.TWO_FINGER_SCROLL)


class ScrollTests(unittest.TestCase):
    def setUp(self):
        self.recogniser = GestureRecognizer()
        self.recogniser.update(Observation(hand(), 0.0))
        entered, self.now = settle(self.recogniser, hand(**TWO_FINGER), 0.2)
        self.assertEqual(kinds(entered.events), ["scroll_begin"])

    def two_finger(self, timestamp: float, offset: Tuple[float, float] = (0.0, 0.0)) -> GestureUpdate:
        return self.recogniser.update(Observation(hand(offset=offset, **TWO_FINGER), timestamp))

    def test_scroll_begin_then_end(self):
        update = self.recogniser.update(Observation(hand(), self.now + 0.1))
        self.assertEqual(update.state, HandState.NEUTRAL)
        self.assertEqual(kinds(update.events), ["scroll_end"])

    def test_no_scroll_event_for_a_still_hand(self):
        update = self.two_finger(self.now + 0.03)
        self.assertEqual(update.events, (), "a stationary two-finger pinch must not scroll")

    def test_scroll_below_the_deadzone_is_silent_then_accumulates(self):
        # Dead zone is 0.05 * 0.15 = 0.0075 in frame units. Two 0.005 steps accumulate to 0.010,
        # which is over the dead zone, so the second frame reports the whole move.
        first = self.two_finger(self.now + 0.03, offset=(0.0, 0.005))
        self.assertEqual(first.events, (), "below the dead zone")
        second = self.two_finger(self.now + 0.06, offset=(0.0, 0.010))
        self.assertEqual(len(second.events), 1)
        self.assertAlmostEqual(second.events[0].value, -0.010 * 900, places=6)

    def test_scroll_sign_and_gain(self):
        update = self.two_finger(self.now + 0.03, offset=(0.0, 0.05))
        self.assertEqual(len(update.events), 1)
        event = update.events[0]
        self.assertEqual(event.kind, EventKind.SCROLL)
        self.assertAlmostEqual(event.value, -0.05 * 900, places=6, msg="hand down must scroll down")

    def test_moving_up_scrolls_the_other_way(self):
        update = self.two_finger(self.now + 0.03, offset=(0.0, -0.05))
        self.assertAlmostEqual(update.events[0].value, 45.0, places=6)

    def test_horizontal_component_is_reported_separately(self):
        update = self.two_finger(self.now + 0.03, offset=(0.05, 0.0))
        self.assertAlmostEqual(update.events[0].secondary, 0.05 * 900, places=6)

    def test_the_reference_resets_when_scroll_begins(self):
        # Move the hand during the hold, then hold still once scroll has begun: a fresh reference
        # means the pre-entry movement must not turn into a scroll jump.
        recogniser = GestureRecognizer()
        recogniser.update(Observation(hand(), 0.0))
        recogniser.update(Observation(hand(offset=(0.0, 0.10), **TWO_FINGER), 0.05))
        entered = recogniser.update(Observation(hand(offset=(0.0, 0.20), **TWO_FINGER), 0.15))
        self.assertEqual(kinds(entered.events), ["scroll_begin"])
        still = recogniser.update(Observation(hand(offset=(0.0, 0.20), **TWO_FINGER), 0.20))
        self.assertEqual(still.events, (), "the entry frame is the reference, not the frame before it")

    def test_scroll_hysteresis_keeps_the_gesture(self):
        update = self.two_finger(self.now + 0.03, offset=(0.0, 0.05))
        self.assertEqual(update.state, HandState.TWO_FINGER_SCROLL)
        in_band = self.recogniser.update(
            Observation(hand(offset=(0.0, 0.05), thumb_index_ratio=BAND, thumb_middle_ratio=BAND), self.now + 0.06)
        )
        self.assertEqual(in_band.state, HandState.TWO_FINGER_SCROLL)

    def test_losing_tracking_ends_the_scroll(self):
        update = self.recogniser.update(Observation(None, self.now + 0.1))
        self.assertEqual(kinds(update.events), ["scroll_end", "tracking_lost"])
        self.assertEqual(update.state, HandState.TRACKING_LOST)


class TrackingLossTests(unittest.TestCase):
    def test_loss_emits_once_and_holds_the_state(self):
        recogniser = GestureRecognizer()
        recogniser.update(Observation(hand(), 0.0))
        first = recogniser.update(Observation(None, 0.1))
        self.assertEqual(kinds(first.events), ["tracking_lost"])
        self.assertEqual(first.state, HandState.TRACKING_LOST)

        second = recogniser.update(Observation(None, 0.2))
        self.assertEqual(second.events, (), "the loss is reported once, not per frame")

    def test_loss_while_neutral_still_reports(self):
        update = GestureRecognizer().update(Observation(None, 0.0))
        self.assertEqual(kinds(update.events), ["tracking_lost"])

    def test_reacquire_reports_restored_and_requires_a_fresh_pinch(self):
        recogniser = GestureRecognizer()
        recogniser.update(Observation(hand(), 0.0))
        settle(recogniser, hand(thumb_index_ratio=PINCHED), 0.1)  # clicks
        recogniser.update(Observation(None, 0.4))

        restored = recogniser.update(Observation(hand(thumb_index_ratio=PINCHED), 0.6))
        self.assertEqual(kinds(restored.events), ["tracking_restored"])
        self.assertEqual(restored.state, HandState.NEUTRAL)

        # Still pinched: holding it must not fire, however long it is held.
        held = recogniser.update(Observation(hand(thumb_index_ratio=PINCHED), 1.6))
        self.assertEqual(held.events, (), "a hand that reappears pinched must be released first")
        self.assertEqual(held.state, HandState.LEFT_PINCH)

        recogniser.update(Observation(hand(), 1.8))  # release
        repinch, _ = settle(recogniser, hand(thumb_index_ratio=PINCHED), 2.0)
        self.assertEqual(clicks(repinch.events), ["left_click"])

    def test_reacquire_with_an_open_hand_clicks_normally_afterwards(self):
        recogniser = GestureRecognizer()
        recogniser.update(Observation(None, 0.0))
        recogniser.update(Observation(hand(), 0.2))  # restored, open hand
        update, _ = settle(recogniser, hand(thumb_index_ratio=PINCHED), 0.4)
        self.assertEqual(clicks(update.events), ["left_click"])


class PauseTests(unittest.TestCase):
    def test_pause_emits_once_and_ignores_the_hand(self):
        recogniser = GestureRecognizer()
        self.assertEqual(kinds(recogniser.set_paused(True)), ["paused"])
        self.assertEqual(recogniser.set_paused(True), (), "pausing twice is a no-op")

        for step in range(20):
            update = recogniser.update(Observation(hand(thumb_index_ratio=PINCHED), step * 0.1))
            self.assertEqual(update.events, (), "nothing may fire while paused")
            self.assertEqual(update.state, HandState.PAUSED)

    def test_resume_returns_to_neutral_and_reports(self):
        recogniser = GestureRecognizer()
        recogniser.set_paused(True)
        self.assertEqual(kinds(recogniser.set_paused(False)), ["resumed"])
        self.assertEqual(recogniser.state, HandState.NEUTRAL)

    def test_a_pinch_held_across_a_resume_does_not_click_even_after_the_hold(self):
        recogniser = GestureRecognizer()
        recogniser.set_paused(True)
        recogniser.update(Observation(hand(thumb_index_ratio=PINCHED), 0.1))
        recogniser.set_paused(False)

        for step in range(30):  # hold the pinch for a second past the resume
            update = recogniser.update(Observation(hand(thumb_index_ratio=PINCHED), 0.2 + step * FRAME))
            self.assertEqual(update.events, (), "the pinch belonged to the paused period")
        self.assertEqual(recogniser.state, HandState.LEFT_PINCH, "the pose is consumed, not queued")

    def test_a_pinch_released_and_pinched_again_after_a_resume_clicks(self):
        recogniser = GestureRecognizer()
        recogniser.set_paused(True)
        recogniser.update(Observation(hand(thumb_index_ratio=PINCHED), 0.1))
        recogniser.set_paused(False)
        recogniser.update(Observation(hand(thumb_index_ratio=PINCHED), 0.5))  # still held: consumed
        recogniser.update(Observation(hand(), 0.6))  # released

        repinch, _ = settle(recogniser, hand(thumb_index_ratio=PINCHED), 0.8)
        self.assertEqual(clicks(repinch.events), ["left_click"])

    def test_an_ordinary_pinch_after_a_resume_still_clicks(self):
        recogniser = GestureRecognizer()
        recogniser.set_paused(True)
        recogniser.set_paused(False)
        recogniser.update(Observation(hand(), 0.1))  # open hand: the pairs re-arm
        update, _ = settle(recogniser, hand(thumb_index_ratio=PINCHED), 0.3)
        self.assertEqual(clicks(update.events), ["left_click"])


class DeterminismTests(unittest.TestCase):
    def scripted_sequence(self) -> List[Tuple[Any, float]]:
        """A fixed scenario: pinch, scroll, loss, recovery — the four states the phase names."""
        frames: List[Tuple[Any, float]] = []
        now = 0.0
        for _ in range(20):
            frames.append((hand(), now))
            now += FRAME
        for _ in range(15):  # left pinch, held
            frames.append((hand(thumb_index_ratio=PINCHED), now))
            now += FRAME
        for _ in range(10):
            frames.append((hand(), now))
            now += FRAME
        for step in range(30):  # two-finger scroll, drifting down
            frames.append((hand(offset=(0.0, step * 0.01), **TWO_FINGER), now))
            now += FRAME
        frames.append((hand(), now))
        now += 0.5
        for _ in range(10):
            frames.append((None, now))  # tracking loss
            now += FRAME
        for _ in range(15):
            frames.append((hand(thumb_middle_ratio=PINCHED), now))
            now += FRAME
        for _ in range(10):
            frames.append((hand(), now))
            now += FRAME
        return frames

    def run_sequence(self) -> List[Tuple[str, Tuple[str, ...]]]:
        recogniser = GestureRecognizer()
        trace = []
        for landmarks, now in self.scripted_sequence():
            update = recogniser.update(Observation(landmarks, now))
            trace.append((update.state.value, tuple(kinds(update.events))))
        return trace

    def test_the_same_sequence_produces_the_same_trace(self):
        self.assertEqual(self.run_sequence(), self.run_sequence())

    def test_the_scripted_sequence_exercises_every_named_state(self):
        states = {state for state, _ in self.run_sequence()}
        for expected in (
            HandState.NEUTRAL.value,
            HandState.LEFT_PINCH.value,
            HandState.TWO_FINGER_SCROLL.value,
            HandState.TRACKING_LOST.value,
        ):
            self.assertIn(expected, states)

    def test_pausing_the_scripted_sequence_adds_the_paused_state(self):
        # Pause mid-scroll (frame 60): the phase's pause state has to be reachable, and pausing
        # must close the scroll that was in flight.
        sequence = self.scripted_sequence()
        recogniser = GestureRecognizer()
        states = set()
        paused = False
        for index, (landmarks, now) in enumerate(sequence):
            if index == 60 and not paused:
                self.assertEqual(kinds(recogniser.set_paused(True)), ["scroll_end", "paused"])
                paused = True
            states.add(recogniser.update(Observation(landmarks, now)).state.value)
        self.assertIn(HandState.PAUSED.value, states)

    def test_the_scripted_sequence_is_long_enough_for_its_sections(self):
        self.assertEqual(len(self.scripted_sequence()), 20 + 15 + 10 + 30 + 1 + 10 + 15 + 10)

    def test_events_only_come_from_the_declared_kinds(self):
        allowed = {kind.value for kind in EventKind}
        for _, events in self.run_sequence():
            for event in events:
                self.assertIn(event, allowed)

    def test_the_scripted_sequence_yields_exactly_the_expected_events(self):
        trace = self.run_sequence()
        fired = [event for _, events in trace for event in events]
        self.assertEqual(fired.count("left_click"), 1, "one deliberate pinch, one click")
        self.assertEqual(fired.count("right_click"), 0)
        self.assertEqual(fired.count("scroll_begin"), 1)
        self.assertEqual(fired.count("scroll_end"), 1, "the loss ends the scroll")
        self.assertEqual(fired.count("tracking_lost"), 1)
        self.assertEqual(fired.count("tracking_restored"), 1)
        self.assertGreater(fired.count("scroll"), 0, "the drift must produce scroll events")


class PressAndHoldTests(unittest.TestCase):
    """A pinch presses and holds the button; it releases only when the pinch itself releases.

    This is the bug fix: ``pyautogui.click()`` is an instant down+up, so a held pinch after the old
    one-shot ``LEFT_CLICK`` never kept the real button down (reported: "it activates left click, then
    immediately releases"). The recogniser now fires ``LEFT_CLICK``/``RIGHT_CLICK`` as a press and a
    matching ``LEFT_RELEASE``/``RIGHT_RELEASE`` when the pose actually opens — tying the release to the
    raw ``_engaged`` flag, not to the hold/cooldown/neutral-between machinery that gates the press, so
    a release can never be blocked by those debounce rules.
    """

    def test_holding_the_pinch_keeps_the_button_down_with_no_extra_events(self):
        recogniser = GestureRecognizer()
        update, now = settle(recogniser, hand(thumb_index_ratio=PINCHED), start=0.0)
        self.assertEqual(kinds(update.events), ["left_click"])
        self.assertTrue(recogniser.pressed("thumb_index"))

        # Hold for a full second (30+ frames): the button must stay down with nothing re-fired.
        for index in range(1, 31):
            held = recogniser.update(Observation(hand(thumb_index_ratio=PINCHED), now + index * FRAME))
            self.assertEqual(held.events, (), "a held pinch must not re-click or re-press")
        self.assertTrue(recogniser.pressed("thumb_index"), "the button must still be down")

    def test_releasing_the_pinch_fires_the_release(self):
        recogniser = GestureRecognizer()
        _update, now = settle(recogniser, hand(thumb_index_ratio=PINCHED), start=0.0)
        released = recogniser.update(Observation(hand(thumb_index_ratio=OPEN), now + FRAME))
        self.assertEqual(kinds(released.events), ["left_release"])
        self.assertFalse(recogniser.pressed("thumb_index"))

    def test_a_right_pinch_presses_and_releases_the_right_button(self):
        recogniser = GestureRecognizer()
        _update, now = settle(recogniser, hand(thumb_middle_ratio=PINCHED), start=0.0)
        self.assertTrue(recogniser.pressed("thumb_middle"))
        released = recogniser.update(Observation(hand(thumb_middle_ratio=OPEN), now + FRAME))
        self.assertEqual(kinds(released.events), ["right_release"])
        self.assertFalse(recogniser.pressed("thumb_middle"))

    def test_a_quick_tap_still_presses_and_releases_right_away(self):
        """A fast pinch-and-release looks exactly like the old one-shot click: down then up."""
        recogniser = GestureRecognizer()
        update, now = settle(recogniser, hand(thumb_index_ratio=PINCHED), start=0.0)
        self.assertEqual(kinds(update.events), ["left_click"])
        released = recogniser.update(Observation(hand(thumb_index_ratio=OPEN), now + FRAME))
        self.assertEqual(kinds(released.events), ["left_release"])

    def test_release_is_not_blocked_by_cooldown(self):
        """The 0.18 s cooldown gates the next *press*; it must never hold a release hostage."""
        recogniser = GestureRecognizer()
        update, now = settle(recogniser, hand(thumb_index_ratio=PINCHED), start=0.0)
        self.assertEqual(kinds(update.events), ["left_click"])
        # Release well inside the cooldown window (0.18 s).
        released = recogniser.update(Observation(hand(thumb_index_ratio=OPEN), now + 0.02))
        self.assertEqual(kinds(released.events), ["left_release"])

    def test_pausing_mid_hold_releases_the_button(self):
        """A pause must never leave the real mouse button stuck down."""
        recogniser = GestureRecognizer()
        settle(recogniser, hand(thumb_index_ratio=PINCHED), start=0.0)
        self.assertTrue(recogniser.pressed("thumb_index"))
        events = recogniser.set_paused(True)
        self.assertIn(EventKind.LEFT_RELEASE, [event.kind for event in events])
        self.assertFalse(recogniser.pressed("thumb_index"))

    def test_tracking_loss_mid_hold_releases_the_button(self):
        """Losing the hand mid-drag must also release, not strand the button down."""
        recogniser = GestureRecognizer()
        _update, now = settle(recogniser, hand(thumb_index_ratio=PINCHED), start=0.0)
        lost = recogniser.update(Observation(landmarks=None, timestamp=now + FRAME))
        self.assertIn("left_release", kinds(lost.events))
        self.assertFalse(recogniser.pressed("thumb_index"))

    def test_resuming_with_the_pinch_already_held_does_not_press_again(self):
        """A pinch held across a pause/resume is consumed, exactly like the old one-shot click."""
        recogniser = GestureRecognizer()
        settle(recogniser, hand(thumb_index_ratio=PINCHED), start=0.0)
        recogniser.set_paused(True)
        recogniser.set_paused(False)
        update = recogniser.update(Observation(hand(thumb_index_ratio=PINCHED), timestamp=1.0))
        self.assertEqual(update.events, (), "the rising edge belonged to the paused period")
        self.assertFalse(recogniser.pressed("thumb_index"), "nothing was actually pressed this time")

    def test_pending_releases_does_not_mutate_state(self):
        recogniser = GestureRecognizer()
        settle(recogniser, hand(thumb_index_ratio=PINCHED), start=0.0)
        first = recogniser.pending_releases()
        second = recogniser.pending_releases()
        self.assertEqual(first, second)
        self.assertTrue(recogniser.pressed("thumb_index"), "merely asking must not release anything")


if __name__ == "__main__":
    unittest.main()
