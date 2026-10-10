"""Cursor reference tests.

The reported bug: the cursor was driven by the index fingertip, so curling or spreading
the fingers dragged it across the screen even though the hand itself had not moved.
These tests pin the fix down — the reference is now the wrist/palm, and the cursor moves
only with the hand.
"""

import unittest
from types import SimpleNamespace
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from kinematic.config import HandMouseConfig
from kinematic.gesture import (
    INDEX_TIP,
    MIDDLE_MCP,
    THUMB_TIP,
    WRIST,
    hand_scale,
    landmark_point,
    palm_center,
    point_distance,
    pointer_delta,
    pointer_reference_point,
    wrist_point,
)
from kinematic.output import RecordingMouseOutput
from kinematic.runtime import HandMouseRuntime

# A synthetic hand: palm at (0.5, 0.6), knuckles above it, fingertip 0.13 further up when
# the fingers are extended. Curling interpolates the fingertip down to the knuckle, which
# is what MediaPipe shows for a fist — and it must not move the cursor.
PALM_BASE = (0.5, 0.6)
WRIST_OFFSET = (0.0, 0.02)
KNUCKLE_OFFSETS = {5: (-0.06, -0.04), 9: (0.0, -0.05), 13: (0.05, -0.04), 17: (0.09, -0.02)}
FINGER_TIPS = {8: (-0.05, -0.17), 12: (0.0, -0.18), 16: (0.05, -0.17), 20: (0.09, -0.15)}
THUMB_EXTENDED = (-0.13, -0.03)


def build_hand(
    offset: Tuple[float, float] = (0.0, 0.0),
    curl: float = 0.0,
    pinched: bool = False,
    index_drop: float = 0.0,
) -> List[SimpleNamespace]:
    """Landmarks for one hand.

    ``offset`` translates the whole hand (wrist, knuckles and fingers together),
    ``curl`` bends the fingers towards the knuckles while leaving the palm fixed, and
    ``index_drop`` moves the index fingertip alone (as a scroll or a wiggle would).
    """
    base = (PALM_BASE[0] + offset[0], PALM_BASE[1] + offset[1])
    points = [(base[0], base[1]) for _ in range(21)]

    points[WRIST] = (base[0] + WRIST_OFFSET[0], base[1] + WRIST_OFFSET[1])
    for index, (dx, dy) in KNUCKLE_OFFSETS.items():
        points[index] = (base[0] + dx, base[1] + dy)

    for index, (dx, dy) in FINGER_TIPS.items():
        knuckle_dx, knuckle_dy = KNUCKLE_OFFSETS[index - 4 if index - 4 in KNUCKLE_OFFSETS else 9]
        # Extend the finger when curl is 0, pull it back to the knuckle when curl is 1.
        points[index] = (
            base[0] + dx + (knuckle_dx - dx) * curl,
            base[1] + dy + (knuckle_dy - dy) * curl,
        )

    thumb = (base[0] + THUMB_EXTENDED[0], base[1] + THUMB_EXTENDED[1])
    if pinched:
        tip = points[INDEX_TIP]
        thumb = (tip[0] - 0.005, tip[1] + 0.005)
    points[THUMB_TIP] = thumb

    points[INDEX_TIP] = (points[INDEX_TIP][0], points[INDEX_TIP][1] + index_drop)
    return [SimpleNamespace(x=x, y=y, z=0.0) for x, y in points]


def make_runtime(output: Optional[RecordingMouseOutput] = None, **overrides: Any) -> HandMouseRuntime:
    config_overrides: dict = {
        "show_preview": False,
        "start_paused": False,
        "smoothing": 1.0,  # deterministic: the cursor lands exactly on the target
    }
    config_overrides.update(overrides)
    return HandMouseRuntime(
        config=HandMouseConfig(**config_overrides),
        output=output if output is not None else RecordingMouseOutput(),
    )


class ReferenceGeometryTests(unittest.TestCase):
    def test_wrist_point_is_landmark_zero(self):
        hand = build_hand()
        self.assertEqual(wrist_point(hand), (hand[WRIST].x, hand[WRIST].y))

    def test_palm_center_is_the_mean_of_wrist_and_knuckles(self):
        hand = build_hand()
        indices = (WRIST, 5, 9, 13, 17)
        expected = (
            sum(hand[i].x for i in indices) / 5,
            sum(hand[i].y for i in indices) / 5,
        )
        self.assertAlmostEqual(palm_center(hand)[0], expected[0], places=9)
        self.assertAlmostEqual(palm_center(hand)[1], expected[1], places=9)

    def test_hand_scale_is_wrist_to_middle_knuckle(self):
        hand = build_hand()
        expected = point_distance(wrist_point(hand), landmark_point(hand, MIDDLE_MCP))
        self.assertAlmostEqual(hand_scale(hand), expected, places=9)

    def test_hand_scale_never_returns_zero(self):
        degenerate = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
        self.assertGreater(hand_scale(degenerate), 0.0)

    def test_reference_dispatch(self):
        hand = build_hand()
        self.assertEqual(pointer_reference_point(hand, "wrist"), wrist_point(hand))
        self.assertEqual(pointer_reference_point(hand, "palm"), palm_center(hand))
        self.assertEqual(pointer_reference_point(hand, "index"), landmark_point(hand, INDEX_TIP))
        self.assertEqual(pointer_reference_point(hand), palm_center(hand))  # palm is the default

    def test_pointer_delta_and_scaling(self):
        dx, dy = pointer_delta((0.1, 0.2), (0.3, 0.5))
        self.assertAlmostEqual(dx, 0.2, places=9)
        self.assertAlmostEqual(dy, 0.3, places=9)
        scaled = pointer_delta((0.0, 0.0), (0.1, 0.1), reference_scale=0.5)
        self.assertAlmostEqual(scaled[0], 0.2, places=9)
        self.assertAlmostEqual(scaled[1], 0.2, places=9)

    def test_finger_curl_moves_the_fingertip_but_not_the_palm(self):
        """The mechanism behind the fix, stated as a test."""
        open_hand = build_hand(curl=0.0)
        fist = build_hand(curl=1.0)

        fingertip_travel = point_distance(landmark_point(open_hand, INDEX_TIP), landmark_point(fist, INDEX_TIP))
        palm_travel = point_distance(palm_center(open_hand), palm_center(fist))
        wrist_travel = point_distance(wrist_point(open_hand), wrist_point(fist))

        self.assertGreater(fingertip_travel, 0.1, "a fist should move the fingertip a long way")
        self.assertEqual(palm_travel, 0.0, "the palm must not move when the fingers curl")
        self.assertEqual(wrist_travel, 0.0, "the wrist must not move when the fingers curl")


class CurlDoesNotMoveTheCursorTests(unittest.TestCase):
    """The reported bug, exactly: fingers move, hand does not, cursor must not move."""

    def test_curling_fingers_leaves_the_cursor_alone(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output)

        for step in range(40):
            runtime.process_landmarks(build_hand(curl=step / 39.0))  # open hand -> fist

        self.assertEqual(output.moves, [], "finger curl must not move the cursor")
        self.assertEqual(runtime.emitted, 0)

    def test_spreading_fingers_leaves_the_cursor_alone(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output)
        for step in range(20):
            runtime.process_landmarks(build_hand(curl=1.0 - step / 19.0))  # fist -> open

        self.assertEqual(output.moves, [])

    def test_pinching_while_the_hand_holds_still_only_clicks(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output, gesture_engine="legacy")  # the inline click path
        runtime.process_landmarks(build_hand(pinched=False), now=0.0)
        runtime.process_landmarks(build_hand(pinched=True), now=1.0)

        self.assertEqual(output.moves, [], "the pinch itself must not drag the cursor")
        self.assertEqual([click.value for click in output.clicks], ["left"])

    def test_the_old_fingertip_reference_reproduces_the_bug(self):
        # Same motion, legacy reference: this is what the user was seeing.
        output = RecordingMouseOutput()
        runtime = make_runtime(output, pointer_reference="index")
        for step in range(40):
            runtime.process_landmarks(build_hand(curl=step / 39.0))

        self.assertGreater(len(output.moves), 0, "the fingertip reference is what caused the drift")


class HandTranslationMovesTheCursorTests(unittest.TestCase):
    def test_translation_shift_is_applied_with_the_configured_gain(self):
        output = RecordingMouseOutput(screen=(1920, 1080))
        # EMA with factor 1.0 is the identity filter: this test is about the gain
        # arithmetic, so the smoothing stage is deliberately neutralised.
        runtime = make_runtime(output, smoothing_mode="ema", smoothing=1.0)
        start = output.position()

        runtime.process_landmarks(build_hand(offset=(0.0, 0.0)))  # anchor
        runtime.process_landmarks(build_hand(offset=(0.01, 0.005)))

        self.assertEqual(len(output.moves), 1)
        expected = (
            start[0] + 0.01 * runtime.config.pointer_sensitivity * 1920,
            start[1] + 0.005 * runtime.config.pointer_sensitivity * 1080,
        )
        self.assertAlmostEqual(output.moves[0].value[0], expected[0], places=6)
        self.assertAlmostEqual(output.moves[0].value[1], expected[1], places=6)

    def test_cursor_follows_the_hand_direction(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output)
        runtime.process_landmarks(build_hand(offset=(0.0, 0.0)))
        before_x = runtime.cursor_x
        runtime.process_landmarks(build_hand(offset=(0.03, 0.0)))
        self.assertGreater(runtime.cursor_x, before_x)

    def test_cursor_is_clamped_to_the_screen(self):
        output = RecordingMouseOutput(screen=(1920, 1080))
        runtime = make_runtime(output)
        runtime.process_landmarks(build_hand(offset=(0.0, 0.0)))
        for _ in range(10):
            runtime.process_landmarks(build_hand(offset=(0.5, 0.5)))
        self.assertLessEqual(runtime.cursor_x, 1920)
        self.assertLessEqual(runtime.cursor_y, 1080)
        self.assertGreaterEqual(runtime.cursor_x, 0)
        self.assertGreaterEqual(runtime.cursor_y, 0)

    def test_jitter_below_the_deadzone_is_held_back_not_lost(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output)
        runtime.process_landmarks(build_hand())  # anchor

        # Twelve frames of 0.001 drift: each is below the 0.003 deadzone, but they add up.
        drift = 0.001
        for step in range(1, 13):
            runtime.process_landmarks(build_hand(offset=(drift * step, 0.0)))

        self.assertGreater(len(output.moves), 0, "slow movement must accumulate rather than vanish")
        self.assertGreater(runtime.cursor_x, output.position()[0] - 1.0)

    def test_hand_reacquire_does_not_jump_the_cursor(self):
        output = RecordingMouseOutput()
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        runtime = make_runtime(output)
        runtime.process_landmarks(build_hand())

        # Hand leaves the frame, then comes back on the other side of it.
        runtime.process_frame(frames[0])  # no hand -> anchor is dropped
        self.assertIsNone(runtime.last_pointer_ref)
        runtime.process_landmarks(build_hand(offset=(0.3, 0.2)))

        self.assertEqual(output.moves, [], "reacquiring a hand must re-anchor, not teleport")

    def test_legacy_absolute_mode_still_maps_position(self):
        output = RecordingMouseOutput(screen=(1920, 1080))
        runtime = make_runtime(output, pointer_mode="absolute", deadzone=0.0)
        runtime.process_landmarks(build_hand(offset=(0.1, 0.0)))
        # palm sits at 0.6, 0.62-ish; absolute mapping multiplies by pointer_scale 1.8.
        self.assertGreater(len(output.moves), 0)
        self.assertGreater(output.moves[-1].value[0], 0)

    def test_hand_size_scaling_option_is_off_by_default(self):
        self.assertFalse(HandMouseConfig().relative_scale_by_hand)


class ScrollDoesNotFightThePointerTests(unittest.TestCase):
    """The legacy index-finger scroll path, pinned to ``gesture_engine="legacy"``.

    From Phase 4 on, scrolling is the two-finger gesture owned by the recogniser (see
    tests/test_phase4_recognizer.py and tests/test_phase4_runtime.py); this class keeps the old path
    — including its still-hand guard — under regression test, because it is still reachable.
    """

    def test_index_finger_still_scrolls_while_the_hand_holds_still(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output, gesture_engine="legacy")
        runtime.process_landmarks(build_hand(index_drop=0.0))
        runtime.process_landmarks(build_hand(index_drop=0.10))  # index finger sweeps down

        self.assertEqual(len(output.scrolls), 1, "the existing scroll gesture must keep working")
        self.assertEqual(output.moves, [], "and it must not drag the cursor")

    def test_scrolling_is_suppressed_while_the_hand_is_translating(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output, gesture_engine="legacy")
        runtime.process_landmarks(build_hand())
        runtime.process_landmarks(build_hand(offset=(0.04, 0.0), index_drop=0.10))

        self.assertEqual(output.scrolls, [], "pointing must not scroll the page")
        self.assertGreater(len(output.moves), 0, "the hand moved, so the cursor should follow")

    def test_guard_can_be_disabled(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output, scroll_requires_still_hand=False, gesture_engine="legacy")
        runtime.process_landmarks(build_hand())
        runtime.process_landmarks(build_hand(offset=(0.04, 0.0), index_drop=0.10))
        self.assertEqual(len(output.scrolls), 1)


if __name__ == "__main__":
    unittest.main()
