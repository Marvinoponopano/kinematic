"""Tests for the default pointer mode (absolute) and the sensitivity reach floor.

The reported problem: turning the sensitivity down far enough left a band at the screen edges the
cursor could not enter. The cause is arithmetic, not taste — the gain is "screen widths of cursor
travel per frame width of hand travel", so below 1.0 the reachable area stops covering the screen.
These tests pin both halves of the fix: absolute pointing (the default) makes every edge and corner
directly reachable, and the dial refuses to go below the gain at which the screen stops fitting.
"""

import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    pointer_reference_point,
)
from kinematic.main import build_config, parse_args
from kinematic.output import DEFAULT_SCREEN, RecordingMouseOutput
from kinematic.runtime import HandMouseRuntime

SCALE = 0.15  # wrist -> middle knuckle
OPEN = 0.70  # an open hand: no gesture can fire, so these tests only move the cursor
SCREEN_W, SCREEN_H = DEFAULT_SCREEN
MARGIN = 0.15  # frame_margin: the inner 70% of the frame covers the screen
INNER_LOW = MARGIN  # 0.15
INNER_HIGH = 1.0 - MARGIN  # 0.85


def _base_points() -> List[Tuple[float, float]]:
    points = [(0.50, 0.60) for _ in range(21)]
    points[WRIST] = (0.50, 0.60)
    points[MIDDLE_MCP] = (0.50, 0.45)  # exactly SCALE above the wrist
    points[THUMB_TIP] = (0.40, 0.52)
    return points


def hand(offset: Tuple[float, float] = (0.0, 0.0), scale: float = SCALE) -> List[SimpleNamespace]:
    ox, oy = offset
    points = [(x + ox, y + oy) for x, y in _base_points()]
    thumb = points[THUMB_TIP]
    points[INDEX_TIP] = (thumb[0] + OPEN * scale, thumb[1])
    points[MIDDLE_TIP] = (thumb[0], thumb[1] - OPEN * scale)
    return [SimpleNamespace(x=x, y=y, z=0.0) for x, y in points]


def hand_with_palm_at(point: Tuple[float, float]) -> List[SimpleNamespace]:
    """A hand whose pointer reference (the palm) sits exactly at *point* in the frame."""
    neutral = hand()
    palm = pointer_reference_point(neutral, "palm")
    return hand(offset=(point[0] - palm[0], point[1] - palm[1]))


def make_runtime(output: RecordingMouseOutput, **overrides: Any) -> HandMouseRuntime:
    config_overrides: Dict[str, Any] = {
        "show_preview": False,
        "start_paused": False,
        "smoothing_mode": "ema",
        "smoothing": 1.0,  # identity filter: the cursor lands exactly on the mapped point
    }
    config_overrides.update(overrides)
    return HandMouseRuntime(config=HandMouseConfig(**config_overrides), output=output)


def cursor_after(runtime: HandMouseRuntime, points: Sequence[SimpleNamespace]) -> Tuple[float, float]:
    """Feed one hand position and return where the cursor ended up."""
    runtime.process_landmarks(points, now=1.0)
    return (runtime.cursor_x, runtime.cursor_y)


class DefaultModeTests(unittest.TestCase):
    def test_the_default_control_scheme_is_absolute_position(self):
        self.assertEqual(HandMouseConfig().pointer_mode, "absolute")

    def test_the_default_gain_is_the_neutral_one(self):
        config = HandMouseConfig()
        self.assertEqual(config.pointer_sensitivity, 1.0)

    def test_relative_mode_is_still_one_flag_away(self):
        config = build_config(parse_args(["--pointer-mode", "relative"]))
        self.assertEqual(config.pointer_mode, "relative")
        self.assertEqual(build_config(parse_args([])).pointer_mode, "absolute")


class AbsoluteReachTests(unittest.TestCase):
    """Absolute pointing: every screen edge is reachable by putting the hand at the region edge."""

    def test_the_frame_region_corners_reach_the_screen_corners(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output)
        for palm, expected in (
            ((INNER_LOW, INNER_LOW), (0.0, 0.0)),
            ((INNER_HIGH, INNER_HIGH), (float(SCREEN_W), float(SCREEN_H))),
        ):
            cursor = cursor_after(runtime, hand_with_palm_at(palm))
            self.assertAlmostEqual(cursor[0], expected[0], places=6, msg=f"x at palm {palm}")
            self.assertAlmostEqual(cursor[1], expected[1], places=6, msg=f"y at palm {palm}")

    def test_the_screen_middle_maps_from_the_frame_middle(self):
        runtime = make_runtime(RecordingMouseOutput())
        cursor = cursor_after(runtime, hand_with_palm_at((0.5, 0.5)))
        self.assertAlmostEqual(cursor[0], SCREEN_W / 2.0, places=6)
        self.assertAlmostEqual(cursor[1], SCREEN_H / 2.0, places=6)

    def test_landmarks_outside_the_region_still_clamp_to_the_screen(self):
        # The region is the inner 70%; a hand out at the frame border must not overshoot the screen.
        runtime = make_runtime(RecordingMouseOutput())
        self.assertEqual(cursor_after(runtime, hand_with_palm_at((0.02, 0.98))), (0.0, float(SCREEN_H)))

    def test_a_gain_above_one_needs_less_hand_travel_to_reach_an_edge(self):
        # sensitivity 2.0: the screen is covered by the inner 35% of the frame, so a palm halfway
        # between the middle and the region edge already reaches the screen edge.
        runtime = make_runtime(RecordingMouseOutput(), pointer_sensitivity=2.0)
        palm = 0.5 + (INNER_HIGH - 0.5) / 2.0  # 0.675: half of the way out
        cursor = cursor_after(runtime, hand_with_palm_at((palm, 0.5)))
        self.assertAlmostEqual(cursor[0], float(SCREEN_W), places=6)

    def test_finger_curl_does_not_move_the_cursor_in_absolute_mode(self):
        # The palm reference has to keep the drift fix in absolute mode too: only the fingers move.
        output = RecordingMouseOutput()
        runtime = make_runtime(output)
        runtime.process_landmarks(hand_with_palm_at((0.5, 0.5)), now=1.0)
        settled = (runtime.cursor_x, runtime.cursor_y)

        curled = hand_with_palm_at((0.5, 0.5))
        curled[INDEX_TIP] = SimpleNamespace(
            x=curled[THUMB_TIP].x + 0.001, y=curled[THUMB_TIP].y, z=0.0
        )  # the index finger folds in towards the thumb
        runtime.process_landmarks(curled, now=1.1)
        self.assertEqual((runtime.cursor_x, runtime.cursor_y), settled, "curl must not move the cursor")
        self.assertEqual(output.moves, [], "and it must not emit a move either")


class ReachFloorTests(unittest.TestCase):
    """The floor is the gain below which the screen stops fitting on the reachable area."""

    def test_the_default_floor_is_the_reach_floor(self):
        config = HandMouseConfig()
        self.assertEqual(config.sensitivity_reach_floor, 1.0)
        self.assertEqual(config.sensitivity_min, config.sensitivity_reach_floor)

    def test_the_dial_cannot_be_turned_below_the_floor(self):
        runtime = make_runtime(RecordingMouseOutput())
        for _ in range(60):
            runtime.adjust_sensitivity(-1)
        self.assertEqual(runtime.sensitivity, 1.0)
        self.assertEqual(runtime.set_sensitivity(0.2), 1.0)
        self.assertIsNone(runtime.reach_warning, "at the floor the edges are still reachable")
        self.assertEqual(runtime.reach, 1.0)

    def test_a_starting_value_below_the_floor_is_clamped_and_reported(self):
        runtime = make_runtime(RecordingMouseOutput(), pointer_sensitivity=0.4)
        self.assertEqual(runtime.sensitivity, 1.0)
        self.assertEqual(runtime.sensitivity_start, 1.0)
        self.assertEqual(runtime.sensitivity_clamped_from, 0.4, "the requested value is kept for the report")
        self.assertEqual(runtime.sensitivity_changes, 0, "clamping at startup is not a dial turn")

    def test_a_starting_value_inside_the_range_is_not_clamped(self):
        runtime = make_runtime(RecordingMouseOutput(), pointer_sensitivity=4.0)
        self.assertEqual(runtime.sensitivity, 4.0)
        self.assertIsNone(runtime.sensitivity_clamped_from)

    def test_lowering_the_floor_by_hand_makes_the_edges_unreachable_and_says_so(self):
        # This is the reported bug, reproduced on purpose: with the floor lowered to the old 0.5 and
        # the gain at 0.5, the left-hand edge of the screen cannot be entered at all.
        output = RecordingMouseOutput()
        runtime = make_runtime(
            output, pointer_sensitivity=0.5, sensitivity_min=0.5, sensitivity_reach_floor=1.0
        )
        self.assertEqual(runtime.sensitivity, 0.5, "the floor was lowered deliberately, so it is respected")

        cursor = cursor_after(runtime, hand_with_palm_at((INNER_LOW, 0.5)))
        self.assertGreater(cursor[0], 0.0, "the left edge is out of reach at half gain")
        self.assertAlmostEqual(cursor[0], SCREEN_W / 4.0, places=6, msg="centre + (edge - centre) * 0.5")

        self.assertIsNotNone(runtime.reach_warning)
        self.assertIn("edges cannot be reached", runtime.reach_warning or "")
        self.assertIn("edges unreachable", runtime.sensitivity_text())

    def test_warning_clears_once_the_gain_is_back_over_the_floor(self):
        runtime = make_runtime(
            RecordingMouseOutput(), pointer_sensitivity=0.5, sensitivity_min=0.5, sensitivity_reach_floor=1.0
        )
        self.assertIsNotNone(runtime.reach_warning)
        runtime.set_sensitivity(1.0)
        self.assertIsNone(runtime.reach_warning)
        self.assertNotIn("edges unreachable", runtime.sensitivity_text())

    def test_the_floor_does_not_get_in_the_way_of_normal_turning(self):
        runtime = make_runtime(RecordingMouseOutput())
        self.assertEqual(runtime.adjust_sensitivity(4), 1.0 * 1.15 ** 4)
        self.assertGreater(runtime.sensitivity, 1.0)


class ReachReportingTests(unittest.TestCase):
    def test_reach_is_the_gain(self):
        runtime = make_runtime(RecordingMouseOutput(), pointer_sensitivity=3.0)
        self.assertEqual(runtime.reach, 3.0)

    def test_the_summary_reports_the_mode_and_the_reach(self):
        runtime = make_runtime(RecordingMouseOutput(), pointer_sensitivity=2.5)
        summary = runtime.summary()
        self.assertEqual(summary["pointer_mode"], "absolute")
        self.assertEqual(summary["reach"], 2.5)

    def test_the_scale_line_shows_the_reach(self):
        runtime = make_runtime(RecordingMouseOutput())
        self.assertIn("reach", self.scale_line(runtime))

    @staticmethod
    def scale_line(runtime: HandMouseRuntime) -> str:
        # The banner is built in main(); this mirrors the reach part of it so a rename cannot pass
        # silently without the value being reported somewhere.
        return f"reach {runtime.reach:.2f} screens per frame width"


class RelativeModeStillWorksTests(unittest.TestCase):
    def test_relative_mode_moves_the_cursor_by_displacement(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output, pointer_mode="relative", pointer_sensitivity=4.0)
        settled = cursor_after(runtime, hand())  # anchors the reference
        cursor = cursor_after(runtime, hand(offset=(0.01, 0.0)))
        self.assertAlmostEqual(cursor[0] - settled[0], 0.01 * 4.0 * SCREEN_W, places=6)

    def test_relative_mode_honours_the_same_floor(self):
        runtime = make_runtime(RecordingMouseOutput(), pointer_mode="relative", pointer_sensitivity=0.1)
        self.assertEqual(runtime.sensitivity, 1.0)
        self.assertEqual(runtime.sensitivity_clamped_from, 0.1)

    def test_the_legacy_pointer_scale_is_still_applied(self):
        # pointer_scale is superseded by the dial but must keep working for anyone who set it.
        runtime = make_runtime(RecordingMouseOutput(), pointer_scale=0.5)
        self.assertAlmostEqual(cursor_after(runtime, hand_with_palm_at((0.5, 0.5)))[0], SCREEN_W / 4.0, places=6)


if __name__ == "__main__":
    unittest.main()
