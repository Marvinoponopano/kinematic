"""Tests for the sensitivity dial: turning the pointer gain up and down while the app runs.

The dial is plain functionality — keys in the preview window (`]` faster, `[` slower, `0` reset) plus
a programmatic API (`set_sensitivity` / `adjust_sensitivity` / `reset_sensitivity`) — and it is drawn as
a value and a log-scale bar in the preview. Nothing here can touch the mouse: the recording adapter
counts what would have happened.
"""

import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from kinematic.config import HandMouseConfig
from kinematic.gesture import INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, THUMB_TIP, WRIST
from kinematic.main import build_config, parse_args
from kinematic.output import RecordingMouseOutput
from kinematic.runtime import HandMouseRuntime

SCALE = 0.15
OPEN = 0.70  # an open hand: no gesture can fire, so the dial tests move the cursor and nothing else


def _base_points() -> List[Tuple[float, float]]:
    points = [(0.50, 0.60) for _ in range(21)]
    points[WRIST] = (0.50, 0.60)
    points[MIDDLE_MCP] = (0.50, 0.45)  # exactly SCALE above the wrist
    points[THUMB_TIP] = (0.40, 0.52)
    return points


def hand(
    offset: Tuple[float, float] = (0.0, 0.0),
    scale: float = SCALE,
) -> List[SimpleNamespace]:
    """An open hand, translated by *offset* (the palm moves by exactly that much)."""
    ox, oy = offset
    points = [(x + ox, y + oy) for x, y in _base_points()]
    thumb = points[THUMB_TIP]
    points[INDEX_TIP] = (thumb[0] + OPEN * scale, thumb[1])
    points[MIDDLE_TIP] = (thumb[0], thumb[1] - OPEN * scale)
    return [SimpleNamespace(x=x, y=y, z=0.0) for x, y in points]


def make_runtime(output: RecordingMouseOutput, **overrides: Any) -> HandMouseRuntime:
    config_overrides: Dict[str, Any] = {
        "show_preview": False,
        "start_paused": False,
        "smoothing_mode": "ema",
        "smoothing": 1.0,  # identity filter: the cursor lands exactly where the gain says
        # The dial is mode-independent (same field, same floor, same keys), but the *travel* it
        # produces is measured against hand displacement, so the travel tests state relative mode
        # explicitly. Absolute mode's gain tests live in test_pointer_modes.py.
        "pointer_mode": "relative",
    }
    config_overrides.update(overrides)
    return HandMouseRuntime(config=HandMouseConfig(**config_overrides), output=output)


class DialDefaultsTests(unittest.TestCase):
    def test_the_dial_has_a_sensible_range_and_step(self):
        config = HandMouseConfig()
        self.assertEqual(config.pointer_sensitivity, 1.0, "the neutral gain: one screen per frame width")
        self.assertEqual(config.sensitivity_step, 1.15)
        self.assertEqual(config.sensitivity_min, 1.0, "the floor: below it the screen edges are unreachable")
        self.assertEqual(config.sensitivity_reach_floor, 1.0)
        self.assertGreater(config.sensitivity_max, config.pointer_sensitivity)
        self.assertTrue(config.show_sensitivity)

    def test_the_keys_are_documented_in_the_config(self):
        config = HandMouseConfig()
        self.assertIn("]", config.sensitivity_keys_up)
        self.assertIn("[", config.sensitivity_keys_down)
        self.assertIn("0", config.sensitivity_keys_reset)
        # The dial must not steal a key the app already uses.
        for key in config.sensitivity_keys_up + config.sensitivity_keys_down + config.sensitivity_keys_reset:
            self.assertNotIn(key, config.toggle_pause_keys)
            self.assertNotIn(key, config.quit_keys)

    def test_the_starting_value_is_remembered_for_the_reset(self):
        runtime = make_runtime(RecordingMouseOutput(), pointer_sensitivity=7.5)
        self.assertEqual(runtime.sensitivity_start, 7.5)
        runtime.adjust_sensitivity(1)
        self.assertAlmostEqual(runtime.reset_sensitivity(), 7.5, places=9)


class SetSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.output = RecordingMouseOutput()
        self.runtime = make_runtime(self.output)

    def test_setting_a_value_inside_the_range_uses_it_exactly(self):
        self.assertEqual(self.runtime.set_sensitivity(12.5), 12.5)
        self.assertEqual(self.runtime.sensitivity, 12.5)
        self.assertEqual(self.runtime.config.pointer_sensitivity, 12.5)

    def test_values_are_clamped_to_the_dial_range(self):
        self.assertEqual(self.runtime.set_sensitivity(0.0), self.runtime.config.sensitivity_min)
        self.assertEqual(self.runtime.set_sensitivity(-5.0), self.runtime.config.sensitivity_min)
        self.assertEqual(self.runtime.set_sensitivity(0.5), self.runtime.config.sensitivity_min)
        self.assertEqual(self.runtime.set_sensitivity(1e6), self.runtime.config.sensitivity_max)

    def test_turning_the_dial_emits_nothing(self):
        for value in (0.5, 5.0, 60.0):
            self.runtime.set_sensitivity(value)
        self.assertEqual(self.output.counts(), {"move": 0, "click": 0, "scroll": 0})
        self.assertEqual(self.runtime.emitted, 0)

    def test_every_change_is_counted(self):
        before = self.runtime.sensitivity_changes
        self.runtime.set_sensitivity(6.0)
        self.runtime.adjust_sensitivity(-1)
        self.assertEqual(self.runtime.sensitivity_changes, before + 2)


class AdjustSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.runtime = make_runtime(RecordingMouseOutput())

    def test_one_notch_up_multiplies_and_one_down_divides(self):
        start = self.runtime.sensitivity
        up = self.runtime.adjust_sensitivity(1)
        self.assertAlmostEqual(up, start * self.runtime.config.sensitivity_step, places=9)
        down = self.runtime.adjust_sensitivity(-1)
        self.assertAlmostEqual(down, start, places=9, msg="a notch up and a notch down cancel out")

    def test_notches_are_multiplicative_so_the_feel_is_even(self):
        # Two notches up is the same factor wherever you start.
        low = make_runtime(RecordingMouseOutput(), pointer_sensitivity=2.0)
        high = make_runtime(RecordingMouseOutput(), pointer_sensitivity=20.0)
        low.adjust_sensitivity(2)
        high.adjust_sensitivity(2)
        self.assertAlmostEqual(low.sensitivity / 2.0, high.sensitivity / 20.0, places=9)

    def test_turning_past_the_end_stops_at_the_end(self):
        for _ in range(200):
            self.runtime.adjust_sensitivity(1)
        self.assertEqual(self.runtime.sensitivity, self.runtime.config.sensitivity_max)
        for _ in range(400):
            self.runtime.adjust_sensitivity(-1)
        self.assertEqual(self.runtime.sensitivity, self.runtime.config.sensitivity_min)

    def test_turning_is_monotone(self):
        values = [self.runtime.sensitivity]
        for _ in range(10):
            values.append(self.runtime.adjust_sensitivity(1))
        self.assertEqual(values, sorted(values))
        self.assertEqual(len(set(values)), len(values), "each notch must actually change the value")

    def test_zero_notches_changes_nothing(self):
        before = self.runtime.sensitivity
        changes = self.runtime.sensitivity_changes
        self.assertEqual(self.runtime.adjust_sensitivity(0), before)
        self.assertEqual(self.runtime.sensitivity_changes, changes)

    def test_a_nonsense_step_is_ignored_rather_than_zeroing_the_pointer(self):
        runtime = make_runtime(RecordingMouseOutput(), sensitivity_step=0.0)
        before = runtime.sensitivity
        self.assertEqual(runtime.adjust_sensitivity(1), before)

    def test_a_custom_range_is_respected(self):
        runtime = make_runtime(RecordingMouseOutput(), sensitivity_min=2.0, sensitivity_max=3.0)
        runtime.adjust_sensitivity(20)
        self.assertEqual(runtime.sensitivity, 3.0)
        runtime.adjust_sensitivity(-20)
        self.assertEqual(runtime.sensitivity, 2.0)


class DialChangesCursorTravelTests(unittest.TestCase):
    """The point of a dial: the same hand movement moves the cursor further or less far."""

    def travel(self, sensitivity: float) -> float:
        output = RecordingMouseOutput()
        runtime = make_runtime(output, pointer_sensitivity=sensitivity)
        start = output.position()[0]  # read it *before* the move: position() is the live pointer
        runtime.process_landmarks(hand(), now=0.0)  # anchors the reference
        runtime.process_landmarks(hand(offset=(0.01, 0.0)), now=0.1)  # one frame of hand travel
        self.assertEqual(len(output.moves), 1)
        return output.moves[0].value[0] - start

    def test_double_the_gain_doubles_the_travel(self):
        slow = self.travel(2.0)
        fast = self.travel(4.0)
        self.assertGreater(slow, 0.0, "the slow case must actually move, or this proves nothing")
        self.assertAlmostEqual(fast, slow * 2.0, places=6)

    def test_travel_matches_the_documented_units(self):
        # pointer_sensitivity is "screen widths of cursor travel per frame width of hand travel".
        screen_width = RecordingMouseOutput().screen_size()[0]
        self.assertAlmostEqual(self.travel(2.5), 0.01 * 2.5 * screen_width, places=6)

    def test_turning_the_dial_mid_motion_does_not_rescale_what_is_already_on_screen(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output, pointer_sensitivity=1.0)
        screen_width = output.screen_size()[0]
        start = output.position()[0]

        runtime.process_landmarks(hand(), now=0.0)  # anchor
        runtime.process_landmarks(hand(offset=(0.01, 0.0)), now=0.1)  # 0.01 * 1.0 * width
        after_first = output.moves[-1].value[0]
        self.assertAlmostEqual(after_first - start, 0.01 * 1.0 * screen_width, places=6)

        runtime.set_sensitivity(4.0)
        runtime.process_landmarks(hand(offset=(0.02, 0.0)), now=0.2)  # 0.01 more, at the new gain
        after_second = output.moves[-1].value[0]
        self.assertAlmostEqual(
            after_second - after_first,
            0.01 * 4.0 * screen_width,
            places=6,
            msg="only the new movement is scaled; the earlier movement must not be redrawn",
        )

    def test_the_dial_works_while_paused_and_still_emits_nothing(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output, start_paused=True, pointer_sensitivity=2.0)
        runtime.process_landmarks(hand(offset=(0.0, 0.0)), now=0.0)
        runtime.adjust_sensitivity(3)
        runtime.process_landmarks(hand(offset=(0.05, 0.0)), now=0.1)
        self.assertEqual(output.counts(), {"move": 0, "click": 0, "scroll": 0})
        self.assertGreater(runtime.sensitivity, 2.0, "the dial is configuration, not output")


class DialKeyTests(unittest.TestCase):
    def setUp(self):
        self.output = RecordingMouseOutput()
        self.runtime = make_runtime(self.output)

    def test_the_up_keys_turn_it_up(self):
        for key in ("]", "=", "+"):
            before = self.runtime.sensitivity
            self.assertFalse(self.runtime.handle_key(ord(key)), "a dial key must not quit")
            self.assertGreater(self.runtime.sensitivity, before, f"{key!r} should be faster")

    def test_the_down_keys_turn_it_down(self):
        # Start above the floor: the default 1.0 *is* the floor, so there is nowhere below it
        # (test_pointer_modes.py covers the floor itself).
        for key in ("[", "-", "_"):
            runtime = make_runtime(RecordingMouseOutput(), pointer_sensitivity=3.0)
            before = runtime.sensitivity
            self.assertFalse(runtime.handle_key(ord(key)))
            self.assertLess(runtime.sensitivity, before, f"{key!r} should be slower")

    def test_the_reset_key_returns_to_the_starting_value(self):
        runtime = make_runtime(self.output, pointer_sensitivity=9.0)
        runtime.adjust_sensitivity(4)
        self.assertFalse(runtime.handle_key(ord("0")))
        self.assertAlmostEqual(runtime.sensitivity, 9.0, places=9)

    def test_the_dial_keys_never_stop_the_loop(self):
        for key in ("]", "[", "0", "=", "-", "+", "_"):
            self.assertFalse(self.runtime.handle_key(ord(key)))

    def test_the_existing_keys_still_work(self):
        # Regression guard: the dial must not have taken over pause or quit.
        self.assertFalse(self.runtime.handle_key(ord(" ")))
        self.assertTrue(self.runtime.paused)
        self.assertFalse(self.runtime.handle_key(ord(" ")))
        self.assertFalse(self.runtime.paused)
        self.assertTrue(self.runtime.handle_key(ord("q")))
        self.assertTrue(self.runtime.handle_key(27))

    def test_keys_the_dial_does_not_own_are_ignored(self):
        before = self.runtime.sensitivity
        for key in ("a", "z", "9", "\t"):
            self.assertFalse(self.runtime.handle_key(ord(key)))
        self.assertEqual(self.runtime.sensitivity, before)

    def test_no_key_can_push_the_value_outside_the_range(self):
        for _ in range(80):
            self.runtime.handle_key(ord("]"))
        self.assertEqual(self.runtime.sensitivity, self.runtime.config.sensitivity_max)
        for _ in range(160):
            self.runtime.handle_key(ord("["))
        self.assertEqual(self.runtime.sensitivity, self.runtime.config.sensitivity_min)

    def test_pressing_a_dial_key_emits_no_mouse_event(self):
        for key in ("]", "[", "0"):
            self.runtime.handle_key(ord(key))
        self.assertEqual(self.output.counts(), {"move": 0, "click": 0, "scroll": 0})


class DialReadoutTests(unittest.TestCase):
    def setUp(self):
        self.runtime = make_runtime(RecordingMouseOutput())

    def test_the_bar_spans_the_range(self):
        self.runtime.set_sensitivity(self.runtime.config.sensitivity_min)
        self.assertEqual(self.runtime.sensitivity_bar(), "#" + "-" * 9, "the floor is the first cell")
        self.runtime.set_sensitivity(self.runtime.config.sensitivity_max)
        self.assertEqual(self.runtime.sensitivity_bar(), "#" * 10)

    def test_the_bar_is_log_scaled_and_never_empty(self):
        # The range 1.0 -> 60 is a factor of 60. It never draws an empty bar: the floor is still a
        # usable gain and an empty bar would read as "off". Log scale makes turns even, so the
        # geometric midpoint (sqrt(60) = 7.75) lands on exactly half the bar.
        self.assertEqual(self.runtime.sensitivity_bar(), "#---------", "the default is the floor")
        self.runtime.set_sensitivity(60 ** 0.5)
        self.assertEqual(self.runtime.sensitivity_bar(), "#####-----", "the geometric midpoint")
        self.runtime.set_sensitivity(5.0)  # the old default, kept as a reference point
        self.assertEqual(self.runtime.sensitivity_bar(), "####------")

    def test_the_bar_is_monotone_and_respects_its_width(self):
        previous = -1
        for value in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 60.0):
            self.runtime.set_sensitivity(value)
            filled = self.runtime.sensitivity_bar().count("#")
            self.assertGreaterEqual(filled, previous)
            previous = filled
        self.assertEqual(len(self.runtime.sensitivity_bar(4)), 4)

    def test_the_readout_shows_the_value_and_the_keys(self):
        self.runtime.set_sensitivity(3.25)
        text = self.runtime.sensitivity_text()
        self.assertIn("3.25", text)
        self.assertIn("]", text)
        self.assertIn("[", text)
        self.assertIn("0 reset", text)


class DialIsDrawnOnThePreviewTests(unittest.TestCase):
    """The dial has to be visible, or you cannot tell where it is set."""

    def preview(self, **overrides: Any) -> np.ndarray:
        runtime = make_runtime(RecordingMouseOutput(), preview_scale=1.0, show_fps=False, **overrides)
        frame = np.zeros((240, 640, 3), dtype=np.uint8)
        return runtime.draw_preview(frame)

    def test_the_dial_line_is_drawn_when_enabled(self):
        frame = self.preview(show_sensitivity=True)
        self.assertTrue((frame != 0).any(), "the sensitivity line must be drawn")
        self.assertTrue((frame[14:24, :, :] != 0).any(), "it goes on the first text line")

    def test_nothing_is_drawn_when_both_overlays_are_off(self):
        frame = self.preview(show_sensitivity=False)
        self.assertFalse((frame != 0).any(), "no overlay, no pixels")

    def test_the_fps_line_keeps_the_first_row_and_the_dial_follows_it(self):
        runtime = make_runtime(RecordingMouseOutput(), preview_scale=1.0, show_fps=True, show_sensitivity=True)
        frame = np.zeros((240, 640, 3), dtype=np.uint8)
        drawn = runtime.draw_preview(frame)
        self.assertTrue((drawn[14:24, :, :] != 0).any(), "fps/gesture line")
        self.assertTrue((drawn[38:48, :, :] != 0).any(), "sensitivity line, one row below")
        self.assertFalse((drawn[96:110, :, :] != 0).any(), "and nothing else while active")


class DialSummaryTests(unittest.TestCase):
    def test_the_summary_reports_where_the_dial_ended_up(self):
        runtime = make_runtime(RecordingMouseOutput(), pointer_sensitivity=5.0)
        runtime.adjust_sensitivity(2)
        summary = runtime.summary()
        self.assertEqual(summary["sensitivity_start"], 5.0)
        self.assertAlmostEqual(summary["sensitivity"], 5.0 * 1.15 ** 2, places=4)
        self.assertEqual(summary["sensitivity_changes"], 1)


class DialCommandLineTests(unittest.TestCase):
    def test_the_starting_gain_can_be_set_from_the_command_line(self):
        config = build_config(parse_args(["--sensitivity", "9.5"]))
        self.assertEqual(config.pointer_sensitivity, 9.5)

    def test_the_step_can_be_set_from_the_command_line(self):
        config = build_config(parse_args(["--sensitivity-step", "1.5"]))
        self.assertEqual(config.sensitivity_step, 1.5)

    def test_defaults_are_untouched_without_the_flags(self):
        config = build_config(parse_args([]))
        self.assertEqual(config.pointer_sensitivity, 1.0)
        self.assertEqual(config.sensitivity_step, 1.15)

    def test_the_app_reports_the_dial_at_startup(self):
        runtime = make_runtime(RecordingMouseOutput(), pointer_sensitivity=12.0)
        self.assertEqual(runtime.sensitivity, 12.0)
        self.assertEqual(runtime.sensitivity_start, 12.0)


if __name__ == "__main__":
    unittest.main()
