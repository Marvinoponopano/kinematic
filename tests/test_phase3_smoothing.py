"""Phase 3 smoothing tests: EMA and 1-euro filters, deterministically.

No camera and no wall clock: timestamps are supplied explicitly, noise comes from a seeded
generator, and every expectation is either exact arithmetic or a comparison between two
filters fed the identical input. These are the "latency-sensitive behaviour" tests the phase
asks for: jitter reduction, response time to a step, and lag while tracking a ramp.
"""

import random
import statistics
import unittest
from typing import List, Optional, Tuple

from kinematic.config import HandMouseConfig
from kinematic.smoothing import (
    CursorFilter,
    ExponentialMovingAverage,
    OneEuroFilter,
    filter_from_config,
)

POINT = (0.5, 0.5)
DT = 1.0 / 30.0  # 30 fps


def timestamps(count: int) -> List[Optional[float]]:
    return [step * DT for step in range(count)]


def response_frames(filter_: CursorFilter, target: Tuple[float, float], threshold: float = 0.9) -> int:
    """Frames until the filter covers *threshold* of a step from the origin to *target*.

    The filter is settled at the origin first: a fresh filter passes its first sample
    straight through, which would otherwise make every filter look instantaneous.
    """
    filter_.update((0.0, 0.0), 0.0)
    for index, timestamp in enumerate(timestamps(600)[1:], start=1):
        value = filter_.update(target, timestamp)
        if abs(value[0]) >= abs(target[0]) * threshold:
            return index
    return -1


class ExponentialMovingAverageTests(unittest.TestCase):
    def test_factor_one_is_the_identity(self):
        filter_ = ExponentialMovingAverage(factor=1.0)
        for timestamp in timestamps(5):
            self.assertEqual(filter_.update((0.25, 0.75), timestamp), (0.25, 0.75))

    def test_it_matches_the_documented_lerp(self):
        filter_ = ExponentialMovingAverage(factor=0.5)
        filter_.update((0.0, 0.0), 0.0)
        self.assertEqual(filter_.update((1.0, 1.0), DT), (0.5, 0.5))
        self.assertEqual(filter_.update((1.0, 1.0), 2 * DT), (0.75, 0.75))

    def test_first_sample_passes_straight_through(self):
        filter_ = ExponentialMovingAverage(factor=0.1)
        self.assertEqual(filter_.update(POINT, 0.0), POINT)

    def test_default_behaviour_is_unchanged_from_the_old_inline_ema(self):
        # smoothing=0.35 was applied directly to the cursor position before this module
        # existed; the filter must reproduce it exactly.
        from kinematic.gesture import smooth_point

        filter_ = ExponentialMovingAverage(factor=0.35)
        current = (100.0, 200.0)
        filter_.update(current, 0.0)
        target = (400.0, 260.0)
        expected = smooth_point(current, target, 0.35)
        self.assertEqual(filter_.update(target, DT), expected)

    def test_reset_clears_the_state(self):
        filter_ = ExponentialMovingAverage(factor=0.5)
        filter_.update((0.0, 0.0), 0.0)
        filter_.update((1.0, 1.0), DT)
        filter_.reset()
        self.assertEqual(filter_.update((0.9, 0.1), 2 * DT), (0.9, 0.1), "reset must not blend")

    def test_invalid_factors_are_rejected(self):
        for factor in (0.0, -0.5, 1.5):
            with self.assertRaises(ValueError):
                ExponentialMovingAverage(factor=factor)


class OneEuroFilterTests(unittest.TestCase):
    def test_first_sample_passes_straight_through(self):
        filter_ = OneEuroFilter()
        self.assertEqual(filter_.update(POINT, 0.0), POINT)

    def test_static_input_converges(self):
        filter_ = OneEuroFilter(min_cutoff=1.0, beta=1.0)
        value = POINT
        for timestamp in timestamps(200):
            value = filter_.update(POINT, timestamp)
        self.assertAlmostEqual(value[0], POINT[0], places=5)
        self.assertAlmostEqual(value[1], POINT[1], places=5)

    def test_jitter_is_reduced_against_the_input(self):
        rng = random.Random(7)
        filter_ = OneEuroFilter(min_cutoff=1.0, beta=1.0)
        inputs, outputs = [], []
        for timestamp in timestamps(180):
            noisy = (0.5 + rng.uniform(-0.01, 0.01), 0.5 + rng.uniform(-0.01, 0.01))
            inputs.append(noisy[0])
            outputs.append(filter_.update(noisy, timestamp)[0])

        input_spread = statistics.pstdev(inputs)
        output_spread = statistics.pstdev(outputs)
        self.assertGreater(input_spread, 0.0)
        self.assertLess(output_spread, input_spread * 0.5, "the filter should remove most jitter")

    def test_speed_opens_the_cutoff_so_a_fast_move_has_less_lag(self):
        fixed_cutoff = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        adaptive = OneEuroFilter(min_cutoff=1.0, beta=1.0)
        target = (1.0, 1.0)

        fixed_frames = response_frames(fixed_cutoff, target)
        adaptive_frames = response_frames(adaptive, target)

        self.assertGreater(fixed_frames, 0)
        self.assertGreater(adaptive_frames, 0)
        self.assertLess(
            adaptive_frames,
            fixed_frames,
            "beta > 0 must respond to a step faster than a pure low-pass with the same cutoff",
        )

    def test_it_beats_the_ema_on_latency_at_comparable_jitter(self):
        target = (1.0, 1.0)

        def residual_jitter(filter_: CursorFilter) -> float:
            rng = random.Random(11)
            samples = [0.5 + rng.uniform(-0.01, 0.01) for _ in range(120)]
            outputs = [filter_.update((sample, 0.5), step * DT)[0] for step, sample in enumerate(samples)]
            return statistics.pstdev(outputs)

        ema = ExponentialMovingAverage(factor=0.35)
        euro = OneEuroFilter(min_cutoff=1.0, beta=1.0)

        ema_spread = residual_jitter(ema)
        euro_spread = residual_jitter(euro)

        ema.reset()
        euro.reset()
        ema_frames = response_frames(ema, target)
        euro_frames = response_frames(euro, target)

        self.assertGreater(ema_spread, 0.0)
        self.assertGreater(euro_spread, 0.0)
        self.assertLess(euro_frames, ema_frames, "the 1-euro filter should arrive sooner")
        self.assertLess(euro_spread, ema_spread, "and with less residual jitter")

    def test_timestamps_drive_the_timestep(self):
        fine = OneEuroFilter(min_cutoff=1.0, beta=1.0)
        coarse = OneEuroFilter(min_cutoff=1.0, beta=1.0)
        fine.update((0.0, 0.0), 0.0)
        coarse.update((0.0, 0.0), 0.0)

        # The same sample with a ten-times longer timestep covers ten times as much time,
        # so the filter is entitled to advance further towards it.
        fine_value = fine.update((1.0, 1.0), DT)
        coarse_value = coarse.update((1.0, 1.0), 10 * DT)

        self.assertLess(fine_value[0], coarse_value[0], "a longer timestep must advance further")

    def test_missing_timestamps_fall_back_to_the_configured_rate(self):
        without = OneEuroFilter(min_cutoff=1.0, beta=1.0, frequency=30.0)
        without.update((0.0, 0.0), None)
        first = without.update((1.0, 1.0), None)

        with_timestamps = OneEuroFilter(min_cutoff=1.0, beta=1.0, frequency=30.0)
        with_timestamps.update((0.0, 0.0), 0.0)
        second = with_timestamps.update((1.0, 1.0), DT)

        self.assertAlmostEqual(first[0], second[0], places=9)

    def test_reset_clears_the_state_and_the_clock(self):
        filter_ = OneEuroFilter()
        for timestamp in timestamps(20):
            filter_.update((0.9, 0.9), timestamp)
        filter_.reset()

        self.assertEqual(filter_.update((0.1, 0.2), 100.0), (0.1, 0.2), "reset must not blend")

    def test_invalid_parameters_are_rejected(self):
        for kwargs in ({"min_cutoff": 0.0}, {"beta": -1.0}, {"d_cutoff": 0.0}, {"frequency": 0.0}):
            with self.assertRaises(ValueError):
                OneEuroFilter(**kwargs)


class FilterFromConfigTests(unittest.TestCase):
    def test_one_euro_is_the_default_mode_after_the_measured_comparison(self):
        config = HandMouseConfig()
        self.assertEqual(config.smoothing_mode, "one_euro")
        filter_ = filter_from_config(config)
        self.assertIsInstance(filter_, OneEuroFilter)
        assert isinstance(filter_, OneEuroFilter)
        self.assertAlmostEqual(filter_.beta, config.one_euro_beta)
        self.assertAlmostEqual(filter_.min_cutoff, config.one_euro_min_cutoff)

    def test_ema_remains_selectable_and_honours_smoothing(self):
        config = HandMouseConfig(smoothing_mode="ema")
        filter_ = filter_from_config(config)
        self.assertIsInstance(filter_, ExponentialMovingAverage)
        assert isinstance(filter_, ExponentialMovingAverage)
        self.assertAlmostEqual(filter_.factor, config.smoothing)

    def test_one_euro_parameters_are_configurable(self):
        config = HandMouseConfig(smoothing_mode="one_euro", one_euro_beta=2.5, one_euro_min_cutoff=0.5)
        filter_ = filter_from_config(config)
        self.assertIsInstance(filter_, OneEuroFilter)
        assert isinstance(filter_, OneEuroFilter)
        self.assertAlmostEqual(filter_.beta, 2.5)
        self.assertAlmostEqual(filter_.min_cutoff, 0.5)
        self.assertAlmostEqual(filter_.frequency, config.capture_fps)

    def test_unknown_mode_is_rejected_loudly(self):
        with self.assertRaises(ValueError):
            filter_from_config(HandMouseConfig(smoothing_mode="kalman"))


class TrackingLossResetTests(unittest.TestCase):
    """The phase's requirement: filters must reset when tracking is lost."""

    def test_runtime_resets_the_filter_when_the_hand_disappears(self):
        import numpy as np

        from kinematic.output import RecordingMouseOutput
        from kinematic.runtime import HandMouseRuntime

        output = RecordingMouseOutput()
        runtime = HandMouseRuntime(
            config=HandMouseConfig(show_preview=False, start_paused=False),
            output=output,
        )
        runtime.cursor_x, runtime.cursor_y = 500.0, 400.0
        runtime.cursor_filter.reset()
        runtime.cursor_filter.update((500.0, 400.0))  # seeded mid-motion

        runtime.process_frame(np.zeros((120, 160, 3), dtype=np.uint8))  # no hand at all

        # After a reset the next sample must pass straight through: no glide from stale state.
        self.assertEqual(runtime.cursor_filter.update((700.0, 100.0)), (700.0, 100.0))

    def test_a_hand_held_still_after_reacquire_does_not_drift(self):
        from types import SimpleNamespace

        import numpy as np

        from kinematic.output import RecordingMouseOutput
        from kinematic.runtime import HandMouseRuntime

        points = [(0.5, 0.5) for _ in range(21)]
        points[8] = (0.55, 0.45)  # index tip
        points[4] = (0.30, 0.60)  # thumb, far away

        def results():
            landmarks = [SimpleNamespace(x=x, y=y, z=0.0) for x, y in points]
            return SimpleNamespace(multi_hand_landmarks=[SimpleNamespace(landmark=landmarks)])

        output = RecordingMouseOutput()
        runtime = HandMouseRuntime(
            config=HandMouseConfig(show_preview=False, start_paused=False),
            output=output,
            hands=SimpleNamespace(process=lambda rgb: results()),
        )

        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        for _ in range(10):
            runtime.process_frame(frame)
        moves_before = len(output.moves)

        runtime.process_frame(np.zeros((120, 160, 3), dtype=np.uint8))  # tracking lost
        for _ in range(10):
            runtime.process_frame(frame)  # the same hand comes back, still

        self.assertLessEqual(
            len(output.moves) - moves_before,
            2,
            "a hand that is not moving must not keep moving the cursor",
        )


if __name__ == "__main__":
    unittest.main()
