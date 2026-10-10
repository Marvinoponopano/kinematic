"""Tests for cursor smoothness: the "it feels stuttery, like it is snapping to a grid" report.

Two separate causes, both pinned here so they cannot come back:

1. **The absolute-mode deadzone froze the reference.** `deadzone` is in *frame* units, so the old
   0.02 held the cursor still until the palm had moved ~49 screen px and then jumped. `deadzone=0`
   is now the default; a non-zero value is a *radius*, not a per-axis box.
2. **The One Euro filter was effectively off.** `beta` is in pixel units: with the old 8.0 and
   screen-pixel inputs the cutoff saturated even at 0.5 px/frame (alpha 0.96), so landmark noise and
   quantisation went straight to the cursor. It is 0 now, and the cutoff is 4 Hz.

The numbers are deterministic (seeded jitter, fake timestamps, recording adapter) and the same ones
`tools/measure_smoothness.py` prints.
"""

import random
import statistics
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from kinematic.config import HandMouseConfig
from kinematic.gesture import (
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    THUMB_TIP,
    WRIST,
    pointer_reference_point,
)
from kinematic.output import RecordingMouseOutput
from kinematic.runtime import HandMouseRuntime
from kinematic.smoothing import ExponentialMovingAverage, OneEuroFilter, filter_from_config

FPS = 30.0
FRAME = 1.0 / FPS
SCALE = 0.15
OPEN = 0.70
SCREEN_W = 1920  # RecordingMouseOutput's default screen
JITTER = 0.002  # landmark jitter stdev, the figure tools/measure_stability.py reports


def _base_points() -> List[Tuple[float, float]]:
    points = [(0.50, 0.60) for _ in range(21)]
    points[WRIST] = (0.50, 0.60)
    points[MIDDLE_MCP] = (0.50, 0.45)
    points[THUMB_TIP] = (0.40, 0.52)
    return points


def hand_with_palm_at(point: Tuple[float, float]) -> List[SimpleNamespace]:
    neutral = [SimpleNamespace(x=x, y=y, z=0.0) for x, y in _base_points()]
    palm = pointer_reference_point(neutral, "palm")
    ox, oy = point[0] - palm[0], point[1] - palm[1]
    points = [(x + ox, y + oy) for x, y in _base_points()]
    thumb = points[THUMB_TIP]
    points[INDEX_TIP] = (thumb[0] + OPEN * SCALE, thumb[1])
    points[MIDDLE_TIP] = (thumb[0], thumb[1] - OPEN * SCALE)
    return [SimpleNamespace(x=x, y=y, z=0.0) for x, y in points]


def make_runtime(output: RecordingMouseOutput, **overrides: Any) -> HandMouseRuntime:
    config_overrides: Dict[str, Any] = {"show_preview": False, "start_paused": False}
    config_overrides.update(overrides)
    return HandMouseRuntime(config=HandMouseConfig(**config_overrides), output=output)


def walk(
    runtime: HandMouseRuntime,
    output: RecordingMouseOutput,
    start: float,
    end: float,
    frames: int = 90,
    jitter: float = JITTER,
    seed: int = 7,
) -> List[float]:
    """Feed a straight-line palm path (with jitter) and return the emitted cursor x positions."""
    rng = random.Random(seed)
    for index in range(frames):
        progress = index / max(1, frames - 1)
        runtime.process_landmarks(
            hand_with_palm_at((start + (end - start) * progress + rng.gauss(0.0, jitter), 0.5 + rng.gauss(0.0, jitter))),
            now=index * FRAME,
        )
    return [event.value[0] for event in output.moves]


def steps(positions: Sequence[float], warmup: int = 10) -> List[float]:
    """Per-frame cursor step sizes, skipping the startup settle so it cannot skew the stats."""
    if len(positions) <= warmup:
        return []
    return [abs(positions[i] - positions[i - 1]) for i in range(max(1, warmup), len(positions))]


def _load_tool() -> Any:
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parent.parent / "tools" / "measure_smoothness.py"
    spec = importlib.util.spec_from_file_location("measure_smoothness", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tool_row(path_name: str, **overrides: Any) -> Dict[str, Any]:
    """One row of tools/measure_smoothness.py — the same numbers the report quotes."""
    module = _load_tool()
    config_overrides: Dict[str, Any] = {"show_preview": False, "start_paused": False, "dry_run": True}
    config_overrides.update(overrides)
    return module.run_path(path_name, HandMouseConfig(**config_overrides), 0.002, 7)


class StillHandTests(unittest.TestCase):
    """A still hand must not jitter the cursor: this is what the filter is for."""

    def test_the_default_filter_actually_smooths(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output)
        positions = walk(runtime, output, 0.5, 0.5)
        sizes = steps(positions)
        self.assertTrue(sizes, "a still hand must still track (it should not be frozen either)")
        self.assertLess(statistics.fmean(sizes), 4.0, "mean cursor step on a still hand, px")
        self.assertLess(max(sizes), 10.0, "largest cursor step on a still hand, px")

    def test_the_old_filter_tuning_was_effectively_no_filtering(self):
        # beta is in pixel units: 8.0 saturates the cutoff at screen-pixel speeds (alpha ~0.96 at
        # 0.5 px/frame), so the old default passed landmark noise straight through. This test is the
        # evidence for one_euro_beta being 0 by default.
        noisy_output, smooth_output = RecordingMouseOutput(), RecordingMouseOutput()
        noisy = make_runtime(noisy_output, one_euro_beta=8.0)
        smooth = make_runtime(smooth_output, one_euro_beta=0.0)
        noisy_steps = steps(walk(noisy, noisy_output, 0.5, 0.5))
        smooth_steps = steps(walk(smooth, smooth_output, 0.5, 0.5))
        self.assertGreater(
            statistics.fmean(noisy_steps),
            2.0 * statistics.fmean(smooth_steps),
            "beta=8 should be measurably worse than beta=0 on a still hand",
        )

    def test_a_lower_cutoff_is_smoother(self):
        swift_output, calm_output = RecordingMouseOutput(), RecordingMouseOutput()
        swift = make_runtime(swift_output, one_euro_min_cutoff=8.0)
        calm = make_runtime(calm_output, one_euro_min_cutoff=2.0)
        self.assertLess(
            statistics.fmean(steps(walk(calm, calm_output, 0.5, 0.5))),
            statistics.fmean(steps(walk(swift, swift_output, 0.5, 0.5))),
            "a lower cutoff must smooth more",
        )


class NoSnappingTests(unittest.TestCase):
    """Deliberate movement must be continuous: no held frames, no jumps."""

    def test_a_slow_sweep_never_holds_the_cursor(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output)
        positions = walk(runtime, output, 0.40, 0.60)
        self.assertEqual(len(positions), 90, "every frame of a slow sweep must move the cursor")
        sizes = steps(positions)
        self.assertLess(max(sizes), 20.0, "no step may be a visible jump")
        self.assertLess(
            statistics.pstdev(sizes) / statistics.fmean(sizes), 0.6, "steps should be similarly sized"
        )

    def test_the_old_deadzone_still_measurably_degrades_a_slow_sweep(self):
        # It no longer freezes outright (the smoother filter creeps toward the frozen anchor), but it
        # is still clearly worse: fewer frames move, the steps that do arrive are bigger and uneven,
        # and the cursor trails the hand by ~39 px more. The default is 0; this is the evidence.
        slow_rows = {
            deadzone: _tool_row("slow", deadzone=deadzone)
            for deadzone in (0.0, 0.02)
        }
        smooth, snapped = slow_rows[0.0], slow_rows[0.02]
        self.assertEqual(smooth["held"], 0)
        self.assertLess(smooth["max"], 15.0)
        self.assertLess(smooth["cv"], 0.55)

        self.assertGreater(snapped["held"], 10, "the deadzone should hold frames")
        self.assertGreater(snapped["max"], 25.0, "the deadzone should produce jumps")
        self.assertGreater(snapped["cv"], 0.8, "the deadzone's steps should be uneven")
        self.assertGreater(snapped["lag"], smooth["lag"] + 20.0, "the deadzone should add trailing")

    def test_the_deadzone_is_a_radius_not_a_per_axis_box(self):
        # A slow diagonal move must not be frozen because one axis is inside the box: with a radius
        # of 0.004 a diagonal step of 0.003 per axis (0.0042 in total) is *outside* it.
        output = RecordingMouseOutput()
        runtime = make_runtime(output, deadzone=0.004)
        rng = random.Random(3)
        for index in range(40):
            progress = index / 39.0
            runtime.process_landmarks(
                hand_with_palm_at((0.40 + 0.10 * progress + rng.gauss(0, JITTER), 0.40 + 0.10 * progress + rng.gauss(0, JITTER))),
                now=index * FRAME,
            )
        self.assertGreater(len(output.moves), 30, "a diagonal move must not be frozen by a box test")

    def test_a_deadzone_of_zero_means_no_freeze_at_all(self):
        self.assertEqual(HandMouseConfig().deadzone, 0.0)
        runtime = make_runtime(RecordingMouseOutput())
        self.assertEqual(runtime.config.deadzone, 0.0)


class FilterSelectionTests(unittest.TestCase):
    def test_the_default_filter_is_one_euro_with_the_measured_tuning(self):
        config = HandMouseConfig()
        self.assertEqual(config.smoothing_mode, "one_euro")
        self.assertEqual(config.one_euro_min_cutoff, 4.0)
        self.assertEqual(config.one_euro_beta, 0.0)

    def test_filter_from_config_builds_each_mode(self):
        one_euro = filter_from_config(HandMouseConfig(smoothing_mode="one_euro", one_euro_min_cutoff=2.5))
        self.assertIsInstance(one_euro, OneEuroFilter)
        self.assertEqual(one_euro.min_cutoff, 2.5)

        ema = filter_from_config(HandMouseConfig(smoothing_mode="ema", smoothing=0.2))
        self.assertIsInstance(ema, ExponentialMovingAverage)

    def test_a_beta_of_zero_is_a_plain_low_pass(self):
        # Documented consequence: with beta=0 the cutoff no longer depends on speed, so the filter
        # absorbs the same *proportion* of every step regardless of how fast the hand moves. With
        # beta > 0 a faster hand is followed more closely (a larger proportion), which is exactly the
        # behaviour that amplified landmark noise on screen-pixel inputs.
        def absorbed(beta: float, step: float) -> float:
            filt = OneEuroFilter(min_cutoff=4.0, beta=beta)
            position = 0.0
            previous_centre = 0.0
            for index in range(40):
                centre = step * (index + 1)
                position = filt.update((centre, 0.0), index * FRAME)[0]
                previous_centre = centre
            return (previous_centre - position) / step  # steady-state lag, in steps

        for beta in (0.0,):
            slow, fast = absorbed(beta, 1.0), absorbed(beta, 20.0)
            self.assertAlmostEqual(slow, fast, delta=0.02, msg=f"beta={beta} must be speed-independent")
        # With the old 8.0 the filter barely smooths at all *at screen-pixel speeds*: at 20 px/frame
        # it absorbs 0.02 of a step, where beta=0 absorbs well over half. That is why beta is 0 here.
        self.assertLess(absorbed(8.0, 20.0), 0.1)
        self.assertGreater(absorbed(0.0, 20.0), 0.5)


class MoveDurationTests(unittest.TestCase):
    """Every emitted move is a teleport, and that is deliberate — see the class comment below."""

    def test_moves_are_teleports(self):
        # pyautogui only tweens above its MINIMUM_DURATION of 0.1 s, and a 0.11 s tween measured
        # 171 ms per move (5.2 camera frame intervals), which would collapse the loop rate. Below
        # 0.1 s it teleports anyway. So there is no usable per-frame tween on this stack and the
        # runtime passes 0; this test pins that the adapter is told so.
        output = RecordingMouseOutput()
        runtime = make_runtime(output)
        walk(runtime, output, 0.40, 0.60, frames=20)
        self.assertTrue(output.moves)
        self.assertTrue(all(event.duration == 0.0 for event in output.moves))


class SmoothnessToolTests(unittest.TestCase):
    def test_the_tool_imports_and_runs_one_path(self):
        row = _tool_row("slow")
        self.assertEqual(row["path"], "slow")
        self.assertEqual(row["held"], 0, "the default pipeline must not hold frames on a slow sweep")
        self.assertLess(row["max"], 20.0)

    def test_the_tool_reports_the_paths_it_documents(self):
        module = _load_tool()
        self.assertEqual(set(module.PATHS), {"still", "slow", "fast"})


if __name__ == "__main__":
    unittest.main()
