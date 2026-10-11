"""Phase 5's last bullet: rate limiting / coalescing cursor moves.

The bullet reads "rate-limit or coalesce movement events **if needed**", so most of what is pinned
here is the *measurement* that decides it, not just the code:

* A limit equal to the frame rate changes nothing (30 Hz on this 30 fps path produces the same
  moves as no limit at all) — so an uncapped default is not the same thing as no mechanism.
* Below the frame rate the cost is exactly the one the smoothness work removed: every emitted move
  grows until one move covers several frames of hand travel (`--rate-sweep` in
  tools/measure_smoothness.py prints it: mean step 6.2 px uncapped, 12.4 px at 20 Hz, 36.7 px at
  5 Hz on the same slow path).
* Nothing is queued: a burst of frames inside one interval produces one move, carrying the *newest*
  position, and never a backlog of stale ones.

Everything below runs with the recording adapter, fake timestamps and no camera.
"""

import inspect
import pathlib
import statistics
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from kinematic.config import HandMouseConfig
from kinematic.gesture import (
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    THUMB_TIP,
    WRIST,
    pointer_reference_point,
)
from kinematic.main import build_config, parse_args
from kinematic.output import RecordingMouseOutput
from kinematic.ratelimit import MoveRateLimiter
from kinematic.runtime import HandMouseRuntime

FPS = 30.0
FRAME = 1.0 / FPS
SCALE = 0.15
OPEN = 0.70  # open hand: gestures stay out of the way
PINCHED = 0.33  # clearly below the 0.40 engage ratio
TWO_FINGER: Dict[str, Any] = dict(thumb_index_ratio=PINCHED, thumb_middle_ratio=PINCHED)

PATHS = {
    # name: (start, end, frames) — the palm walks this straight line across the frame
    "still": (0.50, 0.50, 90),
    "slow": (0.40, 0.60, 90),
}


class NoHands:
    """A hand detector that never sees anything (keeps MediaPipe out of the suite)."""

    def process(self, rgb: Any) -> Any:
        return SimpleNamespace(multi_hand_landmarks=None, multi_handedness=None)


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


def hand(**ratios: float) -> List[SimpleNamespace]:
    """A hand with the requested pinch ratios, built from the same base pose as above."""
    points = _base_points()
    thumb = points[THUMB_TIP]
    points[INDEX_TIP] = (
        thumb[0] + ratios.get("thumb_index_ratio", OPEN) * SCALE,
        thumb[1],
    )
    points[MIDDLE_TIP] = (
        thumb[0],
        thumb[1] - ratios.get("thumb_middle_ratio", OPEN) * SCALE,
    )
    return [SimpleNamespace(x=x, y=y, z=0.0) for x, y in points]


def make_runtime(output: RecordingMouseOutput, **overrides: Any) -> HandMouseRuntime:
    config_overrides: Dict[str, Any] = {"show_preview": False, "start_paused": False}
    config_overrides.update(overrides)
    return HandMouseRuntime(
        config=HandMouseConfig(**config_overrides), output=output, hands=NoHands()
    )


def _mapped(runtime: HandMouseRuntime, frame_point: Tuple[float, float]) -> Tuple[float, float]:
    """Screen point a frame position maps to, including the reach gain (test helper)."""
    from kinematic.mapping import map_point_to_screen

    target = map_point_to_screen(
        frame_point,
        (runtime.screen_w, runtime.screen_h),
        frame_region=runtime.frame_region,
        screen_region=runtime.screen_region,
        clamp=False,
    )
    return runtime.apply_reach_gain(target)


def walk(
    runtime: HandMouseRuntime,
    output: RecordingMouseOutput,
    start: float,
    end: float,
    frames: int = 90,
    then_hold: int = 0,
) -> List[float]:
    """Walk the palm along a straight line, then optionally hold still, returning emitted x."""
    for index in range(frames):
        progress = index / max(1, frames - 1)
        runtime.process_landmarks(
            hand_with_palm_at((start + (end - start) * progress, 0.5)), now=index * FRAME
        )
    for extra in range(then_hold):
        runtime.process_landmarks(hand_with_palm_at((end, 0.5)), now=(frames + extra) * FRAME)
    return [event.value[0] for event in output.moves]


def median_step(positions: Sequence[float], warmup: int = 2) -> float:
    """Median size of an emitted move, skipping the cursor's initial settle."""
    steps = [abs(positions[i] - positions[i - 1]) for i in range(warmup, len(positions))]
    return statistics.median(steps) if steps else 0.0


class MoveRateLimiterTests(unittest.TestCase):
    """The limiter on its own: pure, timestamp-driven, no clock and nothing queued."""

    def test_the_shipped_default_is_off(self):
        # Off is a measurement, not a shrug: the loop is camera-bound at ~30 fps and a move costs
        # ~11 ms, so one move per frame already happens (see --rate-sweep and measure_loop.py).
        self.assertEqual(HandMouseConfig().move_max_hz, 0.0)

    def test_off_means_every_frame_is_offered(self):
        limiter = MoveRateLimiter()
        self.assertFalse(limiter.enabled)
        self.assertEqual(limiter.interval, 0.0)
        for index in range(20):
            self.assertTrue(limiter.offer(index * 0.001))
        self.assertEqual(limiter.suppressed, 0)

    def test_a_limit_exposes_its_interval(self):
        limiter = MoveRateLimiter(10.0)
        self.assertTrue(limiter.enabled)
        self.assertAlmostEqual(limiter.interval, 0.1)

    def test_a_negative_limit_is_the_same_as_off(self):
        limiter = MoveRateLimiter(-5.0)
        self.assertEqual(limiter.max_hz, 0.0)
        self.assertFalse(limiter.enabled)
        self.assertTrue(limiter.offer(0.0))

    def test_the_first_move_is_always_due(self):
        # The cursor must not wait out an interval before it starts following a hand it just saw.
        limiter = MoveRateLimiter(1.0)
        self.assertTrue(limiter.due(0.0))
        self.assertTrue(limiter.offer(0.0))
        self.assertEqual(limiter.last_emit_at, 0.0)

    def test_a_frame_inside_the_interval_is_suppressed_and_counted(self):
        limiter = MoveRateLimiter(10.0)
        self.assertTrue(limiter.offer(0.0))
        self.assertFalse(limiter.offer(0.05))
        self.assertFalse(limiter.offer(0.099))
        self.assertEqual(limiter.suppressed, 2)
        self.assertTrue(limiter.offer(0.1))

    def test_a_frame_exactly_at_the_interval_is_due(self):
        limiter = MoveRateLimiter(10.0)
        limiter.offer(0.0)
        self.assertTrue(limiter.due(0.1), "the boundary belongs to the allowed side")

    def test_a_limit_equal_to_the_frame_rate_lets_every_frame_through(self):
        # The float trap this pins: at 30 Hz on a 30 fps loop the frame deltas miss 1/30 by ~1e-17,
        # and without the tolerance the nominal "one move per frame" setting would drop every other
        # frame. Measured in --rate-sweep: 30 Hz is identical to uncapped.
        limiter = MoveRateLimiter(30.0)
        passed = [limiter.offer(index * FRAME) for index in range(60)]
        self.assertEqual(passed, [True] * 60)

    def test_reset_lets_the_next_move_through_and_clears_the_count(self):
        limiter = MoveRateLimiter(1.0)
        limiter.offer(0.0)
        self.assertFalse(limiter.offer(0.1))
        limiter.reset()
        self.assertIsNone(limiter.last_emit_at)
        self.assertEqual(limiter.suppressed, 0)
        self.assertTrue(limiter.offer(0.1), "the reacquired hand's first move passes through")

    def test_stats_report_what_happened(self):
        limiter = MoveRateLimiter(10.0)
        limiter.offer(0.0)
        limiter.offer(0.05)
        stats = limiter.stats()
        self.assertEqual(stats["max_hz"], 10.0)
        self.assertEqual(stats["suppressed"], 1)
        self.assertEqual(stats["last_emit_at"], 0.0)

    def test_the_limiter_never_imports_the_output_layer_or_a_clock(self):
        # Pure like the filters and the recogniser: no mouse, no camera, no clock of its own.
        import kinematic.ratelimit as module

        source = inspect.getsource(module)
        for forbidden in (
            "import time",
            "import cv2",
            "import numpy",
            "import pyautogui",
            "from .output",
            "import mediapipe",
        ):
            self.assertNotIn(forbidden, source, f"ratelimit.py must not reference {forbidden}")


class RateLimitInTheCursorPathTests(unittest.TestCase):
    """The limiter wired into the runtime's move path."""

    def test_an_uncapped_run_keeps_the_move_every_frame_behaviour(self):
        # The status quo the feature must preserve. 89 of the 90 frames move: the missing one is the
        # move guard doing its job (the settling cursor passes within `min_move_pixels` of the
        # target for a single frame), not the limiter — which suppresses nothing at all here.
        output = RecordingMouseOutput()
        runtime = make_runtime(output)
        positions = walk(runtime, output, *PATHS["slow"][:2], frames=PATHS["slow"][2])
        self.assertGreaterEqual(len(positions), 89)
        self.assertEqual(runtime.move_limiter.max_hz, 0.0)
        self.assertEqual(runtime.move_limiter.suppressed, 0)

    def test_a_limit_at_the_frame_rate_changes_nothing(self):
        uncapped, capped = RecordingMouseOutput(), RecordingMouseOutput()
        plain = walk(make_runtime(uncapped), uncapped, *PATHS["slow"][:2], frames=90)
        limited = walk(make_runtime(capped, move_max_hz=30.0), capped, *PATHS["slow"][:2], frames=90)
        self.assertEqual(len(plain), len(limited))
        self.assertEqual(plain, limited, "30 Hz on a 30 fps loop is the same as no limit")

    def test_a_limit_below_the_frame_rate_thins_the_moves(self):
        plain_output, capped_output = RecordingMouseOutput(), RecordingMouseOutput()
        plain = walk(make_runtime(plain_output), plain_output, *PATHS["slow"][:2], frames=90)
        capped = walk(make_runtime(capped_output, move_max_hz=10.0), capped_output, *PATHS["slow"][:2], frames=90)
        self.assertEqual(len(capped), 30, "10 Hz over 3 s of frames")
        # The first step is the cursor settling onto the hand (large in both runs); the median step
        # is what the cap costs — each remaining move covers several frames of hand travel.
        self.assertGreater(median_step(capped), 2.0 * median_step(plain))

    def test_the_path_is_still_covered_once_the_hand_stops(self):
        # Coalescing defers movement, it does not lose it: with the hand held still at the end, the
        # capped cursor settles on the same place as the uncapped one.
        uncapped, capped = RecordingMouseOutput(), RecordingMouseOutput()
        plain = walk(make_runtime(uncapped), uncapped, 0.40, 0.60, frames=60, then_hold=30)
        limited = walk(make_runtime(capped, move_max_hz=10.0), capped, 0.40, 0.60, frames=60, then_hold=30)
        self.assertLess(len(limited), len(plain), "the cap must actually thin the moves")
        self.assertLess(abs(limited[-1] - plain[-1]), 2.0, "both end on the same spot")

    def test_a_burst_sends_one_move_per_interval_and_never_a_backlog(self):
        # Frames arriving far faster than the interval (a catching-up loop, or an over-fast feed):
        # the limiter emits once per interval, and each emission carries the position of the frame
        # it is emitted on. Nothing is stored, so there is no queue of stale positions to drain.
        output = RecordingMouseOutput()
        runtime = make_runtime(output, move_max_hz=5.0)  # one move every 0.2 s
        start = 2.0
        for index in range(20):  # 20 frames inside the first interval
            runtime.process_landmarks(
                hand_with_palm_at((0.40 + 0.20 * index / 19.0, 0.5)), now=start + index * 0.001
            )
        self.assertEqual(len(output.moves), 1, "one interval, one move — not one per frame")

        for offset in (0.05, 0.10, 0.15):  # still inside the interval, hand parked at the end
            runtime.process_landmarks(hand_with_palm_at((0.60, 0.5)), now=start + offset)
        self.assertEqual(len(output.moves), 1, "frames inside the interval emit nothing")

        runtime.process_landmarks(hand_with_palm_at((0.60, 0.5)), now=start + 0.25)
        self.assertEqual(len(output.moves), 2, "the due frame emits")
        end_x = _mapped(runtime, (0.60, 0.5))[0]
        self.assertGreater(
            output.moves[-1].value[0],
            0.9 * end_x,
            "the emitted position is where the hand is now, not the burst's first frame",
        )
        self.assertLess(abs(output.moves[-1].value[0] - end_x), 25.0)

    def test_clicks_are_not_rate_limited(self):
        # The cap is on cursor moves only: a click is an event, not a sample.
        output = RecordingMouseOutput()
        runtime = make_runtime(output, move_max_hz=2.0)
        for step in range(4):
            runtime.process_landmarks(hand(thumb_index_ratio=PINCHED), now=step * 0.05)
        self.assertEqual([click.value for click in output.clicks], ["left"])

    def test_scrolls_are_not_rate_limited(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output, move_max_hz=2.0)
        for step in range(4):
            runtime.process_landmarks(hand(**TWO_FINGER), now=step * 0.05)
        runtime.process_landmarks(
            [SimpleNamespace(x=p.x, y=p.y + 0.05, z=0.0) for p in hand(**TWO_FINGER)], now=0.5
        )
        self.assertEqual(len(output.scrolls), 1)

    def test_tracking_loss_resets_the_limiter(self):
        # A dropout shorter than the interval must not hold back the reacquired hand's first move.
        output = RecordingMouseOutput()
        runtime = make_runtime(output, move_max_hz=1.0)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        runtime.process_landmarks(hand_with_palm_at((0.40, 0.5)), now=0.0)
        self.assertEqual(len(output.moves), 1)
        runtime.process_frame(frame, now=0.1)  # no hand: the limiter is reset here
        self.assertIsNone(runtime.move_limiter.last_emit_at)

        runtime.process_landmarks(hand_with_palm_at((0.45, 0.5)), now=0.2)
        self.assertEqual(len(output.moves), 2, "the first move after a loss always passes")

    def test_the_summary_reports_the_cap_and_what_it_thinned(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output, move_max_hz=10.0)
        walk(runtime, output, *PATHS["slow"][:2], frames=90)
        summary = runtime.summary()
        self.assertEqual(summary["move_max_hz"], 10.0)
        self.assertEqual(summary["recorded"]["move"], 30)

        # The invariant: every frame that got past the move guard either moved the cursor or was
        # thinned by the limiter — nothing else can happen to a frame.
        plain_output = RecordingMouseOutput()
        guard_passing = walk(make_runtime(plain_output), plain_output, *PATHS["slow"][:2], frames=90)
        self.assertEqual(summary["moves_suppressed"], len(guard_passing) - 30)


class RateLimitCommandLineTests(unittest.TestCase):
    def test_the_cap_can_be_set_from_the_command_line(self):
        config = build_config(parse_args(["--move-max-hz", "12.5"]))
        self.assertEqual(config.move_max_hz, 12.5)

    def test_the_cap_defaults_to_uncapped(self):
        config = build_config(parse_args([]))
        self.assertEqual(config.move_max_hz, 0.0)


def _load_tool() -> Any:
    import importlib.util

    path = pathlib.Path(__file__).resolve().parent.parent / "tools" / "measure_smoothness.py"
    spec = importlib.util.spec_from_file_location("measure_smoothness", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tool_row(path_name: str, **overrides: Any) -> Dict[str, Any]:
    module = _load_tool()
    config_overrides: Dict[str, Any] = {"show_preview": False, "start_paused": False, "dry_run": True}
    config_overrides.update(overrides)
    return module.run_path(path_name, HandMouseConfig(**config_overrides), 0.002, 7)


class RateLimitIsMeasuredTests(unittest.TestCase):
    """The numbers the default is based on, pinned as the same seeded tool rows."""

    def test_a_cap_at_the_loop_rate_is_identical_to_uncapped(self):
        plain, capped = _tool_row("slow", move_max_hz=0.0), _tool_row("slow", move_max_hz=30.0)
        self.assertEqual(plain["moves"], capped["moves"])
        self.assertEqual(plain["mean"], capped["mean"])

    def test_a_cap_costs_step_size_and_nothing_else(self):
        # The trade, in numbers: below the loop rate the moves get fewer and bigger while the
        # trailing at the frames that do move stays put. This is why the default is uncapped.
        plain, capped = _tool_row("slow", move_max_hz=0.0), _tool_row("slow", move_max_hz=10.0)
        self.assertEqual(plain["held"], 0)
        self.assertEqual(capped["moves"], 30)
        self.assertEqual(capped["held"], 60)
        self.assertGreater(capped["mean"], 2.5 * plain["mean"], "steps grow as the cap falls")
        self.assertLess(abs(capped["lag"] - plain["lag"]), 2.0, "the trailing is unchanged")

    def test_the_tool_exposes_a_rate_sweep(self):
        module = _load_tool()
        source = inspect.getsource(module)
        self.assertIn("--rate-sweep", source)
        self.assertIn("move_max_hz", source)


if __name__ == "__main__":
    unittest.main()
