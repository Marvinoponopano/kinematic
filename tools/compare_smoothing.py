"""Compare cursor smoothing filters on identical, seeded synthetic motion.

Answers the question the config's default values hang on: for the same jitter rejection,
which filter lags less? Each metric runs over many seeds and reports the mean and spread, so
no single lucky sequence decides the answer.

    python3 tools/compare_smoothing.py
    python3 tools/compare_smoothing.py --seeds 20 --frames 300 --screen-width 1710

No camera and no mouse output: it feeds explicit points through the pure filters.
"""

import argparse
import random
import statistics
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kinematic.smoothing import CursorFilter, ExponentialMovingAverage, OneEuroFilter  # noqa: E402

Point = Tuple[float, float]
DT = 1.0 / 30.0


def jitter(filter_factory: Callable[[], CursorFilter], seed: int, frames: int, noise: float) -> float:
    """Output spread for a hand that is not moving but is reported noisily."""
    rng = random.Random(seed)
    filter_ = filter_factory()
    for step in range(30):  # settle at the centre first
        filter_.update((0.5, 0.5), step * DT)
    outputs = []
    for step in range(frames):
        sample = (0.5 + rng.uniform(-noise, noise), 0.5 + rng.uniform(-noise, noise))
        outputs.append(filter_.update(sample, (30 + step) * DT)[0])
    return statistics.pstdev(outputs)


def response_frames(filter_factory: Callable[[], CursorFilter], seed: int, frames: int) -> float:
    """Frames to cover 90% of a step, after settling at the origin."""
    filter_ = filter_factory()
    filter_.update((0.0, 0.0), 0.0)
    for step in range(1, frames):
        value = filter_.update((1.0, 1.0), step * DT)
        if value[0] >= 0.9:
            return float(step)
    return float("nan")


def tracking_lag(filter_factory: Callable[[], CursorFilter], seed: int, frames: int, speed: float) -> float:
    """Mean distance behind a hand moving at constant speed (steady-state lag)."""
    filter_ = filter_factory()
    errors = []
    for step in range(frames):
        position = 0.5 + speed * step * DT
        value = filter_.update((position, position), step * DT)
        if step > 20:  # skip the transient
            errors.append(abs(position - value[0]))
    return statistics.fmean(errors)


def summarise(values: List[float]) -> str:
    return f"{statistics.fmean(values):.5f} ± {(statistics.pstdev(values) if len(values) > 1 else 0.0):.5f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare cursor smoothing filters (seeded, pure).")
    parser.add_argument("--seeds", type=int, default=10, help="number of seeds to aggregate (default 10)")
    parser.add_argument("--frames", type=int, default=240, help="frames per run (default 240)")
    parser.add_argument("--noise", type=float, default=0.01, help="landmark noise amplitude (default 0.01)")
    parser.add_argument("--speed", type=float, default=0.25, help="hand speed in normalized units/s (default 0.25)")
    parser.add_argument("--fast-speed", type=float, default=1.0, help="fast hand speed (default 1.0)")
    parser.add_argument("--screen-width", type=int, default=1710, help="screen width, for the pixel column")
    parser.add_argument("--sensitivity", type=float, default=5.0, help="pointer_sensitivity, for the pixel column")
    args = parser.parse_args()

    candidates: Dict[str, Callable[[], CursorFilter]] = {
        "ema factor=0.35 (default)": lambda: ExponentialMovingAverage(factor=0.35),
        "ema factor=0.5": lambda: ExponentialMovingAverage(factor=0.5),
        "one_euro min=1.0 beta=0.0": lambda: OneEuroFilter(min_cutoff=1.0, beta=0.0),
        "one_euro min=1.0 beta=1.0": lambda: OneEuroFilter(min_cutoff=1.0, beta=1.0),
        "one_euro min=1.0 beta=2.0": lambda: OneEuroFilter(min_cutoff=1.0, beta=2.0),
        "one_euro min=1.0 beta=4.0": lambda: OneEuroFilter(min_cutoff=1.0, beta=4.0),
        "one_euro min=1.0 beta=6.0": lambda: OneEuroFilter(min_cutoff=1.0, beta=6.0),
        "one_euro min=1.0 beta=8.0": lambda: OneEuroFilter(min_cutoff=1.0, beta=8.0),
        "one_euro min=0.5 beta=6.0": lambda: OneEuroFilter(min_cutoff=0.5, beta=6.0),
        "one_euro min=2.0 beta=6.0": lambda: OneEuroFilter(min_cutoff=2.0, beta=6.0),
    }

    seeds = list(range(args.seeds))
    print(
        f"seeds={args.seeds} frames={args.frames} landmark noise=±{args.noise} "
        f"hand speed slow={args.speed}/s fast={args.fast_speed}/s  (30 fps, normalized units)\n"
    )
    header = (
        f"{'filter':<28} {'jitter (stdev)':<20} {'90% step (frames)':<19} "
        f"{'lag slow (px)':<15} {'lag fast (px)'}"
    )
    print(header)
    print("-" * len(header))

    results = {}
    for name, factory in candidates.items():
        jitters = [jitter(factory, seed, args.frames, args.noise) for seed in seeds]
        responses = [response_frames(factory, seed, args.frames) for seed in seeds]
        slow_lags = [tracking_lag(factory, seed, args.frames, args.speed) for seed in seeds]
        fast_lags = [tracking_lag(factory, seed, args.frames, args.fast_speed) for seed in seeds]
        results[name] = (jitters, responses, slow_lags, fast_lags)
        px_slow = statistics.fmean(slow_lags) * args.sensitivity * args.screen_width
        px_fast = statistics.fmean(fast_lags) * args.sensitivity * args.screen_width
        print(f"{name:<28} {summarise(jitters):<20} {summarise(responses):<19} {px_slow:14.0f} {px_fast:14.0f}")

    print()
    reference = results["ema factor=0.35 (default)"]
    ref_jitter = statistics.fmean(reference[0])
    ref_slow = statistics.fmean(reference[2]) * args.sensitivity * args.screen_width
    ref_fast = statistics.fmean(reference[3]) * args.sensitivity * args.screen_width
    print(f"reference: ema factor=0.35 -> jitter {ref_jitter:.5f}, lag {ref_slow:.0f} px slow / {ref_fast:.0f} px fast")

    beaten = [
        name
        for name, (jitters, _, slow_lags, fast_lags) in results.items()
        if name != "ema factor=0.35 (default)"
        and statistics.fmean(jitters) < ref_jitter
        and statistics.fmean(slow_lags) * args.sensitivity * args.screen_width <= ref_slow
        and statistics.fmean(fast_lags) * args.sensitivity * args.screen_width <= ref_fast
    ]
    print("\nPareto-better than the current default (less jitter AND no more lag at either speed):")
    for name in beaten:
        print(f"  - {name}")
    if not beaten:
        print("  (none in this sweep: the current default is still the best trade-off here)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
