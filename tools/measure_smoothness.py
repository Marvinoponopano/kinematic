"""Measure how smooth the emitted cursor movement is. No camera, no mouse.

The complaint this exists to answer: "the mouse feels stuttery, like it is snapping to a grid". That is
a measurable property of the *emitted* moves — a smooth pipeline emits small, similarly sized steps every
frame; a snappy one holds the cursor for several frames and then jumps.

    python3 tools/measure_smoothness.py                    # three paths, current defaults
    python3 tools/measure_smoothness.py --deadzone 0.02    # what the old default did
    python3 tools/measure_smoothness.py --sweep            # grid of deadzone / filter settings
    python3 tools/measure_smoothness.py --compare-relative # same paths in relative mode

Reported per path:

* **held frames** — frames where the cursor did not move at all (a "hold" in hold-then-jump);
* **step px** — mean / p50 / p95 / max size of an emitted move, in screen pixels, measured after the
  warm-up (the first frames of a path are the cursor *settling* from wherever it was onto the hand, which
  is a decaying convergence, not stutter — reporting it as a step would mis-rank the filters);
* **step CV** — coefficient of variation of the step sizes. A constant-velocity hand should produce
  near-constant steps (low CV); snapping produces a mix of zeros and big jumps (high CV);
* **lag px** — mean distance between the cursor and the straight line the hand is following, after the
  warm-up, sampled on the frame each move was emitted on. This is what smoothing *costs*: a heavier
  filter trails further behind;
* **ends short px** — how far behind the hand the cursor was on the last frame it moved. Like the lag
  this is sampled on the frames that were emitted, so a rate limit changes *how often* the cursor
  updates (moves/held, step size) rather than how far behind it is when it does.

The hand is synthetic and seeded, so every number here is reproducible without a camera.

Not a smoothing option on this stack: `pyautogui`'s own tween. It only tweened for durations above
`MINIMUM_DURATION = 0.1 s`, and a 0.11 s tween measured **171 ms per move** (5.2 camera frame
intervals) with its sleep floored at `MINIMUM_SLEEP = 0.05 s`; anything at or below 0.1 s teleports
anyway (0.2-0.7 ms). So there is no usable per-frame tween — the smoothness has to come from the
filter and the reference, which is what this tool measures.
"""

import argparse
import math
import pathlib
import random
import statistics
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from kinematic.config import HandMouseConfig  # noqa: E402
from kinematic.gesture import (  # noqa: E402
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    THUMB_TIP,
    WRIST,
    pointer_reference_point,
)
from kinematic.output import RecordingMouseOutput  # noqa: E402
from kinematic.runtime import HandMouseRuntime  # noqa: E402

FPS = 30.0
FRAME = 1.0 / FPS
SCALE = 0.15
OPEN = 0.70  # open hand: gestures stay out of the way

PATHS = {
    # name: (start, end, frames) — the palm walks this straight line across the frame
    "still": (0.50, 0.50, 90),
    "slow": (0.40, 0.60, 90),
    "fast": (0.30, 0.70, 90),
}


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


def ideal_screen_path(name: str, screen_w: int) -> Tuple[float, float]:
    """Where the cursor *should* be at the start and end of the path, in screen pixels."""
    start, end, _ = PATHS[name]
    centre = screen_w / 2.0
    # absolute mode: the inner 70% of the frame covers the screen
    return (centre + (start - 0.5) / 0.70 * screen_w, centre + (end - 0.5) / 0.70 * screen_w)


def run_path(name: str, config: HandMouseConfig, jitter: float, seed: int, warmup: int = 10) -> Dict[str, Any]:
    """Feed one synthetic path through the pipeline and summarise the emitted moves.

    Steps and lag are measured *after* ``warmup`` moves, so the cursor settling onto the hand at the
    start of the path (a decaying convergence from wherever it was) is not mistaken for stepping.
    Lags are sampled at the *frame each move was emitted on*, not at its index in the move list —
    with the rate limit on, move #i happens several frames in, and sampling by move index would
    report the gap between the cursor and a point the hand has not reached yet.
    """
    rng = random.Random(seed)
    output = RecordingMouseOutput()
    runtime = HandMouseRuntime(config=config, output=output)
    start, end, frames = PATHS[name]
    start_px, end_px = ideal_screen_path(name, runtime.screen_w)

    frame_index = [-1]
    emitted_at: List[int] = []
    real_move_to = output.move_to

    def move_to(x: float, y: float, duration: float = 0.0) -> None:
        emitted_at.append(frame_index[0])
        real_move_to(x, y, duration=duration)

    output.move_to = move_to  # type: ignore[assignment]

    for index in range(frames):
        frame_index[0] = index
        progress = index / max(1, frames - 1)
        palm = (
            start + (end - start) * progress + rng.gauss(0.0, jitter),
            0.5 + rng.gauss(0.0, jitter),
        )
        runtime.process_landmarks(hand_with_palm_at(palm), now=index * FRAME)

    positions = [event.value[0] for event in output.moves]
    empty = {"path": name, "moves": 0, "held": frames, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0,
             "cv": 0.0, "lag": 0.0, "settle": 0.0, "travel": 0.0, "end": 0.0, "end_shortfall": 0.0}
    if not positions:
        return empty

    # Where the cursor *should* be on the frame each move was emitted (moves are not one per frame:
    # held frames emit nothing, and a rate limit drops moves outright).
    samples = [start_px + (end_px - start_px) * (frame / max(1, frames - 1)) for frame in emitted_at]
    steady = slice(warmup, len(positions))
    steps = [positions[i] - positions[i - 1] for i in range(max(1, warmup), len(positions))]
    sizes = [abs(step) for step in steps]
    window = range(min(warmup, len(positions) - 1), len(positions))
    lag = statistics.fmean(abs(positions[i] - samples[i]) for i in window)
    settle = abs(positions[min(warmup, len(positions) - 1)] - samples[min(warmup, len(samples) - 1)])
    del steady

    return {
        "path": name,
        "moves": len(output.moves),
        "held": frames - len(output.moves),
        "mean": statistics.fmean(sizes) if sizes else 0.0,
        "p50": statistics.median(sizes) if sizes else 0.0,
        "p95": sorted(sizes)[int(0.95 * (len(sizes) - 1))] if sizes else 0.0,
        "max": max(sizes) if sizes else 0.0,
        "cv": (statistics.pstdev(sizes) / statistics.fmean(sizes)) if len(sizes) > 1 and statistics.fmean(sizes) else 0.0,
        "lag": lag,
        "settle": settle,
        # `travel` is the post-warm-up distance the cursor actually covered, and `end_shortfall` is
        # how far behind the hand it finished. A rate limit defers movement rather than losing it,
        # so both change only a little: what it really changes is `moves`/`held` and the step size.
        "travel": sum(sizes),
        "end": positions[-1],
        "end_shortfall": abs(positions[-1] - samples[-1]),
    }


def report(rows: Sequence[Dict[str, Any]], title: str) -> None:
    print(f"\n{title}")
    print(f"  {'path':6} {'moves':>6} {'held':>5} {'step mean':>10} {'p50':>7} {'p95':>7} {'max':>7} {'step CV':>8} {'lag':>7} {'ends short':>10}")
    for row in rows:
        print(
            f"  {row['path']:6} {row['moves']:6d} {row['held']:5d} {row['mean']:10.1f} {row['p50']:7.1f} "
            f"{row['p95']:7.1f} {row['max']:7.1f} {row['cv']:8.2f} {row['lag']:7.1f} {row['end_shortfall']:10.1f}"
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Measure emitted cursor smoothness (no camera, no mouse).")
    parser.add_argument("--jitter", type=float, default=0.002, help="landmark jitter stdev to add (default 0.002)")
    parser.add_argument("--seed", type=int, default=7, help="seed for the jitter (default 7)")
    parser.add_argument("--deadzone", type=float, default=None, help="override the absolute-mode deadzone")
    parser.add_argument("--smoothing-mode", choices=("ema", "one_euro"), default=None)
    parser.add_argument("--one-euro-beta", type=float, default=None, help="override one_euro_beta (default 8.0)")
    parser.add_argument("--min-cutoff", type=float, default=None, help="override one_euro_min_cutoff (default 1.0)")
    parser.add_argument("--min-move-px", type=float, default=None, help="override min_move_pixels (default 0.5)")
    parser.add_argument("--smoothing", type=float, default=None, help="override the EMA factor (default 0.35)")
    parser.add_argument("--pointer-mode", choices=("absolute", "relative"), default=None)
    parser.add_argument("--move-max-hz", type=float, default=None, help="override move_max_hz (0 = uncapped)")
    parser.add_argument("--sweep", action="store_true", help="compare several configurations")
    parser.add_argument("--rate-sweep", action="store_true", help="compare move_max_hz settings (0 = uncapped)")
    parser.add_argument("--compare-relative", action="store_true", help="also run the paths in relative mode")
    args = parser.parse_args(argv)

    overrides: Dict[str, Any] = {"show_preview": False, "start_paused": False, "dry_run": True}
    if args.deadzone is not None:
        overrides["deadzone"] = args.deadzone
    if args.smoothing_mode is not None:
        overrides["smoothing_mode"] = args.smoothing_mode
    if args.one_euro_beta is not None:
        overrides["one_euro_beta"] = args.one_euro_beta
    if args.min_cutoff is not None:
        overrides["one_euro_min_cutoff"] = args.min_cutoff
    if args.min_move_px is not None:
        overrides["min_move_pixels"] = args.min_move_px
    if args.smoothing is not None:
        overrides["smoothing"] = args.smoothing
    if args.pointer_mode is not None:
        overrides["pointer_mode"] = args.pointer_mode
    if args.move_max_hz is not None:
        overrides["move_max_hz"] = args.move_max_hz
    config = HandMouseConfig(**overrides)

    print(
        f"jitter stdev {args.jitter} ({args.jitter / 0.70 * 1710:.1f} px of screen at gain 1.0), "
        f"seed {args.seed}, {FPS:.0f} fps, mode {config.pointer_mode}"
    )
    print(f"deadzone {config.deadzone} ({config.deadzone / 0.70 * 1710:.1f} px of screen), "
          f"smoothing {config.smoothing_mode} min_cutoff={config.one_euro_min_cutoff} beta={config.one_euro_beta}")
    print(f"move rate limit {config.move_max_hz:g} Hz" + ("" if config.move_max_hz else " (uncapped)"))

    if args.rate_sweep:
        # Phase 5: what a cap on the move rate costs. Held frames rise and steps grow as the cap
        # falls, while travel and the end position stay put — the cursor still covers the path, it
        # just does it in fewer, larger steps. That trade is the reason the default is uncapped.
        for hz in (0.0, 30.0, 20.0, 15.0, 10.0, 5.0):
            sweep_config = HandMouseConfig(**{**overrides, "move_max_hz": hz})
            rows = [run_path(name, sweep_config, args.jitter, args.seed) for name in PATHS]
            label = " (uncapped, the shipped default)" if hz == 0.0 else ""
            report(rows, f"move_max_hz={hz:g}{label}")
        return 0

    if args.sweep:
        for deadzone in (0.02, 0.005, 0.002, 0.0):
            for mode in ("one_euro", "ema"):
                sweep_config = HandMouseConfig(**{**overrides, "deadzone": deadzone, "smoothing_mode": mode})
                rows = [run_path(name, sweep_config, args.jitter, args.seed) for name in PATHS]
                report(rows, f"deadzone={deadzone}  smoothing={mode}")
        return 0

    rows = [run_path(name, config, args.jitter, args.seed) for name in PATHS]
    report(rows, "current configuration")
    if args.compare_relative:
        relative = HandMouseConfig(**{**overrides, "pointer_mode": "relative"})
        report([run_path(name, relative, args.jitter, args.seed) for name in PATHS], "pointer_mode=relative")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
