"""Print the gesture recogniser's transition table for scripted landmark sequences.

No camera, no MediaPipe, no mouse: a synthetic hand is fed through
:class:`kinematic.recognizer.GestureRecognizer` with fake timestamps, and every state change and
event is printed. Use it to see what a pose actually does, and to tune ``pinch_ratio`` /
``pinch_release_ratio`` against your own hand's measured ratios.

    python3 tools/gesture_trace.py                     # every scenario, transitions only
    python3 tools/gesture_trace.py --scenario scroll -v # plus one line per frame
    python3 tools/gesture_trace.py --pinch-ratio 0.35   # try a different threshold
"""

import argparse
import pathlib
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from kinematic.recognizer import (  # noqa: E402
    GestureRecognizer,
    GestureThresholds,
    HandState,
    Observation,
)

SCALE = 0.15  # wrist -> middle knuckle, in normalized frame units
FRAME = 0.033  # ~30 fps

PINCHED = 0.33  # below the 0.40 engage threshold
BAND = 0.47  # inside the hysteresis band
OPEN = 0.70  # above the 0.56 release threshold

WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP, MIDDLE_MCP = 0, 4, 8, 12, 9


class Frame(NamedTuple):
    label: str
    landmarks: Optional[List[SimpleNamespace]]
    timestamp: float


def hand(
    thumb_index_ratio: float = OPEN,
    thumb_middle_ratio: float = OPEN,
    offset: Tuple[float, float] = (0.0, 0.0),
) -> List[SimpleNamespace]:
    """A 21-landmark hand whose pinch distances are exactly ``ratio * SCALE``."""
    ox, oy = offset
    points = [(0.40 + ox, 0.90 + oy) for _ in range(21)]
    points[WRIST] = (0.40 + ox, 0.90 + oy)
    points[MIDDLE_MCP] = (0.40 + ox, 0.90 + oy - SCALE)
    thumb = (0.60 + ox, 0.60 + oy)
    points[THUMB_TIP] = thumb
    points[INDEX_TIP] = (thumb[0] + thumb_index_ratio * SCALE, thumb[1])
    points[MIDDLE_TIP] = (thumb[0], thumb[1] - thumb_middle_ratio * SCALE)
    return [SimpleNamespace(x=x, y=y, z=0.0) for x, y in points]


def frames(pose: str, count: int, start: float, *, step: float = FRAME, offset=(0.0, 0.0)) -> List[Frame]:
    """Repeat one pose for *count* frames."""
    builder = {
        "open": lambda: hand(offset=offset),
        "left": lambda: hand(thumb_index_ratio=PINCHED, offset=offset),
        "right": lambda: hand(thumb_middle_ratio=PINCHED, offset=offset),
        "both": lambda: hand(thumb_index_ratio=PINCHED, thumb_middle_ratio=PINCHED, offset=offset),
        "band": lambda: hand(thumb_index_ratio=BAND, offset=offset),
        "lost": lambda: None,
        "drift": lambda: hand(thumb_index_ratio=PINCHED, thumb_middle_ratio=PINCHED, offset=offset),
    }[pose]
    out = []
    for index in range(count):
        out.append(Frame(pose, builder(), start + index * step))
    return out


def scenarios() -> Dict[str, List[Frame]]:
    """The transitions the phase names, each as a scripted sequence of frames."""
    scripted: Dict[str, List[Frame]] = {}

    # A deliberate tap: pinched long enough to hold (0.10 s past the 0.08 s threshold).
    scripted["hold_time"] = (
        frames("open", 3, 0.0)
        + frames("left", 5, 0.10)  # commits at 0.20; the frames at 0.10/0.133/0.166 are inside the hold
        + frames("open", 3, 0.30)
    )

    # Three taps: the second inside the cooldown, the third after it. Each pinch section spans
    # 4 frames (0.099 s) so it clears the 0.08 s hold.
    scripted["cooldown"] = (
        frames("open", 2, 0.0)  # 0.000, 0.033
        + frames("left", 4, 0.10)  # commits at 0.199 -> left click
        + frames("open", 1, 0.22)  # released: re-arms the pair
        + frames("left", 4, 0.26)  # commits at 0.359, 0.16 s later -> inside the cooldown, dropped
        + frames("open", 1, 0.45)
        + frames("left", 4, 0.50)  # commits at 0.599, 0.40 s later -> left click
    )

    # A hand resting at the engage threshold: hysteresis must stop the chatter.
    chatter = []
    for index in range(24):
        pose = "left" if index % 2 == 0 else "band"
        chatter.append(frames(pose, 1, index * 0.02)[0])
    scripted["chatter"] = chatter

    # Two fingers down, drifting: scroll events, no clicks.
    scripted["scroll"] = (
        frames("open", 2, 0.0)
        + frames("both", 3, 0.05)
        + [Frame("drift", hand(thumb_index_ratio=PINCHED, thumb_middle_ratio=PINCHED, offset=(0.0, step * 0.01)), 0.25 + step * FRAME) for step in range(12)]
    )

    # Losing and reacquiring the hand mid-pinch.
    scripted["tracking_loss"] = (
        frames("open", 2, 0.0) + frames("left", 4, 0.05) + frames("lost", 5, 0.25) + frames("left", 6, 0.45)
    )

    # Paused while pinching, resumed still pinching (consumed), then released and pinched afresh.
    pause_frames: List[Frame] = []
    for index in range(26):
        timestamp = index * FRAME
        if timestamp < 0.40:
            pose, build = "left", "left"  # held across the pause and the resume
        elif timestamp < 0.62:
            pose, build = "open", "open"  # released: re-arms the pair
        else:
            pose, build = "left", "left"  # fresh pinch
        pause_frames.append(Frame(pose, hand(thumb_index_ratio=PINCHED if build == "left" else OPEN), timestamp))
    scripted["pause"] = pause_frames

    # Thumb-middle: the right click.
    scripted["right_pinch"] = frames("open", 2, 0.0) + frames("right", 4, 0.05)

    return scripted


def run_scenario(
    name: str,
    frames_list: Sequence[Frame],
    thresholds: GestureThresholds,
    verbose: bool,
    pause_events: Sequence[Tuple[float, bool]] = (),
) -> Dict[str, Any]:
    recogniser = GestureRecognizer(thresholds)
    pending_pause = list(pause_events)

    print(f"\n=== {name} " + "=" * max(0, 60 - len(name)))
    counts: Dict[str, int] = {}

    for frame in frames_list:
        while pending_pause and frame.timestamp >= pending_pause[0][0]:
            when, paused = pending_pause.pop(0)
            events = recogniser.set_paused(paused)
            for event in events:
                counts[event.kind.value] = counts.get(event.kind.value, 0) + 1
            label = "PAUSE" if paused else "RESUME"
            fired = ", ".join(event.kind.value for event in events)
            print(f"  t={when:5.2f}  {label:17} -> {recogniser.state.value:17} {fired}")

        before = recogniser.state
        update = recogniser.update(Observation(frame.landmarks, frame.timestamp))
        for event in update.events:
            counts[event.kind.value] = counts.get(event.kind.value, 0) + 1
        if update.state is not before or update.events:
            ratios = recogniser.last_ratios
            ratio_text = (
                f"  ratios ti={ratios.get('thumb_index', float('nan')):.2f} "
                f"tm={ratios.get('thumb_middle', float('nan')):.2f}"
                if ratios
                else ""
            )
            events = ", ".join(
                f"{event.kind.value}{f'({event.value:.0f})' if event.value else ''}" for event in update.events
            )
            print(f"  t={frame.timestamp:5.2f}  {before.value:17} -> {update.state.value:17} {events}{ratio_text}")
        elif verbose:
            print(f"  t={frame.timestamp:5.2f}  {update.state.value:17}   (no change)")

    print(f"  final state: {recogniser.state.value}")
    print(f"  events: {counts if counts else 'none'}")
    return {"final_state": recogniser.state, "counts": counts}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Trace the gesture state machine over scripted poses.")
    parser.add_argument("--scenario", default="all", help="scenario name, or 'all' (default)")
    parser.add_argument("-v", "--verbose", action="store_true", help="print every frame, not just transitions")
    parser.add_argument("--pinch-ratio", type=float, default=None, help="override pinch_ratio (default 0.40)")
    parser.add_argument("--hold", type=float, default=None, help="override hold_seconds (default 0.08)")
    parser.add_argument("--cooldown", type=float, default=None, help="override cooldown_seconds (default 0.18)")
    args = parser.parse_args(argv)

    overrides: Dict[str, Any] = {}
    if args.pinch_ratio is not None:
        overrides["pinch_ratio"] = args.pinch_ratio
    if args.hold is not None:
        overrides["hold_seconds"] = args.hold
    if args.cooldown is not None:
        overrides["cooldown_seconds"] = args.cooldown
    thresholds = GestureThresholds(**overrides)

    print(
        f"thresholds: pinch_ratio={thresholds.pinch_ratio} "
        f"release<={thresholds.pinch_ratio * thresholds.release_ratio:.2f} "
        f"hold={thresholds.hold_seconds}s cooldown={thresholds.cooldown_seconds}s"
    )
    print(f"synthetic hand scale={SCALE}  (pose ratios: {PINCHED} pinched / {BAND} band / {OPEN} open)")

    scripted = scenarios()
    names = list(scripted) if args.scenario == "all" else [args.scenario]
    for name in names:
        if name not in scripted:
            parser.error(f"unknown scenario {name!r}; choose from {', '.join(scripted)}")
        # The pause scenario is the only one that needs the pause key pressed mid-sequence.
        pause_events = [(0.0, True), (0.25, False)] if name == "pause" else ()
        run_scenario(name, scripted[name], thresholds, args.verbose, pause_events=pause_events)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
