"""Measure the control loop's real frame rate without touching your mouse.

`pyautogui.moveTo` / `click` / `scroll` are replaced with counters (and, for `moveTo`, a
replay aimed at the cursor's current position — the real cost is paid, nothing moves on
screen), so this can measure the cursor path with no risk to the desktop.

    python3 tools/measure_loop.py                      # real camera, no hand in frame
    python3 tools/measure_loop.py --synthetic-hand --active
    python3 tools/measure_loop.py --synthetic-hand --active --fingers-only   # the bug demo
    python3 tools/measure_loop.py --synthetic-hand --active --pointer-reference index
    python3 tools/measure_loop.py --synthetic-hand --active --dial-notches 6   # measure the dial

The last two are the wrist/palm fix in a nutshell: with `--fingers-only` the synthetic
hand is parked and only its index finger moves. Under the default `palm` reference the
cursor does not move at all; under the legacy `index` reference it is dragged around —
which is what the app used to do.

`--dial-notches N` presses a sensitivity key N times halfway through the run (through the
real key handler, the same path the preview window uses) and reports the mean cursor travel
before and after, so the dial can be seen working as a number.

Two traps this harness exists to avoid:

* Time the loop from *inside* it. `cv2.destroyAllWindows()` on macOS costs ~1.5 s, which
  is enough to make a 30 fps loop look like 22 fps.
* Measure processed frames, not loop iterations. Since the capture thread landed, the loop
  can iterate on "no new frame yet" many times per second.
"""

import argparse
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import pyautogui  # noqa: E402

from kinematic.config import HandMouseConfig  # noqa: E402
from kinematic.main import HandMouseApp  # noqa: E402


class SyntheticHands:
    """Stand-in for MediaPipe that returns one hand with a realistic palm/finger layout.

    ``translate=True`` sweeps the whole hand (which is what moves the cursor).
    ``translate=False`` parks the hand and sweeps only the index fingertip: the reported
    bug's input, kept here so the fix can be demonstrated on demand.
    """

    def __init__(self, real=None, translate: bool = True, tap_period: int = 30, gesture: str = "left") -> None:
        self.real = real
        self.translate = translate
        self.tap_period = tap_period
        self.gesture = gesture
        self.calls = 0

    @staticmethod
    def _hand(offset, index_drop, pinch_target=None):
        base = (0.5 + offset[0], 0.6 + offset[1])
        points = [(base[0], base[1]) for _ in range(21)]
        points[0] = (base[0], base[1] + 0.02)  # wrist
        points[5] = (base[0] - 0.06, base[1] - 0.04)  # knuckles
        points[9] = (base[0], base[1] - 0.05)
        points[13] = (base[0] + 0.05, base[1] - 0.04)
        points[17] = (base[0] + 0.09, base[1] - 0.02)
        points[8] = (base[0] - 0.05, base[1] - 0.17 + index_drop)  # index tip
        points[12] = (base[0], base[1] - 0.18)
        points[16] = (base[0] + 0.05, base[1] - 0.17)
        points[20] = (base[0] + 0.09, base[1] - 0.15)
        thumb = (base[0] - 0.13, base[1] - 0.03)
        if pinch_target == "left":  # left click: thumb meets the index tip
            tip = points[8]
            thumb = (tip[0] - 0.005, tip[1] + 0.005)
        elif pinch_target == "right":  # right click: thumb meets the middle tip, index stays away
            tip = points[12]
            thumb = (tip[0] - 0.005, tip[1] + 0.005)
        elif pinch_target == "two_finger":  # scroll: the thumb meets both tips at once
            middle_of_tips = ((points[8][0] + points[12][0]) / 2.0, (points[8][1] + points[12][1]) / 2.0)
            thumb = middle_of_tips
            points[8] = (middle_of_tips[0] - 0.012, middle_of_tips[1])
            points[12] = (middle_of_tips[0] + 0.012, middle_of_tips[1])
        points[4] = thumb
        return points

    def process(self, rgb):
        self.calls += 1
        if self.real is not None:
            self.real(rgb)  # keep paying the real MediaPipe cost
        step = self.calls
        if self.translate:
            offset = (0.05 * math.sin(step / 10.0), 0.03 * math.cos(step / 13.0))
            index_drop = 0.0
        else:
            offset = (0.0, 0.0)
            index_drop = 0.10 * math.sin(step / 6.0)
        # A realistic tap: pinched for 6 frames every tap_period (0.2 s of a ~1 s cycle). The
        # Phase 4 recogniser only registers a pose that holds for pinch_hold_seconds (0.08 s ≈ 2.4
        # frames at 30 fps), so a single-frame pinch is deliberately ignored.
        pinching = (step % self.tap_period) < 6
        pinch_target = self.gesture if pinching else None
        landmarks = [
            SimpleNamespace(x=x, y=y, z=0.0) for x, y in self._hand(offset, index_drop, pinch_target)
        ]
        return SimpleNamespace(multi_hand_landmarks=[SimpleNamespace(landmark=landmarks)])

    def close(self):
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure the kinematic control loop.")
    parser.add_argument("--frames", type=int, default=120, help="processed frames to measure (default 120)")
    parser.add_argument("--synthetic-hand", action="store_true", help="inject a synthetic hand")
    parser.add_argument(
        "--fingers-only",
        action="store_true",
        help="with --synthetic-hand: park the hand and move only the index finger (the reported bug's input)",
    )
    parser.add_argument("--active", action="store_true", help="start unpaused (default: paused, as shipped)")
    parser.add_argument("--preview", action="store_true", help="show the preview window while measuring")
    parser.add_argument(
        "--pointer-mode",
        choices=("relative", "absolute"),
        default=None,
        help="override the config's pointer mode",
    )
    parser.add_argument(
        "--pointer-reference",
        choices=("palm", "wrist", "index"),
        default=None,
        help="override which hand point drives the cursor (index reproduces the old drift)",
    )
    parser.add_argument(
        "--gesture-engine",
        choices=("recognizer", "legacy"),
        default=None,
        help="override the gesture engine (recognizer is the default; legacy keeps the pre-Phase-4 inline tests)",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=None,
        help="pointer gain to start the run with (default 5.0)",
    )
    parser.add_argument(
        "--sensitivity-step",
        type=float,
        default=None,
        help="multiplier per dial notch (default 1.15)",
    )
    parser.add_argument(
        "--synthetic-pinch",
        choices=("left", "right", "two_finger"),
        default="left",
        help="with --synthetic-hand, which pose the periodic tap uses: left pinch (default), right pinch, or a two-finger pinch (scroll)",
    )
    parser.add_argument(
        "--dial-notches",
        type=int,
        default=0,
        help="turn the sensitivity dial by this many notches halfway through the run (as the preview keys would), then report the change in cursor travel",
    )
    parser.add_argument(
        "--dial-key",
        default=None,
        help="which dial key to press for --dial-notches; defaults to ']' for positive values and '[' for negative",
    )
    args = parser.parse_args()

    here = pyautogui.position()
    real_move_to = pyautogui.moveTo
    calls = {"move": 0, "click": 0, "scroll": 0}
    travel: List[float] = []  # distance of each requested cursor move, in pixels
    previous_point: List[Optional[Tuple[float, float]]] = [None]

    def counted_move_to(x, y, duration=0, **kwargs):
        calls["move"] += 1
        if previous_point[0] is not None:
            travel.append(math.hypot(x - previous_point[0][0], y - previous_point[0][1]))
        previous_point[0] = (x, y)
        return real_move_to(here.x, here.y, duration=duration, **kwargs)

    pyautogui.moveTo = counted_move_to
    pyautogui.click = lambda *a, **k: calls.__setitem__("click", calls["click"] + 1)
    pyautogui.scroll = lambda *a, **k: calls.__setitem__("scroll", calls["scroll"] + 1)

    overrides = {"show_preview": args.preview, "start_paused": not args.active}
    if args.pointer_mode is not None:
        overrides["pointer_mode"] = args.pointer_mode
    if args.pointer_reference is not None:
        overrides["pointer_reference"] = args.pointer_reference
    if args.gesture_engine is not None:
        overrides["gesture_engine"] = args.gesture_engine
    if args.sensitivity is not None:
        overrides["pointer_sensitivity"] = args.sensitivity
    if args.sensitivity_step is not None:
        overrides["sensitivity_step"] = args.sensitivity_step
    config = HandMouseConfig(**overrides)
    app = HandMouseApp(config)
    seen = {"n": 0}
    stamps = []
    real_process = app.runtime.hands.process
    if args.synthetic_hand:
        app.runtime.hands = SyntheticHands(
            real_process,
            translate=not args.fingers_only,
            gesture=args.synthetic_pinch,
        )
        real_process = app.runtime.hands.process

    def process(rgb, _real=real_process, _seen=seen, _stamps=stamps):
        _seen["n"] += 1
        _stamps.append(time.perf_counter())
        return _real(rgb)

    app.runtime.hands.process = process

    real_waitkey = cv2.waitKey
    dial_at = args.frames // 2 if args.dial_notches else None
    dial_done = [False]
    dial_state: dict = {"before_after": None, "key": None}

    def waitkey(delay=1, _seen=seen, _wait=real_waitkey):
        _wait(1)
        # Turn the dial through the real key handler, exactly as the preview window would. The
        # guard matters: waitKey is called once per loop iteration, and after processing frame N
        # the loop keeps iterating on "no new frame yet" until the capture thread delivers N+1 —
        # without it, one intended turn would be applied once per iteration.
        if dial_at is not None and not dial_done[0] and _seen["n"] >= dial_at:
            dial_done[0] = True
            key = args.dial_key or ("]" if args.dial_notches > 0 else "[")
            baseline = len(travel)
            before_sensitivity = app.runtime.sensitivity
            for _ in range(abs(args.dial_notches)):
                app.runtime.handle_key(ord(key))
            dial_state["before_after"] = (baseline, before_sensitivity)
            dial_state["key"] = key
        return ord("q") if _seen["n"] >= args.frames else 255

    cv2.waitKey = waitkey

    app.run()
    span = stamps[-1] - stamps[0]
    fps = (len(stamps) - 1) / span
    print(f"\nprocessed frames : {len(stamps)}")
    print(f"measured span    : {span:.2f} s")
    print(f"loop rate        : {fps:.2f} fps internal (app reports {app.loop_fps:.2f})")
    print(f"pointer          : mode={app.config.pointer_mode} reference={app.config.pointer_reference}")
    print(f"gestures         : engine={app.config.gesture_engine} state={app.gesture_state.value}")
    print(f"sensitivity      : {app.runtime.sensitivity:.3f} "
          f"[{app.runtime.sensitivity_bar()}] start={app.runtime.sensitivity_start:.2f} "
          f"turned={app.runtime.sensitivity_changes}x")
    if dial_state["before_after"] is None:
        if travel:
            print(f"cursor travel    : mean {sum(travel) / len(travel):.1f} px over {len(travel)} moves")
    else:
        baseline, sensitivity = dial_state["before_after"]
        before, after = travel[:baseline], travel[baseline:]
        ratio = app.runtime.sensitivity / sensitivity
        print(f"dial             : {dial_state['key']!r} x{abs(args.dial_notches)} -> sensitivity "
              f"{sensitivity:.3f} -> {app.runtime.sensitivity:.3f} ({ratio:.2f}x)")
        if before and after:
            mean_before = sum(before) / len(before)
            mean_after = sum(after) / len(after)
            print(f"cursor travel    : mean {mean_before:.1f} px over {len(before)} moves before, "
                  f"{mean_after:.1f} px over {len(after)} after "
                  f"({mean_after / mean_before if mean_before else 0:.2f}x)")
    print(f"status           : {app.status}")
    print(f"mouse calls kept away from the OS: {calls}")
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
