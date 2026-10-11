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

`--sweep-reach N` walks the synthetic palm to each corner of the frame region, *holds* it there for
`--reach-dwell` frames and reports how far the cursor actually got. It reports the two phases
separately on purpose: a lagging filter cannot hit the extreme *while the hand is still moving*,
whereas a hand that arrives and stops gets there — and stopping at the edge is what a user does. The
verdict comes from the dwell phase; the travel shortfall is printed next to it so the lag is visible
rather than hidden.

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
from typing import Dict, List, Optional, Tuple

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

    #: The four corners of the frame region, as frame coordinates (the region absolute mode maps).
    REACH_CORNERS: Tuple[Tuple[float, float], ...] = ((0.15, 0.15), (0.85, 0.15), (0.85, 0.85), (0.15, 0.85))

    def __init__(
        self,
        real=None,
        translate: bool = True,
        tap_period: int = 30,
        gesture: str = "left",
        sweep_reach: int = 0,
        reach_dwell: int = 12,
        scroll_axis: Optional[str] = None,
        scroll_amplitude: float = 0.10,
        scroll_frames: int = 120,
    ) -> None:
        self.real = real
        self.translate = translate
        self.tap_period = tap_period
        self.gesture = gesture
        self.sweep_reach = sweep_reach
        self.reach_dwell = reach_dwell
        self.scroll_axis = scroll_axis
        self.scroll_amplitude = scroll_amplitude
        self.scroll_frames = max(2, int(scroll_frames))
        #: Which phase of the reach sweep this frame belongs to ("travel" | "dwell"), read by the
        #: reporting code so the two are not averaged into one misleading number.
        self.reach_phase = "travel"
        self.calls = 0

    def scroll_offset(self, step: int) -> Tuple[float, float]:
        """Where the two-finger reference should be on this step of a controlled scroll sweep.

        A straight line from -amplitude to +amplitude in *frame* units along the chosen axis, so the
        net hand movement is positive (down / right / both) and the emitted sign can be checked.
        """
        span = max(1, self.scroll_frames - 1)
        progress = min(1.0, max(0.0, (step - 1) / span))
        delta = -self.scroll_amplitude + 2.0 * self.scroll_amplitude * progress
        if self.scroll_axis == "vertical":
            return (0.0, delta)
        if self.scroll_axis == "horizontal":
            return (delta, 0.0)
        return (delta, delta)  # diagonal: both axes, same sign

    def reach_target(self, step: int) -> Tuple[float, float]:
        """Where the palm should be on this step of the reach sweep, and whether it is dwelling.

        Each leg travels to a corner over ``sweep_reach`` frames and then holds it for
        ``reach_dwell`` frames, because a hand that stops at the edge is the case that has to reach
        the edge; a continuously moving hand trails it by the filter's lag.
        """
        travel = max(1, self.sweep_reach)
        leg = travel + max(0, self.reach_dwell)
        index = (step // leg) % len(self.REACH_CORNERS)
        within = step % leg
        corner = self.REACH_CORNERS[index]
        previous = self.REACH_CORNERS[index - 1] if index > 0 else (0.5, 0.5)
        if within < travel:
            self.reach_phase = "travel"
            progress = (within + 1) / travel
            return (
                previous[0] + (corner[0] - previous[0]) * progress,
                previous[1] + (corner[1] - previous[1]) * progress,
            )
        self.reach_phase = "dwell"
        return corner

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

    #: Where the palm centre sits relative to the hand's `base` point in :meth:`_hand`, which the
    #: reach sweep needs in order to put the palm itself on a chosen frame position.
    PALM_FROM_BASE = (0.016, -0.026)

    def process(self, rgb):
        self.calls += 1
        if self.real is not None:
            self.real(rgb)  # keep paying the real MediaPipe cost
        step = self.calls
        if self.scroll_axis:
            # Controlled scroll sweep: the hand stays in the two-finger pose and the reference walks
            # a straight line, so the emitted sign and the total distance can both be checked.
            offset = self.scroll_offset(step)
            index_drop = 0.0
        elif self.sweep_reach:
            # Walk the palm to each corner and hold it there (see reach_target). Both axes use their
            # own offset because the palm is not centred on the hand's base point.
            palm = self.reach_target(step)
            offset = (palm[0] - 0.5 - self.PALM_FROM_BASE[0], palm[1] - 0.6 - self.PALM_FROM_BASE[1])
            index_drop = 0.0
        elif self.translate:
            offset = (0.05 * math.sin(step / 10.0), 0.03 * math.cos(step / 13.0))
            index_drop = 0.0
        else:
            offset = (0.0, 0.0)
            index_drop = 0.10 * math.sin(step / 6.0)
        # A realistic tap: pinched for 6 frames every tap_period (0.2 s of a ~1 s cycle). The
        # Phase 4 recogniser only registers a pose that holds for pinch_hold_seconds (0.08 s ≈ 2.4
        # frames at 30 fps), so a single-frame pinch is deliberately ignored.
        if self.scroll_axis:
            pinch_target: Optional[str] = "two_finger"  # held for the whole sweep
        else:
            pinch_target = self.gesture if (step % self.tap_period) < 6 else None
        landmarks = [
            SimpleNamespace(x=x, y=y, z=0.0) for x, y in self._hand(offset, index_drop, pinch_target)
        ]
        return SimpleNamespace(multi_hand_landmarks=[SimpleNamespace(landmark=landmarks)])

    def close(self):
        pass


def scroll_verdict(axis: str, vertical: int, horizontal: int) -> Tuple[bool, str]:
    """Did a straight-line scroll sweep come out with the documented sign on the right axis?

    The sweep's net hand movement is positive (down / right / both), so the expectation is:
    ``vertical`` → the page scrolls **down** (negative units) and nothing sideways; ``horizontal`` →
    positive horizontal units and no vertical wheel events; ``diagonal`` → both, with those signs.
    """
    if axis == "vertical":
        ok = vertical < 0 and horizontal == 0
        detail = f"hand down -> {vertical} vertical (page down), no sideways events"
    elif axis == "horizontal":
        ok = horizontal > 0 and vertical == 0
        detail = f"hand right -> +{horizontal} horizontal (scroll right), no vertical events"
    else:
        ok = vertical < 0 and horizontal > 0
        detail = f"hand down-right -> {vertical} vertical, +{horizontal} horizontal"
    return ok, ("pass - " if ok else "CHECK - ") + detail


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
        "--synthetic-scroll",
        choices=("vertical", "horizontal", "diagonal"),
        default=None,
        help="with --synthetic-hand: hold the two-finger pose and walk the reference in a straight line along this axis, then report the emitted sign and totals per axis",
    )
    parser.add_argument(
        "--scroll-amplitude",
        type=float,
        default=0.10,
        help="half the distance the scroll sweep covers, in frame widths (default 0.10)",
    )
    parser.add_argument(
        "--scroll-frames",
        type=int,
        default=120,
        help="frames one scroll sweep is spread over (default 120, ~4 s at 30 fps)",
    )
    parser.add_argument(
        "--scroll-max-hz",
        type=float,
        default=None,
        help="cap scroll events at this many per second (0 = uncapped, the default); held-back amounts accumulate",
    )
    parser.add_argument(
        "--no-horizontal-scroll",
        action="store_true",
        help="turn sideways scrolling off (the pre-Phase-7 behaviour: the horizontal component is computed and dropped)",
    )
    parser.add_argument(
        "--invert-scroll-vertical",
        action="store_true",
        help="flip the vertical scroll direction",
    )
    parser.add_argument(
        "--invert-scroll-horizontal",
        action="store_true",
        help="flip the horizontal scroll direction",
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
    parser.add_argument(
        "--sweep-reach",
        type=int,
        default=0,
        help="with --synthetic-hand, walk the palm to each corner of the frame region over N frames, hold it for --reach-dwell, and report how far the cursor got (the absolute-mode reachability check)",
    )
    parser.add_argument(
        "--reach-dwell",
        type=int,
        default=12,
        help="frames to hold the palm at each corner during --sweep-reach (default 12, ~0.4 s: what a user does at an edge)",
    )
    parser.add_argument(
        "--move-max-hz",
        type=float,
        default=None,
        help="cap cursor moves at this many per second (0 = uncapped, the default); report the loop rate and moves/frame with and without it",
    )
    parser.add_argument(
        "--mouse-pause",
        type=float,
        default=None,
        help="set pyautogui's PAUSE (its own default is 0.1 s per call) to measure what a slow adapter does to the loop rate",
    )
    args = parser.parse_args()

    here = pyautogui.position()
    real_move_to = pyautogui.moveTo
    calls = {"move": 0, "click": 0, "scroll": 0, "hscroll": 0}
    travel: List[float] = []  # distance of each requested cursor move, in pixels
    previous_point: List[Tuple[float, float]] = []
    reached: Dict[str, List[float]] = {"x": [], "y": []}  # every requested coordinate, for the sweep
    reached_phase: List[str] = []  # "travel" | "dwell", so the two are reported separately
    phase_of: List[str] = ["travel"]  # set by the synt.*hook below; a list so the closure sees updates
    scroll_amounts: List[int] = []  # every vertical scroll amount, for the direction check
    hscroll_amounts: List[int] = []  # every horizontal scroll amount

    def counted_move_to(x, y, duration=0, **kwargs):
        calls["move"] += 1
        if previous_point:
            travel.append(math.hypot(x - previous_point[0][0], y - previous_point[0][1]))
        previous_point[:] = [(x, y)]
        reached["x"].append(x)
        reached["y"].append(y)
        reached_phase.append(phase_of[0])
        return real_move_to(here.x, here.y, duration=duration, **kwargs)

    def counted_scroll(amount, **kwargs):
        calls["scroll"] += 1
        scroll_amounts.append(int(amount))

    def counted_hscroll(amount, **kwargs):
        calls["hscroll"] += 1
        hscroll_amounts.append(int(amount))

    pyautogui.moveTo = counted_move_to
    pyautogui.click = lambda *a, **k: calls.__setitem__("click", calls["click"] + 1)
    pyautogui.scroll = counted_scroll
    pyautogui.hscroll = counted_hscroll

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
    if args.move_max_hz is not None:
        overrides["move_max_hz"] = args.move_max_hz
    if args.mouse_pause is not None:
        overrides["mouse_pause"] = args.mouse_pause
    if args.scroll_max_hz is not None:
        overrides["scroll_max_hz"] = args.scroll_max_hz
    if args.no_horizontal_scroll:
        overrides["scroll_horizontal_gain"] = 0.0
    if args.invert_scroll_vertical:
        overrides["scroll_invert_vertical"] = True
    if args.invert_scroll_horizontal:
        overrides["scroll_invert_horizontal"] = True
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
            sweep_reach=args.sweep_reach,
            reach_dwell=args.reach_dwell,
            scroll_axis=args.synthetic_scroll,
            scroll_amplitude=args.scroll_amplitude,
            scroll_frames=args.scroll_frames,
        )
        reach_phase_source = app.runtime.hands
        real_process = app.runtime.hands.process
    else:
        reach_phase_source = None

    def process(rgb, _real=real_process, _seen=seen, _stamps=stamps, _source=reach_phase_source):
        _seen["n"] += 1
        _stamps.append(time.perf_counter())
        result = _real(rgb)
        if _source is not None:
            phase_of[0] = _source.reach_phase  # the sweep sets this as it builds each frame
        return result

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
    if args.synthetic_hand and not args.synthetic_scroll:
        # The verification half of Phase 6: the synthetic hand taps on a fixed period, so the click
        # count can be checked against the taps that were actually offered. One click per deliberate
        # tap is the recogniser's contract; the legacy engine may fire two per tap (its left and
        # right tests can both see the same pose). Skipped for a scroll sweep, where the hand holds
        # the scroll pose for the whole run and there is no tap to count.
        synthetic = app.runtime.hands
        period = max(1, int(getattr(synthetic, "tap_period", 30) or 30))
        taps = synthetic.calls // period
        clicks, scrolls = calls["click"], calls["scroll"]
        print(f"taps offered     : {taps} ({synthetic.calls} frames, pinched for 6 of every {period})")
        if not taps:
            print("click verdict    : n/a (no tap fitted in the run)")
        elif args.synthetic_pinch == "two_finger":
            if app.config.gesture_engine == "recognizer":
                ok = clicks == 0 and scrolls > 0
                detail = "the scroll pose scrolls and never clicks"
            else:
                # Documented pre-Phase-4 behaviour: the two-finger pose right-clicked instead.
                ok = clicks > 0 and scrolls == 0
                detail = "legacy clicks on this pose instead of scrolling"
            print(f"click verdict    : {'pass' if ok else 'CHECK'} - {detail} "
                  f"({clicks} clicks, {scrolls} scrolls)")
        elif app.config.gesture_engine == "recognizer":
            ok = clicks == taps
            print(f"click verdict    : {'pass' if ok else 'CHECK'} - one click per deliberate tap "
                  f"({clicks} clicks for {taps} taps)")
        else:
            ok = taps <= clicks <= 2 * taps
            print(f"click verdict    : {'ok' if ok else 'CHECK'} for the legacy engine "
                  f"({clicks} clicks for {taps} taps: its left and right tests can both fire on one pose)")
        print(f"gesture counts   : last action {app.last_action!r}, {app.actions()}")
    if args.synthetic_scroll:
        # Phase 7's direction check: a controlled straight-line sweep at a known amplitude, with the
        # wheel calls intercepted, so the sign on each axis can be checked without scrolling anything.
        total_v, total_h = sum(scroll_amounts), sum(hscroll_amounts)
        ok, verdict = scroll_verdict(args.synthetic_scroll, total_v, total_h)
        print(
            f"scroll sweep     : {args.synthetic_scroll} axis, amplitude {args.scroll_amplitude} frame widths "
            f"over {args.scroll_frames} frames"
        )
        print(f"  vertical wheel  : {len(scroll_amounts)} events, {total_v} units total")
        print(f"  horizontal wheel: {len(hscroll_amounts)} events, {total_h} units total")
        print(f"  direction       : {verdict}")
        print(
            f"  event cap       : "
            + (
                f"{app.config.scroll_max_hz:g} Hz, {app.runtime.scroll_limiter.suppressed} frames held and accumulated"
                if app.config.scroll_max_hz
                else "off (one event per due frame)"
            )
        )
    if args.sweep_reach and reached["x"]:
        # Sub-pixel tolerance: the cursor filter lands a hair inside the edge (e.g. 1709.9999), which
        # is the edge as far as the pointer is concerned. The tolerance is the *move guard's* rest
        # band, not an arbitrary half pixel: `min_move_pixels` deliberately stops emitting once the
        # remaining distance to the target is under that, so the cursor rests within it of the target.
        guard = float(getattr(app.config, "min_move_pixels", 0.5) or 0.5)
        tolerance = max(0.5, guard + 0.5)
        screen_w, screen_h = app.runtime.screen_w, app.runtime.screen_h

        def extremes(phase: str) -> Optional[Tuple[float, float, float, float]]:
            xs = [x for x, tag in zip(reached["x"], reached_phase) if tag == phase]
            ys = [y for y, tag in zip(reached["y"], reached_phase) if tag == phase]
            if not xs or not ys:
                return None
            return min(xs), max(xs), min(ys), max(ys)

        def worse_shortfall(span: Optional[Tuple[float, float, float, float]]) -> float:
            if span is None:
                return float("inf")
            min_x, max_x, min_y, max_y = span
            return max(min_x, min_y, screen_w - max_x, screen_h - max_y)

        moving, held = extremes("travel"), extremes("dwell")
        verdict = "all four screen edges reached" if worse_shortfall(held) <= tolerance else "EDGES NOT REACHED"
        print(f"reach            : screen {screen_w}x{screen_h}, tolerance {tolerance:.1f} px (the move guard's rest band)")
        if moving is not None:
            print(
                f"  hand still moving : x {moving[0]:.1f}..{moving[1]:.1f}, y {moving[2]:.1f}..{moving[3]:.1f}"
                f"  (worst shortfall {worse_shortfall(moving):.1f} px - the filter's lag while the hand travels)"
            )
        if held is not None:
            print(
                f"  hand held at edge : x {held[0]:.1f}..{held[1]:.1f}, y {held[2]:.1f}..{held[3]:.1f}"
                f"  (worst shortfall {worse_shortfall(held):.1f} px) -> {verdict}"
            )
        else:
            print(f"  hand held at edge : no dwell frames recorded -> {verdict}")
    print(f"mouse calls kept away from the OS: {calls}")
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
