"""Profile the control loop's stages, with the mouse intercepted.

Where do a frame's milliseconds actually go? This prints the answer per stage — capture wait, flip and
BGR→RGB, MediaPipe, the cursor/gesture logic, the adapter calls and (when shown) the preview — as
mean / median / p95 plus each stage's share of the frame time. Nothing on the desktop moves:
`pyautogui.moveTo`/`click`/`scroll`/`hscroll` are replaced with counters, and `moveTo` replays at the
cursor's current position, so the real cost is paid while the pointer stays where it is.

    python3 tools/profile_loop.py                        # real camera, no hand in frame
    python3 tools/profile_loop.py --synthetic-hand       # no camera either: scripted frames + hand
    python3 tools/profile_loop.py --synthetic-hand --preview
    python3 tools/profile_loop.py --mouse-pause 0.1      # what pyautogui's own default cost
    python3 tools/profile_loop.py --cprofile 20          # top 20 cumulative functions

The breakdown exists to settle arguments about optimisation with numbers. On this machine (2026-10-11,
FaceTime camera at 1280x720) the camera is the answer: `read()` is the 33 ms frame interval and
MediaPipe is ~9 ms, so the loop is camera-bound at ~30 fps and everything else is noise. That is why
the defaults were left alone instead of "optimised" further — and why `--mouse-pause 0.1` (a 113 ms
adapter) is the one configuration where something else does dominate.

Two traps this harness avoids, both documented in AGENTS.md:

* Time from *inside* the loop. ``cv2.destroyAllWindows()`` on macOS costs ~1.5 s, which is enough to
  make a 30 fps loop look like 22 fps.
* Count *processed* frames, not loop iterations: the capture thread means the loop can spin on "no new
  frame yet" many times a second.
"""

import argparse
import contextlib
import cProfile
import io
import math
import pstats
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pyautogui  # noqa: E402

from kinematic.config import HandMouseConfig  # noqa: E402
from kinematic.main import HandMouseApp  # noqa: E402

#: The sub-stages of one processed frame, in the order they run. "logic" is what is left of the frame
#: after subtracting the others: the landmarks conversion, the cursor mapping and filter, the gesture
#: state machine and the bookkeeping around them.
IN_FRAME_STAGES = ("capture", "flip", "rgb", "mediapipe", "output")
#: Stages that run *after* process_frame in the loop (only when the preview is on). They are part of
#: the loop's iteration, not of the timed frame, so they must not be subtracted from `frame`.
LOOP_STAGES = ("preview_draw", "preview_show")


def synthetic_points(phase: float) -> List[Tuple[float, float]]:
    """A 21-landmark hand whose palm sweeps across the frame and whose fingers are spread.

    ``phase`` walks 0..1 across the middle of the frame, so the cursor genuinely moves — a parked hand
    would measure the move-guard's early-out instead of the movement path.
    """
    ox = 0.12 * math.sin(phase * 2.0 * math.pi)
    oy = 0.06 * math.sin(phase * 4.0 * math.pi)
    wrist = (0.50 + ox, 0.60 + oy)
    base = [
        wrist,
        (0.46, 0.56), (0.43, 0.52), (0.41, 0.49), (0.39, 0.46),  # thumb
        (0.45, 0.46), (0.44, 0.39), (0.43, 0.35), (0.42, 0.31),  # index
        (0.50, 0.45), (0.50, 0.37), (0.50, 0.32), (0.50, 0.28),  # middle
        (0.55, 0.46), (0.56, 0.38), (0.57, 0.33), (0.58, 0.29),  # ring
        (0.60, 0.48), (0.62, 0.41), (0.63, 0.36), (0.64, 0.32),  # pinky
    ]
    return [(x + ox, y + oy) for x, y in base]


class SyntheticHands:
    """MediaPipe-shaped stand-in: returns one hand as a protobuf-like object, frame by frame."""

    def __init__(self, frames: int) -> None:
        self._index = 0
        self._frames = frames

    def process(self, rgb: Any) -> Any:
        phase = (self._index % self._frames) / max(1, self._frames)
        self._index += 1
        points = synthetic_points(phase)
        landmarks = [SimpleNamespace(x=x, y=y, z=-0.02) for x, y in points]
        handedness = [SimpleNamespace(classification=[SimpleNamespace(label="Right", score=0.97)])]
        return SimpleNamespace(
            multi_hand_landmarks=[SimpleNamespace(landmark=landmarks)], multi_handedness=handedness
        )

    def close(self) -> None:
        pass


class StageTimer:
    """Accumulates per-stage seconds; every sample is one occurrence inside one processed frame."""

    def __init__(self) -> None:
        self.samples: Dict[str, List[float]] = {}
        #: Only the calls that happened *inside* process_frame. The same function can run again outside
        #: it (the preview flips the frame itself), so the frame's residual has to subtract this bucket
        #: rather than the whole stage — otherwise the residual goes negative.
        self.frame_samples: Dict[str, List[float]] = {}
        self.in_frame = False
        self.frame_times: List[float] = []

    def add(self, name: str, seconds: float) -> None:
        self.samples.setdefault(name, []).append(seconds)
        if self.in_frame:
            self.frame_samples.setdefault(name, []).append(seconds)

    def in_frame_total(self, name: str) -> float:
        return sum(self.frame_samples.get(name, ())) * 1000.0

    @contextlib.contextmanager
    def timed(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - started)

    def wrap(self, name: str, function: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with self.timed(name):
                return function(*args, **kwargs)

        return wrapped

    def stats(self, name: str) -> Optional[Dict[str, float]]:
        samples = self.samples.get(name)
        if not samples:
            return None
        ordered = sorted(samples)
        return {
            "count": len(samples),
            "mean": statistics.fmean(samples) * 1000.0,
            "median": statistics.median(samples) * 1000.0,
            "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] * 1000.0,
            "total": sum(samples) * 1000.0,
        }


def intercept_mouse(timer: Optional[StageTimer] = None) -> Dict[str, int]:
    """Replace pyautogui's calls with counters, keeping the real cost of a move.

    Every call the adapters can make is stubbed here, **including ``mouseDown``/``mouseUp``**: a held
    pinch presses the button instead of clicking it, and a stub list that misses those two lets the
    profiler really press the user's mouse button (it counts a press as a ``click`` and the matching
    release separately, so the numbers keep their meaning).
    """
    calls = {"move": 0, "click": 0, "release": 0, "scroll": 0, "hscroll": 0}
    here = pyautogui.position()
    real_move_to = pyautogui.moveTo

    def counted_move_to(x: Any, y: Any, duration: float = 0.0, **kwargs: Any) -> None:
        calls["move"] += 1
        real_move_to(here[0], here[1], duration=duration)

    def counted_scroll(amount: Any, **kwargs: Any) -> None:
        calls["scroll"] += 1

    def counted_hscroll(amount: Any, **kwargs: Any) -> None:
        calls["hscroll"] += 1

    def counted_press(button: str = "left", **kwargs: Any) -> None:
        calls["click"] += 1

    def counted_release(button: str = "left", **kwargs: Any) -> None:
        calls["release"] += 1

    pyautogui.moveTo = counted_move_to
    pyautogui.click = lambda *a, **k: calls.__setitem__("click", calls["click"] + 1)
    pyautogui.scroll = counted_scroll
    pyautogui.hscroll = counted_hscroll
    pyautogui.mouseDown = counted_press
    pyautogui.mouseUp = counted_release
    return calls


def build_instrumentation(timer: StageTimer, runtime: Any) -> Dict[str, Any]:
    """Patch the frame's stages so each one records its own time. Returns the originals to restore."""
    originals: Dict[str, Any] = {}

    # The camera read (or "no new frame yet") is timed by wrapping the camera object's read.
    originals["flip"] = cv2.flip
    originals["cvtColor"] = cv2.cvtColor
    cv2.flip = timer.wrap("flip", originals["flip"])
    cv2.cvtColor = timer.wrap("rgb", originals["cvtColor"])

    # MediaPipe: the runtime holds the object, so wrap the instance's process method.
    originals["hands_process"] = runtime.hands.process
    runtime.hands.process = timer.wrap("mediapipe", originals["hands_process"])

    # The adapter: what a move/click/scroll really costs, including pyautogui's own PAUSE.
    for name in ("move_to", "click", "press", "release", "scroll", "hscroll"):
        original = getattr(runtime.output, name)
        originals[name] = original
        setattr(runtime.output, name, timer.wrap("output", original))

    # The whole per-frame pipeline, so the shares have a denominator, plus its timestamp for the rate.
    originals["process_frame"] = runtime.process_frame

    def timed_process_frame(*args: Any, **kwargs: Any) -> Any:
        timer.frame_times.append(time.perf_counter())
        timer.in_frame = True
        try:
            with timer.timed("frame"):
                return originals["process_frame"](*args, **kwargs)
        finally:
            timer.in_frame = False

    runtime.process_frame = timed_process_frame

    if runtime.config.show_preview:
        originals["draw_preview"] = runtime.draw_preview
        originals["imshow"] = cv2.imshow
        runtime.draw_preview = timer.wrap("preview_draw", originals["draw_preview"])
        cv2.imshow = timer.wrap("preview_show", originals["imshow"])
    return originals


def restore_instrumentation(runtime: Any, originals: Dict[str, Any]) -> None:
    cv2.flip = originals["flip"]
    cv2.cvtColor = originals["cvtColor"]
    runtime.hands.process = originals["hands_process"]
    for name in ("move_to", "click", "scroll", "hscroll"):
        setattr(runtime.output, name, originals[name])
    runtime.process_frame = originals["process_frame"]
    if "draw_preview" in originals:
        runtime.draw_preview = originals["draw_preview"]
        cv2.imshow = originals["imshow"]


def synthetic_camera(frames: int, width: int, height: int, translate: bool = True) -> Any:
    """A camera whose frames are synthesised, so the tool runs with no hardware at all."""

    class _SyntheticCamera:
        def __init__(self) -> None:
            self._frame = np.full((height, width, 3), 40, dtype=np.uint8)
            self._left = frames
            self._seq = 0
            self.released = False

        @property
        def error(self) -> Optional[str]:
            return None

        def wait_ready(self, timeout: float = 3.0) -> bool:
            return True

        def read(self) -> Tuple[int, Optional[np.ndarray]]:
            if self._left > 0:
                self._left -= 1
                self._seq += 1
            return self._seq, self._frame

        def release(self) -> None:
            self.released = True

    return _SyntheticCamera()


def total_of(timer: StageTimer, stage: str) -> float:
    """Milliseconds spent in *stage* across the run (0.0 when it never ran)."""
    stats = timer.stats(stage)
    return stats["total"] if stats else 0.0


def report(
    timer: StageTimer,
    calls: Dict[str, int],
    elapsed: float,
    frames: int,
    title: str,
    synthetic: bool = False,
) -> None:
    """Print the stage table: milliseconds per frame, share of the iteration, and the loop rate.

    Shares are of the **loop iteration** (the frame plus, when it is on, drawing the preview), which is
    why the preview stages are reported next to the frame total rather than subtracted from it. Only
    the in-frame stages come out of `frame`'s total, which is what makes the `logic` line a real
    residual instead of a negative number.
    """
    frame_stats = timer.stats("frame")
    preview_total = sum(total_of(timer, stage) for stage in LOOP_STAGES)
    loop_total = (frame_stats["total"] if frame_stats else 0.0) + preview_total
    print(f"\n{title}")
    print("=" * len(title))
    print(f"{'stage':<14} {'calls/frame':>11} {'mean ms':>9} {'median ms':>10} {'p95 ms':>8} {'share':>7}")
    for stage in IN_FRAME_STAGES + LOOP_STAGES:
        stats = timer.stats(stage)
        if stats is None:
            continue
        share = (stats["total"] / loop_total * 100.0) if loop_total else 0.0
        print(
            f"{stage:<14} {stats['count'] / max(1, frames):>11.2f} {stats['mean']:>9.3f} "
            f"{stats['median']:>10.3f} {stats['p95']:>8.3f} {share:>6.1f}%"
        )

    known = sum(timer.in_frame_total(stage) for stage in IN_FRAME_STAGES)
    logic = (frame_stats["total"] - known) if frame_stats else 0.0
    if frame_stats and loop_total:
        print(
            f"{'logic':<14} {'-':>11} {logic / frames:>9.3f} {'-':>10} {'-':>8} "
            f"{logic / loop_total * 100:>6.1f}%   (conversion, mapping, filter, gestures)"
        )
    if frame_stats:
        print(
            f"{'frame':<14} {frame_stats['count'] / max(1, frames):>11.2f} {frame_stats['mean']:>9.3f} "
            f"{frame_stats['median']:>10.3f} {frame_stats['p95']:>8.3f} "
            f"{frame_stats['total'] / loop_total * 100 if loop_total else 0:>6.1f}%   (process_frame)"
        )
    work = loop_total / max(1, frames)
    print(f"\nwork per loop iteration: {work:.2f} ms ({frames} processed frames)")

    if len(timer.frame_times) > 1:
        span = timer.frame_times[-1] - timer.frame_times[0]
        rate = (len(timer.frame_times) - 1) / span if span > 0 else 0.0
        interval = span / (len(timer.frame_times) - 1) * 1000.0
        print(
            f"frame interval: {interval:.1f} ms ({rate:.2f} processed frames/second, measured between "
            "frame starts inside the loop)"
        )
        if synthetic:
            print(
                "verdict: synthetic frames — the loop is work-bound by construction (frames arrive "
                "instantly); on the real camera the ~33 ms frame interval dominates instead"
            )
        elif interval > work * 1.5:
            print(
                f"verdict: camera-bound — the interval is {interval / work:.1f}x the work, so the frame "
                "arrival rate sets the speed"
            )
        else:
            print("verdict: work-bound — the pipeline's own stages set the speed, not the camera")
    print(f"wall clock: {elapsed:.2f} s (includes camera open, MediaPipe start-up and teardown)")
    print(
        f"output calls: move={calls['move']} click={calls['click']} scroll={calls['scroll']} "
        f"hscroll={calls['hscroll']} (counted, not sent: nothing on the desktop moved)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile the kinematic control loop's stages.")
    parser.add_argument("--frames", type=int, default=120, help="processed frames to profile (default 120)")
    parser.add_argument("--synthetic-hand", action="store_true", help="use a scripted hand (no camera)")
    parser.add_argument(
        "--synthetic-frames",
        action="store_true",
        help="also synthesise the frames, so the run needs no camera at all (implied by --synthetic-hand)",
    )
    parser.add_argument("--active", action="store_true", help="start unpaused (default: paused, as shipped)")
    parser.add_argument("--preview", action="store_true", help="profile with the preview window drawn")
    parser.add_argument("--pointer-mode", choices=("relative", "absolute"), default=None)
    parser.add_argument("--pointer-reference", choices=("palm", "wrist", "index"), default=None)
    parser.add_argument("--smoothing-mode", choices=("ema", "one_euro"), default=None)
    parser.add_argument("--gesture-engine", choices=("recognizer", "legacy"), default=None)
    parser.add_argument(
        "--mouse-pause",
        type=float,
        default=None,
        help="pyautogui's own per-call pause; 0.1 (its default) is what a slow adapter costs the loop",
    )
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument(
        "--cprofile",
        type=int,
        default=0,
        metavar="N",
        help="also run under cProfile and print the N slowest functions by cumulative time",
    )
    args = parser.parse_args()

    overrides: Dict[str, Any] = {
        "show_preview": bool(args.preview),
        # The profiler measures the *work*: the preview window itself is a display cost, not a
        # pipeline one, and it is timed separately when it is on.
        "start_paused": not args.active,
    }
    if args.pointer_mode is not None:
        overrides["pointer_mode"] = args.pointer_mode
    if args.pointer_reference is not None:
        overrides["pointer_reference"] = args.pointer_reference
    if args.smoothing_mode is not None:
        overrides["smoothing_mode"] = args.smoothing_mode
    if args.gesture_engine is not None:
        overrides["gesture_engine"] = args.gesture_engine
    if args.mouse_pause is not None:
        overrides["mouse_pause"] = args.mouse_pause
    if not args.synthetic_hand:
        overrides["camera_index"] = args.camera
    config = HandMouseConfig(**overrides)

    calls = intercept_mouse()
    app = HandMouseApp(config)
    runtime = app.runtime

    synthetic = args.synthetic_hand or args.synthetic_frames
    camera: Any = None
    if synthetic:
        camera = synthetic_camera(args.frames + 8, config.capture_width, config.capture_height)
    if args.synthetic_hand:
        runtime.hands = SyntheticHands(max(2, args.frames))

    timer = StageTimer()
    originals = build_instrumentation(timer, runtime)

    # The camera side: with a synthetic device the frames are free, but a *real* camera is opened here
    # (rather than by the runtime) so its read() can be timed and it can be released by this tool.
    camera_owner: Any = None
    if synthetic:
        camera_owner = SimpleNamespace(
            read=camera.read, wait_ready=lambda timeout=3.0: True, error=None, release=camera.release
        )
    else:
        from kinematic.capture import ThreadedCamera

        real_camera = ThreadedCamera(
            index=config.camera_index,
            width=config.capture_width,
            height=config.capture_height,
            fps=config.capture_fps,
            buffer_size=config.capture_buffer_size,
        )
        camera_owner = real_camera  # released in the finally below
    camera_owner.read = timer.wrap("capture", camera_owner.read)

    def profiled_run() -> None:
        runtime.run(camera=camera_owner, max_frames=args.frames)

    started = time.perf_counter()
    try:
        if args.cprofile:
            profiler = cProfile.Profile()
            profiler.enable()
            profiled_run()
            profiler.disable()
            buffer = io.StringIO()
            stats = pstats.Stats(profiler, stream=buffer).sort_stats("cumulative")
            stats.print_stats(args.cprofile)
            print(buffer.getvalue())
        else:
            profiled_run()
    finally:
        elapsed = time.perf_counter() - started
        restore_instrumentation(runtime, originals)
        if not synthetic:
            camera_owner.release()
        app.close()

    title = (
        f"{'synthetic hand, ' if args.synthetic_hand else ''}"
        f"{'synthetic frames' if synthetic else f'camera {args.camera}'}"
        f", {'active' if args.active else 'paused'}"
        f"{', preview' if args.preview else ''}"
        f"{f', mouse pause {args.mouse_pause:g}s' if args.mouse_pause else ''}"
        f", {args.frames} frames"
    )
    report(timer, calls, elapsed, args.frames, title, synthetic=synthetic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
