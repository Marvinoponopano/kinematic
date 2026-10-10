"""Entry point for the Kinematic hand-tracking mouse controller.

The pipeline itself lives in :mod:`kinematic.runtime` (coordinator), :mod:`kinematic.capture`
(camera) and :mod:`kinematic.output` (mouse adapters). This module is the command line
plus a small backwards-compatible facade, and it is the only place that decides which
output adapter the app runs with.

    python3 -m kinematic.main            # starts PAUSED; SPACE controls the mouse, ESC/q quits
    python3 -m kinematic.main --active   # start controlling the mouse straight away
    python3 -m kinematic.main --dry-run  # run everything, emit nothing
"""

# Support both invocation styles: as a package module (`python3 -m kinematic.main`)
# and as a plain script (`python3 kinematic/main.py`). A script has no parent
# package, so register the project root on sys.path and give this module a package
# identity, which is what the relative imports are relative to.
if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "kinematic"

import argparse
from typing import Any, Optional

from .capture import ThreadedCamera
from .config import HandMouseConfig
from .output import MouseOutput, RecordingMouseOutput
from .runtime import HandMouseRuntime


class HandMouseApp:
    """Backwards-compatible entry point for the pre-Phase-1 API.

    ``HandMouseApp(config)``, ``app.run()`` and the ``_update_cursor`` /
    ``_handle_clicks`` / ``_handle_scroll`` hooks all keep working; they now delegate
    to :class:`kinematic.runtime.HandMouseRuntime`, which owns the pipeline and the
    pause state. Attributes that used to live on this class (``config``, ``hands``,
    ``paused``, ``cursor_x``, ``loop_fps``, ...) are read through to the runtime.
    """

    def __init__(
        self,
        config: Optional[HandMouseConfig] = None,
        output: Optional[MouseOutput] = None,
    ) -> None:
        self.runtime = HandMouseRuntime(config=config, output=output)

    def __getattr__(self, name: str) -> Any:
        # Only called when normal attribute lookup fails; delegate to the runtime.
        # The guard stops recursion while __init__ is still running.
        if name == "runtime":
            raise AttributeError(name)
        return getattr(self.runtime, name)

    def _update_cursor(self, index_tip: tuple) -> None:
        self.runtime.update_cursor(index_tip)

    def _handle_clicks(self, index_tip: tuple, thumb_tip: tuple) -> None:
        self.runtime.handle_clicks(index_tip, thumb_tip)

    def _handle_scroll(self, index_tip: tuple) -> None:
        self.runtime.handle_scroll(index_tip)

    def run(self, **kwargs: Any) -> None:
        self.runtime.run(**kwargs)

    def close(self) -> None:
        self.runtime.close()


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="kinematic",
        description="Webcam hand-gesture mouse controller. Starts paused; press SPACE to control the mouse.",
    )
    parser.add_argument(
        "--active",
        action="store_true",
        help="start controlling the mouse immediately instead of paused",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run the whole pipeline with a recording output adapter: no mouse output, no Accessibility permission needed",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="do not open the preview window (quit with Ctrl-C instead of q)",
    )
    parser.add_argument("--camera", type=int, default=None, help="camera index to open (default: 0)")
    parser.add_argument(
        "--pointer-mode",
        choices=("relative", "absolute"),
        default=None,
        help="relative (default): the cursor follows hand displacement; absolute: the cursor maps to the hand's position in the frame",
    )
    parser.add_argument(
        "--pointer-reference",
        choices=("palm", "wrist", "index"),
        default=None,
        help="hand point the cursor follows: palm (default, wrist + knuckles), wrist (most stable), index (legacy fingertip)",
    )
    parser.add_argument(
        "--smoothing-mode",
        choices=("ema", "one_euro"),
        default=None,
        help="cursor filter: one_euro (default, less lag at the same jitter) or ema (the previous fixed-factor filter)",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=None,
        help="pointer gain to start with: screen widths of cursor travel per frame width of hand travel (default 5.0)",
    )
    parser.add_argument(
        "--sensitivity-step",
        type=float,
        default=None,
        help="how much one turn of the dial changes the gain, as a multiplier (default 1.15); turn it with ] and [ in the preview, 0 resets",
    )
    parser.add_argument(
        "--gesture-engine",
        choices=("recognizer", "legacy"),
        default=None,
        help="gesture logic: recognizer (default, relative thresholds + state machine) or legacy (the pre-Phase-4 inline pinch/scroll tests)",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> HandMouseConfig:
    overrides: dict = {}
    if args.dry_run:
        overrides["dry_run"] = True
    if args.active:
        overrides["start_paused"] = False
    if args.no_preview:
        overrides["show_preview"] = False
    if args.camera is not None:
        overrides["camera_index"] = args.camera
    if args.pointer_mode is not None:
        overrides["pointer_mode"] = args.pointer_mode
    if args.pointer_reference is not None:
        overrides["pointer_reference"] = args.pointer_reference
    if args.smoothing_mode is not None:
        overrides["smoothing_mode"] = args.smoothing_mode
    if args.gesture_engine is not None:
        overrides["gesture_engine"] = args.gesture_engine
    if args.sensitivity is not None:
        overrides["pointer_sensitivity"] = args.sensitivity
    if args.sensitivity_step is not None:
        overrides["sensitivity_step"] = args.sensitivity_step
    return HandMouseConfig(**overrides)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    app = HandMouseApp(build_config(args))
    runtime = app.runtime

    print(f"kinematic: output={type(runtime.output).__name__} status={runtime.status}")
    print(
        f"kinematic: pointer mode={runtime.config.pointer_mode} "
        f"reference={runtime.config.pointer_reference} "
        f"sensitivity={runtime.config.pointer_sensitivity} "
        f"deadzone={runtime.config.pointer_deadzone} "
        f"smoothing={runtime.config.smoothing_mode}"
    )
    print(
        f"kinematic: gestures engine={runtime.config.gesture_engine} "
        f"pinch_ratio={runtime.config.pinch_ratio} "
        f"hold={runtime.config.pinch_hold_seconds}s "
        f"cooldown={runtime.config.click_cooldown_seconds}s "
        f"scroll_gain={runtime.config.scroll_gain}"
    )
    print(
        f"kinematic: sensitivity {runtime.sensitivity:.2f} "
        f"(range {runtime.config.sensitivity_min}-{runtime.config.sensitivity_max}, "
        f"{runtime.config.sensitivity_step}x per turn)"
    )
    print("kinematic: turn the dial with ] (faster) and [ (slower); 0 resets it to where you started")
    if runtime.paused:
        print("kinematic: paused at startup - press SPACE in the preview window to control the mouse, ESC or q to quit")
    if isinstance(runtime.output, RecordingMouseOutput):
        print("kinematic: dry run - no mouse output will be emitted")

    try:
        app.run()
    except KeyboardInterrupt:
        print("\nkinematic: interrupted")
    finally:
        summary = runtime.summary()
        print(
            f"kinematic: frames={summary['frames']} output_calls={summary['emitted']} "
            f"loop_fps={summary['loop_fps']} status={summary['status']}"
        )
        print(
            f"kinematic: sensitivity ended at {summary['sensitivity']} "
            f"(started at {summary['sensitivity_start']}, turned {summary['sensitivity_changes']}x)"
        )
        if summary["recorded"]:
            print(f"kinematic: recorded (not sent) {summary['recorded']}")
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
