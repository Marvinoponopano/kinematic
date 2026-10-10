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
    return HandMouseConfig(**overrides)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    app = HandMouseApp(build_config(args))
    runtime = app.runtime

    print(f"kinematic: output={type(runtime.output).__name__} status={runtime.status}")
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
        if summary["recorded"]:
            print(f"kinematic: recorded (not sent) {summary['recorded']}")
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
