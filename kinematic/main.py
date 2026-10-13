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
from .diagnostics import check_accessibility, diagnose, format_report, report_fields
from .log import configure as configure_log
from .log import get_logger, log_event
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
    parser.add_argument(
        "--preview-flip",
        choices=("horizontal", "none", "vertical", "both"),
        default=None,
        help="how the preview footage is flipped for display: horizontal (default, the mirror a webcam "
        "selfie shows and the view the pointer follows), none (exactly what the camera delivers), "
        "vertical or both (camera mounted upside down)",
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
    parser.add_argument(
        "--move-max-hz",
        type=float,
        default=None,
        help="cap cursor moves at this many per second, always sending the newest position (default 0 = no cap: the loop is camera-bound and already emits one move per frame)",
    )
    parser.add_argument(
        "--scroll-max-hz",
        type=float,
        default=None,
        help="cap scroll events at this many per second; held-back amounts are accumulated, never dropped (default 0 = no cap)",
    )
    parser.add_argument(
        "--scroll-horizontal-gain",
        type=float,
        default=None,
        help="horizontal scroll units per hand width (default 900); 0 disables sideways scrolling",
    )
    parser.add_argument(
        "--no-horizontal-scroll",
        action="store_true",
        help="never scroll sideways: the pre-Phase-7 behaviour, where only the vertical component was emitted",
    )
    parser.add_argument(
        "--invert-scroll-vertical",
        action="store_true",
        help="flip the vertical scroll direction (default: hand down scrolls the page down)",
    )
    parser.add_argument(
        "--invert-scroll-horizontal",
        action="store_true",
        help="flip the horizontal scroll direction if it feels backwards",
    )
    parser.add_argument(
        "--gesture-pause",
        action="store_true",
        help="let a gesture pause the pointer: hold an open hand (all four fingers straight) for --gesture-pause-hold seconds. Off by default",
    )
    parser.add_argument(
        "--gesture-pause-hold",
        type=float,
        default=None,
        help="seconds the open hand must be held before the gesture pause toggles (default 1.2)",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="check this machine (camera, screen, MediaPipe, permissions) and exit - moves nothing, opens no window",
    )
    parser.add_argument(
        "--no-camera-probe",
        action="store_true",
        help="with --diagnose: skip the camera check (it is the only check that turns the camera light on)",
    )
    parser.add_argument(
        "--log-level",
        choices=("debug", "info", "warning", "error", "critical"),
        default=None,
        help="structured (JSON-lines) log level; the pipeline logs at info/debug, so the default warning prints nothing",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="append the JSON-lines log to this file instead of stderr (one JSON object per line)",
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
    if args.preview_flip is not None:
        overrides["preview_flip"] = args.preview_flip
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
    if args.move_max_hz is not None:
        overrides["move_max_hz"] = args.move_max_hz
    if args.scroll_max_hz is not None:
        overrides["scroll_max_hz"] = args.scroll_max_hz
    if args.scroll_horizontal_gain is not None:
        overrides["scroll_horizontal_gain"] = args.scroll_horizontal_gain
    if args.no_horizontal_scroll:
        overrides["scroll_horizontal_gain"] = 0.0
    if args.invert_scroll_vertical:
        overrides["scroll_invert_vertical"] = True
    if args.invert_scroll_horizontal:
        overrides["scroll_invert_horizontal"] = True
    if args.gesture_pause:
        overrides["gesture_pause_enabled"] = True
    if args.gesture_pause_hold is not None:
        overrides["gesture_pause_hold_seconds"] = args.gesture_pause_hold
    if args.log_level is not None:
        overrides["log_level"] = args.log_level
    if args.log_file is not None:
        overrides["log_file"] = args.log_file
    return HandMouseConfig(**overrides)


def run_diagnostics(config: HandMouseConfig, include_camera: bool = True) -> int:
    """``--diagnose``: check this machine and print the report.

    It never opens the preview window and never moves the pointer; the camera check is the only part
    that touches hardware and ``--no-camera-probe`` skips it. Exit status is 0 when every check
    passed, so a script can gate on it.
    """
    app = HandMouseApp(config)
    runtime = app.runtime
    try:
        checks = diagnose(
            config=config,
            output=runtime.output,
            hands=runtime.hands,
            camera=include_camera,
        )
    finally:
        app.close()
    print(format_report(checks))
    log_event(get_logger("main"), "diagnostics", **report_fields(checks))
    return 0 if all(check.ok for check in checks) else 1


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    # Configure the structured log before anything else, so a session's very first line is logged.
    configure_log(config.log_level, config.log_file or None)

    if args.diagnose:
        return run_diagnostics(config, include_camera=not args.no_camera_probe)

    app = HandMouseApp(config)
    runtime = app.runtime
    logger = get_logger("main")

    print(f"kinematic: output={type(runtime.output).__name__} status={runtime.status}")
    if config.log_level != "warning" or config.log_file:
        print(
            f"kinematic: logging level={config.log_level} -> "
            f"{config.log_file if config.log_file else 'stderr'} (JSON lines)"
        )
    print(
        f"kinematic: pointer mode={runtime.config.pointer_mode} "
        f"reference={runtime.config.pointer_reference} "
        f"sensitivity={runtime.config.pointer_sensitivity} "
        f"smoothing={runtime.config.smoothing_mode}"
        + (
            f"(min_cutoff={runtime.config.one_euro_min_cutoff} beta={runtime.config.one_euro_beta})"
            if runtime.config.smoothing_mode == "one_euro"
            else f"(factor={runtime.config.smoothing})"
        )
    )
    # The two deadzones are different things and were both printed as "deadzone" once, which is how a
    # 49 px freeze in absolute mode went unnoticed: `pointer_deadzone` is the *relative* mode's hand
    # displacement filter, `deadzone` is the absolute mode's reference hold.
    if runtime.config.pointer_mode == "absolute":
        px = runtime.config.deadzone / max(1e-9, 1.0 - 2 * runtime.config.frame_margin) * runtime.screen_w
        print(f"kinematic: absolute mode reference hold={runtime.config.deadzone} ({px:.1f} px of screen), "
              f"frame margin={runtime.config.frame_margin}")
    else:
        print(f"kinematic: relative mode hand deadzone={runtime.config.pointer_deadzone} (displacement below it is held back)")
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
        f"{runtime.config.sensitivity_step}x per turn, reach {runtime.reach:.2f} screens per frame width)"
    )
    print("kinematic: turn the dial with ] (faster) and [ (slower); 0 resets it to where you started")
    horizontal = runtime.config.scroll_horizontal_gain
    print(
        f"kinematic: two-finger scroll vertical gain={runtime.config.scroll_gain} "
        f"horizontal={'OFF (pre-Phase-7)' if horizontal == 0 else f'gain={horizontal}'} "
        f"deadzone={runtime.config.scroll_deadzone_ratio} hand widths "
        f"invert v/h={runtime.config.scroll_invert_vertical}/{runtime.config.scroll_invert_horizontal} "
        f"{'events capped at ' + format(runtime.config.scroll_max_hz, 'g') + ' Hz' if runtime.config.scroll_max_hz else 'no event cap'}"
    )
    if runtime.config.move_max_hz > 0:
        print(
            f"kinematic: cursor moves capped at {runtime.config.move_max_hz:g} Hz "
            "(the newest position wins; measure the cost with tools/measure_smoothness.py --rate-sweep)"
        )
    else:
        print("kinematic: cursor moves not capped (one per frame; --move-max-hz N to cap them)")
    if runtime.config.gesture_pause_enabled:
        print(
            f"kinematic: gesture pause ON - hold an open hand ({runtime.config.gesture_pause_fingers} "
            f"fingers straight, no pinch) for {runtime.config.gesture_pause_hold_seconds}s to pause or "
            "resume; SPACE still works"
        )
    else:
        print("kinematic: gesture pause off (--gesture-pause to hold an open hand and pause)")
    if runtime.config.show_last_action:
        print("kinematic: the preview shows the last click/scroll emitted, which is also how the dry run shows it")
    print(
        f"kinematic: preview flip={runtime.config.preview_flip} "
        "(horizontal = the webcam mirror the pointer follows; none = exactly what the camera delivers)"
    )
    if runtime.sensitivity_clamped_from is not None:
        print(
            f"kinematic: requested sensitivity {runtime.sensitivity_clamped_from} is below the reach "
            f"floor {runtime.config.sensitivity_min} - using {runtime.sensitivity:.2f}; below that the "
            "screen edges cannot be reached"
        )
    if runtime.reach_warning:
        print(f"kinematic: warning - {runtime.reach_warning}")
    if runtime.paused:
        print("kinematic: paused at startup - press SPACE in the preview window to control the mouse, ESC or q to quit")
    if isinstance(runtime.output, RecordingMouseOutput):
        print("kinematic: dry run - no mouse output will be emitted")
    else:
        # Diagnosed *before* the user wonders why nothing moves: without Accessibility on macOS,
        # pyautogui's calls are silently ignored (AGENTS.md gotcha 8.5), which looks like a bug in
        # the mapping. Read-only question, and a real adapter is the only case where it matters.
        access = check_accessibility()
        if not access.ok:
            print(f"kinematic: warning - {access.detail}")
            print(f"kinematic: {access.hint}")

    log_event(
        logger,
        "session_start",
        output=type(runtime.output).__name__,
        status=runtime.status,
        dry_run=isinstance(runtime.output, RecordingMouseOutput),
        sensitivity=runtime.sensitivity,
        gesture_pause=runtime.config.gesture_pause_enabled,
        log_level=config.log_level,
    )

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
            f"kinematic: pause toggles={summary['pause_toggles']} "
            f"(by gesture {summary['gesture_pause_toggles']}, last source {summary['pause_source']}) "
            f"tracking losses={summary['tracking_losses']}"
        )
        print(
            f"kinematic: sensitivity ended at {summary['sensitivity']} "
            f"(started at {summary['sensitivity_start']}, turned {summary['sensitivity_changes']}x)"
        )
        if summary["recorded"]:
            print(f"kinematic: recorded (not sent) {summary['recorded']}")
        if summary["actions"]:
            print(
                f"kinematic: actions emitted {summary['actions']} (last: {summary['last_action']}, "
                f"cursor moves thinned by the rate limit: {summary['moves_suppressed']})"
            )
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
