"""Runtime diagnostics (Phase 9): can this machine run the hand mouse, and if not, why not.

Everything here is **read-only**. No cursor movement, no clicks, no preview window, no permission
prompt: the Accessibility check asks the OS a question (``AXIsProcessTrusted``) rather than trying to
move the pointer and seeing whether it worked. The camera probe is the one part that touches hardware
— opening a device turns on its light and takes a moment to negotiate — so the CLI runs it only for
``--diagnose``, and failures are reported as text with a hint rather than raised.

The report is deliberately printable *and* structured: :func:`format_report` is what a user reads,
:func:`report_fields` is what the JSON log gets, and both come from the same list of
:class:`Check` objects.
"""

import ctypes
import ctypes.util
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import HandMouseConfig
from .output import RecordingMouseOutput


@dataclass(frozen=True)
class Check:
    """One diagnostic result: what was checked, whether it passed, and what to do about it."""

    name: str
    ok: bool
    detail: str
    hint: str = ""

    @property
    def status(self) -> str:
        return "ok" if self.ok else "FAIL"


@dataclass(frozen=True)
class CameraProbe:
    """What the camera check measured, so a caller can reuse the numbers."""

    opened: bool
    width: int = 0
    height: int = 0
    frame_interval_ms: float = 0.0
    brightness: float = 0.0
    hand_found: bool = False
    error: str = ""


# -- individual checks --------------------------------------------------------


def check_platform() -> Check:
    """Which OS and interpreter this is — the first thing to know when something is denied."""
    detail = f"{platform.system()} {platform.release()} · Python {platform.python_version()}"
    ok = sys.version_info >= (3, 9)
    hint = "" if ok else "Python 3.9 or newer is required (dataclasses, typing)."
    return Check("platform", ok, detail, hint)


def check_mediapipe(hands: Optional[Any] = None) -> Check:
    """Import MediaPipe *and* run one frame through the legacy Solutions API.

    Importing is not enough: MediaPipe 0.10.30+ dropped ``mp.solutions`` (see AGENTS.md gotcha 8.2),
    so a fresh install of an unpinned version imports fine and then cannot open a hand tracker. This
    actually constructs ``Hands()`` and processes a blank frame, which is ~10 ms and touches nothing.
    Pass ``hands`` to reuse an already-built tracker (what the app itself will use) instead of paying
    for a second construction.
    """
    try:
        import mediapipe as mp  # noqa: F401  (imported for the version and the API check)
    except Exception as exc:  # pragma: no cover - depends on the environment
        return Check("mediapipe", False, f"import failed: {exc}", "pip install -r requirements.txt")

    version = getattr(mp, "__version__", "unknown")
    if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "hands"):
        return Check(
            "mediapipe",
            False,
            f"{version} has no mp.solutions.hands (the legacy API was removed)",
            "pin it: mediapipe==0.10.14 (or <=0.10.21), or migrate to the Tasks API.",
        )
    try:
        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        if hands is not None:
            hands.process(blank)
        else:
            with mp.solutions.hands.Hands(max_num_hands=1, model_complexity=0) as created:
                created.process(blank)
    except Exception as exc:  # pragma: no cover - depends on the environment
        return Check("mediapipe", False, f"{version} failed on a blank frame: {exc}", "reinstall mediapipe.")
    return Check("mediapipe", True, f"{version} · mp.solutions.hands processes a blank frame")


def accessibility_probe() -> Optional[bool]:
    """Is this process trusted to post events? ``None`` when the platform cannot answer.

    On macOS the answer comes from ``AXIsProcessTrusted()`` in ApplicationServices, loaded through
    ctypes: a read-only question, so no pointer movement and no prompts. Other platforms return
    ``None`` (unknown) rather than inventing an answer.
    """
    if platform.system() != "Darwin":
        return None
    path = ctypes.util.find_library("ApplicationServices")
    if not path:
        return None
    try:
        library = ctypes.cdll.LoadLibrary(path)
        library.AXIsProcessTrusted.restype = ctypes.c_bool
        return bool(library.AXIsProcessTrusted())
    except Exception:
        return None


def check_accessibility(probe: Optional[Callable[[], Optional[bool]]] = None) -> Check:
    """macOS Accessibility permission: required before PyAutoGUI can move the pointer."""
    probe = accessibility_probe if probe is None else probe
    try:
        trusted = probe()
    except Exception as exc:  # pragma: no cover - defensive
        return Check("accessibility", True, f"could not be determined ({exc})")

    if trusted is None:
        return Check("accessibility", True, "not applicable on this platform")
    if trusted:
        return Check("accessibility", True, "granted to this process")
    return Check(
        "accessibility",
        False,
        "not granted to this process — PyAutoGUI's calls are silently ignored until it is",
        "System Settings -> Privacy & Security -> Accessibility -> enable your Terminal/IDE, then restart it.",
    )


def check_screen(output: Optional[Any] = None) -> Check:
    """Screen size, read through the output adapter (``--dry-run`` reports a nominal 1x1 screen).

    ``output`` is any ``MouseOutput``-shaped object — the real adapter, the recording one, or a test
    double — because all this needs is ``screen_size()``.
    """
    if output is None:
        return Check("screen", True, "not measured (no output adapter)")
    try:
        width, height = output.screen_size()
    except Exception as exc:
        return Check("screen", False, f"could not be read: {exc}", "check the display/Quartz session.")
    detail = f"{width} x {height} via {type(output).__name__}"
    if isinstance(output, RecordingMouseOutput):
        # --dry-run has no real display to read, so its size is whatever the adapter was constructed
        # with: say so rather than implying the screen was measured.
        return Check("screen", True, f"{detail} (recording adapter: nominal size)")
    if width <= 1 or height <= 1:
        return Check("screen", True, f"{detail} (nominal)")
    return Check("screen", True, detail)


def check_output(output: Optional[Any] = None) -> Check:
    """Which adapter this run would use, and whether the real one is reachable at all."""
    if output is None:
        return Check("output", True, "not measured (no output adapter)")
    name = type(output).__name__
    if name == "RecordingMouseOutput":
        return Check("output", True, "RecordingMouseOutput (dry run: nothing reaches the desktop)")
    detail = f"{name} (live pointer control)"
    return Check("output", True, detail)


def check_camera(
    config: Optional[HandMouseConfig] = None,
    opener: Optional[Callable[..., Any]] = None,
    hands: Optional[Any] = None,
) -> Tuple[Check, CameraProbe]:
    """Open the camera, read one frame, and report what came back. Returns (Check, CameraProbe).

    A permission problem and a missing device look different on purpose: OpenCV reports
    "not authorized to capture video" when macOS has not granted Camera access, and that is a
    different fix from a camera index that does not exist.
    """
    config = config or HandMouseConfig()
    opened = False
    width = height = 0
    interval_ms = 0.0
    brightness = 0.0
    hand_found = False

    try:
        if opener is not None:
            capture = opener(config.camera_index)
        else:  # pragma: no cover - real hardware
            from .capture import ThreadedCamera

            capture = ThreadedCamera(
                index=config.camera_index,
                width=config.capture_width,
                height=config.capture_height,
                fps=config.capture_fps,
                buffer_size=config.capture_buffer_size,
            )
    except Exception as exc:
        message = str(exc)
        hint = "Camera access: System Settings -> Privacy & Security -> Camera -> enable your Terminal/IDE."
        if "authorized" in message.lower() or "permission" in message.lower():
            hint = "macOS has not granted Camera access to this process. " + hint
        else:
            hint = f"Is a camera connected and is index {config.camera_index} the right one? " + hint
        return Check("camera", False, f"could not open index {config.camera_index}: {message}", hint), CameraProbe(
            opened=False, error=message
        )

    try:
        frame: Any = None
        started = time.monotonic()
        read = getattr(capture, "read", None)
        wait_ready = getattr(capture, "wait_ready", None)
        if callable(wait_ready):
            wait_ready(timeout=3.0)
        if callable(read):
            for _ in range(30):  # a just-opened camera needs a few frames to reach full exposure
                result = read()
                # A ThreadedCamera hands back (sequence, frame); a plain cv2.VideoCapture hands back
                # (ok, frame); a test double may hand back just a frame. Accept all three.
                if isinstance(result, tuple):
                    frame = result[-1]
                else:
                    frame = result
                if frame is not None:
                    break
                time.sleep(0.01)
        interval_ms = (time.monotonic() - started) * 1000.0
        if frame is None:
            return Check(
                "camera",
                False,
                f"opened index {config.camera_index} but no frame arrived",
                "the device may be in use by another app (Zoom, Photo Booth, another Kinematic).",
            ), CameraProbe(opened=True, error="no frame")
        opened = True
        height, width = int(frame.shape[0]), int(frame.shape[1])
        brightness = float(np.asarray(frame).mean())
        if hands is not None:
            result = hands.process(cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2RGB))
            hand_found = bool(getattr(result, "multi_hand_landmarks", None))
    finally:
        release = getattr(capture, "release", None)
        if callable(release):
            release()

    detail = f"{width}x{height}, first frame {interval_ms:.0f} ms, mean brightness {brightness:.0f}/255"
    detail += ", hand in view" if hand_found else ", no hand in view"
    if brightness < 30.0:
        return (
            Check(
                "camera",
                True,
                detail,
                "very dark: MediaPipe needs light to find a hand (add a lamp or raise the exposure).",
            ),
            CameraProbe(True, width, height, interval_ms, brightness, hand_found),
        )
    return (
        Check("camera", True, detail),
        CameraProbe(True, width, height, interval_ms, brightness, hand_found),
    )


def check_configuration(config: Optional[HandMouseConfig] = None) -> Check:
    """The effective settings that decide how the app behaves — the first line of any bug report."""
    config = config or HandMouseConfig()
    detail = " · ".join(
        [
            f"pointer {config.pointer_mode}/{config.pointer_reference}",
            f"sensitivity {config.pointer_sensitivity:g}",
            f"smoothing {config.smoothing_mode}",
            f"gestures {config.gesture_engine}",
            f"gesture-pause {'on' if config.gesture_pause_enabled else 'off'}",
            f"move cap {config.move_max_hz:g} Hz",
            f"scroll cap {config.scroll_max_hz:g} Hz",
            f"preview {'on' if config.show_preview else 'off'}/{config.preview_flip}",
            f"start {'paused' if config.start_paused else 'active'}",
            f"log {config.log_level}",
        ]
    )
    return Check("configuration", True, detail)


# -- the report ---------------------------------------------------------------


def diagnose(
    config: Optional[HandMouseConfig] = None,
    output: Optional[Any] = None,
    hands: Optional[Any] = None,
    opener: Optional[Callable[..., Any]] = None,
    trusted: Optional[Callable[[], Optional[bool]]] = None,
    camera: bool = True,
) -> List[Check]:
    """Run every check in the order a user would want to read them.

    ``hands`` and ``opener`` let a caller (and the test suite) supply a hand tracker and a camera
    without touching hardware; ``camera=False`` skips the only check that opens a device.
    """
    config = config or HandMouseConfig()
    checks = [
        check_platform(),
        check_mediapipe(hands=hands),
        check_output(output),
        check_screen(output),
        check_accessibility(probe=trusted),
        check_configuration(config),
    ]
    if camera:
        camera_check, _probe = check_camera(config=config, opener=opener, hands=hands)
        checks.append(camera_check)
    return checks


def format_report(checks: Sequence[Check], title: str = "Kinematic diagnostics") -> str:
    """Human-readable report: one block per check, with its advice on the line underneath.

    A hint is printed whether or not the check passed: a camera that opened in a dark room passes and
    still needs to say so, which is the whole point of the brightness reading.
    """
    lines = [title, "=" * len(title), ""]
    for check in checks:
        lines.append(f"[{check.status:>4}] {check.name:<14} {check.detail}")
        if check.hint:
            lines.append(f"          -> {check.hint}")
    failures = [check for check in checks if not check.ok]
    lines.append("")
    if failures:
        lines.append(f"{len(failures)} check(s) failed: {', '.join(check.name for check in failures)}")
        lines.append("Fix those before expecting the pointer to move; see README 'Quick Start' for the long form.")
    else:
        lines.append("All checks passed. The app can run: `python3 -m kinematic.main` (starts paused).")
    return "\n".join(lines)


def report_fields(checks: Sequence[Check]) -> Dict[str, Any]:
    """The same result as structured fields, for the JSON log."""
    return {
        "checks": {check.name: check.status for check in checks},
        "failed": [check.name for check in checks if not check.ok],
    }
