"""Phase 9 diagnostics: the checks that answer "why does nothing move?".

The phase asks for diagnostics for camera access, MediaPipe startup, screen size and the macOS
Accessibility permission. Every check here is exercised through an injected probe — a fake camera
opener, a fake hand tracker, a fake permission answer — so the suite needs no hardware, no display and
no permissions, and the *denied* cases (the interesting ones) can be tested at all.

The one real call is :func:`accessibility_probe`, which is read-only: it asks the OS a question rather
than moving the pointer and seeing whether it moved.
"""

import contextlib
import io
import platform
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest import mock

import numpy as np

import kinematic.main as main_module
from kinematic.config import HandMouseConfig
from kinematic.diagnostics import (
    Check,
    accessibility_probe,
    check_accessibility,
    check_camera,
    check_configuration,
    check_mediapipe,
    check_output,
    check_platform,
    check_screen,
    diagnose,
    format_report,
    report_fields,
)
from kinematic.output import RecordingMouseOutput


class FakeCamera:
    """A camera that hands out frames (or nothing) and records that it was released."""

    def __init__(self, frames: Optional[Sequence[Any]] = None) -> None:
        self._frames = list(frames if frames is not None else [np.full((6, 8, 3), 200, dtype=np.uint8)])
        self.released = False

    @property
    def error(self) -> Optional[str]:
        return None

    def wait_ready(self, timeout: float = 3.0) -> bool:
        return bool(self._frames)

    def read(self) -> Tuple[int, Optional[np.ndarray]]:
        if not self._frames:
            return 0, None
        return 1, self._frames[0]

    def release(self) -> None:
        self.released = True


class HandsStub:
    def __init__(self, found: bool = False, explode: bool = False) -> None:
        self.found = found
        self.explode = explode

    def process(self, rgb: Any) -> Any:
        if self.explode:
            raise RuntimeError("tracker unavailable")
        landmarks = [SimpleNamespace(x=0.5, y=0.5, z=0.0)] if self.found else None
        return SimpleNamespace(multi_hand_landmarks=landmarks, multi_handedness=None)


class ScreenStub:
    def __init__(self, size: Tuple[int, int] = (1710, 1112), explode: bool = False) -> None:
        self._size = size
        self._explode = explode

    def screen_size(self) -> Tuple[int, int]:
        if self._explode:
            raise RuntimeError("no display session")
        return self._size


class PlatformTests(unittest.TestCase):
    def test_platform_reports_the_os_and_interpreter(self):
        check = check_platform()
        self.assertTrue(check.ok)
        self.assertIn(platform.system(), check.detail)
        self.assertIn(platform.python_version(), check.detail)


class MediaPipeTests(unittest.TestCase):
    def test_it_uses_the_supplied_tracker_instead_of_building_another(self):
        hands = HandsStub()
        check = check_mediapipe(hands=hands)
        self.assertTrue(check.ok)
        self.assertIn("mp.solutions.hands", check.detail)

    def test_a_tracker_that_fails_is_reported_with_a_fix(self):
        check = check_mediapipe(hands=HandsStub(explode=True))
        self.assertFalse(check.ok)
        self.assertIn("blank frame", check.detail)
        self.assertTrue(check.hint)


class AccessibilityTests(unittest.TestCase):
    def test_granted(self):
        self.assertTrue(check_accessibility(probe=lambda: True).ok)

    def test_denied_mentions_the_settings_pane_and_stays_read_only(self):
        check = check_accessibility(probe=lambda: False)
        self.assertFalse(check.ok)
        self.assertIn("silently ignored", check.detail)
        self.assertIn("Privacy & Security", check.hint)

    def test_unknown_platform_is_not_a_failure(self):
        # Not knowing is not the same as failing: inventing a verdict would be worse than saying so.
        check = check_accessibility(probe=lambda: None)
        self.assertTrue(check.ok)
        self.assertIn("not applicable", check.detail)

    def test_a_probe_that_raises_does_not_break_the_report(self):
        def explode() -> Optional[bool]:
            raise RuntimeError("no frameworks")

        check = check_accessibility(probe=explode)
        self.assertTrue(check.ok)
        self.assertIn("could not be determined", check.detail)

    def test_the_real_probe_answers_without_moving_anything(self):
        # On macOS this is AXIsProcessTrusted(); elsewhere it must decline to answer rather than guess.
        answer = accessibility_probe()
        self.assertIn(answer, (None, True, False))
        if platform.system() != "Darwin":
            self.assertIsNone(answer)


class CameraTests(unittest.TestCase):
    def test_a_good_camera_reports_geometry_brightness_and_no_hand(self):
        camera = FakeCamera()
        check, probe = check_camera(opener=lambda index: camera, hands=HandsStub(found=False))
        self.assertTrue(check.ok)
        self.assertIn("8x6", check.detail)
        self.assertFalse(probe.hand_found)
        self.assertTrue(camera.released, "the probe must not leave the camera open")
        self.assertGreater(probe.brightness, 30.0)

    def test_a_hand_in_view_is_reported(self):
        check, probe = check_camera(opener=lambda index: FakeCamera(), hands=HandsStub(found=True))
        self.assertTrue(check.ok)
        self.assertTrue(probe.hand_found)
        self.assertIn("hand in view", check.detail)

    def test_a_dark_frame_is_flagged_as_low_light(self):
        dark = FakeCamera([np.zeros((6, 8, 3), dtype=np.uint8)])
        check, probe = check_camera(opener=lambda index: dark)
        self.assertTrue(check.ok, "a dark room is not a broken camera")
        self.assertLess(probe.brightness, 30.0)
        self.assertIn("very dark", check.hint)

    def test_a_permission_denial_gets_the_camera_settings_hint(self):
        def denied(index: int) -> Any:
            raise RuntimeError("OpenCV: not authorized to capture video")

        check, probe = check_camera(opener=denied)
        self.assertFalse(check.ok)
        self.assertIn("Privacy & Security -> Camera", check.hint)
        self.assertFalse(probe.opened)

    def test_a_missing_device_gets_a_different_hint(self):
        def missing(index: int) -> Any:
            raise RuntimeError(f"Could not open camera index {index}.")

        check, _probe = check_camera(opener=missing)
        self.assertFalse(check.ok)
        self.assertIn("Is a camera connected", check.hint)

    def test_a_camera_that_never_delivers_a_frame_is_reported(self):
        check, probe = check_camera(opener=lambda index: FakeCamera(frames=[]))
        self.assertFalse(check.ok)
        self.assertIn("no frame arrived", check.detail)
        self.assertTrue(probe.opened)


class ScreenAndOutputTests(unittest.TestCase):
    def test_screen_size_through_the_adapter(self):
        self.assertIn("1710 x 1112", check_screen(ScreenStub()).detail)

    def test_the_recording_adapter_says_its_size_is_nominal(self):
        check = check_screen(RecordingMouseOutput())
        self.assertTrue(check.ok)
        self.assertIn("nominal", check.detail)

    def test_an_unreadable_screen_is_a_failure_with_a_hint(self):
        check = check_screen(ScreenStub(explode=True))
        self.assertFalse(check.ok)
        self.assertTrue(check.hint)

    def test_missing_adapter_is_reported_as_not_measured(self):
        self.assertIn("not measured", check_screen(None).detail)
        self.assertIn("not measured", check_output(None).detail)

    def test_the_adapter_is_named_and_dry_run_is_called_out(self):
        self.assertIn("dry run", check_output(RecordingMouseOutput()).detail)
        self.assertIn("live pointer control", check_output(ScreenStub()).detail)


class ConfigurationTests(unittest.TestCase):
    def test_the_effective_settings_are_listed(self):
        config = HandMouseConfig(gesture_pause_enabled=True, preview_flip="none", log_level="info")
        detail = check_configuration(config).detail
        for fragment in ("absolute/palm", "gestures recognizer", "gesture-pause on", "preview on/none", "log info"):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, detail)

    def test_it_reads_the_defaults_with_no_config(self):
        self.assertIn("pointer absolute/palm", check_configuration().detail)


class ReportTests(unittest.TestCase):
    def test_a_passing_report_says_so(self):
        report = format_report([Check("a", True, "fine"), Check("b", True, "fine too")])
        self.assertIn("[  ok] a", report)
        self.assertIn("All checks passed", report)

    def test_a_failure_is_printed_with_its_hint_and_summarised(self):
        report = format_report([Check("camera", False, "not authorized", "enable Camera access")])
        self.assertIn("[FAIL] camera", report)
        self.assertIn("-> enable Camera access", report)
        self.assertIn("1 check(s) failed: camera", report)

    def test_a_passing_check_still_prints_its_advice(self):
        # The low-light case: the camera opened fine and the advice still matters.
        report = format_report([Check("camera", True, "1280x720, brightness 4/255", "very dark: add a lamp")])
        self.assertIn("[  ok] camera", report)
        self.assertIn("-> very dark: add a lamp", report)
        self.assertIn("All checks passed", report)

    def test_fields_are_structured_for_the_log(self):
        fields = report_fields([Check("a", True, "x"), Check("b", False, "y")])
        self.assertEqual(fields, {"checks": {"a": "ok", "b": "FAIL"}, "failed": ["b"]})


class DiagnoseTests(unittest.TestCase):
    def test_the_full_run_includes_the_camera_last(self):
        checks = diagnose(
            config=HandMouseConfig(),
            output=RecordingMouseOutput(),
            hands=HandsStub(),
            opener=lambda index: FakeCamera(),
            trusted=lambda: True,
        )
        names = [check.name for check in checks]
        self.assertEqual(names[-1], "camera")
        for expected in ("platform", "mediapipe", "output", "screen", "accessibility", "configuration"):
            self.assertIn(expected, names)
        self.assertTrue(all(check.ok for check in checks))

    def test_the_camera_can_be_skipped(self):
        checks = diagnose(output=RecordingMouseOutput(), hands=HandsStub(), trusted=lambda: True, camera=False)
        self.assertNotIn("camera", [check.name for check in checks])

    def test_a_denied_permission_makes_the_run_fail(self):
        checks = diagnose(
            output=RecordingMouseOutput(), hands=HandsStub(), opener=lambda index: FakeCamera(), trusted=lambda: False
        )
        failed = [check.name for check in checks if not check.ok]
        self.assertEqual(failed, ["accessibility"])


class DiagnoseCliTests(unittest.TestCase):
    """`--diagnose` wiring: exit status 0/1, a printed report, and nothing else touched."""

    class FakeApp:
        def __init__(self, config: HandMouseConfig, output: Optional[Any] = None) -> None:
            self.config = config
            self.closed = 0
            self.runtime = SimpleNamespace(output=output or RecordingMouseOutput(), hands=HandsStub())
            self.runtime.close = self.close

        def close(self) -> None:
            self.closed += 1

    def run_main(self, argv: List[str], checks: List[Check]) -> Tuple[int, str, "DiagnoseCliTests.FakeApp"]:
        app = self.FakeApp(HandMouseConfig())
        with mock.patch.object(main_module, "HandMouseApp", lambda config: app), mock.patch.object(
            main_module, "diagnose", lambda **kwargs: checks
        ):
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                status = main_module.main(argv)
        return status, buffer.getvalue(), app

    def test_a_passing_run_exits_zero_and_prints_the_report(self):
        status, output, app = self.run_main(["--diagnose", "--dry-run"], [Check("camera", True, "fine")])
        self.assertEqual(status, 0)
        self.assertIn("All checks passed", output)
        self.assertEqual(app.closed, 1, "diagnostics must release the hand tracker it borrowed")

    def test_a_failing_run_exits_one(self):
        checks = [Check("camera", True, "fine"), Check("accessibility", False, "not granted", "enable it")]
        status, output, _app = self.run_main(["--diagnose"], checks)
        self.assertEqual(status, 1)
        self.assertIn("1 check(s) failed: accessibility", output)

    def test_the_camera_probe_is_optional_on_the_command_line(self):
        captured: Dict[str, Any] = {}

        def capture(**kwargs: Any) -> List[Check]:
            captured.update(kwargs)
            return [Check("platform", True, "fine")]

        app = self.FakeApp(HandMouseConfig())
        with mock.patch.object(main_module, "HandMouseApp", lambda config: app), mock.patch.object(
            main_module, "diagnose", capture
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                main_module.main(["--diagnose", "--no-camera-probe"])
        self.assertFalse(captured["camera"], "--no-camera-probe must skip the only hardware check")

    def test_the_flag_defaults_into_the_config(self):
        args = main_module.parse_args(["--diagnose"])
        self.assertTrue(args.diagnose)
        self.assertFalse(args.no_camera_probe)


if __name__ == "__main__":
    unittest.main()
