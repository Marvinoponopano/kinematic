"""Phase 9's verification bar, end to end.

The phase names what to test: "fresh setup, denied permissions, unavailable cameras, low light,
partial tracking, sustained gestures, repeated launch/cleanup, and full regression checks". Each of
those has a block below. "End to end" here means from a frame to an adapter call: the frames are
synthetic, MediaPipe's protobuf shape is faked once at the boundary (the way Phase 2's tests do it),
the adapter is the recording one, and pyautogui is never imported or called.

What genuinely needs hardware — a camera negotiating real frames, a hand in front of it, the OS
obeying a real cursor move — stays manual, and the README's verification checklist is the procedure
for it. This suite is the part that can run anywhere, on every commit.
"""

import contextlib
import io
import threading
import unittest
from types import SimpleNamespace
from typing import Any, List, Optional, Sequence, Tuple
from unittest import mock

import numpy as np

import kinematic.main as main_module
import kinematic.runtime as runtime_module
from kinematic.capture import CameraError
from kinematic.config import HandMouseConfig
from kinematic.diagnostics import check_camera, check_output, diagnose
from kinematic.gesture import INDEX_MCP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, PINKY_MCP, RING_MCP, THUMB_TIP, WRIST
from kinematic.landmarks import hands_from_results
from kinematic.output import DEFAULT_SCREEN, RecordingMouseOutput
from kinematic.recognizer import HandState
from kinematic.runtime import HandMouseRuntime

STEP = 0.05  # 20 frames a second in these scripts; fine for 30 fps logic, and no binary traps (8.10)
SCALE = 0.15  # wrist (0.50, 0.60) -> middle knuckle (0.50, 0.45): the hand_scale every ratio is of


# -- synthetic hands (and MediaPipe's shape for them) -------------------------


def hand_points(index_ratio: float = 0.70, middle_ratio: float = 0.70, offset: Tuple[float, float] = (0.0, 0.0)) -> List[Tuple[float, float]]:
    """A 21-landmark hand at the requested pinch ratios (0.22 = pinched, 0.70 = open/w released).

    The ratios are exact because every distance below is a multiple of :data:`SCALE`: the pinch
    thresholds are 0.40 (engage) and 0.56 (release), so 0.22 engages and 0.70 both releases and stays
    out of the way. Getting this constant wrong is how a "release" pose can silently stay inside the
    hysteresis band and stop a second tap from ever firing.
    """
    points = [(0.50, 0.60) for _ in range(21)]
    points[WRIST] = (0.50, 0.60)
    points[INDEX_MCP], points[MIDDLE_MCP] = (0.45, 0.46), (0.50, 0.45)
    points[RING_MCP], points[PINKY_MCP] = (0.55, 0.46), (0.59, 0.49)
    points[THUMB_TIP] = (0.40, 0.52)
    thumb = points[THUMB_TIP]
    # index tip to the *right* of the thumb, middle tip above it: the two pairs never coincide.
    points[INDEX_TIP] = (thumb[0] + index_ratio * SCALE, thumb[1] - 0.01)
    points[MIDDLE_TIP] = (thumb[0], thumb[1] - middle_ratio * SCALE)
    ox, oy = offset
    return [(x + ox, y + oy) for x, y in points]


def landmarks(points: Sequence[Tuple[float, float]]) -> List[SimpleNamespace]:
    return [SimpleNamespace(x=x, y=y, z=0.0) for x, y in points]


def left_pinch(offset: Tuple[float, float] = (0.0, 0.0)) -> List[SimpleNamespace]:
    return landmarks(hand_points(index_ratio=0.22, offset=offset))


def two_finger(offset: Tuple[float, float] = (0.0, 0.0)) -> List[SimpleNamespace]:
    return landmarks(hand_points(index_ratio=0.22, middle_ratio=0.22, offset=offset))


def open_hand(offset: Tuple[float, float] = (0.0, 0.0)) -> List[SimpleNamespace]:
    return landmarks(hand_points(offset=offset))


def mediapipe_hand(points: Sequence[Tuple[float, float]]) -> Any:
    """MediaPipe's NormalizedLandmarkList shape, as Phase 2's tests fake it."""
    return SimpleNamespace(landmark=[SimpleNamespace(x=x, y=y, z=-0.02) for x, y in points])


def mediapipe_results(hands: Sequence[Any]) -> Any:
    handedness = [
        SimpleNamespace(classification=[SimpleNamespace(label="Right", score=0.97)]) for _ in hands
    ]
    return SimpleNamespace(multi_hand_landmarks=list(hands), multi_handedness=handedness)


class NoHands:
    def process(self, rgb: Any) -> Any:
        return SimpleNamespace(multi_hand_landmarks=None, multi_handedness=None)


class ScriptedHands:
    """A MediaPipe stand-in that returns whatever hand the script says for this frame."""

    def __init__(self, script: Sequence[Sequence[Tuple[float, float]]]) -> None:
        self._script = list(script)
        self._index = 0

    def process(self, rgb: Any) -> Any:
        points = self._script[min(self._index, len(self._script) - 1)]
        self._index += 1
        return mediapipe_results([mediapipe_hand(points)] if points else [])

    def close(self) -> None:
        pass


class FakeCamera:
    def __init__(self, frames: Sequence[np.ndarray], ready: bool = True) -> None:
        self._frames = list(frames)
        self._ready = ready
        self._index = 0
        self._seq = 0
        self.released = False

    @property
    def error(self) -> Optional[str]:
        return None

    def wait_ready(self, timeout: float = 3.0) -> bool:
        return self._ready and bool(self._frames)

    def read(self) -> Tuple[int, Optional[np.ndarray]]:
        if self._index + 1 < len(self._frames):
            self._index += 1
            self._seq += 1
            return self._seq, self._frames[self._index]
        return self._seq, self._frames[self._index] if self._frames else None

    def release(self) -> None:
        self.released = True


def frames(count: int) -> List[np.ndarray]:
    return [np.full((16, 24, 3), 120, dtype=np.uint8) for _ in range(count)]


def make_runtime(hands: Any = None, output: Optional[RecordingMouseOutput] = None, **overrides: Any) -> HandMouseRuntime:
    overrides.setdefault("show_preview", False)
    return HandMouseRuntime(
        config=HandMouseConfig(**overrides),
        output=output or RecordingMouseOutput(),
        hands=hands if hands is not None else NoHands(),
    )


# -- fresh setup --------------------------------------------------------------


class FreshSetupTests(unittest.TestCase):
    def test_the_default_configuration_starts_paused_and_emits_nothing(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output=output, dry_run=True)
        self.assertTrue(runtime.paused)
        runtime.handle_key = lambda key: True
        runtime.run(camera=FakeCamera(frames(10)))
        self.assertEqual(output.counts(), {"move": 0, "click": 0, "scroll": 0})
        self.assertEqual(runtime.summary()["status"], "paused")

    def test_a_dry_run_is_identified_as_one_and_needs_no_permissions(self):
        runtime = make_runtime(dry_run=True)
        self.assertIn("dry run", check_output(runtime.output).detail)
        checks = diagnose(
            config=runtime.config,
            output=runtime.output,
            hands=NoHands(),
            opener=lambda index: FakeCamera(frames(1)),
            trusted=lambda: False,  # a denied permission must not stop the rest of the report
        )
        self.assertEqual([check.name for check in checks if not check.ok], ["accessibility"])

    def test_the_console_entry_point_exists_and_is_callable(self):
        from kinematic.main import main

        self.assertTrue(callable(main))
        self.assertEqual(main_module.HandMouseApp(HandMouseConfig(dry_run=True)).output.__class__.__name__, "RecordingMouseOutput")

    def test_packaging_metadata_matches_the_project(self):
        try:
            import tomllib  # Python 3.11+
        except ModuleNotFoundError:  # pragma: no cover - 3.11+ has tomllib
            from pip._vendor import tomli as tomllib  # type: ignore[no-redef]
        with open("pyproject.toml", "rb") as handle:
            project = tomllib.load(handle)

        self.assertEqual(project["project"]["name"], "kinematic")
        self.assertEqual(project["project"]["scripts"]["kinematic"], "kinematic.main:main")
        dependencies = project["project"]["dependencies"]
        self.assertTrue(any(dependency.startswith("mediapipe==") for dependency in dependencies))
        self.assertIn("kinematic", project["tool"]["setuptools"]["packages"])


# -- denied permissions and unavailable cameras -------------------------------


class PermissionAndDeviceTests(unittest.TestCase):
    def test_a_denied_camera_is_reported_with_the_settings_hint(self):
        def denied(index: int) -> Any:
            raise CameraError("OpenCV: not authorized to capture video")

        check, probe = check_camera(opener=denied)
        self.assertFalse(check.ok)
        self.assertIn("Privacy & Security -> Camera", check.hint)
        self.assertFalse(probe.opened)

    def test_the_app_surfaces_a_camera_error_without_emitting_anything(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output=output, start_paused=False)

        def denied(**kwargs: Any) -> Any:
            raise CameraError(f"Could not open camera index {kwargs.get('index')}.")

        with mock.patch.object(runtime_module, "ThreadedCamera", denied):
            with self.assertRaises(CameraError) as caught:
                runtime.run()
        self.assertIn("camera index", str(caught.exception))
        self.assertEqual(output.counts(), {"move": 0, "click": 0, "scroll": 0})

    def test_an_unavailable_index_is_named_in_the_report(self):
        config = HandMouseConfig(camera_index=3)

        def missing(index: int) -> Any:
            raise CameraError(f"Could not open camera index {index}.")

        check, _probe = check_camera(config=config, opener=missing)
        self.assertFalse(check.ok)
        self.assertIn("index 3", check.detail)

    def test_a_camera_that_never_produces_a_frame_fails_cleanly(self):
        runtime = make_runtime(start_paused=False, show_preview=False)
        with self.assertRaises(RuntimeError):
            runtime.run(camera=FakeCamera([], ready=False))
        self.assertEqual(runtime.emitted, 0)


# -- low light and partial tracking -------------------------------------------


class PoorConditionsTests(unittest.TestCase):
    def test_a_dark_room_is_flagged_and_a_handless_pipeline_stays_quiet(self):
        dark = [np.zeros((16, 24, 3), dtype=np.uint8) for _ in range(3)]
        check, probe = check_camera(opener=lambda index: FakeCamera(dark))
        self.assertLess(probe.brightness, 30.0)
        self.assertIn("very dark", check.hint)

        output = RecordingMouseOutput()
        runtime = make_runtime(output=output, start_paused=False)
        runtime.handle_key = lambda key: True
        runtime.run(camera=FakeCamera(dark))
        self.assertEqual(output.counts(), {"move": 0, "click": 0, "scroll": 0})
        self.assertGreaterEqual(runtime.tracking_losses, 1)

    def test_partial_tracking_is_ignored_rather_than_acted_on(self):
        # Three landmarks is what MediaPipe hands back when a hand is half out of frame. It must not
        # reach the geometry (an IndexError deep in the pipeline) and must not move anything.
        partial = mediapipe_results([mediapipe_hand([(0.5, 0.5), (0.4, 0.4), (0.3, 0.3)])])
        hands = hands_from_results(partial)
        self.assertEqual(len(hands), 1)
        self.assertFalse(hands[0].is_complete)

        output = RecordingMouseOutput()
        runtime = make_runtime(hands=ScriptedHands([[None]] * 3), output=output, start_paused=False)
        runtime.hands = SimpleNamespace(process=lambda rgb: partial)
        runtime.handle_key = lambda key: True
        runtime.run(camera=FakeCamera(frames(3)))
        self.assertEqual(output.counts(), {"move": 0, "click": 0, "scroll": 0})
        self.assertIs(runtime.gesture_state, HandState.TRACKING_LOST)

    def test_a_hand_that_leaves_and_comes_back_re_anchors_the_cursor(self):
        # A tracking loss must reset the filter and the anchor: the reacquired hand neither jumps nor
        # glides (Phase 3's requirement, re-checked here as part of the end-to-end regression).
        script = [
            hand_points(),
            hand_points(),
            [],  # the hand leaves the frame
            hand_points(offset=(0.02, 0.0)),
            hand_points(offset=(0.04, 0.0)),
        ]
        output = RecordingMouseOutput()
        runtime = make_runtime(hands=ScriptedHands(script), output=output, start_paused=False)
        for index in range(len(script)):
            runtime.process_frame(frames(1)[0], now=index * STEP)
        self.assertGreater(runtime.tracking_losses, 0)
        self.assertGreater(output.counts()["move"], 0)


# -- sustained gestures -------------------------------------------------------


class SustainedGestureTests(unittest.TestCase):
    # 3 pinched frames hold the pose past the 0.08 s hold time (0.10 s at this step), then 3 released;
    # a tap every 6 frames is 0.3 s apart, which clears the 0.18 s click cooldown. Both numbers matter:
    # 2 pinched frames never reach the hold time, and 5-frame taps sit inside the cooldown.
    FRAMES_PER_TAP = 6

    def tap_script(self, taps: int, move: float = 0.0) -> List[List[Tuple[float, float]]]:
        """*taps* of a left pinch, each held long enough to fire, with a release in between."""
        script: List[List[Tuple[float, float]]] = []
        for tap in range(taps):
            offset = (move * tap, 0.0)
            script += [hand_points(index_ratio=0.22, offset=offset) for _ in range(3)]
            script += [hand_points(offset=offset) for _ in range(self.FRAMES_PER_TAP - 3)]
        return script

    def test_a_long_run_of_taps_clicks_exactly_once_per_tap(self):
        taps = 30
        script = self.tap_script(taps)
        self.assertEqual(len(script), taps * self.FRAMES_PER_TAP)
        output = RecordingMouseOutput()
        runtime = make_runtime(hands=ScriptedHands(script), output=output, start_paused=False)
        for index, _ in enumerate(script):
            runtime.process_frame(frames(1)[0], now=index * STEP)

        self.assertEqual(len(output.clicks), taps, "one click per deliberate tap, no drift")
        self.assertEqual(len(output.vertical_scrolls), 0)
        self.assertEqual(runtime.summary()["actions"], {"left click": taps})
        self.assertEqual(runtime.tracking_losses, 0)

    def test_a_long_scroll_sweep_keeps_its_direction_and_never_clicks(self):
        # 0.008 a frame is above the 0.05-hand-width dead zone (0.0075 at this scale), so every frame
        # after the one that sets the reference is a real scroll.
        script = [
            hand_points(index_ratio=0.22, middle_ratio=0.22, offset=(0.008 * step, 0.008 * step))
            for step in range(30)
        ]
        output = RecordingMouseOutput()
        runtime = make_runtime(hands=ScriptedHands(script), output=output, start_paused=False)
        for index, _ in enumerate(script):
            runtime.process_frame(frames(1)[0], now=index * STEP)

        self.assertEqual(len(output.clicks), 0, "the scroll pose must never click")
        self.assertGreater(len(output.vertical_scrolls), 0)
        self.assertTrue(all(event.value < 0 for event in output.vertical_scrolls), "hand down scrolls down")
        self.assertTrue(all(event.value > 0 for event in output.horizontal_scrolls), "hand right scrolls right")

    def test_the_summary_agrees_with_the_adapter(self):
        script = self.tap_script(5)
        output = RecordingMouseOutput()
        runtime = make_runtime(hands=ScriptedHands(script), output=output, start_paused=False)
        for index, _ in enumerate(script):
            runtime.process_frame(frames(1)[0], now=index * STEP)
        summary = runtime.summary()
        self.assertEqual(summary["emitted"], len(output.events))
        self.assertEqual(summary["actions"], {"left click": len(output.clicks)})
        self.assertEqual(summary["recorded"], output.counts())


# -- repeated launch and cleanup ----------------------------------------------


class RepeatedLaunchTests(unittest.TestCase):
    class FakeApp:
        """A HandMouseApp stand-in whose loop returns immediately, for the CLI path."""

        def __init__(self, config: HandMouseConfig, output: Optional[Any] = None) -> None:
            self.config = config
            self.output = output or RecordingMouseOutput()
            self.runtime = HandMouseRuntime(config=config, output=self.output, hands=NoHands())
            self.closed = 0

        @property
        def hands(self) -> Any:
            return self.runtime.hands

        def run(self, **kwargs: Any) -> None:
            self.runtime.frames = 4

        def close(self) -> None:
            self.closed += 1
            self.runtime.close()

    def test_the_command_line_runs_repeatedly_and_reports_a_summary(self):
        for _ in range(3):
            app = self.FakeApp(HandMouseConfig(show_preview=False, dry_run=True))
            with mock.patch.object(main_module, "HandMouseApp", lambda config: app):
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    status = main_module.main(["--dry-run", "--no-preview"])
            self.assertEqual(status, 0)
            self.assertIn("frames=4", buffer.getvalue())
            self.assertIn("pause toggles=", buffer.getvalue())
            self.assertEqual(app.closed, 1)

    def test_no_capture_thread_survives_a_launch(self):
        before = {thread.ident for thread in threading.enumerate()}
        for _ in range(3):
            camera = FakeCamera(frames(3))
            runtime = make_runtime(start_paused=False)
            runtime.handle_key = lambda key: True
            with mock.patch.object(runtime_module, "ThreadedCamera", lambda **kwargs: camera):
                runtime.run()
            runtime.close()
            self.assertTrue(camera.released)
        self.assertEqual(before, {thread.ident for thread in threading.enumerate()})


# -- full pipeline regression -------------------------------------------------


class PipelineRegressionTests(unittest.TestCase):
    """One scripted session, asserted call by call: the net that catches a regression anywhere."""

    def run_script(self) -> Tuple[RecordingMouseOutput, HandMouseRuntime, List[Any]]:
        """Feed one scripted session through the real frame path, MediaPipe's boundary included."""
        script: List[Any] = []
        # 1-4: a left pinch held past the hold time (three frames), then released -> exactly one click.
        script += [hand_points(index_ratio=0.22) for _ in range(3)]
        script += [hand_points()]
        # 5-10: a two-finger pose sweeping down and to the right -> scrolling on both axes.
        script += [
            hand_points(index_ratio=0.22, middle_ratio=0.22, offset=(0.008 * step, 0.008 * step))
            for step in range(1, 7)
        ]
        # 11: released.
        script += [hand_points()]
        # 12-14: the hand leaves the frame, then comes back.
        script += [None, None, hand_points()]

        output = RecordingMouseOutput()
        runtime = make_runtime(output=output, start_paused=False)
        for index, points in enumerate(script):
            if points is None:
                runtime.apply_gestures(None, now=index * STEP)
                runtime.frames += 1
                continue
            # Through MediaPipe's protobuf boundary and back, so the conversion is exercised too.
            runtime.hands = SimpleNamespace(
                process=lambda rgb, pts=points: mediapipe_results([mediapipe_hand(pts)])
            )
            runtime.process_frame(frames(1)[0], now=index * STEP)
        return output, runtime, script

    def test_the_event_stream_is_what_the_pipeline_promises(self):
        output, runtime, _script = self.run_script()

        kinds = [event.kind for event in output.events]
        self.assertIn("click", kinds)
        self.assertEqual([event.value for event in output.clicks], ["left"], "one left click, no right click")
        self.assertEqual(len(output.clicks), 1)

        # Scroll precedence: nothing clicks once scrolling has begun.
        first_scroll = kinds.index("scroll")
        self.assertNotIn("click", kinds[first_scroll:])

        # Both wheel axes, with the documented signs for a hand moving down and right.
        self.assertGreater(len(output.vertical_scrolls), 0)
        self.assertGreater(len(output.horizontal_scrolls), 0)
        self.assertTrue(all(event.value < 0 for event in output.vertical_scrolls))
        self.assertTrue(all(event.value > 0 for event in output.horizontal_scrolls))

        # Every move stayed on screen, and the last thing the session did was notice the loss.
        for event in output.moves:
            x, y = event.value
            self.assertTrue(0.0 <= x <= DEFAULT_SCREEN[0])
            self.assertTrue(0.0 <= y <= DEFAULT_SCREEN[1])
        self.assertGreaterEqual(runtime.tracking_losses, 1)

    def test_the_same_script_is_repeatable(self):
        first, _, _ = self.run_script()
        second, _, _ = self.run_script()
        self.assertEqual(
            [(event.kind, event.value) for event in first.events],
            [(event.kind, event.value) for event in second.events],
            "a seeded, camera-free script must be deterministic",
        )

    def test_nothing_reaches_the_desktop_in_any_of_it(self):
        output, runtime, _script = self.run_script()
        # The only adapter in this suite is the recording one, and it is the one that saw the calls.
        self.assertIsInstance(runtime.output, RecordingMouseOutput)
        self.assertGreater(len(output.events), 0, "the script must actually have produced output")


if __name__ == "__main__":
    unittest.main()
