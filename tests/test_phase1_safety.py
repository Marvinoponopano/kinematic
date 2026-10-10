"""Phase 1 safety tests: the pipeline must be drivable with zero desktop side effects.

Everything here runs without a camera, without a display and without pyautogui ever
touching the machine: the runtime is given a fake capture source, fake landmarks and
a recording output adapter.
"""

import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from kinematic.config import HandMouseConfig
from kinematic.main import HandMouseApp
from kinematic.output import DEFAULT_SCREEN, MouseOutput, RecordingMouseOutput
from kinematic.runtime import HandMouseRuntime


class FakeCamera:
    """Minimal stand-in for ThreadedCamera: replays a fixed list of frames."""

    def __init__(self, frames: Sequence[np.ndarray], ready: bool = True, error: Optional[str] = None) -> None:
        self._frames = list(frames)
        self._ready = ready
        self._error = error
        self._seq = 0
        self._index = -1
        self.width = 640
        self.height = 480
        self.released = False

    @property
    def error(self) -> Optional[str]:
        return self._error

    def wait_ready(self, timeout: float = 3.0) -> bool:
        return self._ready

    def read(self) -> Tuple[int, Optional[np.ndarray]]:
        if self._index + 1 < len(self._frames):
            self._index += 1
            self._seq += 1
            return self._seq, self._frames[self._index]
        # Nothing new: hold the last frame so the loop idles instead of spinning.
        return self._seq, self._frames[self._index] if self._frames else None

    def release(self) -> None:
        self.released = True


class FakeHands:
    """Fake MediaPipe Hands: returns one hand whose landmarks move per call."""

    def __init__(self, points: Optional[List[Tuple[float, float]]] = None, moving: float = 0.0) -> None:
        self.points = points
        self.moving = moving
        self.calls = 0

    def _landmarks(self) -> List[SimpleNamespace]:
        self.calls += 1
        if self.points is not None:
            base = self.points
        else:
            # 21 landmarks. The whole hand translates when `moving` is set (the cursor is
            # driven by the palm, so a hand whose palm never moves must not move it), with
            # the fingertips offset so they stay clear of the pinch/scroll thresholds.
            x = 0.4 + self.moving * self.calls
            base = [(x, 0.5) for _ in range(21)]
            base[8] = (x + 0.05, 0.35)  # index tip
            base[4] = (x - 0.10, 0.55)  # thumb tip
            base[12] = (x + 0.15, 0.35)  # middle tip
        return [SimpleNamespace(x=px, y=py, z=0.0) for px, py in base]

    def process(self, rgb):
        return SimpleNamespace(multi_hand_landmarks=[SimpleNamespace(landmark=self._landmarks())])

    def close(self) -> None:
        pass


def frames(count: int) -> List[np.ndarray]:
    return [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(count)]


def make_runtime(
    output: Optional[MouseOutput] = None,
    hands: Optional[FakeHands] = None,
    **config_overrides: Any,
) -> HandMouseRuntime:
    overrides: Dict[str, Any] = {"show_preview": False}
    overrides.update(config_overrides)
    return HandMouseRuntime(
        config=HandMouseConfig(**overrides),
        output=output if output is not None else RecordingMouseOutput(),
        hands=hands if hands is not None else FakeHands(moving=0.05),
    )


class MockOutputAdapterTests(unittest.TestCase):
    def test_recording_adapter_records_every_call_kind(self):
        output = RecordingMouseOutput()
        output.move_to(10.0, 20.0)
        output.click("left")
        output.click("right")
        output.scroll(-120)

        self.assertEqual(output.counts(), {"move": 1, "click": 2, "scroll": 1})
        self.assertEqual(output.moves[0].value, (10.0, 20.0))
        self.assertEqual(output.scrolls[0].value, -120)

    def test_recording_adapter_tracks_position_without_the_os(self):
        output = RecordingMouseOutput(screen=(800, 600))
        self.assertEqual(output.screen_size(), (800, 600))
        self.assertEqual(output.position(), (400.0, 300.0))  # centre by default
        output.move_to(5.0, 6.0)
        self.assertEqual(output.position(), (5.0, 6.0))

    def test_recording_adapter_needs_no_pyautogui(self):
        # The mock must be usable anywhere, including machines without a desktop.
        import sys

        self.assertNotIn("pyautogui", type(RecordingMouseOutput()).__class__.__module__)
        self.assertIn("output", sys.modules["kinematic.output"].__name__)

    def test_default_screen_is_defined(self):
        self.assertEqual(RecordingMouseOutput().screen_size(), DEFAULT_SCREEN)


class PausedStartupTests(unittest.TestCase):
    def test_runtime_starts_paused_by_default(self):
        runtime = make_runtime()
        self.assertTrue(runtime.paused)
        self.assertFalse(runtime.can_emit)
        self.assertEqual(runtime.status, "paused")

    def test_paused_runtime_emits_nothing_over_many_frames(self):
        """Phase 1 verification: mock startup with a hand in frame produces no mouse events."""
        output = RecordingMouseOutput()
        runtime = make_runtime(output=output)
        camera = FakeCamera(frames(10))

        runtime.run(camera=camera, max_frames=10)

        self.assertEqual(output.counts(), {"move": 0, "click": 0, "scroll": 0})
        self.assertEqual(runtime.emitted, 0)
        self.assertGreater(runtime.frames, 0, "the pipeline should still have analysed frames")
        # run() must not release a camera it did not open (the caller owns it).
        self.assertFalse(camera.released)

    def test_dry_run_config_selects_the_recording_adapter(self):
        runtime = HandMouseRuntime(
            config=HandMouseConfig(dry_run=True, show_preview=False),
            hands=FakeHands(moving=0.05),
        )
        self.assertIsInstance(runtime.output, RecordingMouseOutput)

    def test_active_runtime_does_emit(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output=output, start_paused=False)
        runtime.run(camera=FakeCamera(frames(6)), max_frames=6)
        self.assertGreater(len(output.moves), 0)


class PauseToggleTests(unittest.TestCase):
    def test_toggle_pause_round_trip(self):
        runtime = make_runtime()
        self.assertFalse(runtime.toggle_pause())
        self.assertTrue(runtime.can_emit)
        self.assertTrue(runtime.toggle_pause())
        self.assertFalse(runtime.can_emit)

    def test_pause_and_resume_are_explicit(self):
        runtime = make_runtime(start_paused=False)
        self.assertTrue(runtime.pause())
        self.assertFalse(runtime.resume())

    def test_sustained_pinch_while_paused_does_not_click_on_resume(self):
        # A pinch already held when the user resumes must not fire a click: the rising
        # edge belongs to the paused period. The neutral pose matters — a landmark set
        # where every point is identical is a pinch by both the left and right metrics.
        neutral = [(0.5, 0.5) for _ in range(21)]
        neutral[8] = (0.50, 0.40)  # index tip
        neutral[4] = (0.80, 0.80)  # thumb, far away
        neutral[12] = (0.80, 0.80)  # middle, far away
        pinched = list(neutral)
        pinched[4] = (0.505, 0.405)

        output = RecordingMouseOutput()
        runtime = make_runtime(output=output, hands=FakeHands(points=pinched))
        runtime.run(camera=FakeCamera(frames(3)), max_frames=3)
        self.assertEqual(output.clicks, [], "nothing may fire while paused")

        runtime.resume()
        runtime.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertEqual(output.clicks, [], "a pinch held across the resume must not click")

        now = runtime.last_action_at + 1.0
        runtime.process_landmarks(points_to_landmarks(neutral), now=now)  # release
        runtime.process_landmarks(points_to_landmarks(neutral), now=now + 0.1)
        runtime.process_landmarks(points_to_landmarks(pinched), now=now + 0.2)  # pinch again
        self.assertEqual([click.value for click in output.clicks], ["left"], "release then re-pinch clicks once")

    def test_pinched_frames_click_once_not_per_frame(self):
        output = RecordingMouseOutput()
        points = [(0.5, 0.5) for _ in range(21)]
        points[8] = (0.50, 0.40)
        points[4] = (0.505, 0.405)
        runtime = make_runtime(output=output, hands=FakeHands(points=points), start_paused=False)
        runtime.run(camera=FakeCamera(frames(8)), max_frames=8)
        self.assertEqual(len(output.clicks), 1, "a sustained pinch is one click, not one per frame")


def points_to_landmarks(points: Sequence[Tuple[float, float]]) -> List[SimpleNamespace]:
    return [SimpleNamespace(x=px, y=py, z=0.0) for px, py in points]


class KeyHandlingTests(unittest.TestCase):
    def test_space_and_p_toggle_pause(self):
        runtime = make_runtime()
        self.assertFalse(runtime.handle_key(ord(" ")))
        self.assertFalse(runtime.paused)
        self.assertFalse(runtime.handle_key(ord("p")))
        self.assertTrue(runtime.paused)

    def test_q_and_escape_request_stop(self):
        runtime = make_runtime()
        self.assertTrue(runtime.handle_key(ord("q")))
        self.assertTrue(runtime.handle_key(27))

    def test_other_and_no_key_do_nothing(self):
        runtime = make_runtime(start_paused=False)
        self.assertFalse(runtime.handle_key(ord("z")))
        self.assertFalse(runtime.handle_key(-1))
        self.assertFalse(runtime.paused)

    def test_run_returns_when_a_quit_key_arrives(self):
        runtime = make_runtime()
        camera = FakeCamera(frames(5))
        runtime.handle_key = lambda key: True  # stand-in for the user pressing q
        runtime.run(camera=camera, max_frames=None)  # must not hang


class NoCameraTests(unittest.TestCase):
    def test_camera_that_never_produces_a_frame_raises_and_emits_nothing(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output=output)
        with self.assertRaises(RuntimeError):
            runtime.run(camera=FakeCamera([], ready=False))
        self.assertEqual(output.counts(), {"move": 0, "click": 0, "scroll": 0})

    def test_camera_error_raises_and_emits_nothing(self):
        output = RecordingMouseOutput()
        runtime = make_runtime(output=output, start_paused=False)
        camera = FakeCamera([], error="Unable to read frames from the webcam.")
        with self.assertRaises(RuntimeError):
            runtime.run(camera=camera)
        self.assertEqual(output.counts(), {"move": 0, "click": 0, "scroll": 0})


class AppFacadeTests(unittest.TestCase):
    """The pre-Phase-1 API keeps working; it delegates to the runtime."""

    def test_facade_delegates_state_and_hooks(self):
        app = HandMouseApp(HandMouseConfig(show_preview=False, dry_run=True))
        self.assertIsInstance(app.runtime, HandMouseRuntime)
        self.assertIsInstance(app.output, RecordingMouseOutput)
        self.assertTrue(app.paused)
        self.assertEqual(app.status, "paused")
        self.assertIs(app.config, app.runtime.config)

        app.resume()
        self.assertFalse(app.paused)

        app._update_cursor((0.5, 0.5))
        app._handle_clicks((0.5, 0.4), (0.8, 0.8))
        app._handle_scroll((0.5, 0.5))
        self.assertGreaterEqual(app.emitted, 1)


if __name__ == "__main__":
    unittest.main()
