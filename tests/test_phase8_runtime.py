"""Phase 8's remaining halves: the optional gesture pause, and the pause/teardown guarantees.

The phase's fifth bullet is conditional — "add a gesture pause only after the keyboard control is
reliable" — and this is the only pose in the project that can *change* the pause state, so it is
opt-in, hold-gated, fires once per hold, and works while paused (which is what lets the same pose
resume). What the phase asks to be verified explicitly is pinned at the bottom of this file: toggle
idempotence, recovery after a failed run, and cleanup on a normal exit, on a window close and on an
exception.

Nothing here touches a camera, a window or the mouse: the gestures are synthetic landmark lists, the
output is a recording adapter, and ``cv2`` is patched wherever the loop would draw.
"""

import threading
import unittest
from types import SimpleNamespace
from typing import Any, List, Optional, Sequence, Tuple
from unittest import mock

import numpy as np

import kinematic.runtime as runtime_module
from kinematic.config import HandMouseConfig
from kinematic.gesture import (
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_TIP,
    RING_MCP,
    RING_PIP,
    RING_TIP,
    THUMB_TIP,
    WRIST,
    extended_fingers,
)
from kinematic.output import RecordingMouseOutput
from kinematic.recognizer import (
    EventKind,
    GestureRecognizer,
    GestureThresholds,
    HandState,
    Observation,
    thresholds_from_config,
)
from kinematic.runtime import HandMouseRuntime

STEP = 0.125  # an exact binary fraction, so hold-time arithmetic is not the thing under test (8.10)
HOLD = 0.5  # the gesture-pause hold these tests configure: four steps


# -- synthetic hands ----------------------------------------------------------


def hand_points(
    extended: int = 4,
    pinch_index: bool = False,
    pinch_middle: bool = False,
    offset: Tuple[float, float] = (0.0, 0.0),
) -> List[Tuple[float, float]]:
    """A 21-landmark hand with exactly *extended* straight fingers and no accidental pinch.

    The wrist sits at (0.50, 0.70) and the middle knuckle 0.10 above it, so ``hand_scale`` is exactly
    0.10 and every ratio below is exact. A straight finger's tip is farther from the wrist than its
    middle joint; a curled one's tip comes back toward the palm.
    """
    points = [(0.50, 0.70) for _ in range(21)]
    points[WRIST] = (0.50, 0.70)
    points[INDEX_MCP], points[MIDDLE_MCP] = (0.44, 0.60), (0.50, 0.60)
    points[RING_MCP], points[PINKY_MCP] = (0.56, 0.60), (0.61, 0.62)

    chains = (
        (INDEX_TIP, INDEX_PIP, 0.40),
        (MIDDLE_TIP, MIDDLE_PIP, 0.47),
        (RING_TIP, RING_PIP, 0.54),
        (PINKY_TIP, PINKY_PIP, 0.60),
    )
    for position, (tip, pip, x) in enumerate(chains):
        points[pip] = (x, 0.55)
        points[tip] = (x, 0.42) if position < extended else (x, 0.64)

    thumb = (0.30, 0.70)  # far from every fingertip: nothing engaged
    if pinch_index:
        # 0.0224 from the index tip (ratio 0.224, well inside the 0.40 engage threshold) while staying
        # 0.51 from the middle tip — above 0.40, so this really is a *one-pair* pinch. Landing on an
        # exact threshold here would be the floating-point trap of gotcha 8.10 in ratio form.
        thumb = (points[INDEX_TIP][0] + 0.02, points[INDEX_TIP][1] - 0.01)
    elif pinch_middle:
        thumb = (points[MIDDLE_TIP][0] - 0.02, points[MIDDLE_TIP][1] - 0.01)
    points[THUMB_TIP] = thumb

    ox, oy = offset
    return [(x + ox, y + oy) for x, y in points]


def landmarks(points: Sequence[Tuple[float, float]]) -> List[SimpleNamespace]:
    return [SimpleNamespace(x=x, y=y, z=0.0) for x, y in points]


def open_hand(extended: int = 4, **kwargs: Any) -> List[SimpleNamespace]:
    return landmarks(hand_points(extended=extended, **kwargs))


def pinched_hand(kind: str = "index") -> List[SimpleNamespace]:
    return landmarks(hand_points(pinch_index=kind == "index", pinch_middle=kind == "middle"))


class NoHands:
    def process(self, rgb: Any) -> Any:
        return SimpleNamespace(multi_hand_landmarks=None, multi_handedness=None)


class StubHands:
    """A hand tracker that records how it was torn down (for the repeated-launch checks)."""

    def __init__(self) -> None:
        self.closed = 0

    def process(self, rgb: Any) -> Any:
        return SimpleNamespace(multi_hand_landmarks=None, multi_handedness=None)

    def close(self) -> None:
        self.closed += 1


class FakeCamera:
    """Replays a fixed list of frames; ``owned`` capture is exercised by patching ThreadedCamera."""

    def __init__(
        self,
        frames: Optional[Sequence[np.ndarray]] = None,
        ready: bool = True,
        error: Optional[str] = None,
    ) -> None:
        self._frames = list(frames or [np.zeros((8, 8, 3), dtype=np.uint8)])
        self._ready = ready
        self._error = error
        self._index = 0
        self._seq = 0
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
        return self._seq, self._frames[self._index] if self._frames else None

    def release(self) -> None:
        self.released = True


def make_runtime(**overrides: Any) -> HandMouseRuntime:
    overrides.setdefault("show_preview", False)
    config = HandMouseConfig(**overrides)
    return HandMouseRuntime(config=config, output=RecordingMouseOutput(), hands=NoHands())


def gesture_runtime(**overrides: Any) -> HandMouseRuntime:
    """A runtime with the gesture pause enabled and a hold that fits the test step."""
    overrides.setdefault("gesture_pause_enabled", True)
    overrides.setdefault("gesture_pause_hold_seconds", HOLD)
    return make_runtime(**overrides)


def feed_gestures(
    runtime: HandMouseRuntime,
    frames: int,
    points: Optional[Sequence[Any]] = None,
    start: float = 0.0,
    **hand_kwargs: Any,
) -> List[Any]:
    """Feed *frames* of a hand (an open hand unless *points* is given) and return every event."""
    events: List[Any] = []
    for index in range(frames):
        hand = open_hand(**hand_kwargs) if points is None else points
        runtime.apply_gestures(hand, now=start + index * STEP)
        events.extend(runtime.last_gesture_events)
    return events


# -- the pure pause gesture ---------------------------------------------------


class ExtendedFingerTests(unittest.TestCase):
    def test_counts_straight_fingers(self):
        for count in (0, 1, 2, 3, 4):
            with self.subTest(extended=count):
                self.assertEqual(extended_fingers(open_hand(extended=count)), count)

    def test_ignores_the_thumb(self):
        # The thumb is not in the chain: a hand with four straight fingers counts four, wherever the
        # thumb is (here it is tucked against the palm).
        flat = landmarks(hand_points(extended=4))
        flat[THUMB_TIP] = SimpleNamespace(x=0.50, y=0.66, z=0.0)
        self.assertEqual(extended_fingers(flat), 4)


class PauseGestureThresholdTests(unittest.TestCase):
    def test_off_by_default(self):
        self.assertFalse(GestureThresholds().pause_gesture_enabled)

    def test_finger_count_is_validated(self):
        for bad in (-1, 5):
            with self.subTest(fingers=bad):
                with self.assertRaises(ValueError):
                    GestureThresholds(pause_gesture_fingers=bad)

    def test_negative_hold_is_rejected(self):
        with self.assertRaises(ValueError):
            GestureThresholds(pause_gesture_hold_seconds=-0.1)

    def test_thresholds_come_from_the_config(self):
        config = HandMouseConfig(
            gesture_pause_enabled=True, gesture_pause_fingers=3, gesture_pause_hold_seconds=2.0
        )
        thresholds = thresholds_from_config(config)
        self.assertTrue(thresholds.pause_gesture_enabled)
        self.assertEqual(thresholds.pause_gesture_fingers, 3)
        self.assertAlmostEqual(thresholds.pause_gesture_hold_seconds, 2.0)


class PauseGestureTests(unittest.TestCase):
    """The recogniser's side: one toggle per hold, released to re-arm."""

    def recognizer(self, **kwargs: Any) -> GestureRecognizer:
        kwargs.setdefault("pause_gesture_enabled", True)
        kwargs.setdefault("pause_gesture_hold_seconds", HOLD)
        return GestureRecognizer(GestureThresholds(**kwargs))

    def feed(self, recognizer: GestureRecognizer, frames: int, points: Any, start: float = 0.0) -> List[Any]:
        events: List[Any] = []
        for index in range(frames):
            update = recognizer.update(Observation(landmarks=points, timestamp=start + index * STEP))
            events.extend(update.events)
        return events

    def toggles(self, events: Sequence[Any]) -> int:
        return sum(1 for event in events if event.kind is EventKind.PAUSE_TOGGLE)

    def test_disabled_by_default(self):
        recognizer = GestureRecognizer()
        events = self.feed(recognizer, 20, open_hand())
        self.assertEqual(self.toggles(events), 0)

    def test_an_open_hand_held_long_enough_toggles_once(self):
        recognizer = self.recognizer()
        events = self.feed(recognizer, 4, open_hand())  # 4 x 0.125 = 0.5 s: exactly the hold
        self.assertEqual(self.toggles(events), 0, "the hold time must have elapsed before it fires")
        events += self.feed(recognizer, 1, open_hand(), start=4 * STEP)
        self.assertEqual(self.toggles(events), 1)

    def test_holding_longer_does_not_toggle_again(self):
        recognizer = self.recognizer()
        events = self.feed(recognizer, 40, open_hand())
        self.assertEqual(self.toggles(events), 1)

    def test_releasing_and_holding_again_toggles_a_second_time(self):
        recognizer = self.recognizer()
        events = self.feed(recognizer, 8, open_hand())
        events += self.feed(recognizer, 3, pinched_hand(), start=1.0)  # release the pose
        events += self.feed(recognizer, 8, open_hand(), start=2.0)
        self.assertEqual(self.toggles(events), 2)

    def test_a_flicker_restarts_the_hold(self):
        recognizer = self.recognizer()
        events = self.feed(recognizer, 3, open_hand())  # 0.375 s: not yet
        events += self.feed(recognizer, 1, open_hand(extended=0), start=0.375)  # leave the pose
        events += self.feed(recognizer, 4, open_hand(), start=0.5)  # 4 x 0.125 from the restart
        self.assertEqual(self.toggles(events), 0)
        events += self.feed(recognizer, 1, open_hand(), start=1.0)
        self.assertEqual(self.toggles(events), 1)

    def test_an_engaged_pinch_is_not_the_pause_pose(self):
        # Four straight fingers, but the thumb is on the index tip, so the pair is engaged. The pause
        # pose must refuse it: otherwise a pinch could freeze the pointer.
        recognizer = self.recognizer()
        events = self.feed(recognizer, 30, open_hand(extended=4, pinch_index=True))
        self.assertEqual(self.toggles(events), 0)
        self.assertTrue(recognizer.engaged("thumb_index"))

    def test_the_finger_count_is_configurable(self):
        two = self.recognizer(pause_gesture_fingers=2)
        self.assertEqual(self.toggles(self.feed(two, 10, open_hand(extended=2))), 1)
        three = self.recognizer(pause_gesture_fingers=3)
        self.assertEqual(self.toggles(self.feed(three, 10, open_hand(extended=2))), 0)

    def test_it_works_while_paused(self):
        # The one thing that does work while paused: a resume by gesture would be impossible if the
        # recogniser ignored the pose in the PAUSED state.
        recognizer = self.recognizer()
        recognizer.set_paused(True)
        events = self.feed(recognizer, 30, open_hand())
        self.assertEqual(self.toggles(events), 1)
        self.assertIs(recognizer.state, HandState.PAUSED)

    def test_the_hold_restarts_after_a_tracking_loss(self):
        recognizer = self.recognizer()
        events = self.feed(recognizer, 3, open_hand())  # 0.375 s
        events += self.feed(recognizer, 1, None, start=0.375)
        # The restore frame itself clears the partial hold (a reacquired hand starts from scratch), so
        # the clock restarts on the frame *after* it and the toggle needs the full hold from there.
        events += self.feed(recognizer, 4, open_hand(), start=0.5)
        self.assertEqual(self.toggles(events), 0, "the clock restarts when the hand comes back")
        events += self.feed(recognizer, 2, open_hand(), start=1.0)
        self.assertEqual(self.toggles(events), 1)

    def test_a_hold_in_flight_over_a_pause_transition_is_cancelled(self):
        # Otherwise pressing SPACE while holding an open hand would toggle straight back 0.5 s later.
        recognizer = self.recognizer()
        self.feed(recognizer, 2, open_hand())
        recognizer.set_paused(True)
        self.assertFalse(recognizer.pause_gesture_armed)
        self.assertEqual(self.toggles(self.feed(recognizer, 10, open_hand(), start=1.0)), 0)


# -- the runtime's side -------------------------------------------------------


class GesturePauseRuntimeTests(unittest.TestCase):
    def test_the_gesture_pause_toggles_the_runtime(self):
        runtime = gesture_runtime()
        self.assertTrue(runtime.paused)  # the app starts paused
        feed_gestures(runtime, 5, open_hand())
        self.assertFalse(runtime.paused, "a held open hand resumes")
        self.assertEqual(runtime.gesture_pause_toggles, 1)
        self.assertEqual(runtime.pause_source, "gesture")

    def test_it_pauses_an_active_runtime_and_resumes_again(self):
        runtime = gesture_runtime(start_paused=False)
        feed_gestures(runtime, 5, open_hand())
        self.assertTrue(runtime.paused)
        feed_gestures(runtime, 3, open_hand(extended=0), start=1.0)  # leave the pose
        feed_gestures(runtime, 5, open_hand(), start=2.0)
        self.assertFalse(runtime.paused)
        self.assertEqual(runtime.gesture_pause_toggles, 2)

    def test_it_emits_nothing_to_the_mouse(self):
        output = RecordingMouseOutput()
        runtime = HandMouseRuntime(
            config=HandMouseConfig(
                show_preview=False, gesture_pause_enabled=True, gesture_pause_hold_seconds=HOLD
            ),
            output=output,
            hands=NoHands(),
        )
        feed_gestures(runtime, 5, open_hand())  # the startup pause is resumed by the gesture
        self.assertFalse(runtime.paused)
        self.assertEqual(output.counts(), {"move": 0, "click": 0, "scroll": 0})

    def test_a_click_still_works_and_the_pause_does_not_add_clicks(self):
        output = RecordingMouseOutput()
        runtime = HandMouseRuntime(
            config=HandMouseConfig(
                show_preview=False,
                start_paused=False,
                gesture_pause_enabled=True,
                gesture_pause_hold_seconds=HOLD,
            ),
            output=output,
            hands=NoHands(),
        )
        # A deliberate tap: pinched, held past the hold time, then released.
        runtime.apply_gestures(pinched_hand(), now=0.0)
        runtime.apply_gestures(pinched_hand(), now=0.1)
        runtime.apply_gestures(pinched_hand(), now=0.2)
        runtime.apply_gestures(open_hand(extended=2), now=0.3)
        clicks_sent = output.counts()["click"]
        self.assertEqual(clicks_sent, 1)
        # A held open hand now pauses — and still sends nothing, in particular no click.
        feed_gestures(runtime, 20, open_hand(), start=1.0)
        self.assertTrue(runtime.paused)
        self.assertEqual(output.counts()["click"], clicks_sent)

    def test_it_stays_off_unless_enabled(self):
        runtime = make_runtime(start_paused=False)
        for index in range(60):
            # A moving open hand, so the pointer path is genuinely active.
            runtime.apply_gestures(open_hand(offset=(index * 0.002, 0.0)), now=index * STEP)
        self.assertFalse(runtime.paused)
        self.assertEqual(runtime.gesture_pause_toggles, 0)

    def test_the_summary_reports_it(self):
        runtime = gesture_runtime()
        feed_gestures(runtime, 5, open_hand())
        summary = runtime.summary()
        self.assertTrue(summary["gesture_pause_enabled"])
        self.assertEqual(summary["gesture_pause_toggles"], 1)
        self.assertEqual(summary["pause_source"], "gesture")


# -- Phase 8's verification bar ----------------------------------------------


class PauseToggleTests(unittest.TestCase):
    """Toggle idempotence and the pause-source bookkeeping."""

    def test_direct_pause_and_resume_are_idempotent(self):
        runtime = make_runtime(start_paused=False)
        self.assertTrue(runtime.pause(), "pause() returns the state it settled on")
        self.assertTrue(runtime.pause(), "pausing twice stays paused")
        self.assertFalse(runtime.resume())
        self.assertFalse(runtime.resume(), "resuming twice stays active")
        self.assertEqual(runtime.pause_toggles, 0, "only toggle_pause() counts toggles")

    def test_toggle_twice_returns_to_the_starting_state(self):
        runtime = make_runtime()  # starts paused
        self.assertFalse(runtime.toggle_pause(source="key"))
        self.assertTrue(runtime.toggle_pause(source="key"))
        self.assertEqual(runtime.pause_toggles, 2)
        self.assertEqual(runtime.pause_source, "key")

    def test_space_toggles_and_quit_keys_still_quit(self):
        runtime = make_runtime()
        self.assertFalse(runtime.handle_key(ord(" ")))
        self.assertFalse(runtime.paused)
        self.assertFalse(runtime.handle_key(ord("p")))
        self.assertTrue(runtime.paused)
        self.assertTrue(runtime.handle_key(ord("q")))
        self.assertTrue(runtime.handle_key(27))
        self.assertEqual(runtime.pause_toggles, 2)


class CleanupTests(unittest.TestCase):
    """Recovery, and cleanup on a normal exit, on a window close and on an exception."""

    def test_an_exception_in_the_loop_still_releases_and_destroys(self):
        camera = FakeCamera(error="Unable to read frames from the webcam.")
        runtime = make_runtime(start_paused=False, show_preview=True)
        with mock.patch.object(runtime_module, "ThreadedCamera", lambda **kwargs: camera), mock.patch.object(
            runtime_module.cv2, "destroyAllWindows"
        ) as destroy, mock.patch.object(runtime_module.cv2, "waitKey", lambda ms: -1):
            with self.assertRaises(RuntimeError):
                runtime.run()
        self.assertTrue(camera.released, "a runtime that owns its camera must release it")
        destroy.assert_called_once()

    def test_cleanup_runs_once_per_run(self):
        camera = FakeCamera()
        runtime = make_runtime(start_paused=False, show_preview=True)
        runtime.handle_key = lambda key: True  # the user quits on the first key
        with mock.patch.object(runtime_module.cv2, "destroyAllWindows") as destroy, mock.patch.object(
            runtime_module.cv2, "waitKey", lambda ms: -1
        ):
            runtime.run(camera=camera)
            runtime.run(camera=camera)
        self.assertEqual(destroy.call_count, 2)
        self.assertFalse(camera.released, "a camera the caller passed in is the caller's to release")

    def test_closing_the_window_ends_the_run_cleanly(self):
        camera = FakeCamera()
        runtime = make_runtime(start_paused=False, show_preview=True)
        with mock.patch.object(runtime_module.cv2, "imshow"), mock.patch.object(
            runtime_module.cv2, "destroyAllWindows"
        ) as destroy, mock.patch.object(runtime_module.cv2, "waitKey", lambda ms: -1), mock.patch.object(
            runtime_module.cv2, "getWindowProperty", lambda name, prop: 0.0
        ):
            runtime.run(camera=camera)  # must terminate: the window is gone
        destroy.assert_called_once()

    def test_an_unreadable_window_property_does_not_end_the_run(self):
        # A backend that cannot answer the question must not be read as "the user closed it".
        camera = FakeCamera()
        runtime = make_runtime(start_paused=False, show_preview=True)
        runtime.handle_key = lambda key: True

        def explode(name: str, prop: int) -> float:
            raise runtime_module.cv2.error("no such window")

        with mock.patch.object(runtime_module.cv2, "imshow"), mock.patch.object(
            runtime_module.cv2, "getWindowProperty", explode
        ), mock.patch.object(runtime_module.cv2, "destroyAllWindows"), mock.patch.object(
            runtime_module.cv2, "waitKey", lambda ms: -1
        ):
            runtime.run(camera=camera)  # the quit key ends it, not the property error
        self.assertTrue(runtime.frames >= 0)

    def test_recovery_after_a_failed_run(self):
        broken = make_runtime(start_paused=False, show_preview=False)
        with self.assertRaises(RuntimeError):
            broken.run(camera=FakeCamera([], ready=False))

        # A fresh runtime is unaffected: the failure left no global state behind.
        output = RecordingMouseOutput()
        recovered = HandMouseRuntime(
            config=HandMouseConfig(show_preview=False, start_paused=False),
            output=output,
            hands=NoHands(),
        )
        recovered.process_landmarks(open_hand(offset=(0.2, 0.0)), now=0.0)
        recovered.process_landmarks(open_hand(offset=(0.4, 0.0)), now=STEP)
        self.assertGreater(recovered.emitted, 0)
        self.assertGreater(output.counts()["move"], 0)

    def test_repeated_launch_and_close_leaves_nothing_running(self):
        before = {thread.ident for thread in threading.enumerate()}
        for _ in range(3):
            hands = StubHands()
            camera = FakeCamera([np.zeros((8, 8, 3), dtype=np.uint8)] * 3)
            runtime = HandMouseRuntime(
                config=HandMouseConfig(show_preview=False, start_paused=False),
                output=RecordingMouseOutput(),
                hands=hands,
            )
            runtime.handle_key = lambda key: True
            # Patch the capture factory so the runtime *owns* the camera, which is the real path: an
            # owned camera is the runtime's to release (a camera the caller passed in is not).
            with mock.patch.object(runtime_module, "ThreadedCamera", lambda **kwargs: camera):
                runtime.run()
            runtime.close()
            self.assertTrue(camera.released)
            self.assertEqual(hands.closed, 1, "each launch closes its own hand tracker exactly once")
        after = {thread.ident for thread in threading.enumerate()}
        self.assertEqual(before, after, "no capture thread survives a launch/close cycle")


if __name__ == "__main__":
    unittest.main()
