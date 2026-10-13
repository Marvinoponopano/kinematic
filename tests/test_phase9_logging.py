"""Phase 9 logging: one JSON object per named event, quiet unless asked.

The phase asks for "structured logging and performance metrics". What that means here is pinned
below: the configuration is explicit (level, stream or file), each line is a parseable JSON object
with a named ``event`` and the caller's fields, the package logger does not leak into a host
application's root configuration, and — the part that matters for this project — the hot path costs
nothing at the default level and never writes a line per frame at ``info``.

The runtime's own events (pause, resume, clicks, tracking loss, the exit summary) are asserted
through a recording-adapter runtime, so no camera or mouse is involved.
"""

import io
import json
import logging
import os
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import kinematic.runtime as runtime_module
from kinematic.config import HandMouseConfig
from kinematic.log import LOGGER_NAME, JsonLineFormatter, configure, get_logger, level_number, log_event
from kinematic.output import RecordingMouseOutput
from kinematic.runtime import HandMouseRuntime

STEP = 0.05


def pinch_hand(offset: Tuple[float, float] = (0.0, 0.0)) -> List[SimpleNamespace]:
    """A hand whose thumb touches the index tip and whose middle finger is away.

    The geometry is the same shape the click suite uses — hand scale exactly 0.10 — so the thumb-index
    ratio is 0.22 (engaged) and the thumb-middle ratio is 0.51 (not engaged), which is a left pinch.
    """
    from kinematic.gesture import INDEX_MCP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, PINKY_MCP, RING_MCP, THUMB_TIP, WRIST

    points = [(0.50, 0.60) for _ in range(21)]
    points[WRIST] = (0.50, 0.60)
    points[INDEX_MCP], points[MIDDLE_MCP] = (0.45, 0.46), (0.50, 0.45)
    points[RING_MCP], points[PINKY_MCP] = (0.55, 0.46), (0.59, 0.49)
    points[THUMB_TIP] = (0.40, 0.52)
    thumb = points[THUMB_TIP]
    points[INDEX_TIP] = (thumb[0] + 0.02, thumb[1] - 0.01)
    points[MIDDLE_TIP] = (thumb[0], thumb[1] - 0.70 * 0.10)
    ox, oy = offset
    return [SimpleNamespace(x=x + ox, y=y + oy, z=0.0) for x, y in points]


class NoHands:
    def process(self, rgb: Any) -> Any:
        return SimpleNamespace(multi_hand_landmarks=None, multi_handedness=None)


class FakeCamera:
    def __init__(self, frames: int = 90, error: Optional[str] = None) -> None:
        self._frame = np.zeros((8, 8, 3), dtype=np.uint8)
        self._left = frames
        self._seq = 0
        self._error = error

    @property
    def error(self) -> Optional[str]:
        return self._error

    def wait_ready(self, timeout: float = 3.0) -> bool:
        return True

    def read(self) -> Tuple[int, Optional[np.ndarray]]:
        if self._left > 0:
            self._left -= 1
            self._seq += 1
        return self._seq, self._frame

    def release(self) -> None:
        pass


def lines(stream: io.StringIO) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


def events(stream: io.StringIO) -> List[str]:
    return [entry["event"] for entry in lines(stream)]


class LoggingTestCase(unittest.TestCase):
    """Every test configures the logger itself and restores a silent logger afterwards."""

    def setUp(self) -> None:
        self.stream = io.StringIO()
        self.addCleanup(self.silence)

    def silence(self) -> None:
        logger = get_logger()
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.WARNING)

    def configure(self, level: str = "info") -> io.StringIO:
        configure(level, stream=self.stream)
        return self.stream


class LevelTests(unittest.TestCase):
    def test_names_map_to_levels(self):
        self.assertEqual(level_number("debug"), logging.DEBUG)
        self.assertEqual(level_number("INFO"), logging.INFO)
        self.assertEqual(level_number(" warning "), logging.WARNING)

    def test_an_unknown_name_raises_instead_of_logging_nothing(self):
        with self.assertRaises(ValueError):
            level_number("verbose")


class FormatTests(LoggingTestCase):
    def test_a_line_is_one_json_object_with_the_event_and_fields(self):
        self.configure("info")
        log_event(get_logger("runtime"), "click", kind="left", count=4)
        entry = lines(self.stream)[0]
        self.assertEqual(entry["event"], "click")
        self.assertEqual(entry["level"], "info")
        self.assertEqual(entry["logger"], f"{LOGGER_NAME}.runtime")
        self.assertEqual(entry["kind"], "left")
        self.assertEqual(entry["count"], 4)
        self.assertIn("time", entry)

    def test_the_default_level_drops_the_pipelines_own_events(self):
        # warning + info-level event = nothing written, and no formatting cost either.
        self.configure("warning")
        log_event(get_logger("runtime"), "click", kind="left")
        self.assertEqual(self.stream.getvalue(), "")

    def test_error_level_still_gets_through_at_the_default_level(self):
        self.configure("warning")
        log_event(get_logger("runtime"), "loop_error", level="error", error="boom")
        self.assertEqual(events(self.stream), ["loop_error"])

    def test_configure_twice_replaces_the_handler_instead_of_stacking_it(self):
        self.configure("info")
        second = io.StringIO()
        configure("info", stream=second)
        log_event(get_logger("runtime"), "pause", source="key")
        self.assertEqual(len(lines(second)), 1, "reconfiguring must not duplicate lines")
        self.assertEqual(self.stream.getvalue(), "")

    def test_the_package_logger_does_not_leak_into_the_root_logger(self):
        root = io.StringIO()
        handler = logging.StreamHandler(root)
        logging.getLogger().addHandler(handler)
        self.addCleanup(lambda: logging.getLogger().removeHandler(handler))
        self.configure("info")
        log_event(get_logger("runtime"), "click", kind="left")
        self.assertEqual(root.getvalue(), "", "propagate must stay off")

    def test_unexpected_values_are_serialised_rather_than_crashing_the_log(self):
        formatter = JsonLineFormatter()
        record = logging.LogRecord("kinematic", logging.INFO, __file__, 1, "x", None, None)
        record.event = "weird"
        record.fields = {"value": object()}
        self.assertIn("object object", json.loads(formatter.format(record))["value"])


class FileTests(LoggingTestCase):
    def test_a_file_log_appends_json_lines_and_creates_its_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "nested", "session.jsonl")
            configure("info", path=path)
            logger = get_logger("runtime")
            log_event(logger, "pause", source="gesture")
            log_event(logger, "resume", source="gesture")
            for handler in logger.handlers:
                handler.flush()
            with open(path, "r", encoding="utf-8") as handle:
                entries = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual([entry["event"] for entry in entries], ["pause", "resume"])
        self.assertEqual([entry["source"] for entry in entries], ["gesture", "gesture"])


class RuntimeEventTests(LoggingTestCase):
    """What the pipeline actually writes, at the level a session would run it at."""

    def runtime(self, **overrides: Any) -> HandMouseRuntime:
        overrides.setdefault("start_paused", False)
        overrides.setdefault("show_preview", False)
        return HandMouseRuntime(
            config=HandMouseConfig(**overrides), output=RecordingMouseOutput(), hands=NoHands()
        )

    def test_pause_and_resume_record_their_source(self):
        self.configure("info")
        runtime = self.runtime()
        runtime.toggle_pause(source="key")
        runtime.toggle_pause(source="key")
        entries = lines(self.stream)
        self.assertEqual([entry["event"] for entry in entries], ["pause", "resume"])
        self.assertEqual([entry["source"] for entry in entries], ["key", "key"])

    def test_a_click_is_logged_with_its_kind(self):
        self.configure("info")
        runtime = self.runtime()
        # A pinch held past the hold time: the thumb is on the index tip (ratio 0.22 of the hand's
        # scale) while the middle finger stays away, so exactly the left pair engages.
        pinch = pinch_hand()
        runtime.apply_gestures(pinch, now=0.0)
        runtime.apply_gestures(pinch, now=0.1)
        clicks = [entry for entry in lines(self.stream) if entry["event"] == "click"]
        self.assertEqual(len(clicks), 1)
        self.assertEqual(clicks[0]["kind"], "left")

    def test_tracking_loss_is_counted_and_logged_once(self):
        self.configure("info")
        runtime = self.runtime()
        runtime.apply_gestures(None, now=0.0)
        runtime.apply_gestures(None, now=0.1)  # still lost: no second event
        self.assertEqual(events(self.stream).count("tracking_lost"), 1)
        self.assertEqual(runtime.tracking_losses, 1)

    def test_scroll_amounts_are_logged_per_axis(self):
        self.configure("info")
        runtime = self.runtime()
        runtime.send_scroll((5, 7), now=0.0)
        entry = [line for line in lines(self.stream) if line["event"] == "scroll"][0]
        self.assertEqual(entry["vertical"], 5)
        self.assertEqual(entry["horizontal"], 7)

    def test_no_per_frame_lines_at_info_and_only_heartbeats_at_debug(self):
        self.configure("info")
        runtime = self.runtime()
        camera = FakeCamera(frames=90)
        runtime.handle_key = lambda key: True
        runtime.run(camera=camera)
        per_frame = [line for line in lines(self.stream) if line["event"] in ("move", "frame")]
        self.assertEqual(per_frame, [], "info must not carry per-frame logging")

        self.stream.truncate(0)
        self.stream.seek(0)
        configure("debug", stream=self.stream)
        runtime2 = self.runtime()
        runtime2.handle_key = lambda key: True
        runtime2.run(camera=FakeCamera(frames=90))
        heartbeats = [line for line in lines(self.stream) if line["event"] == "frame"]
        self.assertLessEqual(len(heartbeats), 4, "the heartbeat is periodic, not per frame")

    def test_the_loop_brackets_a_session(self):
        self.configure("info")
        runtime = self.runtime()
        runtime.handle_key = lambda key: True
        runtime.run(camera=FakeCamera(frames=5))
        wrote = events(self.stream)
        self.assertEqual(wrote[0], "loop_start")
        self.assertEqual(wrote[-1], "loop_stop")

    def test_close_writes_the_summary_as_fields(self):
        self.configure("info")
        runtime = self.runtime()
        runtime.frames = 12
        runtime.emitted = 7
        runtime.close()
        summary = [line for line in lines(self.stream) if line["event"] == "summary"][0]
        self.assertEqual(summary["frames"], 12)
        self.assertEqual(summary["emitted"], 7)
        self.assertEqual(summary["status"], "active")

    def test_a_loop_exception_leaves_a_machine_readable_reason(self):
        self.configure("info")
        runtime = self.runtime()
        camera = FakeCamera(frames=0, error="Unable to read frames from the webcam.")
        with self.assertRaises(RuntimeError):
            runtime.run(camera=camera)
        error = [line for line in lines(self.stream) if line["event"] == "loop_error"][0]
        self.assertIn("Unable to read frames", error["error"])
        self.assertEqual(error["level"], "error")

    def test_the_window_close_is_logged(self):
        from unittest import mock

        self.configure("info")
        runtime = self.runtime(show_preview=True)
        with mock.patch.object(runtime_module.cv2, "imshow"), mock.patch.object(
            runtime_module.cv2, "waitKey", lambda ms: -1
        ), mock.patch.object(runtime_module.cv2, "destroyAllWindows"), mock.patch.object(
            runtime_module.cv2, "getWindowProperty", lambda name, prop: 0.0
        ):
            runtime.run(camera=FakeCamera(frames=5))
        self.assertIn("window_closed", events(self.stream))


if __name__ == "__main__":
    unittest.main()
