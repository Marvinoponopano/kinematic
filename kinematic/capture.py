"""Webcam capture for the Kinematic control loop.

``cv2.VideoCapture.read()`` blocks for a whole frame interval — about 33 ms on a
30 fps camera — so reading inline in the control loop makes every iteration pay
capture *and* processing one after the other, roughly halving the achievable loop
rate. This module grabs frames on a background thread instead: the control loop can
spend that wait doing MediaPipe work, and it always receives the **newest** frame
rather than whatever had been queued in the driver buffer (stale queued frames are
the usual reason a pipeline still feels laggy after the frame rate is fixed).

Nothing here touches the mouse; it is only the camera side of the pipeline.
"""

import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class CameraError(RuntimeError):
    """Raised when the camera device cannot be opened."""


class ThreadedCamera:
    """Camera whose ``read()`` never blocks on the device.

    ``read()`` returns ``(sequence, frame)``. The sequence number only advances when
    a genuinely new frame arrived, so a caller that wants to avoid re-processing a
    duplicate frame compares it with the previous value.

    The returned array is the buffer the capture thread filled. Treat it as
    read-only: this project only ever reads from it (copy or resize before drawing).
    """

    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        buffer_size: int = 1,
        backend: int = 0,
    ) -> None:
        self._capture = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
        if not self._capture.isOpened():
            self._capture.release()
            raise CameraError(f"Could not open camera index {index}.")

        for prop, value in (
            (cv2.CAP_PROP_FRAME_WIDTH, width),
            (cv2.CAP_PROP_FRAME_HEIGHT, height),
            (cv2.CAP_PROP_FPS, fps),
            (cv2.CAP_PROP_BUFFERSIZE, buffer_size),
        ):
            if value:
                self._capture.set(prop, float(value))

        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._error: Optional[str] = None
        self._running = True
        self._thread = threading.Thread(target=self._pump, name="kinematic-capture", daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        """Grab frames as fast as the device produces them, keeping only the newest."""
        while self._running:
            ok, frame = self._capture.read()
            if not ok:
                if self._running:
                    with self._lock:
                        self._error = "Unable to read frames from the webcam."
                return
            with self._lock:
                self._frame = frame
                self._seq += 1

    @property
    def error(self) -> Optional[str]:
        with self._lock:
            return self._error

    def read(self) -> Tuple[int, Optional[np.ndarray]]:
        with self._lock:
            return self._seq, self._frame

    def wait_ready(self, timeout: float = 3.0) -> bool:
        """Block until the first frame arrives, or the timeout/device error hits."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            seq, frame = self.read()
            if frame is not None:
                return True
            if self.error:
                return False
            time.sleep(0.005)
        return False

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self._capture.release()
