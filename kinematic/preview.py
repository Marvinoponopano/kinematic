"""Debug-preview drawing for the internal landmark representation.

Drawn with OpenCV directly rather than MediaPipe's protobuf drawing utils, for two reasons:
the preview then depends only on :mod:`kinematic.landmarks`, and drawing becomes testable —
feed it a black frame and a synthetic hand and assert the pixels that should light up.

Purely visual: no mouse output, no camera access.

**Orientation.** ``runtime.process_frame`` flips the captured frame before detection (`cv2.flip(frame,
1)`), so the landmarks this module draws live in the *mirrored* frame. The preview therefore has two
jobs that must agree: show the frame in the orientation the user asked for (`display_frame`) and mirror
the landmarks to match it (`landmark_flip`), or the skeleton is drawn reflected beside the hand instead
of on it. :data:`FLIP_MODES` names the four orientations, each described as a flip of the *captured*
frame:

* ``"horizontal"`` (default) — the mirror a webcam selfie shows, and the view the pointer math is
  built on: move your hand right on screen and the cursor goes right.
* ``"none"`` — exactly as the camera delivers it (what the preview showed before this existed, except
  that the skeleton now lands on the hand).
* ``"vertical"`` / ``"both"`` — for a camera that is mounted upside down.
"""

from typing import List, Sequence, Tuple

import cv2
import numpy as np

from .landmarks import HandLandmarks

#: MediaPipe's 21-point hand skeleton.
HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

BONE_COLOR = (0, 255, 0)
JOINT_COLOR = (0, 200, 255)

#: The ways the preview can be flipped for display, each named by its flip of the captured frame.
FLIP_MODES: Tuple[str, ...] = ("horizontal", "none", "vertical", "both")

#: cv2.flip code for the display frame (None = leave it as captured), and the mirror the landmarks
#: need on top of it so they land on the hand: the landmarks arrive already mirrored (see the module
#: docstring), so a display that is *not* mirrored needs them mirrored once more.
_DISPLAY_FLIP_CODE = {"horizontal": 1, "none": None, "vertical": 0, "both": -1}
_LANDMARK_FLIP = {"horizontal": "none", "none": "horizontal", "vertical": "both", "both": "vertical"}


def _check_mode(flip: str) -> str:
    if flip not in FLIP_MODES:
        raise ValueError(f"unknown preview flip {flip!r}: expected one of {FLIP_MODES}")
    return flip


def display_frame(frame: np.ndarray, flip: str = "horizontal") -> np.ndarray:
    """*frame* in the orientation the preview should show, per :data:`FLIP_MODES`."""
    code = _DISPLAY_FLIP_CODE[_check_mode(flip)]
    return frame if code is None else cv2.flip(frame, code)


def landmark_flip(flip: str = "horizontal") -> str:
    """The mirror the landmarks need to land on the hand in that display orientation."""
    return _LANDMARK_FLIP[_check_mode(flip)]


def flip_point(point: Tuple[float, float], flip: str = "none") -> Tuple[float, float]:
    """Mirror a normalized point: ``"none"``, ``"horizontal"``, ``"vertical"`` or ``"both"``."""
    x, y = point
    if flip in ("horizontal", "both"):
        x = 1.0 - x
    if flip in ("vertical", "both"):
        y = 1.0 - y
    return (x, y)


def to_pixel(point: Tuple[float, float], width: int, height: int) -> Tuple[int, int]:
    """Normalized landmark (0..1) to pixel coordinates of a *width* x *height* frame."""
    return (int(round(point[0] * width)), int(round(point[1] * height)))


def draw_hand(
    frame: np.ndarray,
    hand: HandLandmarks,
    bone_color: Tuple[int, int, int] = BONE_COLOR,
    joint_color: Tuple[int, int, int] = JOINT_COLOR,
    joint_radius: int = 2,
    thickness: int = 1,
    flip: str = "none",
) -> int:
    """Draw one hand on *frame* in place. Returns the number of elements drawn.

    An incomplete detection draws nothing and returns 0, so a partial hand can never
    produce a misleading skeleton. ``flip`` mirrors the landmarks to match the frame's display
    orientation — see :func:`landmark_flip`.
    """
    if not hand.is_complete:
        return 0

    height, width = frame.shape[:2]
    drawn = 0

    def pixel(index: int) -> Tuple[int, int]:
        return to_pixel(flip_point(hand.point(index), flip), width, height)

    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            pixel(start),
            pixel(end),
            bone_color,
            thickness,
            cv2.LINE_AA,
        )
        drawn += 1

    for index in range(len(hand)):
        cv2.circle(
            frame,
            pixel(index),
            joint_radius,
            joint_color,
            -1,
            cv2.LINE_AA,
        )
        drawn += 1

    return drawn


def draw_hands(frame: np.ndarray, hands: Sequence[HandLandmarks], **kwargs: object) -> int:
    """Draw every complete hand in *hands*; returns the total element count."""
    return sum(draw_hand(frame, hand, **kwargs) for hand in hands)  # type: ignore[arg-type]


def drawn_elements_per_hand() -> int:
    """How many lines + joints a complete hand draws (bones + 21 joints)."""
    return len(HAND_CONNECTIONS) + 21
