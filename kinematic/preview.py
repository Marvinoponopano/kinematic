"""Debug-preview drawing for the internal landmark representation.

Drawn with OpenCV directly rather than MediaPipe's protobuf drawing utils, for two reasons:
the preview then depends only on :mod:`kinematic.landmarks`, and drawing becomes testable —
feed it a black frame and a synthetic hand and assert the pixels that should light up.

Purely visual: no mouse output, no camera access.
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
) -> int:
    """Draw one hand on *frame* in place. Returns the number of elements drawn.

    An incomplete detection draws nothing and returns 0, so a partial hand can never
    produce a misleading skeleton.
    """
    if not hand.is_complete:
        return 0

    height, width = frame.shape[:2]
    drawn = 0

    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            to_pixel(hand.point(start), width, height),
            to_pixel(hand.point(end), width, height),
            bone_color,
            thickness,
            cv2.LINE_AA,
        )
        drawn += 1

    for index in range(len(hand)):
        cv2.circle(
            frame,
            to_pixel(hand.point(index), width, height),
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
