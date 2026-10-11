import math
from typing import Any, Iterator, NamedTuple, Optional, Sequence, Tuple

Point = Tuple[float, float]

# MediaPipe hand landmark indices (21 landmarks per hand).
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_MCP = 13
PINKY_MCP = 17

#: Landmarks averaged for the palm point. The wrist joint and the four knuckles barely
#: move when the fingers curl or spread — unlike the fingertips, which move a lot.
PALM_LANDMARKS = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)

#: Floor for hand_scale, so a degenerate detection cannot divide by zero.
MIN_HAND_SCALE = 1e-6


def point_distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def smooth_point(current: Point, target: Point, factor: float) -> Point:
    return (
        current[0] + (target[0] - current[0]) * factor,
        current[1] + (target[1] - current[1]) * factor,
    )


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def landmark_points(landmarks: Sequence[object]) -> list[Point]:
    return [(landmark.x, landmark.y) for landmark in landmarks]


#: The distance metrics :func:`is_pinch` understands. ``"euclidean"`` is the straight-line
#: distance; ``"l1"`` is ``|dx| + |dy|``, which is what the pre-Phase-4 code wrote out inline.
PINCH_METRICS = ("euclidean", "l1")


def is_pinch(landmarks: Sequence[object], threshold: float = 0.06, metric: str = "euclidean") -> bool:
    """True when the thumb and index tips are closer together than *threshold*.

    The metric is an explicit argument rather than an accident of where the test was written:
    ``"euclidean"`` (default) is what the recogniser's hand-scale-relative ratios are built on,
    and ``"l1"`` reproduces the inline ``abs(dx) + abs(dy)`` test exactly, so the legacy engine can
    call this one helper without its feel changing (L1 is always >= the Euclidean distance, so at
    the same threshold it engages slightly less readily).
    """
    if metric not in PINCH_METRICS:
        raise ValueError(f"unknown metric {metric!r}: expected one of {PINCH_METRICS}")
    points = landmark_points(landmarks)
    if len(points) < 21:
        return False
    thumb, index = points[4], points[8]
    if metric == "l1":
        return abs(thumb[0] - index[0]) + abs(thumb[1] - index[1]) < threshold
    return point_distance(thumb, index) < threshold


def is_right_click_gesture(landmarks: Sequence[object], threshold: float = 0.08) -> bool:
    points = landmark_points(landmarks)
    if len(points) < 21:
        return False
    thumb_to_index = point_distance(points[4], points[8])
    index_to_middle = point_distance(points[8], points[12])
    return thumb_to_index < threshold and index_to_middle < threshold


def relative_move_delta(previous: Optional[Point], current: Point) -> Point:
    if previous is None:
        return (0.0, 0.0)
    return (current[0] - previous[0], current[1] - previous[1])


def normalize_screen_point(point: Point, width: int, height: int) -> Point:
    return (point[0] * width, point[1] * height)


def needs_cursor_update(current: Point, target: Point, min_pixels: float = 0.0) -> bool:
    """True when moving the cursor from *current* to *target* is worth the call.

    One pyautogui move costs ~11 ms on macOS, so re-issuing a move that cannot shift
    the cursor by even a fraction of a pixel is pure added latency.
    """
    if min_pixels <= 0:
        return True
    return point_distance(current, target) >= min_pixels


def landmark_point(landmarks: Sequence[Any], index: int) -> Point:
    landmark = landmarks[index]
    return (landmark.x, landmark.y)


def wrist_point(landmarks: Sequence[Any]) -> Point:
    """The wrist joint: the most stable point on the hand under finger movement."""
    return landmark_point(landmarks, WRIST)


def palm_center(landmarks: Sequence[Any]) -> Point:
    """Mean of the wrist and the four knuckles.

    Curling or spreading the fingers moves the fingertips a long way and moves this
    point almost not at all, which is what stops finger articulation from dragging
    the cursor around while the hand itself is still.
    """
    points = [landmark_point(landmarks, index) for index in PALM_LANDMARKS]
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def hand_scale(landmarks: Sequence[Any]) -> float:
    """Wrist → middle-knuckle distance in normalized frame units (a hand-size proxy).

    Used to keep the pointing feel constant as the hand moves nearer to or further
    from the camera, and by later phases for hand-size-relative thresholds.
    """
    return max(point_distance(wrist_point(landmarks), landmark_point(landmarks, MIDDLE_MCP)), MIN_HAND_SCALE)


def pointer_reference_point(landmarks: Sequence[Any], reference: str = "palm") -> Point:
    """The hand point the cursor is derived from.

    ``"palm"`` (default) averages the wrist and knuckles, ``"wrist"`` uses the wrist
    joint alone (even more stable), ``"index"`` keeps the old fingertip behaviour.
    """
    if reference == "wrist":
        return wrist_point(landmarks)
    if reference == "index":
        return landmark_point(landmarks, INDEX_TIP)
    return palm_center(landmarks)


def pointer_delta(previous: Point, current: Point, reference_scale: Optional[float] = None) -> Point:
    """Displacement between two reference points, optionally in units of *reference_scale*."""
    dx = current[0] - previous[0]
    dy = current[1] - previous[1]
    if reference_scale:
        dx /= reference_scale
        dy /= reference_scale
    return (dx, dy)


# --- Phase 3 geometry ---------------------------------------------------------------


class PinchDistances(NamedTuple):
    """The three pinch distances that matter for gestures, in normalized frame units."""

    thumb_index: float
    thumb_middle: float
    index_middle: float


def midpoint(a: Point, b: Point) -> Point:
    """The point halfway between *a* and *b*."""
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def midpoints(points: Sequence[Point]) -> Point:
    """Centroid of a sequence of points (used for the palm and multi-point references)."""
    if not points:
        raise ValueError("midpoints() needs at least one point")
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def pinch_distances(landmarks: Sequence[Any]) -> PinchDistances:
    """Thumb-index, thumb-middle and index-middle distances for one hand."""
    thumb = landmark_point(landmarks, THUMB_TIP)
    index = landmark_point(landmarks, INDEX_TIP)
    middle = landmark_point(landmarks, MIDDLE_TIP)
    return PinchDistances(
        thumb_index=point_distance(thumb, index),
        thumb_middle=point_distance(thumb, middle),
        index_middle=point_distance(index, middle),
    )


def hand_span(landmarks: Sequence[Any]) -> float:
    """Index-knuckle to pinky-knuckle distance: a second, independent hand-size proxy.

    Unlike :func:`hand_scale` (which runs from the wrist to the middle knuckle) this one
    contains no wrist, so the two disagree when the wrist landmark is noisy — a useful
    cross-check for Phase 4's relative thresholds.
    """
    return max(
        point_distance(landmark_point(landmarks, INDEX_MCP), landmark_point(landmarks, PINKY_MCP)),
        MIN_HAND_SCALE,
    )


def clamp_point(point: Point, width: float, height: float) -> Point:
    """Clamp a pixel-space point into a 0..width by 0..height box."""
    return (clamp(point[0], 0.0, width), clamp(point[1], 0.0, height))


def mirror_point(point: Point) -> Point:
    """Mirror a normalized point horizontally (``x`` becomes ``1 - x``).

    The pipeline flips the frame before detection, so landmarks are already mirrored;
    this exists for the raw (pre-flip) coordinates and to make the orientation explicit
    and testable.
    """
    return (1.0 - point[0], point[1])
