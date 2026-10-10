import math
from typing import Optional, Sequence, Tuple

Point = Tuple[float, float]


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


def is_pinch(landmarks: Sequence[object], threshold: float = 0.06) -> bool:
    points = landmark_points(landmarks)
    if len(points) < 21:
        return False
    return point_distance(points[4], points[8]) < threshold


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
