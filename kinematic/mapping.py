"""Screen-region mapping: normalized camera coordinates to screen pixels.

Two rectangles decide the mapping: the part of the camera frame the user can comfortably
reach, and the part of the screen it should cover. Mapping, say, the inner 70% of the frame
onto the whole screen is what puts the screen corners within reach — you never have to move
your hand to the very edge of the frame, where MediaPipe's landmarks get unreliable.

Pure geometry: no camera, no mouse, no OpenCV.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

Point = Tuple[float, float]


@dataclass(frozen=True)
class Region:
    """A rectangle in normalized units (0..1)."""

    left: float = 0.0
    top: float = 0.0
    right: float = 1.0
    bottom: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.left < self.right <= 1.0:
            raise ValueError(f"left/right must satisfy 0 <= left < right <= 1, got {self.left}, {self.right}")
        if not 0.0 <= self.top < self.bottom <= 1.0:
            raise ValueError(f"top/bottom must satisfy 0 <= top < bottom <= 1, got {self.top}, {self.bottom}")

    @classmethod
    def full(cls) -> "Region":
        return cls()

    @classmethod
    def symmetric(cls, margin: float) -> "Region":
        """The full frame minus *margin* on every side (0.15 keeps 0.15..0.85)."""
        if not 0.0 <= margin < 0.5:
            raise ValueError(f"margin must satisfy 0 <= margin < 0.5, got {margin}")
        return cls(left=margin, top=margin, right=1.0 - margin, bottom=1.0 - margin)

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def center(self) -> Point:
        return (self.left + self.width / 2.0, self.top + self.height / 2.0)

    def contains(self, point: Point) -> bool:
        return self.left <= point[0] <= self.right and self.top <= point[1] <= self.bottom

    def clamp_point(self, point: Point) -> Point:
        """Pull a point onto the region's edges if it falls outside."""
        return (
            min(max(point[0], self.left), self.right),
            min(max(point[1], self.top), self.bottom),
        )

    def to_unit(self, point: Point) -> Point:
        """Where *point* sits inside the region, as 0..1 (values outside stay outside)."""
        return ((point[0] - self.left) / self.width, (point[1] - self.top) / self.height)


def map_point_to_screen(
    point: Point,
    screen_size: Tuple[int, int],
    frame_region: Optional[Region] = None,
    screen_region: Optional[Region] = None,
    mirror: bool = False,
    clamp: bool = True,
) -> Point:
    """Map a normalized camera point to a screen pixel.

    ``frame_region`` is the part of the frame that maps onto ``screen_region`` (both default
    to the full area). Points outside the frame region are clamped onto its edge by default,
    so the mapping never produces coordinates beyond ``screen_region``. ``mirror=True``
    flips x first, for raw (pre-flip) landmark coordinates.
    """
    frame_region = frame_region or Region.full()
    screen_region = screen_region or Region.full()

    if mirror:
        point = (1.0 - point[0], point[1])

    unit = frame_region.to_unit(frame_region.clamp_point(point) if clamp else point)

    screen_width, screen_height = screen_size
    x = screen_region.left * screen_width + unit[0] * screen_region.width * screen_width
    y = screen_region.top * screen_height + unit[1] * screen_region.height * screen_height
    return (x, y)


def region_properties(region: Region) -> dict:
    """Small diagnostic helper (used by the tools and tests)."""
    return {
        "left": region.left,
        "top": region.top,
        "right": region.right,
        "bottom": region.bottom,
        "width": region.width,
        "height": region.height,
    }
