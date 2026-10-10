"""Phase 3 geometry tests: distances, midpoints, hand size, clamping, and screen mapping.

Pure functions only — no camera, no landmarks from MediaPipe, no mouse output. The synthetic
hand lives in tests/test_pointer_reference.py; here the points are explicit so every expected
value is arithmetic rather than a recorded number.
"""

import unittest
from types import SimpleNamespace
from typing import List, Optional, Tuple

from kinematic.gesture import (
    INDEX_MCP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    PINKY_MCP,
    THUMB_TIP,
    INDEX_TIP,
    clamp_point,
    hand_scale,
    hand_span,
    midpoint,
    midpoints,
    mirror_point,
    pinch_distances,
    point_distance,
)
from kinematic.mapping import Region, map_point_to_screen, region_properties

HAND_POINTS: List[Tuple[float, float]] = [
    (0.50, 0.60),  # 0 wrist
    (0.46, 0.56), (0.43, 0.52), (0.41, 0.49), (0.40, 0.47),  # 1-4 thumb
    (0.45, 0.45), (0.44, 0.38), (0.43, 0.34), (0.42, 0.30),  # 5-8 index
    (0.50, 0.44), (0.50, 0.36), (0.50, 0.31), (0.50, 0.27),  # 9-12 middle
    (0.55, 0.45), (0.56, 0.37), (0.57, 0.32), (0.58, 0.28),  # 13-16 ring
    (0.60, 0.47), (0.62, 0.40), (0.63, 0.35), (0.64, 0.31),  # 17-20 pinky
]


def hand_of(points: Optional[List[Tuple[float, float]]] = None):
    points = points or HAND_POINTS
    return [SimpleNamespace(x=x, y=y, z=0.0) for x, y in points]


class DistanceTests(unittest.TestCase):
    def test_point_distance_is_euclidean(self):
        self.assertAlmostEqual(point_distance((0.0, 0.0), (3.0, 4.0)), 5.0)
        self.assertAlmostEqual(point_distance((0.1, 0.2), (0.1, 0.2)), 0.0)

    def test_midpoint_of_two_points(self):
        self.assertEqual(midpoint((0.0, 0.0), (1.0, 0.5)), (0.5, 0.25))
        self.assertEqual(midpoint((2.0, 2.0), (2.0, 2.0)), (2.0, 2.0))

    def test_midpoints_is_the_centroid(self):
        self.assertEqual(midpoints([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]), (1 / 3, 1 / 3))

    def test_midpoints_rejects_an_empty_sequence(self):
        with self.assertRaises(ValueError):
            midpoints([])

    def test_pinch_distances_match_the_landmarks(self):
        distances = pinch_distances(hand_of())
        thumb = HAND_POINTS[THUMB_TIP]
        index = HAND_POINTS[INDEX_TIP]
        middle = HAND_POINTS[MIDDLE_TIP]

        self.assertAlmostEqual(distances.thumb_index, point_distance(thumb, index))
        self.assertAlmostEqual(distances.thumb_middle, point_distance(thumb, middle))
        self.assertAlmostEqual(distances.index_middle, point_distance(index, middle))
        # The distances are named, so a mixed-up index cannot pass silently.
        self.assertEqual(distances.thumb_index, distances[0])
        self.assertEqual(distances.index_middle, distances[2])

    def test_hand_size_proxies_are_positive_and_independent(self):
        hand = hand_of()
        self.assertAlmostEqual(hand_scale(hand), point_distance(HAND_POINTS[0], HAND_POINTS[MIDDLE_MCP]))
        self.assertAlmostEqual(hand_span(hand), point_distance(HAND_POINTS[INDEX_MCP], HAND_POINTS[PINKY_MCP]))
        self.assertGreater(hand_span(hand), 0.0)

    def test_hand_size_proxies_have_a_floor(self):
        degenerate = hand_of([(0.5, 0.5)] * 21)
        self.assertGreater(hand_scale(degenerate), 0.0)
        self.assertGreater(hand_span(degenerate), 0.0)

    def test_clamp_point_keeps_points_on_screen(self):
        self.assertEqual(clamp_point((-5.0, 1200.0), 800.0, 600.0), (0.0, 600.0))
        self.assertEqual(clamp_point((400.0, 300.0), 800.0, 600.0), (400.0, 300.0))

    def test_mirror_point_flips_x(self):
        self.assertEqual(mirror_point((0.25, 0.75)), (0.75, 0.75))
        x, y = mirror_point(mirror_point((0.3, 0.4)))
        self.assertAlmostEqual(x, 0.3, places=12)
        self.assertAlmostEqual(y, 0.4, places=12)


class RegionTests(unittest.TestCase):
    def test_full_region(self):
        region = Region.full()
        self.assertEqual(region_properties(region)["width"], 1.0)
        self.assertEqual(region.center, (0.5, 0.5))
        self.assertTrue(region.contains((0.0, 1.0)))

    def test_symmetric_margin_keeps_the_middle(self):
        region = Region.symmetric(0.15)
        self.assertEqual((region.left, region.top, region.right, region.bottom), (0.15, 0.15, 0.85, 0.85))
        self.assertAlmostEqual(region.width, 0.7)
        self.assertEqual(region.center, (0.5, 0.5))

    def test_zero_margin_is_the_full_frame(self):
        self.assertEqual(Region.symmetric(0.0), Region.full())

    def test_invalid_regions_are_rejected(self):
        with self.assertRaises(ValueError):
            Region(left=0.8, right=0.2)
        with self.assertRaises(ValueError):
            Region(top=-0.1)
        with self.assertRaises(ValueError):
            Region.symmetric(0.6)

    def test_contains_and_clamp(self):
        region = Region.symmetric(0.2)
        self.assertTrue(region.contains((0.5, 0.5)))
        self.assertFalse(region.contains((0.1, 0.5)))
        self.assertEqual(region.clamp_point((0.1, 0.9)), (0.2, 0.8))

    def test_to_unit_is_zero_at_the_top_left_of_the_region(self):
        region = Region.symmetric(0.25)
        self.assertEqual(region.to_unit((0.25, 0.25)), (0.0, 0.0))
        self.assertEqual(region.to_unit((0.75, 0.75)), (1.0, 1.0))
        self.assertEqual(region.to_unit((0.5, 0.5)), (0.5, 0.5))


class ScreenMappingTests(unittest.TestCase):
    def test_full_frame_to_full_screen_corners_and_centre(self):
        screen = (1920, 1080)
        self.assertEqual(map_point_to_screen((0.0, 0.0), screen), (0.0, 0.0))
        self.assertEqual(map_point_to_screen((1.0, 1.0), screen), (1920.0, 1080.0))
        self.assertEqual(map_point_to_screen((0.5, 0.5), screen), (960.0, 540.0))

    def test_frame_region_is_expanded_to_fill_the_screen(self):
        screen = (1920, 1080)
        region = Region.symmetric(0.15)  # the reachable middle 70%
        self.assertEqual(map_point_to_screen((0.15, 0.15), screen, frame_region=region), (0.0, 0.0))
        self.assertEqual(map_point_to_screen((0.85, 0.85), screen, frame_region=region), (1920.0, 1080.0))
        self.assertEqual(map_point_to_screen((0.5, 0.5), screen, frame_region=region), (960.0, 540.0))

    def test_points_outside_the_frame_region_are_clamped_onto_its_edge(self):
        screen = (800, 600)
        region = Region.symmetric(0.2)
        self.assertEqual(map_point_to_screen((0.0, 0.0), screen, frame_region=region), (0.0, 0.0))
        self.assertEqual(map_point_to_screen((1.0, 1.0), screen, frame_region=region), (800.0, 600.0))

    def test_clamping_can_be_switched_off(self):
        screen = (800, 600)
        region = Region.symmetric(0.2)
        x, y = map_point_to_screen((0.0, 0.0), screen, frame_region=region, clamp=False)
        self.assertLess(x, 0.0)
        self.assertLess(y, 0.0)

    def test_screen_region_keeps_a_margin_free(self):
        screen = (1000, 500)
        screen_region = Region.symmetric(0.1)
        self.assertEqual(map_point_to_screen((0.0, 0.0), screen, screen_region=screen_region), (100.0, 50.0))
        self.assertEqual(map_point_to_screen((1.0, 1.0), screen, screen_region=screen_region), (900.0, 450.0))

    def test_mirroring_flips_the_horizontal_axis(self):
        screen = (1000, 1000)
        self.assertEqual(map_point_to_screen((0.25, 0.5), screen, mirror=True), (750.0, 500.0))
        self.assertEqual(map_point_to_screen((0.25, 0.5), screen, mirror=False), (250.0, 500.0))

    def test_mapping_is_monotonic_in_both_axes(self):
        screen = (1000, 1000)
        region = Region.symmetric(0.1)
        previous_x = previous_y = -1.0
        for step in range(11):
            # Inside the control region: points outside it legitimately clamp to the same
            # edge, which is not a monotonicity violation but would look like one here.
            point = (0.1 + 0.8 * step / 10.0, 0.1 + 0.8 * step / 10.0)
            x, y = map_point_to_screen(point, screen, frame_region=region)
            self.assertGreater(x, previous_x)
            self.assertGreater(y, previous_y)
            previous_x, previous_y = x, y

    def test_absolute_mapping_is_now_region_based_by_default(self):
        # The old mapping multiplied the frame point by pointer_scale 1.8 and clamped, which
        # skewed the reachable region off-centre. Both defaults are now calibrated values.
        from kinematic.config import HandMouseConfig

        config = HandMouseConfig()
        self.assertEqual(config.pointer_scale, 1.0)
        self.assertAlmostEqual(config.frame_margin, 0.15)
        self.assertEqual(config.screen_margin, 0.0)


if __name__ == "__main__":
    unittest.main()
