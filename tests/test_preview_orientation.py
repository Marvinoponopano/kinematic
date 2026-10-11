"""Preview orientation: which way the footage is flipped, and the skeleton landing on the hand.

Two things have to agree, and before this they did not: the frame the preview *shows* and the frame the
landmarks were *detected* in. `runtime.process_frame` flips the captured frame for MediaPipe
(`cv2.flip(frame, 1)`), so the landmarks are in mirrored coordinates while `draw_preview` used to draw
the captured frame — which put the skeleton reflected beside the hand.

These tests therefore work from the physical frame of reference:

* ``world``  — the frame as the camera delivers it. A marker block stands in for the hand's position.
* ``detected`` — what MediaPipe saw, i.e. ``cv2.flip(world, 1)``. The landmark coordinates are the
  mirror of the hand's world position, which is what ``process_frame`` stores.
* the preview must show the marker and the skeleton in the *same place*, whichever orientation it is
  flipped to.

No camera, no window, no mouse: every assertion is about pixels on a numpy array.
"""

import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from kinematic.config import HandMouseConfig
from kinematic.landmarks import HandLandmarks, Landmark
from kinematic.main import build_config, parse_args
from kinematic.output import RecordingMouseOutput
from kinematic.preview import (
    FLIP_MODES,
    display_frame,
    draw_hand,
    flip_point,
    landmark_flip,
    to_pixel,
)
from kinematic.runtime import HandMouseRuntime

WIDTH, HEIGHT = 320, 240
#: Where the "hand" is in the world, normalized. Left third, lower half: asymmetric on both axes.
WORLD_HAND = (0.15, 0.75)
MARKER_COLUMNS = (40, 60)  # pixels 40..60 of a 320-wide frame ~= x 0.15
MARKER_ROWS = (170, 190)  # pixels 170..190 of a 240-high frame ~= y 0.75


def world_frame() -> np.ndarray:
    """A black frame with a white block where the hand is (the camera's own orientation)."""
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[MARKER_ROWS[0]:MARKER_ROWS[1], MARKER_COLUMNS[0]:MARKER_COLUMNS[1]] = 255
    return frame


def detected_hand() -> HandLandmarks:
    """The hand as MediaPipe would report it: every landmark mirrored into the detected frame."""
    x, y = WORLD_HAND
    landmarks = tuple(Landmark(1.0 - x, y, 0.0) for _ in range(21))
    return HandLandmarks(landmarks, handedness="Right", score=0.99)


def skeleton_pixels(frame: np.ndarray) -> np.ndarray:
    """Boolean mask of the skeleton: high green (bones) or high red with no blue (joints)."""
    blue, green, red = frame[:, :, 0].astype(int), frame[:, :, 1].astype(int), frame[:, :, 2].astype(int)
    return np.asarray(((green > 150) & (blue < 120)) | ((red > 200) & (blue < 120)))


def marker_pixels(frame: np.ndarray) -> np.ndarray:
    return np.asarray((frame > 200).all(axis=2))


def centroid_x(mask: np.ndarray) -> float:
    columns = np.nonzero(mask)[1]
    return float(columns.mean()) if columns.size else float("nan")


def centroid_y(mask: np.ndarray) -> float:
    rows = np.nonzero(mask)[0]
    return float(rows.mean()) if rows.size else float("nan")


class DisplayFrameTests(unittest.TestCase):
    """The orientation of the picture itself, pixel by pixel."""

    def test_horizontal_is_the_mirror_of_the_captured_frame(self):
        world = world_frame()
        shown = display_frame(world, "horizontal")
        self.assertTrue((shown == np.flip(world, axis=1)).all())
        self.assertGreater(centroid_x(shown > 200), WIDTH / 2, "the marker moves to the right side")

    def test_none_is_exactly_what_the_camera_delivered(self):
        world = world_frame()
        shown = display_frame(world, "none")
        self.assertIs(shown, world, "no copy: the captured frame is shown as it is")
        self.assertLess(centroid_x(shown > 200), WIDTH / 2)

    def test_vertical_flips_top_to_bottom_and_leaves_x_alone(self):
        shown = display_frame(world_frame(), "vertical")
        self.assertAlmostEqual(centroid_x(shown > 200), (MARKER_COLUMNS[0] + MARKER_COLUMNS[1] - 1) / 2, delta=1.5)
        self.assertLess(centroid_y(shown > 200), HEIGHT / 2, "the marker was low, now it is high")

    def test_both_flips_both_axes(self):
        shown = display_frame(world_frame(), "both")
        self.assertGreater(centroid_x(shown > 200), WIDTH / 2)
        self.assertLess(centroid_y(shown > 200), HEIGHT / 2)

    def test_every_mode_is_offered_and_nothing_else(self):
        self.assertEqual(FLIP_MODES, ("horizontal", "none", "vertical", "both"))
        for mode in FLIP_MODES:
            display_frame(world_frame(), mode)
        with self.assertRaises(ValueError):
            display_frame(world_frame(), "sideways")

    def test_the_default_is_the_webcam_mirror(self):
        self.assertEqual(HandMouseConfig().preview_flip, "horizontal")


class SkeletonLandsOnTheHandTests(unittest.TestCase):
    """The property the user actually sees: skeleton and hand in the same place, in every mode."""

    def shown_with_skeleton(self, mode: str) -> np.ndarray:
        preview = display_frame(world_frame(), mode)
        draw_hand(preview, detected_hand(), flip=landmark_flip(mode))
        return preview

    def test_the_skeleton_matches_the_hand_in_every_orientation(self):
        for mode in FLIP_MODES:
            with self.subTest(mode=mode):
                shown = self.shown_with_skeleton(mode)
                hand_gap = abs(centroid_x(skeleton_pixels(shown)) - centroid_x(marker_pixels(shown)))
                self.assertLess(hand_gap, 6.0, f"{mode}: the skeleton must sit on the hand, not beside it")

    def test_the_skeleton_needs_no_extra_mirror_in_the_default_mode(self):
        # The landmarks are already in the mirrored frame, so the mirror view is the one they fit.
        self.assertEqual(landmark_flip("horizontal"), "none")
        self.assertEqual(landmark_flip("none"), "horizontal")
        self.assertEqual(landmark_flip("vertical"), "both")
        self.assertEqual(landmark_flip("both"), "vertical")

    def test_the_old_composition_was_reflected(self):
        # What the code did before: draw the detected landmarks straight onto the captured frame.
        # Pinning it here so nobody "simplifies" the flip away again.
        shown = world_frame()
        draw_hand(shown, detected_hand(), flip="none")
        hand_gap = abs(centroid_x(skeleton_pixels(shown)) - centroid_x(marker_pixels(shown)))
        self.assertGreater(hand_gap, 100.0, "without the display flip the skeleton is on the wrong side")

    def test_flip_point_mirrors_the_requested_axes(self):
        self.assertEqual(flip_point((0.25, 0.25), "none"), (0.25, 0.25))
        self.assertEqual(flip_point((0.25, 0.25), "horizontal"), (0.75, 0.25))
        self.assertEqual(flip_point((0.25, 0.25), "vertical"), (0.25, 0.75))
        self.assertEqual(flip_point((0.25, 0.25), "both"), (0.75, 0.75))

    def test_drawing_still_respects_the_frame_size(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        draw_hand(frame, detected_hand(), flip="horizontal")
        columns = np.nonzero(skeleton_pixels(frame))[1]
        self.assertTrue(columns.size)
        self.assertTrue(0 <= columns.min() and columns.max() < 200)


class PreviewWiringTests(unittest.TestCase):
    """`draw_preview` composes the two halves and draws the readout un-mirrored."""

    def runtime(self, **overrides: Any) -> HandMouseRuntime:
        config: Dict[str, Any] = {
            "show_preview": True,
            "preview_scale": 1.0,
            "show_landmarks": False,
            # Unpaused: the PAUSED banner is red text, which the skeleton mask below would count.
            "start_paused": False,
        }
        config.update(overrides)
        runtime = HandMouseRuntime(
            config=HandMouseConfig(**config),
            output=RecordingMouseOutput(),
            hands=SimpleNamespace(process=lambda rgb: SimpleNamespace(multi_hand_landmarks=None, multi_handedness=None)),
        )
        runtime.current_hands = [detected_hand()]
        return runtime

    def test_the_preview_is_composed_on_the_displayed_orientation(self):
        runtime = self.runtime(show_landmarks=True, show_fps=False, show_sensitivity=False)
        shown = runtime.draw_preview(world_frame())
        self.assertGreater(centroid_x(marker_pixels(shown)), WIDTH / 2, "the default is the mirror view")
        gap = abs(centroid_x(skeleton_pixels(shown)) - centroid_x(marker_pixels(shown)))
        self.assertLess(gap, 6.0, "and the skeleton is on the hand")

    def test_the_preview_flip_can_be_turned_off(self):
        runtime = self.runtime(preview_flip="none", show_landmarks=True, show_fps=False, show_sensitivity=False)
        shown = runtime.draw_preview(world_frame())
        self.assertLess(centroid_x(marker_pixels(shown)), WIDTH / 2, "as the camera delivered it")
        gap = abs(centroid_x(skeleton_pixels(shown)) - centroid_x(marker_pixels(shown)))
        self.assertLess(gap, 6.0, "and the skeleton still lands on the hand")

    def test_the_readout_is_drawn_after_the_flip_so_it_is_readable(self):
        # The same text pixels in both orientations proves it was not mirrored with the image: a
        # flipped fps line would come out as a different bitmap.
        mirrored = self.runtime(preview_flip="horizontal", show_landmarks=False, show_fps=True)
        raw = self.runtime(preview_flip="none", show_landmarks=False, show_fps=True)
        first, second = mirrored.draw_preview(world_frame()), raw.draw_preview(world_frame())
        self.assertTrue((first[14:24, :, :] != 0).any(), "the fps line is drawn")
        self.assertTrue((first[14:24, :, :] == second[14:24, :, :]).all(), "and it is not mirrored")

    def test_the_scaled_preview_keeps_the_orientation(self):
        runtime = self.runtime(preview_scale=0.5, show_landmarks=False, show_fps=False, show_sensitivity=False)
        shown = runtime.draw_preview(world_frame())
        self.assertEqual(shown.shape, (HEIGHT // 2, WIDTH // 2, 3))
        self.assertGreater(centroid_x(marker_pixels(shown)), WIDTH / 4, "still the mirror view")

    def test_no_landmarks_still_returns_the_flipped_frame(self):
        runtime = self.runtime(show_landmarks=False, show_fps=False, show_sensitivity=False, preview_scale=1.0)
        shown = runtime.draw_preview(world_frame())
        self.assertEqual(int(shown.sum()), int(world_frame().sum()), "no overlay, no extra ink")


class PreviewFlipCommandLineTests(unittest.TestCase):
    def test_the_flip_can_be_chosen_from_the_command_line(self):
        for mode in FLIP_MODES:
            config = build_config(parse_args(["--preview-flip", mode]))
            self.assertEqual(config.preview_flip, mode)

    def test_the_default_is_untouched_without_the_flag(self):
        self.assertEqual(build_config(parse_args([])).preview_flip, "horizontal")


if __name__ == "__main__":
    unittest.main()
