"""Phase 2 tests: the internal landmark representation and the landmark overlay.

Everything here runs without MediaPipe, a camera or a display: MediaPipe's output shape is
faked at the boundary, and drawing is asserted by inspecting the pixels of a black frame.
"""

import unittest
from types import SimpleNamespace
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from kinematic.config import HandMouseConfig
from kinematic.landmarks import (
    HAND_LANDMARK_COUNT,
    HandLandmarks,
    Landmark,
    hand_from_mediapipe,
    hands_from_results,
)
from kinematic.output import RecordingMouseOutput
from kinematic.preview import HAND_CONNECTIONS, draw_hand, draw_hands, drawn_elements_per_hand, to_pixel
from kinematic.runtime import HandMouseRuntime

HAND = [
    (0.50, 0.60),  # 0 wrist
    (0.46, 0.56), (0.43, 0.52), (0.41, 0.49), (0.39, 0.46),  # 1-4 thumb
    (0.45, 0.45), (0.44, 0.38), (0.43, 0.34), (0.42, 0.30),  # 5-8 index
    (0.50, 0.44), (0.50, 0.36), (0.50, 0.31), (0.50, 0.27),  # 9-12 middle
    (0.55, 0.45), (0.56, 0.37), (0.57, 0.32), (0.58, 0.28),  # 13-16 ring
    (0.60, 0.47), (0.62, 0.40), (0.63, 0.35), (0.64, 0.31),  # 17-20 pinky
]


def mediapipe_hand(points: Sequence[Tuple[float, float]], z: float = -0.02) -> Any:
    """Mimic MediaPipe's NormalizedLandmarkList (only the attributes we read)."""
    return SimpleNamespace(landmark=[SimpleNamespace(x=x, y=y, z=z) for x, y in points])


def mediapipe_results(hands: Sequence[Any], labels: Optional[Sequence[str]] = None) -> Any:
    handedness = []
    for index, _ in enumerate(hands):
        label = (labels or ["Right"] * len(hands))[index]
        handedness.append(SimpleNamespace(classification=[SimpleNamespace(label=label, score=0.97)]))
    return SimpleNamespace(multi_hand_landmarks=list(hands), multi_handedness=handedness)


class ConversionTests(unittest.TestCase):
    def test_hand_from_mediapipe_produces_plain_types(self):
        hand = hand_from_mediapipe(mediapipe_hand(HAND), handedness="Left", score=0.9)

        self.assertIsInstance(hand, HandLandmarks)
        self.assertEqual(len(hand), HAND_LANDMARK_COUNT)
        self.assertTrue(all(isinstance(landmark, Landmark) for landmark in hand.landmarks))
        self.assertEqual(hand.handedness, "Left")
        self.assertAlmostEqual(hand.score, 0.9)
        self.assertAlmostEqual(hand[0].x, 0.50)
        self.assertAlmostEqual(hand[0].y, 0.60)
        self.assertAlmostEqual(hand[0].z, -0.02)

    def test_hands_from_results_carries_handedness_and_score(self):
        results = mediapipe_results([mediapipe_hand(HAND)], labels=["Right"])
        hands = hands_from_results(results)

        self.assertEqual(len(hands), 1)
        self.assertEqual(hands[0].handedness, "Right")
        self.assertAlmostEqual(hands[0].score, 0.97)

    def test_hands_from_results_handles_no_hand_and_missing_handedness(self):
        self.assertEqual(hands_from_results(SimpleNamespace(multi_hand_landmarks=None)), [])
        self.assertEqual(hands_from_results(SimpleNamespace()), [])

        hands = hands_from_results(SimpleNamespace(multi_hand_landmarks=[mediapipe_hand(HAND)]))
        self.assertEqual(hands[0].handedness, "unknown")
        self.assertEqual(hands[0].score, 0.0)

    def test_conversion_does_not_keep_a_reference_to_the_proto(self):
        raw = mediapipe_hand(HAND)
        hand = hand_from_mediapipe(raw)
        raw.landmark[0].x = 0.99  # mutate the "proto" afterwards
        self.assertAlmostEqual(hand[0].x, 0.50, msg="the conversion must copy, not alias")

    def test_partial_detection_is_flagged_incomplete(self):
        hand = hand_from_mediapipe(mediapipe_hand(HAND[:5]))
        self.assertFalse(hand.is_complete)
        self.assertTrue(hand_from_mediapipe(mediapipe_hand(HAND)).is_complete)

    def test_sequence_protocol_matches_a_raw_landmark_list(self):
        hand = hand_from_mediapipe(mediapipe_hand(HAND))
        self.assertEqual(hand.point(8), HAND[8])
        self.assertEqual(hand.points(), list(HAND))
        self.assertEqual([(landmark.x, landmark.y) for landmark in hand], list(HAND))


class PreviewDrawingTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((240, 320, 3), dtype=np.uint8)
        self.hand = hand_from_mediapipe(mediapipe_hand(HAND))

    def test_drawing_marks_pixels_and_reports_the_element_count(self):
        drawn = draw_hand(self.frame, self.hand)

        self.assertEqual(drawn, drawn_elements_per_hand())
        self.assertEqual(drawn, len(HAND_CONNECTIONS) + HAND_LANDMARK_COUNT)
        self.assertGreater(int(self.frame.sum()), 0, "something should have been painted")

    def test_a_joint_lands_where_the_landmark_says(self):
        draw_hand(self.frame, self.hand, joint_radius=1)
        x, y = to_pixel(self.hand.point(8), 320, 240)
        self.assertGreater(int(self.frame[y, x].sum()), 0, "the index fingertip should be drawn")

    def test_incomplete_hand_draws_nothing(self):
        partial = hand_from_mediapipe(mediapipe_hand(HAND[:5]))
        drawn = draw_hand(self.frame, partial)

        self.assertEqual(drawn, 0)
        self.assertEqual(int(self.frame.sum()), 0, "a partial detection must not draw a skeleton")

    def test_draw_hands_sums_every_hand(self):
        hands = [self.hand, hand_from_mediapipe(mediapipe_hand(HAND))]
        self.assertEqual(draw_hands(self.frame, hands), 2 * drawn_elements_per_hand())

    def test_to_pixel_scales_with_the_frame(self):
        self.assertEqual(to_pixel((0.5, 0.5), 320, 240), (160, 120))
        self.assertEqual(to_pixel((0.5, 0.5), 640, 480), (320, 240))
        self.assertEqual(to_pixel((0.0, 1.0), 320, 240), (0, 240))


class RuntimeUsesTheInternalRepresentationTests(unittest.TestCase):
    def make_runtime(self, results: Any, output: Optional[RecordingMouseOutput] = None) -> HandMouseRuntime:
        output = output if output is not None else RecordingMouseOutput()
        runtime = HandMouseRuntime(
            config=HandMouseConfig(show_preview=False, start_paused=False, smoothing=1.0),
            output=output,
            hands=SimpleNamespace(process=lambda rgb, _results=results: _results),
        )
        return runtime

    def test_a_frame_is_converted_and_drives_the_cursor(self):
        runtime = self.make_runtime(mediapipe_results([mediapipe_hand(HAND)]))
        runtime.process_frame(np.zeros((240, 320, 3), dtype=np.uint8))
        runtime.process_frame(np.zeros((240, 320, 3), dtype=np.uint8))

        self.assertEqual(len(runtime.current_hands), 1)
        self.assertIsInstance(runtime.current_hands[0], HandLandmarks)
        self.assertIsInstance(runtime.current_landmarks, HandLandmarks)
        self.assertEqual(runtime.current_landmarks.handedness, "Right")

    def test_an_incomplete_hand_is_ignored_rather_than_crashing(self):
        output = RecordingMouseOutput()
        runtime = self.make_runtime(mediapipe_results([mediapipe_hand(HAND[:3])]), output)

        runtime.process_frame(np.zeros((240, 320, 3), dtype=np.uint8))

        self.assertEqual(output.counts(), {"move": 0, "click": 0, "scroll": 0})
        self.assertEqual(runtime.current_landmarks, [])
        self.assertEqual(len(runtime.current_hands), 1, "the detection is still reported for the preview")

    def test_preview_overlay_draws_the_hand(self):
        runtime = self.make_runtime(mediapipe_results([mediapipe_hand(HAND)]))
        runtime.config.show_preview = True
        runtime.config.preview_scale = 0.5
        runtime.config.show_fps = False
        runtime.process_frame(np.zeros((240, 320, 3), dtype=np.uint8))

        preview = runtime.draw_preview(np.zeros((240, 320, 3), dtype=np.uint8), camera=None)

        self.assertEqual(preview.shape, (120, 160, 3))
        self.assertGreater(int(preview.sum()), 0, "the scaled preview should carry the skeleton")

    def test_overlay_can_be_switched_off(self):
        runtime = self.make_runtime(mediapipe_results([mediapipe_hand(HAND)]))
        runtime.config.show_preview = True
        runtime.config.preview_scale = 0.5
        runtime.config.show_fps = False
        runtime.config.show_landmarks = False
        runtime.process_frame(np.zeros((240, 320, 3), dtype=np.uint8))

        preview = runtime.draw_preview(np.zeros((240, 320, 3), dtype=np.uint8), camera=None)

        self.assertEqual(int(preview.sum()), 0, "nothing should be drawn when the overlay is off")


if __name__ == "__main__":
    unittest.main()
