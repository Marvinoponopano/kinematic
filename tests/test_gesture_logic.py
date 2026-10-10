import unittest

from kinematic.gesture import is_pinch, is_right_click_gesture, point_distance, smooth_point


class FakeLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class GestureLogicTests(unittest.TestCase):
    def test_pinch_detection(self):
        landmarks = [FakeLandmark(0.5, 0.5)] * 21
        landmarks[4] = FakeLandmark(0.50, 0.52)
        landmarks[8] = FakeLandmark(0.51, 0.53)
        self.assertTrue(is_pinch(landmarks, threshold=0.05))

    def test_right_click_gesture_detection(self):
        landmarks = [FakeLandmark(0.5, 0.5)] * 21
        landmarks[4] = FakeLandmark(0.49, 0.51)
        landmarks[8] = FakeLandmark(0.50, 0.50)
        landmarks[12] = FakeLandmark(0.51, 0.50)
        self.assertTrue(is_right_click_gesture(landmarks, threshold=0.08))

    def test_smooth_point_blends_toward_target(self):
        current = (0.0, 0.0)
        target = (10.0, 10.0)
        smoothed = smooth_point(current, target, 0.5)
        self.assertEqual(smoothed, (5.0, 5.0))

    def test_point_distance_uses_hypotenuse(self):
        a = (0.0, 0.0)
        b = (3.0, 4.0)
        self.assertEqual(point_distance(a, b), 5.0)


if __name__ == "__main__":
    unittest.main()
