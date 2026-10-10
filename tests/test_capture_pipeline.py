import unittest

from kinematic.capture import CameraError, ThreadedCamera
from kinematic.config import HandMouseConfig
from kinematic.gesture import needs_cursor_update


class PerformanceDefaultsTests(unittest.TestCase):
    def test_mouse_pause_default_is_zero(self):
        # pyautogui's own default (0.1 s) is slept before every public call, which is
        # 100 ms added to each cursor move/click/scroll.
        self.assertEqual(HandMouseConfig().mouse_pause, 0.0)

    def test_lite_model_complexity_is_the_default(self):
        self.assertEqual(HandMouseConfig().model_complexity, 0)

    def test_preview_is_scaled_not_full_size(self):
        config = HandMouseConfig()
        self.assertTrue(0 < config.preview_scale <= 1.0)

    def test_capture_geometry_is_sane(self):
        config = HandMouseConfig()
        self.assertGreater(config.capture_width, 0)
        self.assertGreater(config.capture_height, 0)
        self.assertEqual(config.capture_buffer_size, 1)


class CursorUpdateGuardTests(unittest.TestCase):
    def test_zero_threshold_always_updates(self):
        self.assertTrue(needs_cursor_update((0.0, 0.0), (0.0, 0.0), 0.0))

    def test_tiny_move_is_skipped(self):
        self.assertFalse(needs_cursor_update((100.0, 100.0), (100.2, 100.0), 0.5))

    def test_move_at_or_above_threshold_updates(self):
        self.assertTrue(needs_cursor_update((100.0, 100.0), (100.5, 100.0), 0.5))

    def test_distance_is_euclidean_not_axis_aligned(self):
        # 3-4-5 triangle: a (3, 4) offset is 5 px, not 4 or 7.
        self.assertFalse(needs_cursor_update((0.0, 0.0), (3.0, 4.0), 5.1))
        self.assertTrue(needs_cursor_update((0.0, 0.0), (3.0, 4.0), 5.0))


class ThreadedCameraErrorTests(unittest.TestCase):
    def test_missing_camera_raises_camera_error(self):
        with self.assertRaises(CameraError):
            ThreadedCamera(index=99)


if __name__ == "__main__":
    unittest.main()
