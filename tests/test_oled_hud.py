"""The "oled" preview HUD (kinematic/hud.py): the landing page's look, with the classic HUD's rules.

The classic HUD's own pixel tests pin ``hud_style="classic"``; these cases pin the oled one. What they
hold it to is the same contract the classic overlay has — the skeleton lands on the hand in every
flip mode, the text is never mirrored, nothing is drawn when every overlay is off, an incomplete hand
draws nothing, and drawing never reaches the mouse — plus what is new: the status panel's rows and
colours, the amber contact spark, the control region and the paused banner.
"""

import pathlib
import time
import unittest
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np

from kinematic.config import HandMouseConfig
from kinematic.hud import (
    ACCENT,
    AMBER,
    HUD_STYLES,
    HudPanel,
    check_style,
    draw_hand_oled,
    draw_paused_banner,
    draw_status_panel,
)
from kinematic.landmarks import HandLandmarks, Landmark
from kinematic.main import build_config, parse_args
from kinematic.output import RecordingMouseOutput
from kinematic.recognizer import HandState
from kinematic.runtime import HandMouseRuntime

WIDTH, HEIGHT = 640, 360

# An open hand, in normalized detected-frame coordinates, sitting at the frame's *left* (x ~0.2-0.35),
# so a mirror of it would land at the right: asymmetric on purpose, like test_preview_orientation.
_OPEN = [
    (200, 420), (160, 392), (124, 352), (102, 314), (84, 280), (166, 272), (156, 208), (151, 166),
    (147, 126), (203, 264), (203, 194), (203, 146), (203, 104), (239, 272), (246, 206), (251, 164),
    (255, 128), (272, 292), (290, 242), (300, 208), (308, 176),
]


def open_hand(pts=None) -> HandLandmarks:
    pts = pts or _OPEN
    return HandLandmarks(tuple(Landmark(0.10 + x / 400 * 0.35, 0.10 + y / 460 * 0.80, 0.0) for x, y in pts))


def black() -> np.ndarray:
    return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


def teal_pixels(frame: np.ndarray) -> np.ndarray:
    b, g, r = (frame[:, :, i].astype(int) for i in range(3))
    return np.asarray((b > 150) & (g > 170) & (r < 130))


def amber_pixels(frame: np.ndarray) -> np.ndarray:
    b, g, r = (frame[:, :, i].astype(int) for i in range(3))
    return np.asarray((r > 200) & (g > 130) & (b < 120))


def centroid_x(mask: np.ndarray) -> float:
    columns = np.nonzero(mask)[1]
    return float(columns.mean()) if columns.size else float("nan")


def make_runtime(**overrides: Any) -> HandMouseRuntime:
    config: Dict[str, Any] = {
        "show_preview": True,
        "preview_scale": 1.0,
        "start_paused": False,
        "show_landmarks": False,
        "show_fps": False,
        "show_sensitivity": False,
        "show_last_action": False,
        "show_control_region": False,
    }
    config.update(overrides)
    return HandMouseRuntime(
        config=HandMouseConfig(**config),
        output=RecordingMouseOutput(),
        hands=SimpleNamespace(process=lambda rgb: None),
    )


class StyleSelectionTests(unittest.TestCase):
    def test_oled_is_the_default(self):
        self.assertEqual(HandMouseConfig().hud_style, "oled")
        self.assertEqual(build_config(parse_args([])).hud_style, "oled")

    def test_the_style_can_be_chosen_from_the_command_line(self):
        for style in HUD_STYLES:
            self.assertEqual(build_config(parse_args(["--hud-style", style])).hud_style, style)

    def test_an_unknown_style_is_refused_rather_than_silently_drawn(self):
        with self.assertRaises(ValueError):
            check_style("neon")
        runtime = make_runtime(hud_style="neon")
        with self.assertRaises(ValueError):
            runtime.draw_preview(black())

    def test_classic_is_still_the_original_overlay(self):
        # Same inputs, different looks: the switch really selects a different renderer.
        frames = []
        for style in HUD_STYLES:
            runtime = make_runtime(hud_style=style, show_fps=True)
            frames.append(runtime.draw_preview(black()))
        self.assertFalse((frames[0] == frames[1]).all())


class NothingToShowTests(unittest.TestCase):
    def test_every_overlay_off_draws_nothing(self):
        shown = make_runtime().draw_preview(black())
        self.assertEqual(int(shown.sum()), 0)

    def test_the_input_frame_is_never_drawn_on(self):
        frame = black()
        runtime = make_runtime(show_fps=True, show_landmarks=True, start_paused=True)
        runtime.current_hands = [open_hand()]
        runtime.draw_preview(frame)
        self.assertEqual(int(frame.sum()), 0)

    def test_drawing_emits_nothing(self):
        runtime = make_runtime(show_fps=True, show_landmarks=True, show_sensitivity=True)
        runtime.current_hands = [open_hand()]
        for _ in range(10):
            runtime.draw_preview(black())
        self.assertEqual(runtime.output.counts(), {"move": 0, "click": 0, "scroll": 0})

    def test_the_module_cannot_reach_the_mouse_or_mediapipe(self):
        source = (pathlib.Path(__file__).resolve().parents[1] / "kinematic" / "hud.py").read_text()
        for name in ("import pyautogui", "import mediapipe", "from .output", "from .runtime"):
            self.assertNotIn(name, source)


class SkeletonTests(unittest.TestCase):
    def test_the_skeleton_lands_on_the_hand_in_the_mirror_view(self):
        runtime = make_runtime(show_landmarks=True)
        runtime.current_hands = [open_hand()]
        shown = runtime.draw_preview(black())
        # horizontal display: the landmarks are already mirrored, so they are drawn where they are.
        self.assertLess(centroid_x(teal_pixels(shown)), WIDTH / 2)

    def test_the_skeleton_follows_the_display_flip(self):
        runtime = make_runtime(show_landmarks=True, preview_flip="none")
        runtime.current_hands = [open_hand()]
        shown = runtime.draw_preview(black())
        self.assertGreater(centroid_x(teal_pixels(shown)), WIDTH / 2, "mirrored once more, onto the hand")

    def test_an_incomplete_hand_draws_nothing(self):
        frame = black()
        partial = HandLandmarks(open_hand().landmarks[:3])
        self.assertEqual(draw_hand_oled(frame, partial), 0)
        self.assertEqual(int(frame.sum()), 0)

    def test_a_complete_hand_draws_bones_and_joints(self):
        frame = black()
        self.assertEqual(draw_hand_oled(frame, open_hand()), 21 + 21)
        self.assertTrue(teal_pixels(frame).any(), "teal joints and tips")

    def test_an_engaged_pinch_gets_an_amber_spark_and_neutral_does_not(self):
        neutral, pinched = black(), black()
        draw_hand_oled(neutral, open_hand(), gesture="neutral")
        draw_hand_oled(pinched, open_hand(), gesture="left_pinch")
        self.assertFalse(amber_pixels(neutral).any())
        self.assertTrue(amber_pixels(pinched).any())

    def test_the_runtime_passes_the_gesture_state_through(self):
        runtime = make_runtime(show_landmarks=True)
        runtime.current_hands = [open_hand()]
        runtime.gesture_state = HandState.TWO_FINGER_SCROLL
        self.assertTrue(amber_pixels(runtime.draw_preview(black())).any())


class StatusPanelTests(unittest.TestCase):
    def test_rows_appear_only_for_what_is_enabled(self):
        self.assertEqual(draw_status_panel(black(), HudPanel()), 0)
        self.assertEqual(draw_status_panel(black(), HudPanel(fps_line="30.0 fps")), 1)
        self.assertEqual(draw_status_panel(black(), HudPanel(fps_line="30.0 fps", sensitivity=1.0)), 2)
        self.assertEqual(
            draw_status_panel(black(), HudPanel(fps_line="30.0 fps", sensitivity=1.0, last_action="last: x")), 3
        )

    def test_a_paused_session_always_shows_its_status_in_amber(self):
        frame = black()
        self.assertEqual(draw_status_panel(frame, HudPanel(paused=True)), 1)
        top_left = frame[: HEIGHT // 4, : WIDTH // 2]
        self.assertTrue(amber_pixels(top_left).any())
        self.assertFalse(teal_pixels(top_left).any())

    def test_an_active_session_is_teal(self):
        frame = black()
        draw_status_panel(frame, HudPanel(fps_line="30.0 fps"))
        self.assertTrue(teal_pixels(frame).any())
        self.assertFalse(amber_pixels(frame).any())

    def test_the_last_action_row_waits_for_the_first_action(self):
        runtime = make_runtime(show_last_action=True)
        self.assertEqual(int(runtime.draw_preview(black()).sum()), 0, "nothing before a click")
        runtime.last_action, runtime.last_action_at = "left click", time.monotonic()
        runtime.action_counts = {"left click": 1}
        self.assertTrue(amber_pixels(runtime.draw_preview(black())).any())

    def test_the_readout_is_drawn_after_the_flip(self):
        mirrored = make_runtime(show_fps=True, show_sensitivity=True, preview_flip="horizontal")
        raw = make_runtime(show_fps=True, show_sensitivity=True, preview_flip="none")
        first, second = mirrored.draw_preview(black()), raw.draw_preview(black())
        self.assertTrue(first.any())
        self.assertTrue((first == second).all(), "the panel is the same bitmap in both orientations")

    def test_the_panel_is_translucent_not_opaque(self):
        grey = np.full((HEIGHT, WIDTH, 3), 200, dtype=np.uint8)
        draw_status_panel(grey, HudPanel(fps_line="30.0 fps"))
        # The panel's left padding (between the accent bar and the text): darker than the frame, not black.
        inside = grey[14:18, 15:20].astype(int)
        self.assertTrue(((inside > 20) & (inside < 120)).all(), inside.ravel().tolist())


class OverlayTests(unittest.TestCase):
    def test_the_paused_banner_sits_at_the_bottom(self):
        frame = black()
        draw_paused_banner(frame)
        rows = np.nonzero(amber_pixels(frame))[0]
        self.assertTrue(rows.size)
        self.assertGreater(rows.min(), HEIGHT * 0.75)

    def test_a_paused_runtime_draws_the_banner(self):
        shown = make_runtime(start_paused=True).draw_preview(black())
        self.assertTrue(amber_pixels(shown[int(HEIGHT * 0.75):]).any())

    def test_the_control_region_is_drawn_in_absolute_mode_at_the_margin(self):
        shown = make_runtime(show_control_region=True, pointer_mode="absolute").draw_preview(black())
        lit = np.nonzero(shown.any(axis=2))
        left_edge = int(round(0.15 * WIDTH))
        self.assertTrue(lit[1].size)
        self.assertLessEqual(abs(int(lit[1].min()) - left_edge), 1, "the dashes start at frame_margin")

    def test_the_control_region_is_not_drawn_in_relative_mode(self):
        shown = make_runtime(show_control_region=True, pointer_mode="relative").draw_preview(black())
        self.assertEqual(int(shown.sum()), 0)


if __name__ == "__main__":
    unittest.main()
