"""The "oled" preview HUD: the landing page's Dark OLED look, drawn with plain OpenCV.

The site (``index.html``) sets the palette: pure black, one dominant teal (the logo's ``#54d6c7``),
amber as the single sharp accent for "pressed" and "paused", soft-white text and a muted grey for
secondary information. This module carries that over to the preview window:

* a translucent black status panel with a teal accent bar instead of bare coloured text lines;
* a drawn log-scale bar for the sensitivity dial instead of ``[###-------]``;
* the hand skeleton as soft-white bones, black joints ringed in teal and glowing teal fingertips,
  with an amber spark where the thumb meets the finger of an engaged pinch;
* the absolute-mode control region (the inner 70% of the frame that covers the screen) as a dashed
  teal rectangle, so it is visible where the hand has to stay;
* the PAUSED banner as an amber strip at the bottom.

**What OpenCV cannot match:** ``cv2.putText`` only has the built-in Hershey vector fonts, and this
OpenCV build has no ``cv2.freetype``, so the site's Clash Display / Satoshi cannot be used. Letter
spacing on the labels is emulated by drawing them a character at a time. There is no blur either,
so the glow is a dim halo drawn under each bright dot rather than a real gaussian bloom.

Purely visual, like :mod:`kinematic.preview`: no mouse output, no camera access. Every function
draws in place on the frame it is given, and draws nothing for a section that has nothing to show.
The ``"classic"`` HUD (``HandMouseConfig.hud_style``) is still in ``runtime.draw_preview``,
unchanged.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .landmarks import HandLandmarks
from .mapping import Region
from .preview import HAND_CONNECTIONS, flip_point, to_pixel

#: The preview HUD styles: "oled" (this module) and "classic" (the original coloured text lines).
HUD_STYLES: Tuple[str, ...] = ("oled", "classic")

Color = Tuple[int, int, int]

# The site's palette, converted to OpenCV's BGR order.
BLACK: Color = (0, 0, 0)
TEXT: Color = (234, 237, 237)        # #ededea
MUTED: Color = (152, 145, 141)       # #8d9198
ACCENT: Color = (199, 214, 84)       # #54d6c7, the logo's teal
ACCENT_DIM: Color = (70, 75, 29)     # the teal at ~35%: the fingertip halos
ACCENT_MID: Color = (125, 135, 53)   # the teal at ~63%: the dashed control region and its label
AMBER: Color = (71, 181, 255)        # #ffb547
AMBER_DIM: Color = (25, 63, 89)
LINE: Color = (40, 40, 40)           # the site's 8% white hairline, on black
TRACK: Color = (72, 72, 72)          # the dial's empty track: must read against the darkened panel
BONE: Color = (150, 152, 152)        # the site's 55% white bones, on black

#: Fingertip landmarks, drawn larger and glowing, as on the site's demo hand.
FINGERTIPS: Tuple[int, ...] = (4, 8, 12, 16, 20)

#: For an engaged gesture, the landmark pairs whose meeting point gets the amber spark.
_CONTACTS = {
    "left_pinch": ((4,), (8,)),
    "right_pinch": ((4,), (12,)),
    "two_finger_scroll": ((4,), (8, 12)),
}

_BODY_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LABEL_FONT = cv2.FONT_HERSHEY_DUPLEX


def check_style(style: str) -> str:
    if style not in HUD_STYLES:
        raise ValueError(f"unknown HUD style {style!r}: expected one of {HUD_STYLES}")
    return style


def ui_scale(frame: np.ndarray) -> float:
    """One size unit for the HUD: 1.0 on the default 640-wide preview, clamped for tiny/huge frames."""
    return min(max(frame.shape[1] / 640.0, 0.6), 2.0)


@dataclass(frozen=True)
class HudPanel:
    """What the status panel shows. ``None`` (or False) leaves that row out."""

    paused: bool = False
    fps_line: Optional[str] = None          # e.g. "29.8 fps  1280x720" (None = show_fps off)
    gesture: Optional[str] = None           # the recogniser state, e.g. "left_pinch"
    sensitivity: Optional[float] = None     # None = the dial is not shown
    sensitivity_position: float = 0.0       # 0..1 along the dial's log scale
    reach_warning: bool = False
    last_action: Optional[str] = None       # the full last-action line, or None before the first


# -- text helpers --------------------------------------------------------------


def _text_width(text: str, scale: float, font: int = _BODY_FONT, thickness: int = 1) -> int:
    return cv2.getTextSize(text, font, scale, thickness)[0][0]


def _tracked_width(text: str, scale: float, spacing: float, font: int = _LABEL_FONT) -> int:
    if not text:
        return 0
    return sum(_text_width(ch, scale, font) for ch in text) + int(round(spacing * (len(text) - 1)))


def _put(frame: np.ndarray, text: str, org: Tuple[int, int], scale: float, color: Color,
         font: int = _BODY_FONT, thickness: int = 1) -> int:
    """Draw *text* with its baseline at *org*; returns the x just after it."""
    cv2.putText(frame, text, org, font, scale, color, thickness, cv2.LINE_AA)
    return org[0] + _text_width(text, scale, font, thickness)


def _put_tracked(frame: np.ndarray, text: str, org: Tuple[int, int], scale: float, color: Color,
                 spacing: float, font: int = _LABEL_FONT) -> int:
    """Letter-spaced uppercase label (the site's mono eyebrows), one character at a time."""
    x, y = org
    for index, ch in enumerate(text):
        x = _put(frame, ch, (x, y), scale, color, font)
        if index < len(text) - 1:
            x += int(round(spacing))
    return x


def _darken(frame: np.ndarray, x0: int, y0: int, x1: int, y1: int, opacity: float) -> None:
    """Blend a rectangle towards black: the translucent panel behind the text."""
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    roi[:] = cv2.convertScaleAbs(roi, alpha=1.0 - opacity)


def _panel(frame: np.ndarray, x0: int, y0: int, x1: int, y1: int, bar: Color, unit: float) -> None:
    """Translucent black panel, hairline border and a 2-unit accent bar on its left edge."""
    _darken(frame, x0, y0, x1, y1, 0.72)
    cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), LINE, 1, cv2.LINE_AA)
    cv2.rectangle(frame, (x0, y0), (x0 + max(2, int(round(2 * unit))) - 1, y1 - 1), bar, -1)


# -- the status panel ----------------------------------------------------------


def _status_row(frame: np.ndarray, panel: HudPanel, x: int, y: int, unit: float) -> int:
    color = AMBER if panel.paused else ACCENT
    radius = max(2, int(round(3.5 * unit)))
    cv2.circle(frame, (x + radius, y - radius - 1), radius, color, -1, cv2.LINE_AA)
    x += 2 * radius + int(8 * unit)
    x = _put_tracked(frame, "PAUSED" if panel.paused else "ACTIVE", (x, y), 0.42 * unit, color, 2.0 * unit)
    if panel.fps_line:
        x = _put(frame, f"   {panel.fps_line}", (x, y), 0.42 * unit, MUTED)
    if panel.gesture:
        x = _put(frame, f"   {panel.gesture.replace('_', ' ')}", (x, y), 0.42 * unit, TEXT)
    return x


def _dial_row(frame: np.ndarray, panel: HudPanel, x: int, y: int, unit: float) -> int:
    x = _put_tracked(frame, "SENS", (x, y), 0.38 * unit, MUTED, 2.0 * unit)
    x = _put(frame, f"  {panel.sensitivity:5.2f}  ", (x, y), 0.42 * unit, TEXT)
    track_w, track_h = int(84 * unit), max(2, int(round(3 * unit)))
    top = y - int(6 * unit)
    cv2.rectangle(frame, (x, top), (x + track_w, top + track_h), TRACK, -1)
    # Never drawn empty, for the same reason as runtime.sensitivity_bar(): the floor is a usable gain.
    filled = max(track_h, int(round(min(max(panel.sensitivity_position, 0.0), 1.0) * track_w)))
    cv2.rectangle(frame, (x, top), (x + filled, top + track_h), ACCENT, -1)
    x += track_w
    x = _put(frame, "   ] faster  [ slower  0 reset", (x, y), 0.38 * unit, MUTED)
    if panel.reach_warning:
        x = _put(frame, "   ! edges unreachable", (x, y), 0.38 * unit, AMBER)
    return x


def draw_status_panel(frame: np.ndarray, panel: HudPanel) -> int:
    """The top-left panel. Returns how many rows it drew (0 = nothing at all, frame untouched)."""
    rows = []
    if panel.fps_line is not None or panel.paused:
        rows.append(_status_row)
    if panel.sensitivity is not None:
        rows.append(_dial_row)
    if panel.last_action:
        rows.append(lambda f, p, x, y, u: _put(f, p.last_action, (x, y), 0.42 * u, AMBER))
    if not rows:
        return 0

    unit = ui_scale(frame)
    margin, pad_x, pad_y, row_h = int(10 * unit), int(14 * unit), int(10 * unit), int(22 * unit)
    x0, y0 = margin, margin
    text_x = x0 + pad_x + int(2 * unit)

    # Measure on a scratch canvas so the panel can be sized to its content before it is drawn.
    scratch = np.zeros((row_h * len(rows) + 2 * pad_y, frame.shape[1] * 2, 3), dtype=np.uint8)
    right = max(row(scratch, panel, text_x, pad_y + row_h * (i + 1) - int(6 * unit), unit)
                for i, row in enumerate(rows))
    x1 = min(frame.shape[1] - margin, right + pad_x)
    y1 = y0 + 2 * pad_y + row_h * len(rows) - int(4 * unit)

    _panel(frame, x0, y0, x1, y1, AMBER if panel.paused else ACCENT, unit)
    for i, row in enumerate(rows):
        row(frame, panel, text_x, y0 + pad_y + row_h * (i + 1) - int(6 * unit), unit)
    return len(rows)


def draw_paused_banner(frame: np.ndarray, text: str = "PAUSED  -  press SPACE to control the mouse") -> None:
    """An amber strip centred at the bottom: the one instruction a paused session needs."""
    unit = ui_scale(frame)
    height, width = frame.shape[:2]
    scale = 0.45 * unit
    text_w = _tracked_width(text, scale, 1.2 * unit)
    pad_x, pad_y = int(16 * unit), int(10 * unit)
    box_h = int(16 * unit) + 2 * pad_y
    x0 = max(0, (width - text_w) // 2 - pad_x)
    x1 = min(width, x0 + text_w + 2 * pad_x)
    y1 = height - int(12 * unit)
    y0 = y1 - box_h
    _panel(frame, x0, y0, x1, y1, AMBER, unit)
    _put_tracked(frame, text, (x0 + pad_x, y1 - pad_y - int(2 * unit)), scale, AMBER, 1.2 * unit)


# -- the frame overlays ----------------------------------------------------------


def draw_control_region(frame: np.ndarray, region: Region, flip: str = "none") -> None:
    """The absolute-mode control region as a dashed teal rectangle, with a small label.

    *region* is in the landmarks' (mirrored) frame; *flip* is the same landmark flip the skeleton
    uses, so the rectangle lands where the hand has to be in the displayed orientation.
    """
    height, width = frame.shape[:2]
    corners = [to_pixel(flip_point(p, flip), width, height)
               for p in ((region.left, region.top), (region.right, region.bottom))]
    x0, x1 = sorted(c[0] for c in corners)
    y0, y1 = sorted(c[1] for c in corners)
    unit = ui_scale(frame)
    dash, gap = max(3, int(7 * unit)), max(3, int(6 * unit))

    def dashed(a: Tuple[int, int], b: Tuple[int, int]) -> None:
        length = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
        for start in range(0, length, dash + gap):
            end = min(start + dash, length)
            t0, t1 = start / max(length, 1), end / max(length, 1)
            p0 = (int(a[0] + (b[0] - a[0]) * t0), int(a[1] + (b[1] - a[1]) * t0))
            p1 = (int(a[0] + (b[0] - a[0]) * t1), int(a[1] + (b[1] - a[1]) * t1))
            cv2.line(frame, p0, p1, ACCENT_MID, 1, cv2.LINE_AA)

    dashed((x0, y0), (x1, y0))
    dashed((x1, y0), (x1, y1))
    dashed((x1, y1), (x0, y1))
    dashed((x0, y1), (x0, y0))
    percent = int(round((region.right - region.left) * 100))
    label = f"CONTROL REGION  {percent}%"
    scale, spacing = 0.34 * unit, 1.5 * unit
    lx, ly = x0 + int(8 * unit), y1 - int(8 * unit)
    _darken(frame, lx - int(4 * unit), ly - int(12 * unit), lx + _tracked_width(label, scale, spacing, _BODY_FONT)
            + int(10 * unit), ly + int(5 * unit), 0.6)
    _put_tracked(frame, label, (lx, ly), scale, ACCENT_MID, spacing, font=_BODY_FONT)


def _glow_dot(frame: np.ndarray, center: Tuple[int, int], radius: int, color: Color, halo: Color) -> None:
    cv2.circle(frame, center, radius * 2, halo, -1, cv2.LINE_AA)
    cv2.circle(frame, center, radius, color, -1, cv2.LINE_AA)


def draw_hand_oled(frame: np.ndarray, hand: HandLandmarks, flip: str = "none",
                   gesture: Optional[str] = None) -> int:
    """The site's demo hand: soft-white bones, teal-ringed joints, glowing tips, an amber contact.

    Same contract as :func:`kinematic.preview.draw_hand`: an incomplete detection draws nothing
    and returns 0; otherwise it returns the number of elements drawn.
    """
    if not hand.is_complete:
        return 0
    height, width = frame.shape[:2]
    unit = ui_scale(frame)

    def pixel(index: int) -> Tuple[int, int]:
        return to_pixel(flip_point(hand.point(index), flip), width, height)

    drawn = 0
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, pixel(start), pixel(end), BONE, max(1, int(round(1.4 * unit))), cv2.LINE_AA)
        drawn += 1

    joint_r = max(2, int(round(3 * unit)))
    tip_r = max(2, int(round(4 * unit)))
    for index in range(len(hand)):
        if index in FINGERTIPS:
            _glow_dot(frame, pixel(index), tip_r, ACCENT, ACCENT_DIM)
        else:
            cv2.circle(frame, pixel(index), joint_r, BLACK, -1, cv2.LINE_AA)
            cv2.circle(frame, pixel(index), joint_r, ACCENT, 1, cv2.LINE_AA)
        drawn += 1

    contact = _CONTACTS.get(gesture or "")
    if contact is not None:
        def mean(indices: Sequence[int]) -> np.ndarray:
            return np.mean([pixel(i) for i in indices], axis=0)
        point = (mean(contact[0]) + mean(contact[1])) / 2.0
        _glow_dot(frame, (int(point[0]), int(point[1])), tip_r + 1, AMBER, AMBER_DIM)
        drawn += 1
    return drawn
