from dataclasses import dataclass


@dataclass
class HandMouseConfig:
    camera_index: int = 0
    window_name: str = "Kinematic"
    smoothing: float = 0.35
    deadzone: float = 0.02
    click_threshold: float = 0.06
    right_click_threshold: float = 0.08
    scroll_threshold: float = 0.04
    debounce_seconds: float = 0.18
    pointer_scale: float = 1.8
