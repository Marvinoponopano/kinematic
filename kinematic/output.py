"""Mouse output adapters.

The pipeline never calls pyautogui directly: it goes through a :class:`MouseOutput`,
so a dry run, the paused startup path and the test suite can run the whole pipeline
without ever touching the real cursor.

``RealMouseOutput`` is the only adapter in the project that can affect the machine,
and it is only constructed when the caller asks for it (or when the config leaves
``dry_run`` off and the caller does not inject an adapter).
"""

import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

DEFAULT_SCREEN: Tuple[int, int] = (1920, 1080)


@dataclass(frozen=True)
class MouseEvent:
    """One recorded output call."""

    kind: str  # "move" | "click" | "scroll"
    value: Any  # (x, y) for move, button name for click, amount for scroll
    at: float  # time.monotonic() when it happened


class MouseOutput:
    """Interface every output adapter implements."""

    def screen_size(self) -> Tuple[int, int]:
        raise NotImplementedError

    def position(self) -> Tuple[float, float]:
        raise NotImplementedError

    def move_to(self, x: float, y: float, duration: float = 0.0) -> None:
        raise NotImplementedError

    def click(self, button: str = "left") -> None:
        raise NotImplementedError

    def scroll(self, amount: int) -> None:
        raise NotImplementedError


class RealMouseOutput(MouseOutput):
    """pyautogui-backed adapter: the real cursor, clicks and scroll.

    Recovery story while this adapter is live: the app starts paused, ``SPACE``
    toggles the pause (the preview window keeps keyboard focus while the pointer is
    being driven), ``ESC`` or ``q`` quits, and ``Ctrl-C`` in the terminal always
    works. ``FAILSAFE`` is left configurable: with it on, slamming the pointer into a
    screen corner also aborts.
    """

    def __init__(self, pause: float = 0.0, failsafe: bool = False) -> None:
        import pyautogui  # imported lazily so dry runs need no display/permissions

        self._pyautogui = pyautogui
        pyautogui.FAILSAFE = failsafe
        # pyautogui sleeps PAUSE seconds before every public call; its default of 0.1
        # is 100 ms added to each cursor move, click and scroll.
        pyautogui.PAUSE = pause

    def screen_size(self) -> Tuple[int, int]:
        size = self._pyautogui.size()
        return int(size[0]), int(size[1])

    def position(self) -> Tuple[float, float]:
        point = self._pyautogui.position()
        return float(point[0]), float(point[1])

    def move_to(self, x: float, y: float, duration: float = 0.0) -> None:
        self._pyautogui.moveTo(x, y, duration=duration)

    def click(self, button: str = "left") -> None:
        self._pyautogui.click(button=button)

    def scroll(self, amount: int) -> None:
        self._pyautogui.scroll(amount)


class RecordingMouseOutput(MouseOutput):
    """Mock adapter: records every call and never touches the operating system.

    Used by the test suite, by the ``--dry-run`` CLI flag and anywhere else the
    pipeline should be exercised with no desktop side effects.
    """

    def __init__(self, screen: Tuple[int, int] = DEFAULT_SCREEN) -> None:
        self.events: List[MouseEvent] = []
        self._screen = (int(screen[0]), int(screen[1]))
        self._position = (self._screen[0] / 2.0, self._screen[1] / 2.0)

    def screen_size(self) -> Tuple[int, int]:
        return self._screen

    def position(self) -> Tuple[float, float]:
        return self._position

    def move_to(self, x: float, y: float, duration: float = 0.0) -> None:
        self._position = (float(x), float(y))
        self.events.append(MouseEvent("move", self._position, time.monotonic()))

    def click(self, button: str = "left") -> None:
        self.events.append(MouseEvent("click", button, time.monotonic()))

    def scroll(self, amount: int) -> None:
        self.events.append(MouseEvent("scroll", int(amount), time.monotonic()))

    def of_kind(self, kind: str) -> List[MouseEvent]:
        return [event for event in self.events if event.kind == kind]

    @property
    def moves(self) -> List[MouseEvent]:
        return self.of_kind("move")

    @property
    def clicks(self) -> List[MouseEvent]:
        return self.of_kind("click")

    @property
    def scrolls(self) -> List[MouseEvent]:
        return self.of_kind("scroll")

    def counts(self) -> dict:
        return {
            "move": len(self.moves),
            "click": len(self.clicks),
            "scroll": len(self.scrolls),
        }

    def clear(self) -> None:
        self.events.clear()


def recording_output_if(condition: bool, screen: Optional[Tuple[int, int]] = None) -> Optional[RecordingMouseOutput]:
    """Small helper: a recording adapter when *condition* holds, else None."""
    if not condition:
        return None
    return RecordingMouseOutput() if screen is None else RecordingMouseOutput(screen)
