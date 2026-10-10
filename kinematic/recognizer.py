"""Pure gesture recogniser: a timestamp-driven state machine for one hand.

Rules this module enforces, all of them testable without a camera:

* **Relative thresholds.** Every distance is divided by the hand's scale (wrist → middle knuckle),
  so the same pinch works whether the hand is near or far from the camera. Nothing here is in
  pixels or absolute landmark units.
* **Explicit priority.** Two-finger (scroll) beats thumb-index (left pinch) beats thumb-middle
  (right pinch). ``GESTURE_PRIORITY`` states that order in one place.
* **Hysteresis per pair.** A pair engages below ``pinch_ratio`` and only releases above
  ``pinch_ratio * release_ratio``, so a hand resting at the threshold cannot chatter.
* **Hold, cooldown, re-arm.** A pose must hold for ``hold_seconds`` before it registers; a click
  inside ``cooldown_seconds`` of the previous one is dropped rather than deferred; and a gesture
  cannot follow a *different* gesture without passing through neutral first.
* **No side effects.** No mouse, no camera, no clock of its own: timestamps come in as arguments and
  events come out. The runtime decides what an event means.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Tuple

from .gesture import hand_scale, landmark_point, midpoint, pinch_distances, INDEX_TIP, MIDDLE_TIP


class HandState(str, Enum):
    """What the hand is currently doing."""

    NEUTRAL = "neutral"
    LEFT_PINCH = "left_pinch"
    RIGHT_PINCH = "right_pinch"
    TWO_FINGER_SCROLL = "two_finger_scroll"
    PAUSED = "paused"
    TRACKING_LOST = "tracking_lost"


class EventKind(str, Enum):
    """What the runtime should do about it."""

    LEFT_CLICK = "left_click"
    RIGHT_CLICK = "right_click"
    SCROLL_BEGIN = "scroll_begin"
    SCROLL = "scroll"
    SCROLL_END = "scroll_end"
    TRACKING_LOST = "tracking_lost"
    TRACKING_RESTORED = "tracking_restored"
    PAUSED = "paused"
    RESUMED = "resumed"


#: Which gesture wins when more than one pose is recognisable. First match takes it.
GESTURE_PRIORITY: Tuple[HandState, ...] = (
    HandState.TWO_FINGER_SCROLL,
    HandState.LEFT_PINCH,
    HandState.RIGHT_PINCH,
)


@dataclass(frozen=True)
class GestureEvent:
    """One thing that happened. ``value`` carries scroll units, ``secondary`` the horizontal part."""

    kind: EventKind
    value: float = 0.0
    secondary: float = 0.0


@dataclass(frozen=True)
class GestureThresholds:
    """All distances are fractions of the hand's scale, never absolute."""

    pinch_ratio: float = 0.40
    #   A pair counts as pinched when its distance / hand_scale is below this. 0.40 keeps the old
    #   absolute 0.06 threshold at a nominal hand scale of 0.15.
    release_ratio: float = 1.4
    #   It stops counting as pinched only above pinch_ratio * this — the hysteresis band.
    hold_seconds: float = 0.08
    #   How long a pose must persist before it registers (about 2.4 frames at 30 fps).
    cooldown_seconds: float = 0.18
    #   Two clicks are never closer than this; a click inside the window is dropped, not deferred.
    scroll_deadzone_ratio: float = 0.05
    #   Scroll reference movement below this (in hand widths) is ignored, and accumulates.
    scroll_gain: float = 900.0
    #   Scroll units per hand-width of reference movement (the old hard-coded -900).
    require_neutral_between: bool = True
    #   A gesture cannot follow a different gesture without a neutral frame in between.

    def __post_init__(self) -> None:
        if self.pinch_ratio <= 0.0:
            raise ValueError(f"pinch_ratio must be > 0, got {self.pinch_ratio}")
        if self.release_ratio < 1.0:
            raise ValueError(f"release_ratio must be >= 1, got {self.release_ratio}")
        if self.hold_seconds < 0.0:
            raise ValueError(f"hold_seconds must be >= 0, got {self.hold_seconds}")
        if self.cooldown_seconds < 0.0:
            raise ValueError(f"cooldown_seconds must be >= 0, got {self.cooldown_seconds}")


@dataclass(frozen=True)
class Observation:
    """One frame's input: the hand (or ``None`` when tracking is lost) and the timestamp."""

    landmarks: Optional[Sequence[Any]] = None
    timestamp: float = 0.0


@dataclass(frozen=True)
class GestureUpdate:
    """What the recogniser concluded this frame."""

    state: HandState
    events: Tuple[GestureEvent, ...] = ()


def thresholds_from_config(config: Any) -> GestureThresholds:
    """Build thresholds from a :class:`kinematic.config.HandMouseConfig`."""
    return GestureThresholds(
        pinch_ratio=float(getattr(config, "pinch_ratio", 0.40)),
        release_ratio=float(getattr(config, "pinch_release_ratio", 1.4)),
        hold_seconds=float(getattr(config, "pinch_hold_seconds", 0.08)),
        cooldown_seconds=float(getattr(config, "click_cooldown_seconds", 0.18)),
        scroll_deadzone_ratio=float(getattr(config, "scroll_deadzone_ratio", 0.05)),
        scroll_gain=float(getattr(config, "scroll_gain", 900.0)),
    )


class GestureRecognizer:
    """Feed observations in, get a state and events out. Holds only transition state."""

    def __init__(self, thresholds: Optional[GestureThresholds] = None) -> None:
        self.thresholds = thresholds or GestureThresholds()
        self.state: HandState = HandState.NEUTRAL
        self._paused = False
        self._engaged: Dict[str, bool] = {"thumb_index": False, "thumb_middle": False}
        self._armed: Dict[str, bool] = {"thumb_index": True, "thumb_middle": True}
        self._pending: Optional[HandState] = None
        self._pending_since: Optional[float] = None
        self._last_click_at: Optional[float] = None
        self._scroll_reference: Optional[Tuple[float, float]] = None
        self._last_scale: Optional[float] = None
        self.last_ratios: Dict[str, float] = {}

    # -- introspection --------------------------------------------------------

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def last_hand_scale(self) -> Optional[float]:
        """Hand scale of the most recent frame (diagnostics and tests)."""
        return self._last_scale

    def engaged(self, pair: str) -> bool:
        return bool(self._engaged.get(pair, False))

    def armed(self, pair: str) -> bool:
        """False until the pair has been released at least once since it was consumed."""
        return bool(self._armed.get(pair, True))

    # -- control --------------------------------------------------------------

    def reset(self, disarm: bool = False) -> None:
        """Drop engagement and pending state (tracking loss, pause, re-arm).

        ``disarm=True`` also marks both pairs as consumed, which is what stops a pinch that is
        *already* held when the hand reappears (or the user resumes) from firing: it has to be
        released and pinched again. A pair that is open at that moment re-arms on the next frame
        by itself, so an ordinary pinch straight after a resume still works.
        """
        self._engaged = {"thumb_index": False, "thumb_middle": False}
        if disarm:
            self._armed = {"thumb_index": False, "thumb_middle": False}
        self._clear_pending()
        self._scroll_reference = None

    def set_paused(self, paused: bool, timestamp: Optional[float] = None) -> Tuple[GestureEvent, ...]:
        """Pause or resume. Resuming never fires the gesture that was held while paused."""
        paused = bool(paused)
        if paused == self._paused:
            return ()
        self._paused = paused
        was_scrolling = self.state is HandState.TWO_FINGER_SCROLL
        self.reset(disarm=True)
        events = []
        if paused and was_scrolling:
            events.append(GestureEvent(EventKind.SCROLL_END))
        if paused:
            self.state = HandState.PAUSED
            events.append(GestureEvent(EventKind.PAUSED))
        else:
            self.state = HandState.NEUTRAL
            events.append(GestureEvent(EventKind.RESUMED))
        return tuple(events)

    # -- the machine ----------------------------------------------------------

    def update(self, observation: Observation) -> GestureUpdate:
        now = observation.timestamp

        if self._paused:
            self._clear_pending()
            self.state = HandState.PAUSED
            return GestureUpdate(state=self.state)

        if observation.landmarks is None:
            if self.state is HandState.TRACKING_LOST:
                return GestureUpdate(state=self.state)
            was_scrolling = self.state is HandState.TWO_FINGER_SCROLL
            self.reset(disarm=True)
            self.state = HandState.TRACKING_LOST
            events = []
            if was_scrolling:
                events.append(GestureEvent(EventKind.SCROLL_END))
            events.append(GestureEvent(EventKind.TRACKING_LOST))
            return GestureUpdate(state=self.state, events=tuple(events))

        events = []
        if self.state is HandState.TRACKING_LOST:
            # Coming back from a loss always re-arms from neutral: a hand that is already pinched
            # when it reappears must be released and pinched again before it clicks. The state has
            # to leave TRACKING_LOST here, or the restore would repeat every frame and the pending
            # pose would be reset before it could ever hold.
            self.reset(disarm=True)
            self.state = HandState.NEUTRAL
            events.append(GestureEvent(EventKind.TRACKING_RESTORED))

        self._last_scale = hand_scale(observation.landmarks)
        distances = pinch_distances(observation.landmarks)
        self.last_ratios = {
            "thumb_index": distances.thumb_index / self._last_scale,
            "thumb_middle": distances.thumb_middle / self._last_scale,
        }
        self._update_engagement("thumb_index", self.last_ratios["thumb_index"])
        self._update_engagement("thumb_middle", self.last_ratios["thumb_middle"])

        target = self._resolve_state()
        events.extend(self._advance(target, observation, now))
        return GestureUpdate(state=self.state, events=tuple(events))

    # -- internals ------------------------------------------------------------

    def _update_engagement(self, pair: str, ratio: float) -> None:
        """Hysteresis: engage under pinch_ratio, release only above pinch_ratio * release_ratio."""
        engage_below = self.thresholds.pinch_ratio
        release_above = engage_below * self.thresholds.release_ratio
        if self._engaged[pair]:
            if ratio > release_above:
                self._engaged[pair] = False
                self._armed[pair] = True  # released: the pair may fire again
        elif ratio < engage_below:
            self._engaged[pair] = True
        else:
            # Observed clearly disengaged (or merely in the band): the pair is available again.
            # Without this, a pair disarmed by a tracking restore could never re-arm while the
            # hand stayed open, and no pinch afterwards would ever click.
            self._armed[pair] = True

    @staticmethod
    def _pair_for(state: HandState) -> str:
        return "thumb_middle" if state is HandState.RIGHT_PINCH else "thumb_index"

    def _resolve_state(self) -> HandState:
        """Explicit priority over the engaged pairs; never two fingers half-way."""
        both = self._engaged["thumb_index"] and self._engaged["thumb_middle"]
        if both:
            return HandState.TWO_FINGER_SCROLL
        if self._engaged["thumb_index"]:
            return HandState.LEFT_PINCH
        if self._engaged["thumb_middle"]:
            return HandState.RIGHT_PINCH
        return HandState.NEUTRAL

    def _clear_pending(self) -> None:
        self._pending = None
        self._pending_since = None

    def _advance(self, target: HandState, observation: Observation, now: float) -> list:
        """Move the machine towards *target*, honouring hold, cooldown and re-arm rules."""
        events: list = []

        if target is self.state:
            self._clear_pending()
            if self.state is HandState.TWO_FINGER_SCROLL:
                events.extend(self._scroll(observation))
            return events

        if target is HandState.NEUTRAL:
            # Releasing is immediate — a hold time on the way out would only add latency — but
            # leaving a scroll gesture closes it explicitly.
            if self.state is HandState.TWO_FINGER_SCROLL:
                events.append(GestureEvent(EventKind.SCROLL_END))
                self._scroll_reference = None
            self._clear_pending()
            self.state = HandState.NEUTRAL
            return events

        # A different gesture: it has to hold for hold_seconds before it registers.
        if self._pending is not target:
            self._pending = target
            self._pending_since = now
            if self.thresholds.hold_seconds > 0.0:
                return events
        elif self._pending_since is not None and (now - self._pending_since) < self.thresholds.hold_seconds:
            return events

        if self.thresholds.require_neutral_between and self.state is not HandState.NEUTRAL:
            # Swapping straight from one pinch to another is ignored: release first.
            return events

        self._clear_pending()
        self.state = target

        if target is HandState.TWO_FINGER_SCROLL:
            events.append(GestureEvent(EventKind.SCROLL_BEGIN))
            self._scroll_reference = self._scroll_reference_point(observation)
            return events

        kind = EventKind.LEFT_CLICK if target is HandState.LEFT_PINCH else EventKind.RIGHT_CLICK
        pair = self._pair_for(target)
        if not self._armed[pair]:
            # Consumed: the pair is already held (across a resume or a tracking loss) or a click
            # inside the cooldown window was dropped. Either way the user must release and pinch
            # again, which keeps the event count predictable instead of deferring the click.
            return events
        if self._cooling_down(now):
            # Inside the cooldown the pose is consumed without firing: the user must release and
            # pinch again, which keeps the event count predictable.
            self._armed[pair] = False
            return events
        self._last_click_at = now
        self._armed[pair] = False
        events.append(GestureEvent(kind))
        return events

    def _cooling_down(self, now: float) -> bool:
        if self._last_click_at is None:
            return False
        return (now - self._last_click_at) < self.thresholds.cooldown_seconds

    def _scroll_reference_point(self, observation: Observation) -> Tuple[float, float]:
        """The two-finger midpoint: the documented reference for scroll movement."""
        landmarks = observation.landmarks
        assert landmarks is not None, "the scroll reference needs a hand in frame"
        return midpoint(landmark_point(landmarks, INDEX_TIP), landmark_point(landmarks, MIDDLE_TIP))

    def _scroll(self, observation: Observation) -> list:
        reference = self._scroll_reference_point(observation)
        if self._scroll_reference is None:
            self._scroll_reference = reference
            return []

        dx = reference[0] - self._scroll_reference[0]
        dy = reference[1] - self._scroll_reference[1]
        deadzone = self.thresholds.scroll_deadzone_ratio * (self._last_scale or 0.0)
        vertical = -dy * self.thresholds.scroll_gain
        horizontal = dx * self.thresholds.scroll_gain

        if abs(dy) < deadzone and abs(dx) < deadzone:
            # Below the dead zone: hold the reference so slow movement accumulates rather than
            # being thrown away.
            return []

        self._scroll_reference = reference
        return [GestureEvent(EventKind.SCROLL, value=vertical, secondary=horizontal)]


def timestamp() -> float:
    """Convenience for callers that have no clock of their own to pass in."""
    return time.monotonic()
