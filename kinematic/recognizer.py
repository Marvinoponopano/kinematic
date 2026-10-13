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
* **A pinch presses and holds, it does not just click.** Entering ``LEFT_PINCH``/``RIGHT_PINCH`` fires
  a one-shot press event (``LEFT_CLICK``/``RIGHT_CLICK`` — same names, now meaning "button down"); the
  matching ``LEFT_RELEASE``/``RIGHT_RELEASE`` fires the instant the pair's physical engagement ends,
  independent of the hold/cooldown/neutral-between rules that gate the press. A quick pinch-and-release
  is still a down immediately followed by an up — an ordinary click — but a held pinch keeps the button
  down for as long as the pose is held, which is what makes dragging possible.
* **The gesture pause is opt-in and hold-gated.** With ``pause_gesture_enabled`` (off by default) an
  open hand — all four fingers straight — held for ``pause_gesture_hold_seconds`` emits exactly one
  ``PAUSE_TOGGLE`` per hold, and it works while paused as well as while active, so the same gesture
  resumes. It is off by default because a pose that can freeze the pointer must be a deliberate
  choice rather than something a stretch can do by accident.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Tuple

from .gesture import (
    extended_fingers,
    hand_scale,
    landmark_point,
    midpoint,
    pinch_distances,
    INDEX_TIP,
    MIDDLE_TIP,
)


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
    LEFT_RELEASE = "left_release"
    RIGHT_RELEASE = "right_release"
    SCROLL_BEGIN = "scroll_begin"
    SCROLL = "scroll"
    SCROLL_END = "scroll_end"
    TRACKING_LOST = "tracking_lost"
    TRACKING_RESTORED = "tracking_restored"
    PAUSED = "paused"
    RESUMED = "resumed"
    PAUSE_TOGGLE = "pause_toggle"


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
    #   Scroll reference movement below this (in hand widths) is ignored, and accumulates. Applied
    #   per axis, so sideways wobble during a vertical scroll does not scroll sideways.
    scroll_gain: float = 900.0
    #   Vertical scroll units per hand-width of reference movement (the old hard-coded 900).
    scroll_horizontal_gain: float = 900.0
    #   Horizontal scroll units per hand-width. 0 turns horizontal scrolling off, which reproduces
    #   the pre-Phase-7 behaviour of dropping the sideways component (it was computed and reported,
    #   never emitted).
    invert_vertical: bool = False
    #   Flip the vertical direction: the default is hand down = page down.
    invert_horizontal: bool = False
    #   Flip the horizontal direction. Which way Quartz's positive horizontal wheel value moves the
    #   content is inherited from macOS rather than measured here (scrolling the user's screen to
    #   find out is not a thing this suite does), so this is the one-field switch for it.
    require_neutral_between: bool = True
    #   A gesture cannot follow a different gesture without a neutral frame in between.
    pause_gesture_enabled: bool = False
    #   Phase 8's optional gesture pause. Off by default: enabling it means a pose can stop the
    #   pointer, so it is opt-in, hold-gated and only ever toggles.
    pause_gesture_fingers: int = 4
    #   How many of the four fingers (index → pinky) must be straight for the pause pose. 4 = a flat
    #   open hand, which no click pose can look like (both pinch poses have a curled index or middle).
    pause_gesture_hold_seconds: float = 1.2
    #   How long the open hand must be held before it toggles. Long on purpose: the pose is what the
    #   hand passes through on its way to any pinch, and a brief stretch must not freeze the pointer.

    def __post_init__(self) -> None:
        if self.pinch_ratio <= 0.0:
            raise ValueError(f"pinch_ratio must be > 0, got {self.pinch_ratio}")
        if self.release_ratio < 1.0:
            raise ValueError(f"release_ratio must be >= 1, got {self.release_ratio}")
        if self.hold_seconds < 0.0:
            raise ValueError(f"hold_seconds must be >= 0, got {self.hold_seconds}")
        if self.cooldown_seconds < 0.0:
            raise ValueError(f"cooldown_seconds must be >= 0, got {self.cooldown_seconds}")
        if self.scroll_gain < 0.0:
            raise ValueError(f"scroll_gain must be >= 0 (use invert_vertical), got {self.scroll_gain}")
        if self.scroll_horizontal_gain < 0.0:
            raise ValueError(
                f"scroll_horizontal_gain must be >= 0 (use invert_horizontal), got {self.scroll_horizontal_gain}"
            )
        if not 0 <= self.pause_gesture_fingers <= 4:
            raise ValueError(f"pause_gesture_fingers must be 0..4, got {self.pause_gesture_fingers}")
        if self.pause_gesture_hold_seconds < 0.0:
            raise ValueError(
                f"pause_gesture_hold_seconds must be >= 0, got {self.pause_gesture_hold_seconds}"
            )


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
        scroll_horizontal_gain=float(getattr(config, "scroll_horizontal_gain", 900.0)),
        invert_vertical=bool(getattr(config, "scroll_invert_vertical", False)),
        invert_horizontal=bool(getattr(config, "scroll_invert_horizontal", False)),
        pause_gesture_enabled=bool(getattr(config, "gesture_pause_enabled", False)),
        pause_gesture_fingers=int(getattr(config, "gesture_pause_fingers", 4)),
        pause_gesture_hold_seconds=float(getattr(config, "gesture_pause_hold_seconds", 1.2)),
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
        self._pause_since: Optional[float] = None
        self._pause_armed = True
        self._pressed: Dict[str, bool] = {"thumb_index": False, "thumb_middle": False}
        self.last_ratios: Dict[str, float] = {}
        self.last_extended_fingers: Optional[int] = None

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

    def pressed(self, pair: str) -> bool:
        """True while a prior press on this pair has no matching release yet (it is "held down")."""
        return bool(self._pressed.get(pair, False))

    @property
    def pause_gesture_armed(self) -> bool:
        """False while the open-hand pose that just toggled is still being held."""
        return self._pause_armed

    # -- control --------------------------------------------------------------

    def reset(self, disarm: bool = False) -> None:
        """Drop engagement and pending state (tracking loss, pause, re-arm).

        ``disarm=True`` also marks both pairs as consumed, which is what stops a pinch that is
        *already* held when the hand reappears (or the user resumes) from firing: it has to be
        released and pinched again. A pair that is open at that moment re-arms on the next frame
        by itself, so an ordinary pinch straight after a resume still works.

        This also force-releases any button a prior press left down, since ``_engaged`` is being
        reset to "nothing pinched" here too. Callers that need the matching ``..._RELEASE`` event —
        anywhere a hand can be mid-drag when this runs — should call :meth:`pending_releases` *before*
        ``reset()`` and include the result in what they return; ``reset()`` itself cannot emit events
        (several of its callers do not build an event list at all).
        """
        self._engaged = {"thumb_index": False, "thumb_middle": False}
        self._pressed = {"thumb_index": False, "thumb_middle": False}
        if disarm:
            self._armed = {"thumb_index": False, "thumb_middle": False}
        self._clear_pending()
        self._scroll_reference = None
        # A partial open-hand hold is dropped (the hand that was feeding it is gone or the mode
        # changed), but the pose stays *armed*: unlike a pinch, re-establishing it costs a full hold
        # time, so there is nothing to protect against. set_paused() is where a hold in flight over a
        # pause transition is disarmed, because that is the case where the pose is still being held.
        self._pause_since = None

    def pending_releases(self) -> Tuple[GestureEvent, ...]:
        """Release events for whatever is currently pressed, without changing any state.

        Call this immediately before :meth:`reset` wherever a hand can be mid-drag when the state is
        about to be cleared out from under it (pause, tracking loss) — otherwise the physical button
        stays down on the real desktop with nothing left in this object that remembers to lift it.
        """
        return tuple(
            GestureEvent(self._release_kind(pair)) for pair in ("thumb_index", "thumb_middle") if self._pressed[pair]
        )

    def set_paused(self, paused: bool, timestamp: Optional[float] = None) -> Tuple[GestureEvent, ...]:
        """Pause or resume. Resuming never fires the gesture that was held while paused."""
        paused = bool(paused)
        if paused == self._paused:
            return ()
        if self._pause_since is not None:
            # The open-hand pose is being held right now, and this transition came from elsewhere (the
            # SPACE key or the API). Cancel *and* disarm it, or the same continuing hold would toggle
            # straight back a moment later.
            self._pause_armed = False
        releases = self.pending_releases()
        self._paused = paused
        was_scrolling = self.state is HandState.TWO_FINGER_SCROLL
        self.reset(disarm=True)
        events = list(releases)
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

        # Checked before the paused branch on purpose: the pause gesture is the one thing that works
        # while paused, which is what makes the same pose able to resume. (The keyboard works too.)
        pause_events = self._pause_events(observation, now)

        if self._paused:
            self._clear_pending()
            self.state = HandState.PAUSED
            # Keep the engagement view current while paused: the pause pose's own rule is "no pinch
            # engaged", and releasing it (returning to a different pose) is what re-arms the gesture,
            # so the two pairs still have to be tracked. No click or scroll logic runs here — only
            # what the pause gesture itself reads.
            if observation.landmarks is not None:
                self._observe_hand(observation)
            return GestureUpdate(state=self.state, events=pause_events)

        if observation.landmarks is None:
            if self.state is HandState.TRACKING_LOST:
                return GestureUpdate(state=self.state, events=pause_events)
            was_scrolling = self.state is HandState.TWO_FINGER_SCROLL
            releases = self.pending_releases()
            self.reset(disarm=True)
            self.state = HandState.TRACKING_LOST
            events = list(pause_events) + list(releases)
            if was_scrolling:
                events.append(GestureEvent(EventKind.SCROLL_END))
            events.append(GestureEvent(EventKind.TRACKING_LOST))
            return GestureUpdate(state=self.state, events=tuple(events))

        events = list(pause_events)
        if self.state is HandState.TRACKING_LOST:
            # Coming back from a loss always re-arms from neutral: a hand that is already pinched
            # when it reappears must be released and pinched again before it clicks. The state has
            # to leave TRACKING_LOST here, or the restore would repeat every frame and the pending
            # pose would be reset before it could ever hold. Nothing is pressed at this point (the
            # branch above already released it on the way into the loss), so there is nothing to
            # release again here.
            self.reset(disarm=True)
            self.state = HandState.NEUTRAL
            events.append(GestureEvent(EventKind.TRACKING_RESTORED))

        self._observe_hand(observation)
        events.extend(self._check_releases())
        target = self._resolve_state()
        events.extend(self._advance(target, observation, now))
        return GestureUpdate(state=self.state, events=tuple(events))

    # -- internals ------------------------------------------------------------

    def _observe_hand(self, observation: Observation) -> None:
        """Measure the hand and update the two pairs' engagement (the one place that does it)."""
        landmarks = observation.landmarks
        assert landmarks is not None
        self._last_scale = hand_scale(landmarks)
        distances = pinch_distances(landmarks)
        self.last_ratios = {
            "thumb_index": distances.thumb_index / self._last_scale,
            "thumb_middle": distances.thumb_middle / self._last_scale,
        }
        self._update_engagement("thumb_index", self.last_ratios["thumb_index"])
        self._update_engagement("thumb_middle", self.last_ratios["thumb_middle"])

    def _check_releases(self) -> Tuple[GestureEvent, ...]:
        """Release the moment a pressed pair's physical engagement ends.

        Deliberately independent of the state machine's ``target``/``self.state`` bookkeeping (hold
        time, cooldown, "no gesture switch without neutral"): those rules exist to debounce *presses*,
        not to decide when the real mouse button should come up. Tying the release to anything but the
        raw ``_engaged`` flag risks a stuck button — e.g. the thumb crossing from the index tip
        straight to the middle tip inside one frame would otherwise be blocked by
        ``require_neutral_between`` and leave the left button "held" by the state machine's
        bookkeeping even though the index pinch is physically gone.
        """
        events = []
        for pair in ("thumb_index", "thumb_middle"):
            if self._pressed[pair] and not self._engaged[pair]:
                self._pressed[pair] = False
                events.append(GestureEvent(self._release_kind(pair)))
        return tuple(events)

    @staticmethod
    def _release_kind(pair: str) -> EventKind:
        return EventKind.RIGHT_RELEASE if pair == "thumb_middle" else EventKind.LEFT_RELEASE

    def _pause_events(self, observation: Observation, now: float) -> Tuple[GestureEvent, ...]:
        """The opt-in open-hand pause: one ``PAUSE_TOGGLE`` per hold, released to re-arm.

        The pose is "at least ``pause_gesture_fingers`` of the four fingers straight and no pinch
        engaged" — no click pose looks like it, because both pinch poses have the index or the middle
        finger curled. The hold time is what keeps the hand passing *through* an open shape on its way
        to a pinch from toggling anything.
        """
        if not self.thresholds.pause_gesture_enabled:
            return ()
        landmarks = observation.landmarks
        if landmarks is None:
            self._pause_since = None
            self._pause_armed = True
            self.last_extended_fingers = None
            return ()

        extended = extended_fingers(landmarks)
        self.last_extended_fingers = extended
        open_hand = (
            extended >= self.thresholds.pause_gesture_fingers
            and not self._engaged["thumb_index"]
            and not self._engaged["thumb_middle"]
        )
        if not open_hand:
            # Left the pose: re-armed, and the hold clock starts again from scratch next time.
            self._pause_since = None
            self._pause_armed = True
            return ()
        if not self._pause_armed:
            return ()
        if self._pause_since is None:
            self._pause_since = now
            return ()
        if (now - self._pause_since) >= self.thresholds.pause_gesture_hold_seconds:
            self._pause_armed = False
            self._pause_since = None
            return (GestureEvent(EventKind.PAUSE_TOGGLE),)
        return ()

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
        self._pressed[pair] = True
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

        # The dead zone is applied *per axis*: a hand moving vertically with a little sideways wobble
        # must scroll vertically only. While the horizontal component was dropped by the runtime the
        # two axes moved together (one over the zone meant both were sent), which would now put that
        # wobble on screen as sideways scrolling.
        moving_vertically = abs(dy) >= deadzone
        moving_horizontally = abs(dx) >= deadzone
        if not moving_vertically and not moving_horizontally:
            # Below the dead zone on both axes: hold the reference so slow movement accumulates
            # rather than being thrown away.
            return []

        vertical = self._axis_amount(dy, self.thresholds.scroll_gain, self.thresholds.invert_vertical, -1.0)
        horizontal = self._axis_amount(
            dx, self.thresholds.scroll_horizontal_gain, self.thresholds.invert_horizontal, 1.0
        )
        self._scroll_reference = reference
        return [
            GestureEvent(
                EventKind.SCROLL,
                value=vertical if moving_vertically else 0.0,
                secondary=horizontal if moving_horizontally else 0.0,
            )
        ]

    @staticmethod
    def _axis_amount(delta: float, gain: float, invert: bool, sign: float) -> float:
        """One axis' scroll amount: *delta* × *gain* × the axis' convention, flipped by ``invert``.

        ``sign`` is the OS wheel convention for a hand moving in the *positive* direction on that
        axis: **-1** for vertical (a hand moving down scrolls the page down, while a positive wheel
        value scrolls up) and **+1** for horizontal. ``invert`` is the per-axis direction switch, and
        it lives here rather than in the adapter because this module is where a hand movement becomes
        a scroll amount.
        """
        return delta * gain * (-sign if invert else sign)


def timestamp() -> float:
    """Convenience for callers that have no clock of their own to pass in."""
    return time.monotonic()
