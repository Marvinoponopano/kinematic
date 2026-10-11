"""Output rate limiting: cap how often the pipeline may emit, without losing information.

Phase 5 asks for "rate-limit or coalesce movement events **if needed**" and Phase 7 for rate limiting
on the scroll path, so this module answers "if needed" with a measurement rather than a taste decision:

* At the shipped defaults neither limiter is **needed**. The loop is camera-bound at ~30 fps and one
  ``pyautogui.moveTo`` / ``scroll`` costs ~11 ms, so at most one call per frame already happens and the
  output path is roughly a third of the frame budget (`tools/measure_loop.py` measures both).
* They **are** needed when the adapter gets slow: with ``mouse_pause`` at pyautogui's own 0.1 s default
  every call costs ~113 ms, longer than the 33 ms frame interval, and the loop then blocks on the output
  instead of on the camera. A limit below that rate drops the calls that cannot be delivered in time and
  keeps the loop responsive.
* A limit only helps **below** the rate the loop is already achieving — capping at or above it changes
  nothing at all (measured: 6.76 fps with a cap of 8 Hz on a path that ran at 6.78 fps uncapped).

The two limiters make opposite choices about a frame they hold back, because what they carry is
different in kind:

* :class:`MoveRateLimiter` carries **cursor positions** — absolute values — so a held frame is
  *superseded*: the newest position wins and nothing is queued, which is what stops a slow loop from
  sending a backlog of stale positions.
* :class:`ScrollRateLimiter` carries **scroll amounts** — deltas — so a held frame is *accumulated*:
  throwing an increment away would lose that much scrolling, so the total goes out on the next due
  frame and the distance is preserved exactly.

Both are pure and timestamp-driven (timestamps come in as arguments, like the smoothing filters and the
gesture recogniser), both are resettable, and both default to off — rate limiting below the frame rate is
the hold-then-jump behaviour the smoothness work removed (fewer, larger steps).
``tools/measure_smoothness.py --rate-sweep`` prints what each cursor cap costs in held frames, step size
and trailing.
"""

from typing import Optional, Tuple

#: Slack for the interval comparison. At a limit equal to the loop rate (30 Hz on a 30 fps loop)
#: the frame-to-frame differences of floating-point timestamps miss the interval by ~1e-17, and
#: without this slack the nominal "one emission per frame" setting would drop every other frame.
INTERVAL_TOLERANCE = 1e-9


class _EmissionClock:
    """Shared timing for an emission limit: at most ``max_hz`` emissions per second (0 = no limit)."""

    def __init__(self, max_hz: float = 0.0) -> None:
        self.max_hz = max(0.0, float(max_hz))
        self.suppressed = 0
        self._last_emit_at: Optional[float] = None

    # -- introspection --------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self.max_hz > 0.0

    @property
    def interval(self) -> float:
        """Seconds between two allowed emissions (``0.0`` while the limiter is off)."""
        return 0.0 if self.max_hz <= 0.0 else 1.0 / self.max_hz

    @property
    def last_emit_at(self) -> Optional[float]:
        """Timestamp of the last allowed emission, or None before the first one."""
        return self._last_emit_at

    def due(self, now: float) -> bool:
        """True when an emission is allowed at *now*.

        The first call is always due: nothing should wait out an interval before it starts following
        something it has just seen.
        """
        if not self.enabled:
            return True
        if self._last_emit_at is None:
            return True
        return (float(now) - self._last_emit_at) >= self.interval - INTERVAL_TOLERANCE

    def reset(self) -> None:
        """Forget the clock so the next emission passes through immediately, and drop the counters.

        Called with the cursor filter when tracking is lost: without it, a dropout *shorter* than the
        interval (a one-frame flicker at a 10 Hz limit) would hold back the first emission of the
        reacquired hand.
        """
        self._last_emit_at = None
        self.suppressed = 0

    def stats(self) -> dict:
        return {
            "max_hz": self.max_hz,
            "suppressed": self.suppressed,
            "last_emit_at": self._last_emit_at,
        }

    # -- internals ------------------------------------------------------------

    def _note_emission(self, now: float) -> None:
        self._last_emit_at = float(now)

    def _hold(self) -> None:
        self.suppressed += 1


class MoveRateLimiter(_EmissionClock):
    """Cap cursor moves at ``max_hz`` per second by *superseding* the frames it holds back.

    Nothing is stored: a frame inside the interval emits nothing and its position is overtaken by the
    next frame that is due, so there is no backlog to drain and no stale position to send late.
    """

    def offer(self, now: float) -> bool:
        """One frame's decision: True = emit this (newest) position, False = it is superseded."""
        if not self.enabled:
            return True
        if not self.due(now):
            self._hold()
            return False
        self._note_emission(now)
        return True


class ScrollRateLimiter(_EmissionClock):
    """Cap scroll events at ``max_hz`` per second by *accumulating* the frames it holds back.

    Scroll amounts are increments, not positions: a held frame's amount is added to the pending total
    and sent on the next due frame, so the event rate comes down while the distance scrolled stays
    what the hand asked for. (This is why it is not a :class:`MoveRateLimiter`: a cursor position
    supersedes, an increment adds up.)

    It accumulates in floats and emits whole units, keeping the sub-unit remainder. That matters more
    than it looks: the recogniser emits fractional amounts (a hand width of movement is ``scroll_gain``
    units, so a slow frame is well under one unit), and truncating each event to an int lost ~25% of
    the distance on a measured sweep — 135 units of the 180 the hand asked for. The rounding error now
    carries over instead of being thrown away.
    """

    def __init__(self, max_hz: float = 0.0) -> None:
        super().__init__(max_hz)
        self._pending: Tuple[float, float] = (0.0, 0.0)

    @property
    def pending(self) -> Tuple[float, float]:
        """Scroll units not yet sent, as (vertical, horizontal) — including the sub-unit remainder."""
        return self._pending

    def add(
        self,
        vertical: float = 0.0,
        horizontal: float = 0.0,
        now: float = 0.0,
    ) -> Optional[Tuple[int, int]]:
        """Offer this frame's amounts; get back whole units to send now, or None if nothing is due.

        The remainder (held frames *and* the sub-unit fraction) stays pending, so a caller that treats
        the None case as "nothing to send this frame" still scrolls the same total distance — just in
        fewer, larger events.
        """
        self._pending = (self._pending[0] + float(vertical), self._pending[1] + float(horizontal))
        if self.enabled and not self.due(now):
            self._hold()
            return None

        whole = (int(self._pending[0]), int(self._pending[1]))
        if not (whole[0] or whole[1]):
            # Under a unit on both axes: keep it and do not start the clock, or a still hand would
            # delay the next real scroll.
            return None
        self._pending = (self._pending[0] - whole[0], self._pending[1] - whole[1])
        if self.enabled:
            self._note_emission(now)
        return whole

    def reset(self) -> None:
        super().reset()
        self._pending = (0.0, 0.0)

    def flush(self) -> Optional[Tuple[int, int]]:
        """Send everything accumulated *now*, whole units only, ignoring the interval.

        The scroll gesture has ended (the hand left the pose), so the cap must not hold the remainder
        back: without this, whatever was pending at that moment would never be scrolled. The sub-unit
        fraction stays pending — it belongs to movement that is over.
        """
        whole = (int(self._pending[0]), int(self._pending[1]))
        if not (whole[0] or whole[1]):
            return None
        self._pending = (self._pending[0] - whole[0], self._pending[1] - whole[1])
        return whole

    def stats(self) -> dict:
        stats = super().stats()
        stats["pending"] = self._pending
        return stats
