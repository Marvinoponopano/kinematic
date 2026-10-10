"""Cursor smoothing filters.

Both filters are pure: they take a point (and optionally a timestamp) and return a point.
No camera, no mouse, no wall clock of their own — which is what makes them deterministic in
tests and lets the same code run in the live loop.

* :class:`ExponentialMovingAverage` — the fixed-factor EMA the app started with. Cheap and
  predictable, but a single factor has to trade jitter against lag.
* :class:`OneEuroFilter` — the 1€ filter (Casiez, Roussel & Vogel, CHI 2012). Its cutoff
  rises with speed, so a still hand is smoothed hard and a moving hand is followed closely.
  That is the fix for "smooth enough to be usable, but too laggy to be precise".

:func:`filter_from_config` is the only place the config's mode string is interpreted.
"""

import math
from typing import Any, Optional, Tuple, Union

from .gesture import smooth_point

Point = Tuple[float, float]


class CursorFilter:
    """Interface: feed points in, get smoothed points out, and be resettable."""

    name = "filter"

    def update(self, point: Point, timestamp: Optional[float] = None) -> Point:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class _LowPass:
    """Per-axis first-order low pass with a caller-supplied alpha (shared by both filters)."""

    def __init__(self) -> None:
        self._value: Optional[Point] = None

    @property
    def value(self) -> Optional[Point]:
        return self._value

    def apply(self, point: Point, alpha: Union[float, Point]) -> Point:
        alpha_x, alpha_y = alpha if isinstance(alpha, tuple) else (alpha, alpha)
        if self._value is None:
            self._value = point
            return point
        self._value = (
            alpha_x * point[0] + (1.0 - alpha_x) * self._value[0],
            alpha_y * point[1] + (1.0 - alpha_y) * self._value[1],
        )
        return self._value

    def reset(self) -> None:
        self._value = None


class ExponentialMovingAverage(CursorFilter):
    """``out += (in - out) * factor`` — the previous behaviour, kept as the default."""

    name = "ema"

    def __init__(self, factor: float = 0.35) -> None:
        if not 0.0 < factor <= 1.0:
            raise ValueError(f"factor must satisfy 0 < factor <= 1, got {factor}")
        self.factor = factor
        self._filter = _LowPass()

    def update(self, point: Point, timestamp: Optional[float] = None) -> Point:
        if self._filter.value is None:
            self._filter.apply(point, 1.0)
            return point
        # Delegated to gesture.smooth_point so the default behaviour is bit-for-bit what
        # the app produced before this module existed.
        smoothed = smooth_point(self._filter.value, point, self.factor)
        self._filter.apply(smoothed, 1.0)
        return smoothed

    def reset(self) -> None:
        self._filter.reset()


def _alpha(cutoff: float, dt: float) -> float:
    """Smoothing factor for a cutoff frequency (Hz) and timestep (s)."""
    if cutoff <= 0.0 or dt <= 0.0:
        return 1.0
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter(CursorFilter):
    """The 1€ filter: low jitter when slow, low lag when fast.

    ``min_cutoff`` is the cutoff when the hand is still (lower = smoother), ``beta`` is how
    much the cutoff opens up with speed (higher = more responsive), ``d_cutoff`` filters the
    speed estimate itself, and ``frequency`` is the fallback rate used when the caller has
    no timestamps.
    """

    name = "one_euro"

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 1.0,
        d_cutoff: float = 1.0,
        frequency: float = 30.0,
    ) -> None:
        if min_cutoff <= 0.0:
            raise ValueError(f"min_cutoff must be > 0, got {min_cutoff}")
        if beta < 0.0:
            raise ValueError(f"beta must be >= 0, got {beta}")
        if d_cutoff <= 0.0:
            raise ValueError(f"d_cutoff must be > 0, got {d_cutoff}")
        if frequency <= 0.0:
            raise ValueError(f"frequency must be > 0, got {frequency}")

        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.frequency = frequency

        self._position = _LowPass()
        self._speed = _LowPass()
        self._previous: Optional[Point] = None
        self._last_timestamp: Optional[float] = None

    def _timestep(self, timestamp: Optional[float]) -> float:
        if timestamp is None or self._last_timestamp is None:
            dt = 1.0 / self.frequency
        else:
            dt = timestamp - self._last_timestamp
        if timestamp is not None:
            self._last_timestamp = timestamp
        return dt if dt > 0.0 else 1.0 / self.frequency

    def update(self, point: Point, timestamp: Optional[float] = None) -> Point:
        dt = self._timestep(timestamp)

        if self._previous is None:
            speed = (0.0, 0.0)
        else:
            speed = ((point[0] - self._previous[0]) / dt, (point[1] - self._previous[1]) / dt)
        self._previous = point

        speed_hat = self._speed.apply(speed, _alpha(self.d_cutoff, dt))
        cutoff = (
            self.min_cutoff + self.beta * abs(speed_hat[0]),
            self.min_cutoff + self.beta * abs(speed_hat[1]),
        )
        # Per-axis: a hand moving mainly sideways should not be smoothed less vertically.
        return self._position.apply(point, (_alpha(cutoff[0], dt), _alpha(cutoff[1], dt)))

    def reset(self) -> None:
        self._position.reset()
        self._speed.reset()
        self._previous = None
        self._last_timestamp = None


def filter_from_config(config: Any, frequency: Optional[float] = None) -> CursorFilter:
    """Build the filter the config asks for. ``"ema"`` and ``"one_euro"`` are supported."""
    mode = str(getattr(config, "smoothing_mode", "ema"))
    rate = frequency if frequency else float(getattr(config, "capture_fps", 30) or 30)

    if mode == "ema":
        return ExponentialMovingAverage(factor=float(getattr(config, "smoothing", 0.35)))
    if mode == "one_euro":
        return OneEuroFilter(
            min_cutoff=float(getattr(config, "one_euro_min_cutoff", 1.0)),
            beta=float(getattr(config, "one_euro_beta", 1.0)),
            d_cutoff=float(getattr(config, "one_euro_d_cutoff", 1.0)),
            frequency=rate,
        )
    raise ValueError(f"unknown smoothing_mode {mode!r}: expected 'ema' or 'one_euro'")
