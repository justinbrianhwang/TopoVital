"""Absolute record coordinates vs. relative 10 s window coordinates (brief §5)."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


def ms_to_samples(ms: float, fs: int) -> int:
    """round(ms * fs / 1000) with round-half-up on exact rationals (D-03)."""
    x = Fraction(str(ms)) * fs / 1000
    return int(x + Fraction(1, 2)) if x >= 0 else -int(-x + Fraction(1, 2))


@dataclass(frozen=True)
class EventWindow:
    r_abs: int            # absolute index of target R annotation
    g_ms: float
    fs: int
    g_samples: int
    boundary_abs: int     # r_abs - g_samples
    start_abs: int        # boundary_abs - half_window*fs   (inclusive)
    end_abs: int          # start_abs + window*fs           (exclusive)

    @property
    def realized_g_ms(self) -> float:
        return 1000.0 * self.g_samples / self.fs

    @property
    def n_samples(self) -> int:
        return self.end_abs - self.start_abs

    def to_rel(self, abs_idx: int) -> int:
        return abs_idx - self.start_abs

    def in_window(self, abs_idx: int) -> bool:
        return self.start_abs <= abs_idx < self.end_abs

    def in_pre_half(self, abs_idx: int) -> bool:
        """[start, boundary): an annotation exactly on the boundary is NOT pre-half."""
        return self.start_abs <= abs_idx < self.boundary_abs

    def in_post_half(self, abs_idx: int) -> bool:
        """[boundary, end)."""
        return self.boundary_abs <= abs_idx < self.end_abs

    def within_record(self, sig_len: int) -> bool:
        """No padding / wrap-around: the whole window must lie in [0, sig_len)."""
        return self.start_abs >= 0 and self.end_abs <= sig_len


def event_window(r_abs: int, g_ms: float, fs: int, half_window_s: float = 5,
                 window_s: float = 10) -> EventWindow:
    g = ms_to_samples(g_ms, fs)
    boundary = r_abs - g
    start = boundary - ms_to_samples(half_window_s * 1000, fs)
    return EventWindow(r_abs, g_ms, fs, g, boundary, start,
                       start + ms_to_samples(window_s * 1000, fs))
