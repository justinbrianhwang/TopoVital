import pytest

from topovital.timeaxis import event_window, ms_to_samples


@pytest.mark.parametrize("g_ms,expected", [(200, 72), (250, 90), (300, 108)])
def test_g_samples_at_360hz(g_ms, expected):
    assert ms_to_samples(g_ms, 360) == expected


def test_rounding_half_up():
    assert ms_to_samples(12.5, 40) == 1  # 0.5 -> 1 (half-up)
    assert ms_to_samples(37.5, 40) == 2  # 1.5 -> 2


@pytest.mark.parametrize("g_ms", [200, 250, 300])
def test_absolute_and_relative_coordinates(g_ms):
    fs, r = 360, 100_000
    w = event_window(r, g_ms, fs)
    g = ms_to_samples(g_ms, fs)
    assert w.boundary_abs == r - g
    assert w.start_abs == w.boundary_abs - 5 * fs
    assert w.end_abs == w.start_abs + 10 * fs
    assert w.n_samples == 3600
    assert w.to_rel(w.boundary_abs) == 5 * fs              # boundary always at 5 s
    assert w.to_rel(r) == 5 * fs + g                       # target R at 5 s + g
    assert w.realized_g_ms == pytest.approx(g_ms)


def test_boundary_membership_half_open():
    w = event_window(100_000, 250, 360)
    assert w.in_window(w.start_abs) and not w.in_window(w.end_abs)
    assert not w.in_pre_half(w.boundary_abs) and w.in_post_half(w.boundary_abs)
    assert w.in_pre_half(w.boundary_abs - 1)
    assert w.in_post_half(w.r_abs)


def test_window_outside_record_detected_not_padded():
    assert not event_window(1000, 250, 360).within_record(650_000)
    assert not event_window(649_900, 250, 360).within_record(650_000)
    assert event_window(100_000, 250, 360).within_record(650_000)
    w = event_window(650_000 - 5 * 360 + 90, 250, 360)   # end exactly at sig_len
    assert w.end_abs == 650_000 and w.within_record(650_000)
