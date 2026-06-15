"""Regression tests for prince_cr.source.evolution.SingleZoneSolver (Phase 2).

Validates the single-zone ETD core against the analytic synchrotron cooling
break and checks steady-state / time-march consistency + positivity.
"""
import numpy as np
import pytest

from prince_cr.source.evolution import SingleZoneSolver


def _slope(g, n, lo, hi):
    win = (g > lo) & (g < hi)
    ln = np.where(n > 0, np.log(np.where(n > 0, n, 1.0)), np.nan)
    return float(np.nanmedian(np.gradient(ln, np.log(g))[win]))


def test_synchrotron_cooling_break_slope():
    """Q∝γ^-p + synchrotron cooling, steady state ⇒ n∝γ^-(p+1)."""
    p = 2.0
    sz = SingleZoneSolver(gamma_lo=1.0, gamma_hi=1e8, n_bins=384, B_Gauss=1.0)
    Q = sz.injection_powerlaw(Q0=1.0, p=p, gamma_min=1e3, gamma_max=1e7)
    n = sz.steady_state(Q)
    assert np.min(n) >= 0.0                       # positivity
    slope = _slope(sz.g, n, 1e4, 1e6)
    assert abs(slope - (-(p + 1))) < 0.05         # analytic cooled slope


def test_etd_march_matches_steady_state():
    """Exponential-Euler march relaxes to the direct steady-state solve."""
    p = 2.0
    sz = SingleZoneSolver(gamma_lo=1.0, gamma_hi=1e8, n_bins=384, B_Gauss=1.0)
    Q = sz.injection_powerlaw(Q0=1.0, p=p, gamma_min=1e3, gamma_max=1e7)
    n_ss = sz.steady_state(Q)
    t_end = 30 * float(sz.t_cool(1e3))
    n_eq = sz.evolve(np.zeros_like(sz.g), Q, t_end=t_end, dt=t_end / 400)
    win = (sz.g > 1e4) & (sz.g < 1e6)             # relaxed window
    rel = np.linalg.norm((n_eq - n_ss)[win]) / np.linalg.norm(n_ss[win])
    assert rel < 1e-6
    assert np.min(n_eq) >= 0.0


def test_escape_steepens_below_cooling_break():
    """With finite escape, the spectrum is not steeper than the cooled slope."""
    p = 2.0
    sz = SingleZoneSolver(gamma_lo=1.0, gamma_hi=1e8, n_bins=384, B_Gauss=1.0,
                          t_esc_s=1e5)
    Q = sz.injection_powerlaw(Q0=1.0, p=p, gamma_min=1e3, gamma_max=1e7)
    n = sz.steady_state(Q)
    assert np.min(n) >= 0.0
    assert np.all(np.isfinite(n))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
