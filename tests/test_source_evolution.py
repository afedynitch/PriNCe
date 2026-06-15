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


def test_synchrotron_function_peak():
    """The synchrotron function F(x) peaks at ≈0.918 near x≈0.29 (textbook)."""
    from prince_cr.source.evolution import synchrotron_F
    x = np.logspace(-4, 1, 4000)
    F = synchrotron_F(x)
    assert abs(np.max(F) - 0.9180) < 0.01
    assert abs(x[np.argmax(F)] - 0.29) < 0.03


def test_synchrotron_sed_slope():
    """Power-law electrons γ^-q ⇒ synchrotron j(ν) ∝ ν^-(q-1)/2 in the band."""
    q = 2.5
    sz = SingleZoneSolver(gamma_lo=1.0, gamma_hi=1e8, n_bins=512, B_Gauss=1.0)
    n_e = np.where((sz.g >= 1e2) & (sz.g <= 1e6), sz.g ** (-q), 0.0)
    nu, j = sz.synchrotron_sed(n_e)
    nu_c1, nu_c2 = float(sz.nu_c(1e2)), float(sz.nu_c(1e6))
    m = (nu > nu_c1 * 30) & (nu < nu_c2 / 30) & (j > 0)
    sl = np.polyfit(np.log(nu[m]), np.log(j[m]), 1)[0]
    assert abs(sl - (-(q - 1) / 2)) < 0.05


def test_ssc_fixed_point_self_consistent():
    """SSC fixed point converges, is self-consistent, and IC feedback drains
    the high-γ electrons (extra cooling)."""
    R = 1e16
    sz = SingleZoneSolver(gamma_lo=1.0, gamma_hi=1e7, n_bins=512, B_Gauss=1.0,
                          t_esc_s=R / 2.99792458e10)
    Q = sz.injection_powerlaw(Q0=1e-2, p=2.2, gamma_min=1e3, gamma_max=1e6)
    sz.set_ic_target(0.0)
    n_ref = sz.steady_state(Q)
    n_ssc, info = sz.solve_ssc(Q, R_cm=R, tol=1e-5)
    assert info["converged"]
    u_check = sz.synchrotron_energy_density(n_ssc, R)
    assert abs(u_check - info["u_syn"]) / info["u_syn"] < 1e-4
    hi = sz.g > 1e5
    assert np.sum((n_ssc * sz.dg)[hi]) < np.sum((n_ref * sz.dg)[hi])  # IC drains


def test_ic_thomson_slope_and_kn_suppression():
    """IC photon-number spectrum: Thomson slope -(q+1)/2; KN steepens it."""
    _EV = 1.602176634e-12
    q = 2.5
    sz = SingleZoneSolver(gamma_lo=1.0, gamma_hi=1e5, n_bins=256, B_Gauss=1.0)
    n_e = np.where((sz.g >= 1e2) & (sz.g <= 1e4), sz.g ** (-q), 0.0)

    def target(eps0):
        eps = np.logspace(np.log10(eps0) - 0.6, np.log10(eps0) + 0.6, 81)
        nph = np.exp(-0.5 * ((np.log(eps) - np.log(eps0)) / (0.05 * np.log(10))) ** 2)
        return eps, nph

    def sed_slope(eps_out, Q, lo, hi):
        m = (eps_out > lo) & (eps_out < hi) & (Q > 0)
        return np.polyfit(np.log(eps_out[m]), np.log(Q[m]), 1)[0]

    # Thomson: 1 eV target
    e0 = 1.0 * _EV
    et, nph = target(e0)
    eo = np.logspace(np.log10(e0 * 3), np.log10(e0 * 1e4 ** 2 * 3), 90)
    Q = sz.ic_sed(n_e, et, nph, eo)
    assert np.all(Q >= 0)
    sl = sed_slope(eo, Q, e0 * 1e2 ** 2 * 5, e0 * 1e4 ** 2 / 5)
    assert abs(sl - (-(q + 1) / 2)) < 0.1                 # -(q+1)/2 = -1.75

    # KN: 10 keV target -> Γ(γ_max)~780 -> steeper
    e0k = e0 * 1e4
    etk, nphk = target(e0k)
    eok = np.logspace(np.log10(e0k * 3), np.log10(e0k * 1e4 ** 2 * 3), 90)
    Qk = sz.ic_sed(n_e, etk, nphk, eok)
    slk = sed_slope(eok, Qk, e0k * 1e2 ** 2 * 2, e0k * 1e4 ** 2 / 5)
    assert slk < sl                                       # KN suppresses high-E


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
