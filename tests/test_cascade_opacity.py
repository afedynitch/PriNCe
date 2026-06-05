"""Unit tests for prince_cr.cascade.opacity (gamma-gamma pair-production
optical depth). The sigma_gg and CMB-based checks need no external data;
the EBL spot-check is skipped if the EBL spline file is unavailable."""

import numpy as np
import pytest

from prince_cr import photonfields as pf
from prince_cr.cascade import attenuation, sigma_gg, tau_gg
from prince_cr.cascade.opacity import M_E, SIGMA_THOMSON


def test_sigma_thomson_value():
    # (8 pi / 3) r_e^2 = 6.6524e-25 cm^2
    assert SIGMA_THOMSON == pytest.approx(6.6524587e-25, rel=1e-5)


def test_sigma_gg_threshold_and_peak():
    s_th = 4.0 * M_E**2
    assert sigma_gg([0.99 * s_th])[0] == 0.0  # below threshold
    assert sigma_gg([s_th])[0] == 0.0  # exactly at threshold
    s = np.logspace(np.log10(s_th) + 1e-3, np.log10(s_th) + 3, 500)
    sig = sigma_gg(s)
    # Breit-Wheeler peaks at ~0.26 sigma_T near s ~ 2 * (4 m_e^2)
    assert sig.max() == pytest.approx(0.26 * SIGMA_THOMSON, rel=0.05)
    assert 1.5 < s[np.argmax(sig)] / s_th < 2.5


def test_tau_zero_at_zero_redshift():
    cmb = pf.CMBPhotonSpectrum()
    assert np.all(tau_gg(np.array([1e3, 1e6]), 0.0, cmb) == 0.0)


def test_tau_monotonic_in_redshift_and_energy():
    cmb = pf.CMBPhotonSpectrum()
    E = np.array([1e5, 3e5, 1e6])  # 0.1-1 PeV: CMB pair-production band
    taus = [tau_gg(E, z, cmb, eps_min=1e-14, eps_max=1e-9) for z in (0.05, 0.1, 0.2)]
    # increasing with z at fixed E
    assert np.all(taus[0] < taus[1]) and np.all(taus[1] < taus[2])
    # increasing with E across the rising edge
    assert np.all(np.diff(taus[1]) > 0)


def test_cmb_transparent_tev_opaque_pev():
    cmb = pf.CMBPhotonSpectrum()
    tev, pev = tau_gg(np.array([1e3, 1e6]), 0.1, cmb, eps_min=1e-14, eps_max=1e-9)
    assert tev < 1e-3  # universe transparent to TeV photons on the CMB
    assert pev > 1e2  # and extremely opaque to PeV photons


def test_attenuation_matches_exp_tau():
    cmb = pf.CMBPhotonSpectrum()
    E = np.array([5e5])
    t = tau_gg(E, 0.1, cmb, eps_min=1e-14, eps_max=1e-9)
    assert attenuation(E, 0.1, cmb, eps_min=1e-14, eps_max=1e-9) == pytest.approx(
        np.exp(-t)
    )


def test_ebl_horizon_order_of_magnitude():
    """EBL gamma-ray horizon: tau=1 near ~TeV at z=0.1 (Dominguez). Skips if
    the EBL spline data file is not installed."""
    try:
        dom = pf.CIBDominguez2D()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"EBL spline unavailable: {exc}")
    tau_1tev = tau_gg(np.array([1e3]), 0.1, dom)[0]
    assert 0.5 < tau_1tev < 2.0  # horizon within a factor ~2 of 1 TeV
