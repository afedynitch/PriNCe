"""Unit tests for the Phase-B EM-cascade kernels and 1D cascade
(prince_cr.cascade.kernels, prince_cr.cascade.cascade)."""

import numpy as np
import pytest

from prince_cr import photonfields as pf
from prince_cr.cascade.kernels import (
    C_CM,
    M_E,
    SIGMA_THOMSON,
    ic_energy_loss_rate,
    pair_injection_spectrum,
)


@pytest.fixture(scope="module")
def cmb_field():
    cmb = pf.CMBPhotonSpectrum()
    eps = np.logspace(-15, -9, 700)
    n = np.asarray(cmb.get_photon_density(eps, 0.0))
    return cmb, eps, n


def test_ic_loss_matches_thomson(cmb_field):
    """IC energy-loss rate matches (4/3) sigma_T c gamma^2 U_rad in the
    Thomson regime (low gamma)."""
    _, eps, n = cmb_field
    U = np.trapezoid(eps * n, eps)
    for E_e in (50.0, 100.0):  # GeV — deep Thomson on CMB
        g = E_e / M_E
        analytic = 4.0 / 3.0 * SIGMA_THOMSON * C_CM * g**2 * U
        assert ic_energy_loss_rate(E_e, eps, n) == pytest.approx(analytic, rel=0.02)


def test_ic_loss_kn_suppressed(cmb_field):
    """At high gamma the IC loss is Klein-Nishina suppressed below the
    Thomson extrapolation."""
    _, eps, n = cmb_field
    U = np.trapezoid(eps * n, eps)
    E_e = 1e4  # 10 TeV — KN onset on CMB
    g = E_e / M_E
    thomson = 4.0 / 3.0 * SIGMA_THOMSON * C_CM * g**2 * U
    assert ic_energy_loss_rate(E_e, eps, n) < 0.95 * thomson


def test_pair_injection_energy_conservation(cmb_field):
    """Produced e+/e- carry the photon energy: <E_e> = E0/2 (symmetric
    spectrum), to a few percent in the PeV cascade band."""
    _, eps, n = cmb_field
    E0 = 1e6  # 1 PeV
    E_e = np.logspace(np.log10(1.01 * M_E), np.log10(E0), 500)
    dN = pair_injection_spectrum(E_e, E0, eps, n)
    num = np.trapezoid(dN, E_e)
    ene = np.trapezoid(E_e * dN, E_e)
    assert num > 0
    assert ene / num == pytest.approx(E0 / 2.0, rel=0.03)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.xfail(
    reason="run_cascade is deprecated and carries the energy_ratio~2 bug; "
    "superseded by cascade.kinetic_cascade_transfer (validated vs Kalashev Fig 2).",
    strict=False,
)
def test_cascade_universal_spectrum():
    """The saturated cascade conserves energy and yields the universal
    E^-2 spectrum below the absorption break, independent of E_inj."""
    from prince_cr.cascade.cascade import absorption_energy, run_cascade

    field = pf.CombinedPhotonField([pf.CMBPhotonSpectrum, pf.CIBDominguez2D])
    slopes = []
    for E_inj in (1e6, 1e7):
        r = run_cascade(E_inj, 0.1, field, n_grid=70)
        assert r["energy_ratio"] == pytest.approx(1.0, rel=0.01)
        E, d = r["E"], r["dNdE"]
        m = (E > 1) & (E < r["E_abs"]) & (d > 0)
        slopes.append(np.polyfit(np.log(E[m]), np.log(d[m]), 1)[0])
    # universal cascade spectrum: dN/dE ~ E^-2 (within ~0.15)
    for s in slopes:
        assert -2.2 < s < -1.8
    # universal: slope independent of injection energy
    assert abs(slopes[0] - slopes[1]) < 0.1


def test_absorption_energy_decreases_with_redshift():
    from prince_cr.cascade.cascade import absorption_energy

    field = pf.CombinedPhotonField([pf.CMBPhotonSpectrum, pf.CIBDominguez2D])
    e_lo = absorption_energy(1.0, field)
    e_hi = absorption_energy(0.03, field)
    assert e_hi > e_lo  # higher z -> lower transparency energy
    assert 50.0 < absorption_energy(1.0, field) < 300.0  # ~100 GeV at z=1


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_cascade_linear_spectrum_injection():
    """run_cascade is linear: cascade(δ_A+δ_B) == cascade(δ_A)+cascade(δ_B),
    so an injection spectrum is handled in one pass (inheriting the
    CRPropa-validated mono response). Also checks energy conservation for a
    power-law injection."""
    import numpy as np
    from prince_cr import photonfields as pf
    from prince_cr.cascade.cascade import run_cascade

    field = pf.CombinedPhotonField([pf.CMBPhotonSpectrum, pf.CIBDominguez2D])
    z, Etop, ng = 0.1, 1e7, 60
    E = np.logspace(0, np.log10(Etop), ng)
    dE = np.gradient(E)

    def delta(Etgt):
        v = np.zeros(ng)
        i = np.argmin(np.abs(E - Etgt))
        v[i] = 1.0 / dE[i]
        return v

    kw = dict(n_grid=ng, e_min=1.0)
    rA = run_cascade(Etop, z, field, inject_dNdE=delta(1e6), **kw)["dNdE"]
    rB = run_cascade(Etop, z, field, inject_dNdE=delta(3e4), **kw)["dNdE"]
    rAB = run_cascade(Etop, z, field, inject_dNdE=delta(1e6) + delta(3e4), **kw)["dNdE"]
    m = (rA + rB) > 1e-12 * np.max(rA + rB)
    assert np.max(np.abs(rAB[m] / (rA[m] + rB[m]) - 1.0)) < 1e-6

    res = run_cascade(Etop, z, field, inject_dNdE=E**-2.0, **kw)
    assert res["energy_ratio"] == pytest.approx(1.0, rel=0.02)
