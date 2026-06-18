"""Regression tests for prince_cr.source.coupled_cascade (the keystone
in-source coupled lepto-hadronic γγ-regenerating cascade engine).

These exercise the field-agnostic machinery only (no FLUKA DB / decay tables):
the leptonic fixed point, the γγ absorption rate, the energy-conserving γγ→e±
pair injection, and the cascade filling-down of an injected high-energy photon
component. The full hadronic validation vs AM3 lives in the run driver
(runs/2026-06-15_am3-insource/inputs/it6g_coupled_cascade.py).
"""
import numpy as np

from prince_cr.source.coupled_cascade import (
    CoupledCascadeSolver,
    gamma_gamma_abs_inv,
)
from prince_cr.source.evolution import CompositePhotonField

_ME_C2_GeV = 0.51099895e-3
_C = 2.99792458e10


def _ccs(small=True):
    return CoupledCascadeSolver(
        R_cm=1e15, B_Gauss=10.0,
        gamma_e=(1.0, 1e6, 160) if small else (1.0, 2e9, 320),
        E_ph_GeV=(1e-12, 1e5, 140) if small else (1e-13, 1e6, 280),
    )


def _powerlaw_injection(ccs, Q0=1e-10, gmax=1e3):
    g = ccs.sze.g
    Q = ccs.sze.injection_powerlaw(Q0=Q0, p=1.9, gamma_min=2.0, gamma_max=gmax,
                                   cutoff_steepness=1.0)
    return lambda field: Q


def test_coupled_leptonic_converges_positive():
    """Pure leptonic SSC fixed point: converges to a positive, finite field."""
    ccs = _ccs()
    res = ccs.solve(_powerlaw_injection(ccs), photon_injection=None,
                    n_iter=8, relax=0.6, tol=2e-3, verbose=False)
    n_g = res["n_gamma"]
    assert np.all(np.isfinite(n_g)) and np.all(n_g >= 0.0)
    assert np.all(np.isfinite(res["n_e"])) and np.all(res["n_e"] >= 0.0)
    assert res["history"][-1]["U_gamma_GeV_cm3"] > 0.0
    # the synchrotron field should peak in the IR/optical (sub-eV) for B=10 G
    E, sed = res["E_ph"], res["E_ph"] ** 2 * n_g
    assert E[np.argmax(sed)] < 1e-6      # < keV; synchrotron-dominated


def test_gg_absorption_scales_with_field():
    """t_γγ⁻¹ is positive/finite and grows with the soft-photon density."""
    E = np.logspace(2, 5, 30)                  # 100 GeV .. 100 TeV test photons
    eps = np.logspace(-9, -3, 120)             # eV .. MeV soft targets
    n_soft = 1e6 * (eps / 1e-6) ** -2.0        # a soft power law

    class _F:
        E_min_GeV, E_max_GeV = eps[0], eps[-1]

        def __init__(self, scale):
            self.scale = scale

        def get_photon_density(self, e, z=0.0):
            e = np.atleast_1d(np.asarray(e, float))
            return self.scale * np.interp(e, eps, n_soft, left=0.0, right=0.0)

    t1 = gamma_gamma_abs_inv(E, _F(1.0), eps, n_mu=32)
    t2 = gamma_gamma_abs_inv(E, _F(3.0), eps, n_mu=32)
    assert np.all(np.isfinite(t1)) and np.all(t1 >= 0.0)
    assert np.any(t1 > 0.0)
    assert np.allclose(t2, 3.0 * t1, rtol=1e-6)   # linear in the target density


def test_pair_injection_energy_conservation():
    """γγ→e± injection conserves energy: the e± energy-injection rate equals the
    absorbed-photon energy rate (energy-conserving pair_matrix)."""
    ccs = _ccs()
    # a field: strong soft photons (eV–MeV) + a hard photon bump at ~1 GeV; the
    # pairs (≤1 GeV ⇒ γ_e ≤ 2e3) sit well inside the lepton grid (γ_hi=1e6), so
    # any energy mismatch is the pair_matrix step, not grid truncation.
    E = ccs.E_ph
    n = 1e4 * (E / 1e-6) ** -2.0 * np.exp(-(1e-9 / np.clip(E, 1e-30, None)))
    n = np.where((E > 1e-9) & (E < 1e-2), n, 0.0)
    n = n + 1e-8 * np.exp(-((np.log(E) - np.log(1e0)) ** 2) / 0.2)
    ccs.field.set_internal(E, n)
    tgg = gamma_gamma_abs_inv(E, ccs.field, ccs._eps_soft, n_mu=48)
    Q_e = ccs._pair_injection(n, tgg)                 # dN/dγ on lepton grid
    g = ccs.sze.g
    e_pairs = _ME_C2_GeV * float(np.trapezoid(g * Q_e, g))   # GeV cm⁻³ s⁻¹
    e_abs = float(np.trapezoid(E * (n * tgg), E))            # GeV cm⁻³ s⁻¹
    assert e_pairs > 0.0 and np.isfinite(e_pairs)
    assert abs(e_pairs - e_abs) / e_abs < 0.12        # ≤12% (grid + interp)


def test_high_energy_photon_injection_cascades_down():
    """A hard photon injection raises the field BELOW it via the γγ cascade
    (pairs → synchrotron/IC), not just at the injection energy."""
    ccs = _ccs()
    lep = _powerlaw_injection(ccs, Q0=1e-10, gmax=1e3)
    base = ccs.solve(lep, photon_injection=None, n_iter=6, relax=0.6,
                     tol=3e-3, verbose=False)
    n_base = base["n_gamma"].copy()

    ccs2 = _ccs()
    # inject a hard component at ~10 TeV that must γγ-absorb and cascade down
    def phot_inj(field):
        Eg = np.logspace(2, 5, 60)
        Q = 1e-6 * np.exp(-((np.log(Eg) - np.log(1e4)) ** 2) / 0.3)
        return Eg, Q

    casc = ccs2.solve(lep, photon_injection=phot_inj, n_iter=8, relax=0.5,
                      tol=3e-3, verbose=False)
    n_casc = casc["n_gamma"]
    E = ccs.E_ph
    # in the GeV–TeV band the field must rise above the leptonic baseline
    band = (E > 1.0) & (E < 1e3)
    assert np.any(n_casc[band] > 2.0 * np.maximum(n_base[band], 1e-300))
    # and the regenerated cascade must add power at lower (MeV–GeV) energies too
    low = (E > 1e-4) & (E < 1.0)
    assert np.sum(n_casc[low] * E[low]) > np.sum(n_base[low] * E[low])
