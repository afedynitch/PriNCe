"""Bethe-Heitler pair-production e± SOURCE for the co-evolved EM cascade.

PriNCe already has the BH proton energy-LOSS side: ``interaction_rates.
ContinuousPairProductionLossRate`` uses the Blumenthal-1970 / Chodorowski-1992
/ Dermer-Menon-2009 φ-function (same coefficients as AM3). What was missing is
the e± SOURCE — the pairs produced by p + γ_bg → p + e⁺e⁻ were never injected
into the cascade; the lost proton energy just vanished.

This module supplies that source, mirroring AM3's method (``ampy/bethe_heitler.py``
/ ``src/BetheHeitler.cc``): the e± spectrum SHAPE comes from the exact
differential Born cross section (the "W function", Blumenthal 1970 eq. 10 ≡
Kelner & Aharonian 2008 eq. 62), and the per-proton-bin TOTAL injected energy is
pinned to PriNCe's existing (validated) BH proton energy-loss rate. Pinning the
energy to the loss vector guarantees exact energy conservation between the
proton sink and the e± source and sidesteps the absolute-prefactor bookkeeping.

All internal quantities are in units of m_e c² (electron, photon energies) and
the proton Lorentz factor γ_p, as in AM3. ``dsigma_BH``/``kernel_BH_elec`` are
ported verbatim from ``AM3/ampy/bethe_heitler.py`` (Klinger et al. 2023).
"""

import math

import numpy as np
from scipy.integrate import trapezoid as trapz

from prince_cr.data import PRINCE_UNITS

M_E = PRINCE_UNITS.m_electron  # GeV
M_P = PRINCE_UNITS.m_proton    # GeV


# ---------------------------------------------------------------------------
# W-function differential cross section (Blumenthal 1970 / Kelner-Aharonian 2008)
# Ported verbatim from AM3 ampy/bethe_heitler.py :: dsigma_BH.
# ---------------------------------------------------------------------------
def dsigma_BH(eps_pho, gamma_e, xi):
    """d²σ/(dγ_e dξ) in units (3/16π)·σ_T·α; 0 outside the kinematic range.

    eps_pho, gamma_e in m_e c²; xi = cosθ_e in the photon rest frame.
    """
    if eps_pho < 2.0001 or eps_pho > 600.0:
        return 0.0
    REG = 1e-30
    k, g, u = eps_pho, gamma_e, xi
    p = math.sqrt(max(g * g - 1.0, 0.0) + REG)
    g1 = k - g
    p1 = math.sqrt(max(g1 * g1 - 1.0, 0.0) + REG)
    D = g - u * p
    T = math.sqrt(max(k * k + p * p - 2.0 * k * p * u, 0.0))
    if T < 1e-30:
        return 0.0
    d1T = math.log((T + p1) / max(T - p1, REG))
    y1 = math.log(max((g1 + p1) / max(g1 - p1, REG), REG)) / max(p1, REG)
    Y = 2.0 * math.log(max((g * g1 + p * p1 + 1.0) / k, REG)) / max(p * p, REG)
    A0 = p * p1 / (k * k * k)
    A1 = -4.0 * (1.0 - u * u) * (2.0 * g * g + 1.0) / max(p * p * D ** 4, REG)
    A2 = (5.0 * g * g - 2.0 * g * g1 + 3.0) / max(p * p * D * D, REG)
    A3 = (p * p - k * k) / max(T * T * D * D, REG)
    A4 = 2.0 * g1 / max(p * p * D, REG)
    A50 = Y / max(p * p1, REG)
    A51 = 2.0 * g * (1.0 - u * u) * (3.0 * k + p * p * g1) / max(D ** 4, REG)
    A52 = (2.0 * g * g * (g * g + g1 * g1) - 7.0 * g * g - 3.0 * g * g1
           - g1 * g1 + 1.0) / max(D * D, REG)
    A53 = k * (g * g - g * g1 - 1.0) / max(D, REG)
    A6 = -(d1T / max(p1 * T, REG)) * (2.0 / max(D * D, REG)
                                      - 3.0 * k / max(D, REG)
                                      - k * (p * p - k * k) / max(T * T * D, REG))
    A7 = -2.0 * y1 / max(D, REG)
    return A0 * (A1 + A2 + A3 + A4 + A50 * (A51 + A52 + A53) + A6 + A7)


def _rk4_4(ff, lower, upper, v0, v1, v2, n_bins=5):
    if upper <= lower:
        return 0.0
    dx = (upper - lower) / n_bins
    x, y = lower, 0.0
    for _ in range(n_bins):
        k1 = dx * ff(v0, v1, v2, x)
        k2 = dx * ff(v0, v1, v2, x + 0.5 * dx)
        k4 = dx * ff(v0, v1, v2, x + dx)
        y += (k1 + k4) / 6.0 + 2.0 * k2 / 3.0
        x += dx
    return y


def _rk4_3(ff, lower, upper, v0, v1, n_bins=5):
    if upper <= lower:
        return 0.0
    dx = (upper - lower) / n_bins
    x, y = lower, 0.0
    for _ in range(n_bins):
        k1 = dx * ff(v0, v1, x)
        k2 = dx * ff(v0, v1, x + 0.5 * dx)
        k4 = dx * ff(v0, v1, x + dx)
        y += (k1 + k4) / 6.0 + 2.0 * k2 / 3.0
        x += dx
    return y


def _bh_inner(gamma_p, gamma_e, eps_pho, eps_e):
    REG = 1e-20
    p_ = math.sqrt(max(eps_e * eps_e - 1.0, 0.0) + REG)
    if p_ < REG:
        return 0.0
    xi = (gamma_p * eps_e - gamma_e) / (gamma_p * p_)
    if xi < -1.0 or xi > 1.0 or eps_e < 1.0 or eps_e > eps_pho - 1.0:
        return 0.0
    return eps_pho * dsigma_BH(eps_pho, eps_e, xi) / p_


def _bh_outer(gamma_p, gamma_e, eps_pho):
    REG = 1e-5
    lower = (gamma_p * gamma_p + gamma_e * gamma_e) / (2.0 * gamma_p * gamma_e) + REG
    upper = eps_pho - 1.0 - REG
    if upper - lower < -1e-4 * lower:
        return 0.0
    if (upper - lower) / lower < 1e-4:
        return 0.0
    return _rk4_4(_bh_inner, lower, upper, gamma_p, gamma_e, eps_pho, n_bins=5)


def kernel_BH_elec(gamma_e, gamma_p, eps_pho, n_bins=5):
    """Single-lepton BH generation kernel dN/dγ_e (AM3 units; shape only here).

    Ported from AM3 ``kernel_BH_elec`` / ``BetheHeitler::bh_kernal_elec_gen``.
    Returns 0 outside the kinematic range.
    """
    upper = 2.0 * gamma_p * eps_pho
    lower = (gamma_p + gamma_e) ** 2 / (2.0 * gamma_p * gamma_e)
    if (upper - lower) / max(lower, 1e-30) < 1e-8:
        return 0.0
    y = _rk4_3(_bh_outer, lower, upper, gamma_p, gamma_e, n_bins=n_bins)
    return y * gamma_e / (2.0 * gamma_p ** 3 * eps_pho ** 2)


def bh_pair_shape_matrix(E_e, E_p, eps, n_eps, e_p_min=1e5):
    """Energy-normalized BH e± injection shape ``R[i, j]`` [1/GeV] per proton.

    ``R[i, j]`` is the e⁺(or e⁻) spectrum dN/dE_e (one lepton sign) produced by
    a proton of energy ``E_p[j]`` on the photon field ``n_eps(eps)``, integrated
    over the W-kernel and normalized per column so that the energy carried by
    the *pair* (e⁺+e⁻ = 2× one sign) equals the proton energy E_p[j]:
    ``2·∫ E_e R[i,j] dE_e = E_p[j]``. So multiplying a column by the proton's
    BH energy-loss RATE and splitting equally into e⁻ and e⁺ deposits exactly
    that lost energy as pairs (energy conservation; absolute prefactor cancels).

    Args:
        E_e, E_p (ndarray): lepton-out / proton-in energy grids [GeV].
        eps (ndarray): target photon energy grid [GeV].
        n_eps (ndarray): photon density n(eps) [GeV^-1 cm^-3] at the redshift.

    Returns:
        ndarray (len(E_e), len(E_p)): R[i, j] [1/GeV], energy-normalized columns
        (all-zero columns where BH is kinematically inactive).
    """
    ge = E_e / M_E              # lepton γ grid  (γ_e = E_e/m_e)
    gp = E_p / M_P              # proton γ grid  (γ_p = E_p/m_p — NOT m_e!)
    ke = eps / M_E             # LAB photon energy in m_e units (≪1 for CMB/EBL)
    ln_eps = np.log(eps)
    # kernel_BH_elec takes the LAB photon energy and boosts internally; the
    # rest-frame threshold ω=2γ_pε>2 (and the Born cap ω<600) are enforced
    # inside dsigma_BH. So include every populated photon bin; per-(γ_p) the
    # threshold ε>1/γ_p makes the kernel return 0 below it.
    ok_eps = (n_eps > 0) & (eps > 0)
    R = np.zeros((E_e.size, E_p.size))
    eidx = np.nonzero(ok_eps)[0]
    if not eidx.size:
        return R
    n_eps_ok = n_eps[eidx]
    eps_ok = eps[eidx]
    ke_ok = ke[eidx]
    ln_eps_ok = ln_eps[eidx]
    for j, gpj in enumerate(gp):
        if E_p[j] < e_p_min:
            continue
        # e± kinematically capped at the proton energy; the kernel is zero for
        # E_e ≳ E_p and far below the γ_e~γ_p peak. Restrict the E_e band.
        ei = np.nonzero((E_e <= E_p[j]) & (E_e > 1e-6 * E_p[j]))[0]
        col = np.zeros(E_e.size)
        for i in ei:
            gei = ge[i]
            integ = np.array([kernel_BH_elec(gei, gpj, kk) for kk in ke_ok])
            col[i] = trapz(integ * n_eps_ok * eps_ok, ln_eps_ok)
        e_carried = 2.0 * trapz(E_e * col, E_e)  # pair energy (both signs)
        if e_carried > 0:
            R[:, j] = col * (E_p[j] / e_carried)
    return R
