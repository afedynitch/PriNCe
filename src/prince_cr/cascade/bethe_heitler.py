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


# ---------------------------------------------------------------------------
# Vectorized kernel (field-free) — AM3-style factorization.
#
# kernel_BH_elec(gamma_e, gamma_p, eps) carries NO photon field, so the
# (gamma_e, gamma_p, eps) kernel tensor is z-INDEPENDENT. The scalar path
# above rebuilds it inside every per-z bh_pair_shape_matrix call; these
# vectorized builders compute the whole grid in one numpy pass and a module
# cache keeps it across z-nodes (the field enters only in the contraction).
# Validated element-wise against the scalar functions to ~1e-15
# (runs/2026-06-13_em-cupy-figure-refresh). The scalar functions are kept as
# the reference implementation.
# ---------------------------------------------------------------------------
def _dsigma_BH_vec(k, g, u):
    """Vectorized :func:`dsigma_BH`. k=eps_pho (ω), g=gamma_e, u=xi — arrays."""
    REG = 1e-30
    with np.errstate(all="ignore"):
        p = np.sqrt(np.maximum(g * g - 1.0, 0.0) + REG)
        g1 = k - g
        p1 = np.sqrt(np.maximum(g1 * g1 - 1.0, 0.0) + REG)
        D = g - u * p
        T = np.sqrt(np.maximum(k * k + p * p - 2.0 * k * p * u, 0.0))
        d1T = np.log((T + p1) / np.maximum(T - p1, REG))
        y1 = np.log(np.maximum((g1 + p1) / np.maximum(g1 - p1, REG), REG)) / np.maximum(p1, REG)
        Y = 2.0 * np.log(np.maximum((g * g1 + p * p1 + 1.0) / k, REG)) / np.maximum(p * p, REG)
        A0 = p * p1 / (k * k * k)
        A1 = -4.0 * (1.0 - u * u) * (2.0 * g * g + 1.0) / np.maximum(p * p * D**4, REG)
        A2 = (5.0 * g * g - 2.0 * g * g1 + 3.0) / np.maximum(p * p * D * D, REG)
        A3 = (p * p - k * k) / np.maximum(T * T * D * D, REG)
        A4 = 2.0 * g1 / np.maximum(p * p * D, REG)
        A50 = Y / np.maximum(p * p1, REG)
        A51 = 2.0 * g * (1.0 - u * u) * (3.0 * k + p * p * g1) / np.maximum(D**4, REG)
        A52 = (2.0 * g * g * (g * g + g1 * g1) - 7.0 * g * g - 3.0 * g * g1
               - g1 * g1 + 1.0) / np.maximum(D * D, REG)
        A53 = k * (g * g - g * g1 - 1.0) / np.maximum(D, REG)
        A6 = -(d1T / np.maximum(p1 * T, REG)) * (2.0 / np.maximum(D * D, REG)
                                                 - 3.0 * k / np.maximum(D, REG)
                                                 - k * (p * p - k * k) / np.maximum(T * T * D, REG))
        A7 = -2.0 * y1 / np.maximum(D, REG)
        res = A0 * (A1 + A2 + A3 + A4 + A50 * (A51 + A52 + A53) + A6 + A7)
    ok = (k >= 2.0001) & (k <= 600.0) & (T >= 1e-30)
    return np.where(ok, res, 0.0)


def _bh_inner_vec(gp, ge, omega, eps_e):
    """Vectorized :func:`_bh_inner`."""
    REG = 1e-20
    with np.errstate(all="ignore"):
        p_ = np.sqrt(np.maximum(eps_e * eps_e - 1.0, 0.0) + REG)
        xi = (gp * eps_e - ge) / (gp * p_)
        val = omega * _dsigma_BH_vec(omega, eps_e, xi) / p_
    ok = (p_ >= REG) & (xi >= -1.0) & (xi <= 1.0) & (eps_e >= 1.0) & (eps_e <= omega - 1.0)
    return np.where(ok, val, 0.0)


def _rk4_vec(node_fn, lo, hi, n_bins=5):
    """Vectorized RK4 (Simpson-weighted, n_bins) over per-element [lo, hi];
    replicates the scalar :func:`_rk4_3`/:func:`_rk4_4` accumulation. Zero where
    hi <= lo."""
    valid = hi > lo
    dx = np.where(valid, (hi - lo) / n_bins, 0.0)
    y = np.zeros_like(lo)
    x = lo.copy()
    for _ in range(n_bins):
        k1 = dx * node_fn(x)
        k2 = dx * node_fn(x + 0.5 * dx)
        k4 = dx * node_fn(x + dx)
        y = y + (k1 + k4) / 6.0 + 2.0 * k2 / 3.0
        x = x + dx
    return np.where(valid, y, 0.0)


def _bh_outer_vec(gp, ge, omega):
    """Vectorized :func:`_bh_outer` (inner RK4 over eps_e in [eps_lo, ω-1])."""
    REG = 1e-5
    lo = (gp * gp + ge * ge) / (2.0 * gp * ge) + REG
    hi = omega - 1.0 - REG
    ok = ((hi - lo) >= -1e-4 * lo) & ((hi - lo) / lo >= 1e-4)
    lo_e = np.where(ok, lo, 1.0)
    hi_e = np.where(ok, hi, 0.0)
    res = _rk4_vec(lambda ee: _bh_inner_vec(gp, ge, omega, ee), lo_e, hi_e)
    return np.where(ok, res, 0.0)


def bh_kernel_tensor(E_e, E_p, eps):
    """Field-free BH kernel tensor ``K[i, j, k] = kernel_BH_elec(E_e_i, E_p_j,
    eps_k)`` built in one vectorized pass over the full grid. z-INDEPENDENT."""
    ge = (E_e / M_E)[:, None, None]
    gp = (E_p / M_P)[None, :, None]
    ke = (eps / M_E)[None, None, :]
    GE, GP, KE = np.broadcast_arrays(ge, gp, ke)
    upper = 2.0 * GP * KE
    lower = (GP + GE) ** 2 / (2.0 * GP * GE)
    ok = (upper - lower) / np.maximum(lower, 1e-30) >= 1e-8
    lo = np.where(ok, lower, 1.0)
    hi = np.where(ok, upper, 0.0)
    y = _rk4_vec(lambda om: _bh_outer_vec(GP, GE, om), lo, hi)
    K = y * GE / (2.0 * GP**3 * KE**2)
    return np.where(ok, K, 0.0)


# Module cache for the field-free kernel tensor, keyed by the log-grid
# signature (shape + endpoints). _em_bh_at calls bh_pair_shape_matrix once per
# z-node with the same E_e/E_p/eps grids, so the kernel builds once and every
# subsequent z reuses it. Clear with bh_kernel_cache_clear().
_BH_KERNEL_CACHE = {}


def _grid_key(a):
    a = np.asarray(a)
    return (a.size, float(a[0]), float(a[-1]))


def bh_kernel_cache_clear():
    _BH_KERNEL_CACHE.clear()


def _bh_kernel_cached(E_e, E_p, eps):
    from prince_cr.cascade.kernels import (
        kernel_cache_path, kernel_disk_load, kernel_disk_save,
    )

    key = (_grid_key(E_e), _grid_key(E_p), _grid_key(eps))
    K = _BH_KERNEL_CACHE.get(key)
    if K is not None:
        return K
    # Disk cache: the BH kernel is the decisive one to persist (~37 s build,
    # ~8 MB file). Field-free → reusable across runs/redshifts.
    path = kernel_cache_path("bh", key)
    disk = kernel_disk_load(path)
    if disk is not None:
        _BH_KERNEL_CACHE[key] = disk[0]
        return disk[0]
    K = bh_kernel_tensor(E_e, E_p, eps)
    kernel_disk_save(path, (K,))
    _BH_KERNEL_CACHE[key] = K
    return K


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
    ln_eps = np.log(eps)
    # Field-free kernel tensor K[i, j, k] = kernel_BH_elec(E_e_i, E_p_j, eps_k),
    # built once (vectorized) and cached across z-nodes — the photon field is
    # NOT in K (AM3-style factorization). The rest-frame threshold ω=2γ_pε>2
    # and the Born cap ω<600 are enforced inside the kernel; per-(γ_p) the
    # ε>1/γ_p threshold makes K vanish below it.
    ok_eps = (n_eps > 0) & (eps > 0)
    R = np.zeros((E_e.size, E_p.size))
    if not np.any(ok_eps):
        return R
    K = _bh_kernel_cached(E_e, E_p, eps)        # (n_Ee, n_Ep, n_eps), z-independent
    # col[i, j] = ∫ K[i,j,·] n(ε) ε d(ln ε)  — the only z-dependent step (matvec).
    wln = (n_eps * eps)[None, None, :]
    col = trapz((K * wln)[:, :, ok_eps], ln_eps[ok_eps], axis=2)   # (n_Ee, n_Ep)
    # E_e band (e± capped at E_p; kernel ~0 far below the γ_e~γ_p peak) + e_p_min
    band = (E_e[:, None] <= E_p[None, :]) & (E_e[:, None] > 1e-6 * E_p[None, :])
    col = np.where(band & (E_p[None, :] >= e_p_min), col, 0.0)
    # per-column energy normalization: 2·∫E_e col dE_e = E_p (pair, both signs)
    e_carried = 2.0 * trapz(E_e[:, None] * col, E_e, axis=0)       # (n_Ep,)
    good = e_carried > 0
    R[:, good] = col[:, good] * (E_p[good] / e_carried[good])[None, :]
    return R


def bh_pair_shape_matrix_batched(E_e, E_p, eps, n_eps_stack, e_p_min=1e5, xp=None):
    """Batched :func:`bh_pair_shape_matrix` over a stack of photon fields.

    ``n_eps_stack`` is ``(n_eps, Nz)``. Returns ``R`` of shape ``(Nz, n_Ee,
    n_Ep)`` — the energy-normalized BH e± shape at each redshift, built in one
    pass: the cached field-free kernel is contracted against the stacked fields
    (one einsum) and the per-(column, z) energy normalization is vectorized.
    Reproduces the per-z function to fp. ``xp`` = numpy (default) or cupy."""
    if xp is None:
        xp = np
    K = _bh_kernel_cached(E_e, E_p, eps)            # (i, j, k) field-free
    # ln-eps trapezoid weights (sum w*y == trapz(y, ln eps)); fold in eps so the
    # contraction equals trapz(K n(eps) eps, ln eps).
    ln_eps = np.log(eps)
    w = np.empty_like(ln_eps)
    w[1:-1] = (ln_eps[2:] - ln_eps[:-2]) / 2.0
    w[0] = (ln_eps[1] - ln_eps[0]) / 2.0
    w[-1] = (ln_eps[-1] - ln_eps[-2]) / 2.0
    Kw = xp.asarray(K * (eps * w)[None, None, :])
    N = xp.asarray(n_eps_stack)
    Ee = xp.asarray(E_e); Ep = xp.asarray(E_p)
    col = xp.einsum('ijk,kz->zij', Kw, N)           # (Nz, i, j)
    band = (Ee[:, None] <= Ep[None, :]) & (Ee[:, None] > 1e-6 * Ep[None, :]) \
        & (Ep[None, :] >= e_p_min)
    col = xp.where(band[None], col, 0.0)
    # e_carried[z,j] = 2 ∫ E_e col dE_e  (trapz over the i / E_e axis)
    y = Ee[None, :, None] * col
    e_carried = 2.0 * xp.sum((y[:, 1:, :] + y[:, :-1, :])
                             * (Ee[1:] - Ee[:-1])[None, :, None] / 2.0, axis=1)
    R = xp.where(e_carried[:, None, :] > 0,
                 col * (Ep[None, None, :] / xp.where(e_carried > 0, e_carried, 1.0)[:, None, :]),
                 0.0)
    return R
