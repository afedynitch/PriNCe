"""Closure / regression tests for the Bethe-Heitler e± injection kernel
(prince_cr.cascade.bethe_heitler).

Covers the 2026-06-13 AM3-style field-factorization + vectorization:
  - the vectorized field-free kernel reproduces the scalar reference,
  - the kernel tensor is photon-field-independent and cached across z,
  - bh_pair_shape_matrix conserves energy (the pinned per-column norm),
  - kinematic support (e_p_min, the e± < E_p band).
These need no FLUKA DB — the kernels are pure functions of the photon field.
"""
import numpy as np
import pytest

from prince_cr import photonfields as pf
from prince_cr.cascade import bethe_heitler as BH
from prince_cr.cascade.bethe_heitler import (
    M_E,
    M_P,
    bh_kernel_tensor,
    bh_kernel_cache_clear,
    bh_pair_shape_matrix,
    kernel_BH_elec,
)


@pytest.fixture(scope="module")
def grids():
    # Coarse log grids (fast) that still straddle the BH pair threshold
    # (omega = 2 gamma_p eps/m_e > 2): protons to 1e14 eV/nucleon, eps to 1e-7 GeV.
    E_e = np.logspace(np.log10(M_E), 14, int((14 - np.log10(M_E)) * 4) + 1)
    E_p = np.logspace(5, 14, int((14 - 5) * 4) + 1)
    eps = np.logspace(-12.5, -7.0, 20)
    return E_e, E_p, eps


@pytest.fixture(scope="module")
def cmb_n(grids):
    _, _, eps = grids
    n = np.asarray(pf.CMBPhotonSpectrum().get_photon_density(eps, 0.0), dtype="double")
    return np.where(np.isfinite(n) & (n > 0), n, 0.0)


def test_vectorized_kernel_matches_scalar(grids):
    """The vectorized field-free kernel tensor reproduces the scalar
    kernel_BH_elec element-wise (the reference implementation)."""
    E_e, E_p, eps = grids
    ge, gp, ke = E_e / M_E, E_p / M_P, eps / M_E
    K = bh_kernel_tensor(E_e, E_p, eps)
    assert K.shape == (E_e.size, E_p.size, eps.size)
    rng = np.random.default_rng(1)
    nz = 0
    worst = 0.0
    for _ in range(500):
        i = int(rng.integers(E_e.size))
        j = int(rng.integers(E_p.size))
        k = int(rng.integers(eps.size))
        s = kernel_BH_elec(ge[i], gp[j], ke[k])
        v = K[i, j, k]
        if abs(s) > 0 or abs(v) > 0:
            nz += 1
            worst = max(worst, abs(v - s) / max(abs(s), 1e-300))
    assert nz > 20, "sample hit no nonzero kernel entries — check grids"
    assert worst < 1e-12, f"vectorized kernel deviates from scalar: {worst:.3e}"


def test_kernel_is_field_independent_and_cached(grids):
    """The kernel tensor depends only on the energy grids, not the photon
    field, and _bh_kernel_cached returns the same cached object for repeated
    same-grid calls (so it builds once across z-nodes)."""
    E_e, E_p, eps = grids
    bh_kernel_cache_clear()
    K1 = BH._bh_kernel_cached(E_e, E_p, eps)
    K2 = BH._bh_kernel_cached(E_e.copy(), E_p.copy(), eps.copy())
    assert K1 is K2, "cache miss on identical grids — kernel would rebuild per z"
    assert len(BH._BH_KERNEL_CACHE) == 1


def test_pair_shape_energy_conservation(grids, cmb_n):
    """Each active column carries exactly the proton energy into the pair:
    2 * integral(E_e * R[:, j]) dE_e == E_p[j] (both e+ and e-)."""
    E_e, E_p, eps = grids
    R = bh_pair_shape_matrix(E_e, E_p, eps, cmb_n)
    active = np.where(R.sum(axis=0) > 0)[0]
    assert active.size > 0, "no active proton columns — check field/grids"
    for j in active:
        e_pair = 2.0 * np.trapezoid(E_e * R[:, j], E_e)
        assert e_pair == pytest.approx(E_p[j], rel=1e-9)


def test_kinematic_support(grids, cmb_n):
    """R vanishes below e_p_min and above the e± < E_p kinematic cap."""
    E_e, E_p, eps = grids
    e_p_min = 1e9
    R = bh_pair_shape_matrix(E_e, E_p, eps, cmb_n, e_p_min=e_p_min)
    assert np.all(R[:, E_p < e_p_min] == 0.0), "production below e_p_min"
    band = E_e[:, None] <= E_p[None, :]
    assert np.all(R[~band] == 0.0), "e± produced above the proton energy"
    assert np.all(R >= 0.0)


def test_field_normalization_cancels(grids, cmb_n):
    """R is a per-column energy-normalized SHAPE: scaling the photon field by a
    constant leaves R unchanged (the field amplitude cancels in the pinned
    normalization; only its spectral shape matters)."""
    E_e, E_p, eps = grids
    R1 = bh_pair_shape_matrix(E_e, E_p, eps, cmb_n)
    R2 = bh_pair_shape_matrix(E_e, E_p, eps, 3.7 * cmb_n)
    m = R1 > R1.max() * 1e-10
    assert np.allclose(R2[m], R1[m], rtol=1e-9)
