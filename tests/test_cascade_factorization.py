"""Regression tests for the AM3-style photon-field factorization of the IC /
γγ cascade kernels (prince_cr.cascade.cascade, 2026-06-13).

The factored builders contract a cached field-free kernel with n(eps); these
guard that they reproduce the scalar per-column reference implementations
(ic_emission_spectrum / ic_energy_loss_rate / pair_injection_spectrum) and that
the kernel cache builds once per grid. No FLUKA DB needed.
"""
import numpy as np
import pytest

from prince_cr import photonfields as pf
from prince_cr.cascade import cascade as C
from prince_cr.cascade.kernels import (
    M_E,
    ic_emission_spectrum,
    ic_energy_loss_rate,
    pair_injection_spectrum,
)


@pytest.fixture(scope="module")
def setup():
    # coarse grid + CMB+EBL; straddles the IC/γγ kinematic ranges
    E = np.logspace(np.log10(M_E), 13, int((13 - np.log10(M_E)) * 6) + 1)
    eps = np.logspace(-15, -9, 200)
    field = pf.CombinedPhotonField([pf.CMBPhotonSpectrum, pf.CIBGilmore2D])
    n_eps = np.asarray(field.get_photon_density(eps, 0.3), dtype="double")
    n_eps = np.where(np.isfinite(n_eps) & (n_eps > 0), n_eps, 0.0)
    C.cascade_kernel_cache_clear()
    return E, eps, n_eps


def _relmax(a, b):
    m = np.abs(b) > np.abs(b).max() * 1e-10
    return np.abs(a[m] / b[m] - 1.0).max() if m.any() else 0.0


def test_pair_matrix_matches_scalar(setup):
    E, eps, n_eps = setup
    P_fac = C.pair_matrix(E, E, eps, n_eps)
    P_ref = np.zeros((E.size, E.size))
    for j, Eg in enumerate(E):
        P_ref[:, j] = pair_injection_spectrum(E, Eg, eps, n_eps)
    assert np.count_nonzero(P_fac) == np.count_nonzero(P_ref)
    assert _relmax(P_fac, P_ref) < 1e-12


def test_emission_matrix_matches_scalar(setup):
    E, eps, n_eps = setup
    emis_fac = C._ic_emission_matrix(E, E, eps, n_eps)
    emis_ref = np.zeros((E.size, E.size))
    for j, Ej in enumerate(E):
        emis_ref[:, j] = ic_emission_spectrum(E, Ej, eps, n_eps)
    assert _relmax(emis_fac, emis_ref) < 1e-12


def test_loss_vector_matches_scalar(setup):
    E, eps, n_eps = setup
    loss_fac = C._ic_loss_vector(E, eps, n_eps)
    loss_ref = np.array([ic_energy_loss_rate(Ee, eps, n_eps) for Ee in E])
    m = np.abs(loss_ref) > np.abs(loss_ref).max() * 1e-10
    assert np.abs(loss_fac[m] / loss_ref[m] - 1.0).max() < 1e-12


def test_kernel_cache_builds_once(setup):
    E, eps, n_eps = setup
    C.cascade_kernel_cache_clear()
    A1, _ = C._ic_emis_kernel(E, E, eps)
    A2, _ = C._ic_emis_kernel(E.copy(), E.copy(), eps.copy())
    assert A1 is A2, "emission kernel rebuilt on identical grids (would lose the per-z win)"
    Ap1, _ = C._pair_kernel(E, E, eps)
    Ap2, _ = C._pair_kernel(E.copy(), E.copy(), eps.copy())
    assert Ap1 is Ap2


def test_factored_field_linearity(setup):
    """Emission and pair matrices are linear in the photon field (the matvec
    structure): scaling n(eps) scales the matrix by the same factor."""
    E, eps, n_eps = setup
    assert np.allclose(C._ic_emission_matrix(E, E, eps, 2.5 * n_eps),
                       2.5 * C._ic_emission_matrix(E, E, eps, n_eps), rtol=1e-12)
    assert np.allclose(C.pair_matrix(E, E, eps, 2.5 * n_eps),
                       2.5 * C.pair_matrix(E, E, eps, n_eps), rtol=1e-12)
