"""Tests for the EM cascade inside the transport solver (M1: γ absorption).

Validates that the parallel EMInteractionRate is correctly wired into the
ETD2 solver and that an injected γ population is absorbed as exp(-∫sink dl),
with redshift handled natively by the solver. Skipped if the FLUKA smoke DB
is unavailable."""

import os

import numpy as np
import pytest

import prince_cr.config as config

_SMOKE = os.path.expanduser(
    "~/work/devel/UH-UHECR-Fluka-Prince/runs/2026-05-04_pfu-v1-smoke"
)


@pytest.fixture(scope="module")
def em_run():
    if not os.path.isdir(_SMOKE):
        pytest.skip("FLUKA smoke DB not available")
    config.fluka_db_path = _SMOKE
    config.fluka_db_fname = "prince_db_v1_smoke.h5"
    # Use the production grid (8 bins/decade) so the moderate-τ absorption
    # transition is resolved; conftest's coarse grid jumps past it. Restore.
    saved = (config.cosmic_ray_grid, config.photon_grid)
    config.cosmic_ray_grid = (3, 14, 8)
    config.photon_grid = (-15, -6, 8)
    from prince_cr import photonfields as pf
    from prince_cr.core import PriNCeRun

    field = pf.CombinedPhotonField([pf.CMBPhotonSpectrum, pf.CIBDominguez2D])
    run = PriNCeRun(max_mass=4, photon_field=field, enable_em_cascade=True)
    yield run
    config.cosmic_ray_grid, config.photon_grid = saved


def test_em_off_is_default():
    if not os.path.isdir(_SMOKE):
        pytest.skip("FLUKA smoke DB not available")
    config.fluka_db_path = _SMOKE
    config.fluka_db_fname = "prince_db_v1_smoke.h5"
    from prince_cr import photonfields as pf
    from prince_cr.core import PriNCeRun

    run = PriNCeRun(max_mass=4, photon_field=pf.CMBPhotonSpectrum())
    assert run.em_int_rates is None  # default off, nuclear path untouched


def test_em_jacobian_gamma_sink(em_run):
    """EM Jacobian γ-block diagonal equals -dτ/dl from the validated kernel."""
    from prince_cr.cascade.opacity import _kernel_per_length

    z = 0.1
    jac = em_run.em_int_rates.get_hadr_jacobian(z, 1.0, force_update=True)
    g = em_run.spec_man.pdgid2sref[22]
    diag = jac.toarray().diagonal()[g.sl]
    eps = np.logspace(-15, -7, 256)
    mu = np.linspace(-1, 1, 128)
    expect = -np.array(
        [_kernel_per_length(E, z, em_run.photon_field, eps, mu) for E in em_run.cr_grid.grid]
    )
    assert np.allclose(diag, expect, rtol=1e-9)
    assert np.all(diag <= 0)  # loss term


def test_em_absorption_in_transport(em_run):
    """Injected γ absorbed as exp(-∫sink dl) inside the ETD2 solver."""
    from scipy.integrate import trapezoid as trapz

    from prince_cr.cascade.opacity import _kernel_per_length
    from prince_cr.cosmology import H
    from prince_cr.data import PRINCE_UNITS
    from prince_cr.solvers import UHECRPropagationSolverETD2

    run = em_run
    g = run.spec_man.pdgid2sref[22]
    E = run.cr_grid.grid
    z_s = 0.1
    eps = np.logspace(-15, -7, 256)
    mu = np.linspace(-1, 1, 128)
    zz = np.linspace(0, z_s, 600)
    c = PRINCE_UNITS.c
    tau = np.array(
        [
            trapz(
                [_kernel_per_length(Ei, z, run.photon_field, eps, mu) * c / ((1 + z) * H(z)) for z in zz],
                zz,
            )
            for Ei in E
        ]
    )

    solver = UHECRPropagationSolverETD2(
        initial_z=z_s,
        final_z=0.0,
        prince_run=run,
        enable_pairprod_losses=False,
        enable_adiabatic_losses=False,
        enable_partial_diff_jacobian=False,
        enable_injection_jacobian=False,
    )
    solver.recomp_z_threshold = 1e-5
    init = np.zeros(run.dim_states)
    init[g.sl] = 1.0
    solver._init_state = lambda: init.copy()
    solver.solve(dz=2e-4)
    surv = solver.state[g.sl]

    # compare where tau is resolvable (grid-resolution-robust window)
    m = (tau > 0.02) & (tau < 10.0)
    assert m.sum() >= 1, "no grid bin with resolvable tau (check grid)"
    ratio = surv[m] / np.exp(-tau[m])
    assert np.max(np.abs(ratio - 1.0)) < 0.03  # transport reproduces exp(-tau)
