"""Regression baseline for the ETD2 solver.

Pins the ETD2 result on a realistic AugerFitSource setup at the production
grid, so future solver changes can be validated against a frozen reference
state.

The reference state is provided by the ``baseline_state`` fixture
(see conftest.py), cached to disk under ``tests/data/baseline_state.npz``.
"""

from __future__ import annotations

import numpy as np

from prince_cr.cr_sources import AugerFitSource
from prince_cr.solvers import UHECRPropagationSolverETD2


AUGER_PARAMS = {
    101: (0.96, 10**9.68, 20.0),
    402: (0.96, 10**9.68, 50.0),
    1407: (0.96, 10**9.68, 30.0),
}


def _make_solver(prince_run, **kw):
    solver = UHECRPropagationSolverETD2(
        initial_z=1.0,
        final_z=0.0,
        prince_run=prince_run,
        enable_pairprod_losses=True,
        enable_adiabatic_losses=True,
        enable_injection_jacobian=True,
        enable_partial_diff_jacobian=True,
        **kw,
    )
    solver.add_source_class(
        AugerFitSource(prince_run, norm=1e-50, params=AUGER_PARAMS)
    )
    return solver


def _solve_and_get_state(solver, dz=1e-3):
    solver.solve(dz=dz, verbose=False, summary=False, progressbar=False)
    return solver.state.copy()


def test_baseline_state_is_finite_and_nonzero(baseline_state):
    assert np.all(np.isfinite(baseline_state))
    assert np.any(baseline_state != 0.0)


def test_default_etd2_matches_baseline(
    cached_prince_run, baseline_state, bench_propagation
):
    """Default-config ETD2 should agree with the tight-cache reference.

    Loose-tolerance regression: catches accidental ETD2 behavior changes
    (operator construction, cache logic, sign conventions).
    """
    solver = _make_solver(cached_prince_run)
    with bench_propagation("ETD2 default config"):
        state = _solve_and_get_state(solver)

    nz = baseline_state != 0.0
    rel = np.abs(state[nz] - baseline_state[nz]) / np.abs(baseline_state[nz])
    # The default config differs from the reference by the cache-staleness
    # systematic of order ``recomp_z_threshold`` (0.01) — see
    # test_etd2_self_convergence_with_threshold.
    L2 = np.linalg.norm(state - baseline_state) / np.linalg.norm(baseline_state)
    assert L2 < 5e-3, f"L2 relative error {L2:.3e}"


def test_pure_adiabatic_loss_runs_without_nan(cached_prince_run):
    """Pure adiabatic loss path runs end-to-end without NaN/Inf.

    With photo-hadronic interactions and pair production disabled and no
    injection, the state remains zero throughout; the test exercises the
    solver wiring on an analytically-trivial case.
    """
    solver = UHECRPropagationSolverETD2(
        initial_z=0.05,
        final_z=0.0,
        prince_run=cached_prince_run,
        enable_pairprod_losses=False,
        enable_adiabatic_losses=True,
        enable_photohad_losses=False,
        enable_injection_jacobian=False,
        enable_partial_diff_jacobian=True,
    )
    solver.solve(dz=1e-4, verbose=False)
    assert np.all(np.isfinite(solver.state))
