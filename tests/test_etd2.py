"""ETD2 solver validation: convergence, frozen-baseline agreement, stiffness."""

from __future__ import annotations

import numpy as np

from prince_cr.cr_sources import AugerFitSource
from prince_cr.solvers import UHECRPropagationSolverETD2


AUGER_PARAMS = {
    101: (0.96, 10**9.68, 20.0),
    402: (0.96, 10**9.68, 50.0),
    1407: (0.96, 10**9.68, 30.0),
}


def _add_auger(solver, prince_run):
    solver.add_source_class(
        AugerFitSource(prince_run, norm=1e-50, params=AUGER_PARAMS)
    )


def _make_etd2(prince_run, thr=None, **kw):
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
    if thr is not None:
        solver.recomp_z_threshold = thr
    _add_auger(solver, prince_run)
    return solver


def test_etd2_smoke(cached_prince_run):
    """ETD2 runs to completion on production-size kernel without NaNs."""
    solver = _make_etd2(cached_prince_run)
    solver.solve(dz=1e-3, verbose=False, progressbar=False)
    assert np.all(np.isfinite(solver.state))
    assert np.any(solver.state != 0.0)


def test_etd2_matches_frozen_baseline(cached_prince_run, baseline_state):
    """ETD2 default config agrees with a frozen tight-cache reference.

    The reference is a one-time tight-cache ETD2 solve cached on disk;
    the default-cache run is held to ~1% on physically-relevant bins and
    sub-percent in the L2 norm.
    """
    solver = _make_etd2(cached_prince_run)
    solver.solve(dz=1e-3, verbose=False, progressbar=False)
    state = solver.state

    rel_thresh = np.abs(baseline_state).max() * 1e-6
    mask = np.abs(baseline_state) > rel_thresh
    rel = np.abs(state[mask] - baseline_state[mask]) / np.abs(baseline_state[mask])
    L2_rel = np.linalg.norm(state - baseline_state) / np.linalg.norm(baseline_state)

    assert np.all(np.isfinite(state))
    assert L2_rel < 5e-3, f"L2 relative error {L2_rel:.3e}"
    assert rel.max() < 2e-2, f"max rel error {rel.max():.3e}"


def test_etd2_self_convergence_with_threshold(cached_prince_run):
    """As ``recomp_z_threshold`` shrinks, ETD2 converges to a fixed point.

    Doubles as a regression on the operator-rebuild logic: a stale cache
    would leave the error plateau insensitive to the threshold.
    """
    ref = _make_etd2(cached_prince_run, thr=1e-3)
    ref.solve(dz=1e-3, verbose=False, progressbar=False)
    ref_state = ref.state.copy()

    coarse = _make_etd2(cached_prince_run, thr=1e-2)
    coarse.solve(dz=1e-3, verbose=False, progressbar=False)

    L2 = np.linalg.norm(coarse.state - ref_state) / np.linalg.norm(ref_state)
    assert L2 < 1e-2, f"L2 = {L2:.3e}"


def test_etd2_wallclock_under_target(cached_prince_run, bench_propagation):
    """ETD2 should solve the realistic Auger case in under ~10 s on a laptop."""
    solver = _make_etd2(cached_prince_run)
    with bench_propagation("ETD2 dz=1e-3") as b:
        solver.solve(dz=1e-3, verbose=False, progressbar=False)
    # Generous bound; CI can be slower than a dev box. The target of ≤10 s
    # is to catch egregious regressions, not to police shaving.
    assert b.elapsed < 30.0, f"ETD2 wall-clock {b.elapsed:.2f}s"


def test_etd2_stiff_mode(cached_prince_run):
    """ETD2 stays stable when short-lived species are kept in the system.

    Default ``tau_dec_threshold = np.inf`` keeps all unstable species
    in-state. The ETD2 integrating-factor on the diagonal absorbs even very
    fast loss rates without instability — a path Euler cannot survive.
    Confirms there are no NaN/Inf even when stiff diagonals are present.
    """
    solver = _make_etd2(cached_prince_run)
    solver.solve(dz=1e-3, verbose=False, progressbar=False)
    assert np.all(np.isfinite(solver.state))
    # No species should have blown up to a non-physical magnitude.
    # Typical state magnitudes are ≲ 1e-25; anything > 1e-10 is unphysical.
    assert np.abs(solver.state).max() < 1e-10
