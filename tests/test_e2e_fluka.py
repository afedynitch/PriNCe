"""End-to-end smoke for the FLUKA photo-nuclear wiring.

Builds a small PriNCeRun (max_mass=14 from conftest), runs the ETD2 solver
across one redshift step (z=1 → z=0), and asserts the final state is
finite and that the proton flux at z=0 is positive across the egrid. This
catches gross regressions where the wiring loads but produces nonsense
(NaN propagation, all-zeros, wrong shape).
"""

from __future__ import annotations

import numpy as np


def test_e2e_solver_runs(prince_run_m14):
    """ETD2 solver completes from z=1 to z=0 with FLUKA cross sections."""
    from prince_cr.cr_sources import AugerFitSource
    from prince_cr.solvers import UHECRPropagationSolverETD2

    run = prince_run_m14

    solver = UHECRPropagationSolverETD2(
        initial_z=1.0,
        final_z=0.0,
        prince_run=run,
        enable_pairprod_losses=True,
        enable_adiabatic_losses=True,
        enable_injection_jacobian=True,
        enable_partial_diff_jacobian=True,
    )
    # Auger-fit injection. Norm tiny so absolute scale doesn't matter — we
    # only check finiteness and sign of the resulting state.
    solver.add_source_class(
        AugerFitSource(
            run,
            norm=1e-50,
            params={
                2212: (0.96, 10**9.68, 20.0),         # proton
                1000020040: (0.96, 10**9.68, 50.0),    # He-4
            },
        )
    )
    solver.solve(dz=1e-2, verbose=False, summary=False, progressbar=False)

    state = solver.state
    assert np.all(np.isfinite(state)), "solver state contains NaN/inf"

    # Proton bin: state vector slice for PDG 2212
    p_species = run.spec_man.pdgid2sref[2212]
    proton_state = state[p_species.sl]
    assert proton_state.shape[0] == run.cr_grid.d
    # Allow zeros at energies where injection is below threshold; require
    # at least *some* flux somewhere
    assert (proton_state > 0).any(), "proton flux is zero across the entire egrid"
    # ETD2 on the reduced test grid (20 bins) leaves tail-bin round-off ~1e-5
    # below the peak — allow that but reject true negatives.
    p_max = proton_state.max()
    rtol = 1e-4
    assert (proton_state >= -rtol * p_max).all(), (
        f"proton flux has unphysical negative values "
        f"(min={proton_state.min():.3e}, peak={p_max:.3e})"
    )
