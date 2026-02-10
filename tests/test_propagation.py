# Test whether cross section are correctly created.

import numpy as np
import pytest

from prince_cr.solvers import UHECRPropagationSolverBDF
from prince_cr.cr_sources import AugerFitSource


@pytest.fixture(scope="module")
def prince_run_talys(prince_run_m14):
    return prince_run_m14


def test_kernel_1(prince_run_talys):
    ph_dim = prince_run_talys.ph_grid.d
    assert prince_run_talys.int_rates._batch_matrix.shape[1] == ph_dim
    assert prince_run_talys.int_rates._batch_matrix.shape[0] > 0
    assert prince_run_talys.int_rates._batch_rows.shape[0] > 0
    assert prince_run_talys.int_rates._batch_cols.shape[0] > 0
    assert prince_run_talys.int_rates._batch_vec.shape[0] > 0


def test_propagation(prince_run_talys):
    solver = UHECRPropagationSolverBDF(
        initial_z=1.0,
        final_z=0.0,
        prince_run=prince_run_talys,
        enable_pairprod_losses=True,
        enable_adiabatic_losses=True,
        enable_injection_jacobian=True,
        enable_partial_diff_jacobian=True,
    )

    solver.add_source_class(
        AugerFitSource(
            prince_run_talys,
            norm=1e-50,
            params={
                101: (0.96, 10**9.68, 20.0),
                402: (0.96, 10**9.68, 50.0),
                1407: (0.96, 10**9.68, 30.0),
            },
        )
    )
    solver.solve(dz=1e-3, verbose=False, progressbar=False)

    assert solver.known_species == prince_run_talys.spec_man.known_species
