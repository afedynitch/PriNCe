# Test whether cross section are correctly created.

import numpy as np

import prince_cr.config as config
from prince_cr import cross_sections, photonfields, core
from prince_cr.solvers import UHECRPropagationSolverBDF
from prince_cr.cr_sources import AugerFitSource
import pytest

config.x_cut = 1e-4
config.x_cut_proton = 1e-2
config.tau_dec_threshold = np.inf
config.max_mass = 14
config.debug_level = 0  # suppress info statements


@pytest.fixture
def pf():
    return photonfields.CombinedPhotonField(
        [photonfields.CMBPhotonSpectrum, photonfields.CIBGilmore2D]
    )


@pytest.fixture
def cs():
    return cross_sections.CompositeCrossSection(
        [
            (0.0, cross_sections.TabulatedCrossSection, ("CRP2_TALYS",)),
            (0.14, cross_sections.SophiaSuperposition, ()),
        ]
    )


@pytest.fixture
def prince_run_talys(pf, cs):
    return core.PriNCeRun(max_mass=14, photon_field=pf, cross_sections=cs)


def test_kernel_1(prince_run_talys):
    assert prince_run_talys.int_rates._batch_matrix.shape == (287664, 72)
    assert prince_run_talys.int_rates._batch_rows.shape == (287664,)
    assert prince_run_talys.int_rates._batch_cols.shape == (287664,)
    assert prince_run_talys.int_rates._batch_vec.shape == (287664,)


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
