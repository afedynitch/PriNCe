"""Additional tests for prince_cr.solvers.propagation module.

Covers UHECRPropagationSolver base-class behavior and ETD2 init/options.
"""

import numpy as np
import pytest

from prince_cr.solvers.propagation import UHECRPropagationSolverETD2
from prince_cr.cr_sources import SimpleSource


@pytest.fixture(scope="module")
def prince_run(prince_run_m4):
    return prince_run_m4


def _solver(cls, prince_run, **kw):
    """Helper: build a solver with sensible defaults for the small grid."""
    return cls(
        initial_z=1.0,
        final_z=0.0,
        prince_run=prince_run,
        **kw,
    )


class TestUHECRPropagationSolverInit:
    def test_basic_init(self, prince_run):
        solver = _solver(UHECRPropagationSolverETD2, prince_run)
        assert solver.initial_z == 1.0
        assert solver.final_z == 0.0
        assert solver.dim_states == prince_run.dim_states

    def test_known_species(self, prince_run):
        solver = _solver(UHECRPropagationSolverETD2, prince_run)
        assert solver.known_species == prince_run.spec_man.known_species

    def test_dldz(self, prince_run):
        solver = _solver(UHECRPropagationSolverETD2, prince_run)
        result = solver.dldz(0.0)
        assert np.isfinite(result)
        assert result < 0  # dldz should be negative

    def test_hooks(self, prince_run):
        solver = _solver(UHECRPropagationSolverETD2, prince_run)
        # These should not raise
        solver.pre_step_hook(0.5)
        solver.post_step_hook(0.5)

    def test_res_property(self, prince_run):
        solver = _solver(UHECRPropagationSolverETD2, prince_run)
        solver.state = np.random.rand(prince_run.dim_states) * 1e-20
        res = solver.res
        assert res is not None
        assert res.state is solver.state

    def test_add_source(self, prince_run):
        solver = _solver(UHECRPropagationSolverETD2, prince_run)
        src = SimpleSource(
            prince_run,
            params={2212: (2.0, 1e10, 1.0)},
            m="flat",
        )
        solver.add_source_class(src)
        assert len(solver.list_of_sources) == 1

    def test_disable_adiabatic(self, prince_run):
        solver = _solver(
            UHECRPropagationSolverETD2,
            prince_run,
            enable_adiabatic_losses=False,
        )
        assert solver.enable_adiabatic_losses is False

    def test_disable_pairprod(self, prince_run):
        solver = _solver(
            UHECRPropagationSolverETD2,
            prince_run,
            enable_pairprod_losses=False,
        )
        assert solver.enable_pairprod_losses is False

    def test_disable_photohad(self, prince_run):
        solver = _solver(
            UHECRPropagationSolverETD2,
            prince_run,
            enable_photohad_losses=False,
        )
        assert solver.enable_photohad_losses is False


class TestInjection:
    def test_single_source(self, prince_run):
        solver = _solver(UHECRPropagationSolverETD2, prince_run)
        solver.add_source_class(
            SimpleSource(prince_run, params={2212: (2.0, 1e10, 1.0)}, m="flat")
        )
        result = solver.injection(0.01, 0.5)
        assert result.shape == (prince_run.dim_states,)

    def test_multiple_sources(self, prince_run):
        solver = _solver(UHECRPropagationSolverETD2, prince_run)
        solver.add_source_class(
            SimpleSource(prince_run, params={2212: (2.0, 1e10, 1.0)}, m="flat")
        )
        solver.add_source_class(
            SimpleSource(prince_run, params={2212: (2.5, 1e11, 0.5)}, m="flat")
        )
        result = solver.injection(0.01, 0.5)
        assert result.shape == (prince_run.dim_states,)


class TestETD2SolverOptions:
    def test_z_offset(self, prince_run):
        solver = _solver(
            UHECRPropagationSolverETD2,
            prince_run,
            z_offset=0.1,
        )
        assert solver.z_offset == 0.1
        assert solver.initial_z == 1.1
        assert solver.final_z == 0.1

    def test_smoke_solve(self, prince_run):
        """ETD2 runs end-to-end on the test fixture without NaN."""
        solver = UHECRPropagationSolverETD2(
            initial_z=0.05,
            final_z=0.0,
            prince_run=prince_run,
        )
        solver.add_source_class(
            SimpleSource(
                prince_run,
                params={2212: (2.0, 1e10, 1e-50)},
                m="flat",
            )
        )
        solver.solve(dz=0.005, verbose=False, progressbar=False)
        assert np.all(np.isfinite(solver.state))


