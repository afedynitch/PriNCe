"""Additional tests for prince_cr.solvers.propagation module."""

import numpy as np
import pytest

import prince_cr.config as config
from prince_cr.solvers.propagation import (
    UHECRPropagationSolverBDF,
    UHECRPropagationSolverEULER,
)
from prince_cr.cr_sources import SimpleSource


@pytest.fixture(scope="module")
def prince_run(prince_run_m4):
    return prince_run_m4


class TestUHECRPropagationSolverInit:
    def test_basic_init(self, prince_run):
        solver = UHECRPropagationSolverBDF(
            initial_z=1.0,
            final_z=0.0,
            prince_run=prince_run,
        )
        assert solver.initial_z == 1.0
        assert solver.final_z == 0.0
        assert solver.dim_states == prince_run.dim_states

    def test_known_species(self, prince_run):
        solver = UHECRPropagationSolverBDF(
            initial_z=1.0,
            final_z=0.0,
            prince_run=prince_run,
        )
        assert solver.known_species == prince_run.spec_man.known_species

    def test_dldz(self, prince_run):
        solver = UHECRPropagationSolverBDF(
            initial_z=1.0,
            final_z=0.0,
            prince_run=prince_run,
        )
        result = solver.dldz(0.0)
        assert np.isfinite(result)
        assert result < 0  # dldz should be negative

    def test_hooks(self, prince_run):
        solver = UHECRPropagationSolverBDF(
            initial_z=1.0,
            final_z=0.0,
            prince_run=prince_run,
        )
        # These should not raise
        solver.pre_step_hook(0.5)
        solver.post_step_hook(0.5)

    def test_res_property(self, prince_run):
        solver = UHECRPropagationSolverBDF(
            initial_z=1.0,
            final_z=0.0,
            prince_run=prince_run,
        )
        solver.state = np.random.rand(prince_run.dim_states) * 1e-20
        res = solver.res
        assert res is not None
        assert res.state is solver.state

    def test_add_source(self, prince_run):
        solver = UHECRPropagationSolverBDF(
            initial_z=1.0,
            final_z=0.0,
            prince_run=prince_run,
        )
        src = SimpleSource(
            prince_run,
            params={101: (2.0, 1e10, 1.0)},
            m="flat",
        )
        solver.add_source_class(src)
        assert len(solver.list_of_sources) == 1

    def test_disable_adiabatic(self, prince_run):
        solver = UHECRPropagationSolverBDF(
            initial_z=1.0,
            final_z=0.0,
            prince_run=prince_run,
            enable_adiabatic_losses=False,
        )
        assert solver.enable_adiabatic_losses is False

    def test_disable_pairprod(self, prince_run):
        solver = UHECRPropagationSolverBDF(
            initial_z=1.0,
            final_z=0.0,
            prince_run=prince_run,
            enable_pairprod_losses=False,
        )
        assert solver.enable_pairprod_losses is False

    def test_disable_photohad(self, prince_run):
        solver = UHECRPropagationSolverBDF(
            initial_z=1.0,
            final_z=0.0,
            prince_run=prince_run,
            enable_photohad_losses=False,
        )
        assert solver.enable_photohad_losses is False


class TestSemiLagrangian:
    def test_no_losses(self, prince_run):
        solver = UHECRPropagationSolverBDF(
            initial_z=1.0,
            final_z=0.0,
            prince_run=prince_run,
            enable_adiabatic_losses=False,
            enable_pairprod_losses=False,
        )
        state = np.ones(prince_run.dim_states) * 1e10
        result = solver.semi_lagrangian(0.01, 1.0, state)
        np.testing.assert_array_equal(result, state)

    def test_with_adiabatic(self, prince_run):
        solver = UHECRPropagationSolverBDF(
            initial_z=1.0,
            final_z=0.0,
            prince_run=prince_run,
            enable_adiabatic_losses=True,
            enable_pairprod_losses=False,
        )
        state = np.ones(prince_run.dim_states) * 1e10
        result = solver.semi_lagrangian(0.001, 1.0, state.copy())
        assert result.shape == state.shape

    def test_with_pairprod(self, prince_run):
        solver = UHECRPropagationSolverBDF(
            initial_z=1.0,
            final_z=0.0,
            prince_run=prince_run,
            enable_adiabatic_losses=False,
            enable_pairprod_losses=True,
        )
        state = np.ones(prince_run.dim_states) * 1e10
        result = solver.semi_lagrangian(0.001, 1.0, state.copy())
        assert result.shape == state.shape

    def test_gradient_method(self, prince_run):
        old_method = config.semi_lagr_method
        config.semi_lagr_method = "gradient"
        solver = UHECRPropagationSolverBDF(
            initial_z=1.0,
            final_z=0.0,
            prince_run=prince_run,
        )
        state = np.ones(prince_run.dim_states) * 1e10
        result = solver.semi_lagrangian(0.001, 1.0, state.copy())
        assert result.shape == state.shape
        config.semi_lagr_method = old_method

    def test_linear_method(self, prince_run):
        old_method = config.semi_lagr_method
        config.semi_lagr_method = "linear"
        solver = UHECRPropagationSolverBDF(
            initial_z=1.0,
            final_z=0.0,
            prince_run=prince_run,
        )
        state = np.ones(prince_run.dim_states) * 1e10
        result = solver.semi_lagrangian(0.001, 1.0, state.copy())
        assert result.shape == state.shape
        config.semi_lagr_method = old_method

    def test_quadratic_method(self, prince_run):
        old_method = config.semi_lagr_method
        config.semi_lagr_method = "quadratic"
        solver = UHECRPropagationSolverBDF(
            initial_z=1.0,
            final_z=0.0,
            prince_run=prince_run,
        )
        state = np.ones(prince_run.dim_states) * 1e10
        result = solver.semi_lagrangian(0.001, 1.0, state.copy())
        assert result.shape == state.shape
        config.semi_lagr_method = old_method

    def test_cubic_method(self, prince_run):
        old_method = config.semi_lagr_method
        config.semi_lagr_method = "cubic"
        solver = UHECRPropagationSolverBDF(
            initial_z=1.0,
            final_z=0.0,
            prince_run=prince_run,
        )
        state = np.ones(prince_run.dim_states) * 1e10
        result = solver.semi_lagrangian(0.001, 1.0, state.copy())
        assert result.shape == state.shape
        config.semi_lagr_method = old_method

    def test_4th_order_method(self, prince_run):
        old_method = config.semi_lagr_method
        config.semi_lagr_method = "4th_order"
        solver = UHECRPropagationSolverBDF(
            initial_z=1.0,
            final_z=0.0,
            prince_run=prince_run,
        )
        state = np.ones(prince_run.dim_states) * 1e10
        result = solver.semi_lagrangian(0.001, 1.0, state.copy())
        assert result.shape == state.shape
        config.semi_lagr_method = old_method

    def test_intp_numpy_method(self, prince_run):
        old_method = config.semi_lagr_method
        config.semi_lagr_method = "intp_numpy"
        solver = UHECRPropagationSolverBDF(
            initial_z=1.0,
            final_z=0.0,
            prince_run=prince_run,
        )
        state = np.ones(prince_run.dim_states) * 1e10
        result = solver.semi_lagrangian(0.001, 1.0, state.copy())
        assert result.shape == state.shape
        config.semi_lagr_method = old_method

    def test_finite_diff_method(self, prince_run):
        old_method = config.semi_lagr_method
        config.semi_lagr_method = "finite_diff"
        solver = UHECRPropagationSolverBDF(
            initial_z=1.0,
            final_z=0.0,
            prince_run=prince_run,
        )
        state = np.ones(prince_run.dim_states) * 1e10
        result = solver.semi_lagrangian(0.001, 1.0, state.copy())
        assert result.shape == state.shape
        config.semi_lagr_method = old_method

    def test_unknown_method_raises(self, prince_run):
        old_method = config.semi_lagr_method
        config.semi_lagr_method = "unknown_method"
        solver = UHECRPropagationSolverBDF(
            initial_z=1.0,
            final_z=0.0,
            prince_run=prince_run,
        )
        state = np.ones(prince_run.dim_states) * 1e10
        with pytest.raises(Exception, match="Unknown semi-lagrangian"):
            solver.semi_lagrangian(0.001, 1.0, state.copy())
        config.semi_lagr_method = old_method


class TestInjection:
    def test_single_source(self, prince_run):
        solver = UHECRPropagationSolverBDF(
            initial_z=1.0,
            final_z=0.0,
            prince_run=prince_run,
        )
        solver.add_source_class(
            SimpleSource(prince_run, params={101: (2.0, 1e10, 1.0)}, m="flat")
        )
        result = solver.injection(0.01, 0.5)
        assert result.shape == (prince_run.dim_states,)

    def test_multiple_sources(self, prince_run):
        solver = UHECRPropagationSolverBDF(
            initial_z=1.0,
            final_z=0.0,
            prince_run=prince_run,
        )
        solver.add_source_class(
            SimpleSource(prince_run, params={101: (2.0, 1e10, 1.0)}, m="flat")
        )
        solver.add_source_class(
            SimpleSource(prince_run, params={101: (2.5, 1e11, 0.5)}, m="flat")
        )
        result = solver.injection(0.01, 0.5)
        assert result.shape == (prince_run.dim_states,)


class TestEulerSolver:
    def test_init(self, prince_run):
        solver = UHECRPropagationSolverEULER(
            initial_z=0.1,
            final_z=0.0,
            prince_run=prince_run,
        )
        assert solver.initial_z == 0.1
        assert solver.final_z == 0.0

    def test_solve_short(self, prince_run):
        solver = UHECRPropagationSolverEULER(
            initial_z=0.01,
            final_z=0.0,
            prince_run=prince_run,
            enable_injection_jacobian=False,
            enable_partial_diff_jacobian=False,
        )
        solver.add_source_class(
            SimpleSource(
                prince_run,
                params={101: (2.0, 1e10, 1e-50)},
                m="flat",
            )
        )
        solver.solve(dz=0.005, verbose=False, progressbar=False)
        assert solver.state is not None

    def test_solve_with_initial_inj(self, prince_run):
        solver = UHECRPropagationSolverEULER(
            initial_z=0.01,
            final_z=0.0,
            prince_run=prince_run,
            enable_injection_jacobian=True,
            enable_partial_diff_jacobian=True,
        )
        solver.add_source_class(
            SimpleSource(
                prince_run,
                params={101: (2.0, 1e10, 1e-50)},
                m="flat",
            )
        )
        solver.solve(
            dz=0.005,
            verbose=False,
            initial_inj=False,
            disable_inj=True,
            progressbar=False,
        )
        assert solver.state is not None


class TestBDFSolverOptions:
    def test_custom_tolerances(self, prince_run):
        solver = UHECRPropagationSolverBDF(
            initial_z=1.0,
            final_z=0.0,
            prince_run=prince_run,
            atol=1e30,
            rtol=1e-5,
        )
        assert solver.atol == 1e30
        assert solver.rtol == 1e-5

    def test_z_offset(self, prince_run):
        solver = UHECRPropagationSolverBDF(
            initial_z=1.0,
            final_z=0.0,
            prince_run=prince_run,
            z_offset=0.1,
        )
        assert solver.z_offset == 0.1
        assert solver.initial_z == 1.1
        assert solver.final_z == 0.1


class TestScipyBackend:
    def test_scipy_eqn_derivative(self, prince_run):
        """Test scipy backend derivative function."""
        import prince_cr.config as config

        old_backend = config.linear_algebra_backend
        try:
            config.linear_algebra_backend = "scipy"
            solver = UHECRPropagationSolverBDF(
                initial_z=0.1,
                final_z=0.0,
                prince_run=prince_run,
            )
            solver.add_source_class(
                SimpleSource(
                    prince_run,
                    params={101: (2.0, 1e10, 1e-50)},
                    m="flat",
                )
            )
            # Initialize jacobian first
            solver._update_jacobian(0.1)
            solver.current_z_rates = 0.1
            # Call eqn_derivative with state to cover scipy path
            state = np.zeros(solver.dim_states)
            result = solver.eqn_derivative(0.05, state)
            assert result.shape == state.shape
        finally:
            config.linear_algebra_backend = old_backend

    def test_scipy_eqn_derivative_with_partial_diff(self, prince_run):
        """Test scipy backend with partial differential jacobian."""
        import prince_cr.config as config

        old_backend = config.linear_algebra_backend
        try:
            config.linear_algebra_backend = "scipy"
            solver = UHECRPropagationSolverEULER(
                initial_z=0.01,
                final_z=0.0,
                prince_run=prince_run,
                enable_injection_jacobian=True,
                enable_partial_diff_jacobian=True,
            )
            solver.add_source_class(
                SimpleSource(
                    prince_run,
                    params={101: (2.0, 1e10, 1e-50)},
                    m="flat",
                )
            )
            solver._update_jacobian(0.01)
            solver.current_z_rates = 0.01
            state = np.zeros(solver.dim_states)
            result = solver.eqn_derivative(0.01, state)
            assert result.shape == state.shape
        finally:
            config.linear_algebra_backend = old_backend
