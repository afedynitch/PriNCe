"""Tests for prince_cr.solvers.partial_diff module."""

import numpy as np
import pytest

import prince_cr.config as config

config.debug_level = 0

from prince_cr.data import EnergyGrid
from prince_cr.solvers.partial_diff import DifferentialOperator, SemiLagrangianSolver


@pytest.fixture
def cr_grid():
    return EnergyGrid(3, 10, 8)


@pytest.fixture
def semi_lag(cr_grid):
    return SemiLagrangianSolver(cr_grid)


class TestSemiLagrangianSolver:
    def test_init(self, semi_lag, cr_grid):
        assert semi_lag.grid is cr_grid
        assert semi_lag.grid_log.shape == cr_grid.grid.shape
        assert semi_lag.bins_log.shape == cr_grid.bins.shape

    def test_get_shifted_state(self, semi_lag, cr_grid):
        state = np.ones(cr_grid.d) * 1e10
        conloss = np.ones(cr_grid.d + 1) * 0.01
        newgrid_log, newstate_log = semi_lag.get_shifted_state(conloss, state)
        assert newgrid_log.shape == state.shape
        assert newstate_log.shape == state.shape

    def test_interpolate(self, semi_lag, cr_grid):
        state = np.ones(cr_grid.d) * 1e10
        conloss = np.ones(cr_grid.d + 1) * 0.01
        result = semi_lag.interpolate(conloss, state)
        assert result.shape == state.shape

    def test_interpolate_gradient(self, semi_lag, cr_grid):
        state = np.ones(cr_grid.d) * 1e10
        conloss = np.ones(cr_grid.d + 1) * 0.01
        result = semi_lag.interpolate_gradient(conloss, state)
        assert result.shape == state.shape

    def test_interpolate_linear_weights(self, semi_lag, cr_grid):
        state = np.ones(cr_grid.d) * 1e10
        conloss = np.ones(cr_grid.d + 1) * 0.01
        result = semi_lag.interpolate_linear_weights(conloss, state)
        assert result.shape == state.shape

    def test_interpolate_quadratic_weights(self, semi_lag, cr_grid):
        state = np.ones(cr_grid.d) * 1e10
        conloss = np.ones(cr_grid.d + 1) * 0.01
        result = semi_lag.interpolate_quadratic_weights(conloss, state)
        assert result.shape == state.shape

    def test_interpolate_cubic_weights(self, semi_lag, cr_grid):
        state = np.ones(cr_grid.d) * 1e10
        conloss = np.ones(cr_grid.d + 1) * 0.01
        result = semi_lag.interpolate_cubic_weights(conloss, state)
        assert result.shape == state.shape

    def test_interpolate_4thorder_weights(self, semi_lag, cr_grid):
        state = np.ones(cr_grid.d) * 1e10
        conloss = np.ones(cr_grid.d + 1) * 0.01
        result = semi_lag.interpolate_4thorder_weights(conloss, state)
        assert result.shape == state.shape

    def test_interpolate_5thorder_weights(self, semi_lag, cr_grid):
        state = np.ones(cr_grid.d) * 1e10
        conloss = np.ones(cr_grid.d + 1) * 0.01
        result = semi_lag.interpolate_5thorder_weights(conloss, state)
        assert result.shape == state.shape

    def test_power_law_conservation(self, semi_lag, cr_grid):
        """Test that a power-law spectrum shifted slightly is well approximated."""
        state = cr_grid.grid ** (-2.0) * 1e30
        conloss = np.ones(cr_grid.d + 1) * 1e-3
        result = semi_lag.interpolate(conloss, state)
        # Should be similar to original
        ratio = result[5:-5] / state[5:-5]
        assert np.all(ratio > 0.9) and np.all(ratio < 1.1)


class TestDifferentialOperator:
    def test_init(self, cr_grid):
        nspec = 3
        diffop = DifferentialOperator(cr_grid, nspec)
        assert diffop.operator is not None
        expected_size = cr_grid.d * nspec
        assert diffop.operator.shape == (expected_size, expected_size)

    def test_operator_sparse(self, cr_grid):
        diffop = DifferentialOperator(cr_grid, 1)
        from scipy.sparse import issparse

        assert issparse(diffop.operator)

    def test_solve_coefficients_insufficient_stencils(self, cr_grid):
        diffop = DifferentialOperator(cr_grid, 1)
        with pytest.raises(Exception, match="Not enough stencils"):
            diffop.solve_coefficients(np.array([0]), degree=1)
