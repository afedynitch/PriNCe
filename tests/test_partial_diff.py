"""Tests for prince_cr.solvers.partial_diff module."""

import numpy as np
import pytest

from prince_cr.data import EnergyGrid
from prince_cr.solvers.partial_diff import DifferentialOperator


@pytest.fixture
def cr_grid():
    return EnergyGrid(3, 10, 8)


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
