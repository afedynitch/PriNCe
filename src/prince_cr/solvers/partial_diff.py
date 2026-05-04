import numpy as np


class DifferentialOperator(object):
    def __init__(self, cr_grid, nspec):
        self.ebins = cr_grid.bins
        self.egrid = cr_grid.grid
        self.ewidths = cr_grid.widths
        self.dim_e = cr_grid.d
        # binsize in log(e)
        self.log_width = np.log(self.ebins[1] / self.ebins[0])

        self.nspec = nspec
        # Keep the FD operator as scipy CSR. Backend conversion (MKL handle,
        # cupy CSR) is the solver's job — done at the ETD2 boundary in
        # `propagation.UHECRPropagationSolverETD2._ensure_D_split`. The
        # previous in-class conversion to ``cupyx.scipy.sparse.csr_matrix``
        # collided with ETD2's `split_operator` (scipy `sp.diags` does not
        # accept cupy arrays).
        self.operator = self.construct_differential_operator()

    def construct_differential_operator(self):
        from scipy.sparse import block_diag, coo_matrix

        # Bulk: 4th-order asymmetric upwind stencil on log-E (correct for losses
        # that move flux toward lower E). Rows 0..2 and dim_e-3..dim_e-1 use
        # 3- and 4-point one-sided stencils (2nd-order at the edges).
        diags_leftmost = [1, 2, 3]
        coeffs_leftmost = [-3, 4, -1]
        denom_leftmost = 2.0
        diags_left_1 = [-1, 0, 1, 2]
        coeffs_left_1 = [-2, -3, 6, -1]
        denom_left_1 = 6.0
        diags_left_2 = [-1, 0, 1, 2]
        coeffs_left_2 = [-2, -3, 6, -1]
        denom_left_2 = 6.0

        diags = [-1, 0, 1, 2, 3]
        coeffs = [-3, -10, 18, -6, 1]
        denom = 12.0

        # Last rows at the right of operator matrix
        diags_right_2 = [-d for d in diags_left_2[::-1]]
        coeffs_right_2 = [-d for d in coeffs_left_2[::-1]]
        denom_right_2 = denom_left_2
        diags_right_1 = [-d for d in diags_left_1[::-1]]
        coeffs_right_1 = [-d for d in coeffs_left_1[::-1]]
        denom_right_1 = denom_left_1
        diags_rightmost = [-d for d in diags_leftmost[::-1]]
        coeffs_rightmost = [-d for d in coeffs_leftmost[::-1]]
        denom_rightmost = denom_leftmost

        h = self.log_width
        dim_e = self.dim_e
        last = dim_e - 1

        op_matrix = np.zeros((dim_e, dim_e))
        op_matrix[0, np.asarray(diags_leftmost)] = np.asarray(coeffs_leftmost) / (
            denom_leftmost * h
        )
        op_matrix[1, 1 + np.asarray(diags_left_1)] = np.asarray(coeffs_left_1) / (
            denom_left_1 * h
        )
        op_matrix[2, 2 + np.asarray(diags_left_2)] = np.asarray(coeffs_left_2) / (
            denom_left_2 * h
        )
        op_matrix[last, last + np.asarray(diags_rightmost)] = np.asarray(
            coeffs_rightmost
        ) / (denom_rightmost * h)
        op_matrix[last - 1, last - 1 + np.asarray(diags_right_1)] = np.asarray(
            coeffs_right_1
        ) / (denom_right_1 * h)
        op_matrix[last - 2, last - 2 + np.asarray(diags_right_2)] = np.asarray(
            coeffs_right_2
        ) / (denom_right_2 * h)
        for row in range(3, dim_e - 3):
            op_matrix[row, row + np.asarray(diags)] = np.asarray(coeffs) / (denom * h)
        # Construct an operator by left multiplication of the back-substitution
        # dlnE to dE. The right energy loss has to be later multiplied in every step
        single_op = coo_matrix(op_matrix * (1.0 / self.egrid)[:, None])

        # construct the operator for the whole matrix, by repeating
        return block_diag(self.nspec * [single_op]).tocsr()

    def solve_coefficients(self, stencils, degree=1):
        """Calculates the finite difference coefficients for given stencils.

        Note: The function sets up a linear equation system and solves it numerically
              Do not expect the result to be 100% accurate, as the coefficients are
              usually fractions.

        Args:
            stencils (list of integers): position of stencils on regular grid
            degree (integer): degree of derviative, default: 1
        """
        if len(stencils) < degree + 1:
            raise Exception(
                "Not enough stencils to solve for dervative of "
                + "degree {:}, stencils given: {}".format(degree, stencils)
            )

        from math import factorial

        # setup of equation system
        exponents = np.arange(len(stencils))
        matrix = np.power.outer(stencils, exponents).T
        right = np.zeros_like(stencils)
        right[degree] = factorial(degree)

        # solution
        return np.linalg.solve(matrix, right)
