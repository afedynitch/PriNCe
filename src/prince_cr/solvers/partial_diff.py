import numpy as np


class DifferentialOperator(object):
    def __init__(self, cr_grid, nspec, spec_man=None, grids=None):
        self.ebins = cr_grid.bins
        self.egrid = cr_grid.grid
        self.ewidths = cr_grid.widths
        self.dim_e = cr_grid.d
        # binsize in log(e)
        self.log_width = np.log(self.ebins[1] / self.ebins[0])

        self.nspec = nspec
        # Tier 3: when a species manager + per-home-grid registry are supplied,
        # each species' block is built on its OWN grid (cr for nuclei, em for
        # γ/e±). Without them, every block uses ``cr_grid`` — the single-grid
        # path, bit-identical to the pre-Tier-3 ``block_diag(nspec*[op])``.
        self.spec_man = spec_man
        self.grids = grids if grids is not None else {"default": cr_grid}
        # Keep the FD operator as scipy CSR. Backend conversion (MKL handle,
        # cupy CSR) is the solver's job — done at the ETD2 boundary in
        # `propagation.UHECRPropagationSolverETD2._ensure_D_split`. The
        # previous in-class conversion to ``cupyx.scipy.sparse.csr_matrix``
        # collided with ETD2's `split_operator` (scipy `sp.diags` does not
        # accept cupy arrays).
        self.operator = self.construct_differential_operator()

    def construct_differential_operator(self):
        from scipy.sparse import block_diag

        if self.spec_man is None:
            # Single-grid path: one op repeated nspec times.
            single_op = self._build_single_op(self.grids["default"])
            return block_diag(self.nspec * [single_op]).tocsr()

        # Heterogeneous path: one op per distinct home grid (cached), assembled
        # in princeidx order. With every species on "default" this produces the
        # identical block_diag as the single-grid path above.
        op_cache = {}
        blocks = []
        for spec in sorted(self.spec_man.species_refs, key=lambda s: s.princeidx):
            tag = spec.grid_tag
            if tag not in op_cache:
                op_cache[tag] = self._build_single_op(self.grids[tag])
            blocks.append(op_cache[tag])
        return block_diag(blocks).tocsr()

    def _build_single_op(self, grid):
        """Build the (d_e × d_e) finite-difference loss operator for one energy
        grid. Numerics are identical to the original single-grid construction;
        only the grid arrays (``bins``/``grid``/``d``) are parameterised."""
        from scipy.sparse import coo_matrix

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

        h = np.log(grid.bins[1] / grid.bins[0])
        dim_e = grid.d
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
        single_op = coo_matrix(op_matrix * (1.0 / grid.grid)[:, None])
        return single_op

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
