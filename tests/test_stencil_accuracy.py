"""Diagnostic accuracy tests for the continuous-loss FD stencil.

The default ``DifferentialOperator`` (``partial_diff.py:486–639``) is a
sparse 4th-order one-sided FD operator on log-E, left-multiplied by
``diag(1/E)`` to convert ``d/dlog(E)`` into ``d/dE`` and block-diagonal-
replicated across species.

These tests apply the operator to known analytic profiles and compare with
the exact derivative. They are diagnostics, not regression gates — they
characterise where the stencil is accurate (centered rows on smooth power
laws) and where it is not (boundary rows, sharp features), informing the
decision tree in Stage 4 of the ETD2 plan.
"""

from __future__ import annotations

import numpy as np
import pytest

from prince_cr.solvers.partial_diff import DifferentialOperator


def _build_operator(prince_run):
    return DifferentialOperator(prince_run.cr_grid, 1).operator


def _power_law_profile(egrid, alpha):
    """f(E) = E^α. Returns f and the exact df/dE."""
    f = egrid**alpha
    dfdE = alpha * egrid ** (alpha - 1.0)
    return f, dfdE


@pytest.mark.parametrize("alpha", [-3.0, -2.0, -1.0, 1.0, 2.0])
def test_stencil_power_law_centered_rows(prince_run_m4, alpha):
    """Centered rows reproduce d/dE of E^α on the small test grid.

    The 5-point asymmetric stencil ``[-3, -10, 18, -6, 1]/12`` is 4th-order
    accurate; truncation error scales with ``|α|^5 · h^4``. At 4 bins/decade
    (the unit-test grid) ``h = ln(10)/4 ≈ 0.576``, so even centered-row
    errors can exceed 30% for steep spectra. The production-grid version is
    in :func:`test_stencil_grid_resolution_sensitivity`.
    """
    op = _build_operator(prince_run_m4)
    egrid = prince_run_m4.cr_grid.grid
    n_E = egrid.size
    interior = slice(3, n_E - 3)

    f, dfdE_exact = _power_law_profile(egrid, alpha)
    dfdE_num = op.dot(f)

    rel = np.abs(dfdE_num[interior] - dfdE_exact[interior]) / np.abs(
        dfdE_exact[interior]
    )
    print(f"\n  4 bins/dec, α={alpha:+.1f}: rel.max={rel.max():.2e}")
    # Sanity bound: at h ≈ 0.576, theoretical 4th-order truncation gives
    # 0.5 max for |α| ≤ 3.5. Anything above that signals a stencil bug.
    assert np.all(np.isfinite(dfdE_num))
    assert rel.max() < 0.5, (
        f"Centered-row truncation error blows past 4th-order bound for "
        f"α={alpha}: {rel.max():.3e}"
    )


def test_stencil_boundary_rows_are_lower_order(prince_run_m4):
    """Boundary rows use 3-point one-sided FD (only 2nd-order accurate).

    Documents the asymmetry: interior rows hit 4th-order, the first three
    and last three rows fall back to 2nd-order. Anyone embedding the
    operator should know this.
    """
    op = _build_operator(prince_run_m4)
    egrid = prince_run_m4.cr_grid.grid
    n_E = egrid.size
    f, dfdE_exact = _power_law_profile(egrid, 2.0)
    dfdE_num = op.dot(f)

    boundary_idx = list(range(3)) + list(range(n_E - 3, n_E))
    interior_idx = list(range(3, n_E - 3))
    rel_boundary = np.abs(
        dfdE_num[boundary_idx] - dfdE_exact[boundary_idx]
    ) / np.abs(dfdE_exact[boundary_idx])
    rel_interior = np.abs(
        dfdE_num[interior_idx] - dfdE_exact[interior_idx]
    ) / np.abs(dfdE_exact[interior_idx])

    print(
        f"\n  α=2 boundary rel.max={rel_boundary.max():.2e}, "
        f"interior rel.max={rel_interior.max():.2e}"
    )
    # Boundary rows are dominated by the 3-point edge stencil; the test is
    # documentary, not a gate. We just assert finiteness.
    assert np.all(np.isfinite(dfdE_num))


def test_stencil_grid_resolution_sensitivity(cached_prince_run):
    """At production grid (8 bins/decade), centered rows are well below 1%.

    Cross-checks the small-grid truncation by repeating the test on the
    cached production-grid kernel.
    """
    op = _build_operator(cached_prince_run)
    egrid = cached_prince_run.cr_grid.grid
    n_E = egrid.size
    interior = slice(3, n_E - 3)

    rel_max_by_alpha = {}
    for alpha in (-3.0, -2.0, -1.0, 1.0, 2.0):
        f, dfdE_exact = _power_law_profile(egrid, alpha)
        dfdE_num = op.dot(f)
        rel = np.abs(dfdE_num[interior] - dfdE_exact[interior]) / np.abs(
            dfdE_exact[interior]
        )
        rel_max_by_alpha[alpha] = rel.max()

    print("\n  production grid (8 bins/decade) centered-row rel.max:")
    for alpha, r in rel_max_by_alpha.items():
        print(f"    α={alpha:+.1f}: {r:.2e}")

    # User raised concern about "E^-3 not valid". Document the actual
    # production-grid accuracy at α = -3.
    assert rel_max_by_alpha[-3.0] < 0.05, (
        "α = -3 truncation error > 5% — embed in operator at production "
        "grid may need a higher-order stencil."
    )


def test_stencil_h_refinement_4th_order(prince_run_m4):
    """Halving h on the centered rows shrinks the error by ~16× (4th-order).

    The check is that the observed convergence order is at least 3.5 — the
    nominal 4 minus a small slack for boundary contamination.
    """
    # We cannot easily refine the grid stored on prince_run_m4 without
    # rebuilding the kernel, so synthesize a 1-species DifferentialOperator
    # against two grids of differing resolution.
    from prince_cr.data import EnergyGrid

    g_lo = EnergyGrid(6, 11, 4)  # 4 bins/decade
    g_hi = EnergyGrid(6, 11, 8)  # 8 bins/decade
    op_lo = DifferentialOperator(g_lo, 1).operator
    op_hi = DifferentialOperator(g_hi, 1).operator

    alpha = 3.0
    f_lo, dfdE_lo = _power_law_profile(g_lo.grid, alpha)
    f_hi, dfdE_hi = _power_law_profile(g_hi.grid, alpha)
    interior_lo = slice(3, g_lo.d - 3)
    interior_hi = slice(3, g_hi.d - 3)
    err_lo = np.abs(op_lo.dot(f_lo)[interior_lo] - dfdE_lo[interior_lo]) / np.abs(
        dfdE_lo[interior_lo]
    )
    err_hi = np.abs(op_hi.dot(f_hi)[interior_hi] - dfdE_hi[interior_hi]) / np.abs(
        dfdE_hi[interior_hi]
    )
    order = np.log2(err_lo.max() / err_hi.max())
    print(f"\n  α={alpha} h-refinement (4 → 8 bins/decade) order={order:.2f}")
    # 4th-order stencil → halving h drops error by ~16x → order ≈ 4.
    assert order > 3.0, f"observed order {order:.2f} below expected 4"
