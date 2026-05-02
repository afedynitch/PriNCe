"""Regression: legacy and toeplitz kernel paths must agree to machine precision.

Builds two `PhotoNuclearInteractionRate` instances against the same cross sections
and photon field, one with each `config.kernel_method`, and asserts that:
  - `_batch_matrix` is the same up to row reordering
  - `coupling_mat` (after `_init_coupling_mat`) is identical (data, indices, indptr)

Row reordering tolerance: lex-sort by (row, col) before comparison, so we don't
care about the order in which rows happen to be appended during the inner loop.
"""

from __future__ import annotations

import numpy as np
import pytest

import prince_cr.config as cfg
from prince_cr import core, interaction_rates


def _build_run(method, max_mass, pf, cs):
    saved = cfg.kernel_method
    try:
        cfg.kernel_method = method
        run = core.PriNCeRun(max_mass=max_mass, photon_field=pf, cross_sections=cs)
        return run
    finally:
        cfg.kernel_method = saved


def _canonical_triplet(ir):
    """Return (rows, cols, data) sorted lexicographically by (row, col).

    `_batch_matrix` is shape (n, dph). `_batch_rows`/`_batch_cols` are length n.
    """
    rows = np.asarray(ir._batch_rows)
    cols = np.asarray(ir._batch_cols)
    data = np.asarray(ir._batch_matrix)
    order = np.lexsort((cols, rows))
    return rows[order], cols[order], data[order, :]


@pytest.mark.parametrize("max_mass", [1, 4, 14])
def test_legacy_vs_toeplitz_batch_matrix(max_mass, pf, cs):
    run_legacy = _build_run("legacy", max_mass, pf, cs)
    run_toep = _build_run("toeplitz", max_mass, pf, cs)

    r_l, c_l, d_l = _canonical_triplet(run_legacy.int_rates)
    r_t, c_t, d_t = _canonical_triplet(run_toep.int_rates)

    # Same number of stored rows
    assert r_l.shape == r_t.shape, (
        f"row count differs: legacy={r_l.shape} toeplitz={r_t.shape}"
    )
    # Same (row, col) sparsity pattern
    np.testing.assert_array_equal(r_l, r_t)
    np.testing.assert_array_equal(c_l, c_t)
    # Numerical agreement on the data: combined relative + absolute tolerance.
    # Many entries are physically zero or sub-1e-30; we only require relative
    # agreement once the magnitude is above the floating-point noise floor.
    diff = np.abs(d_l - d_t)
    abs_floor = 1e-25  # well below any physical rate × scale_fac
    rtol = 1e-10
    over = np.abs(d_l) > abs_floor
    if np.any(over):
        rel_over = (diff[over] / np.abs(d_l[over])).max()
        assert rel_over < rtol, (
            f"max relative error (above {abs_floor:g}): {rel_over:.3e}"
        )
    assert diff.max() < abs_floor or np.all(over[diff > abs_floor]), (
        f"unexpected disagreement: max abs error = {diff.max():.3e}"
    )


@pytest.mark.parametrize("max_mass", [4, 14])
def test_legacy_vs_toeplitz_coupling_mat(max_mass, pf, cs):
    """After `_init_coupling_mat`, the CSR matrices must be identical."""
    run_legacy = _build_run("legacy", max_mass, pf, cs)
    run_toep = _build_run("toeplitz", max_mass, pf, cs)

    cm_l = run_legacy.int_rates.coupling_mat
    cm_t = run_toep.int_rates.coupling_mat

    # Same shape
    assert cm_l.shape == cm_t.shape
    # Same sparsity pattern (CSR uses sorted (row, col), so indices/indptr match)
    np.testing.assert_array_equal(cm_l.indices, cm_t.indices)
    np.testing.assert_array_equal(cm_l.indptr, cm_t.indptr)


def test_grid_compatibility_assert_passes_on_default():
    """`_assert_log_grids_compatible` is silent when grids match."""
    from prince_cr.data import EnergyGrid

    class _Fake:
        pass

    obj = _Fake()
    obj.e_cosmicray = EnergyGrid(3, 14, 8)
    obj.e_photon = EnergyGrid(-15, -6, 8)
    # Should not raise
    interaction_rates.PhotoNuclearInteractionRate._assert_log_grids_compatible(obj)


def test_grid_compatibility_assert_raises_on_mismatch():
    """`_assert_log_grids_compatible` raises RuntimeError when bins/decade differ."""
    from prince_cr.data import EnergyGrid

    class _Fake:
        pass

    obj = _Fake()
    obj.e_cosmicray = EnergyGrid(3, 14, 8)
    obj.e_photon = EnergyGrid(-15, -6, 4)  # different bins/decade
    with pytest.raises(RuntimeError, match="bins-per-decade"):
        interaction_rates.PhotoNuclearInteractionRate._assert_log_grids_compatible(obj)


def test_init_matrices_raises_on_grid_mismatch(pf):
    """Top-level: building a PriNCeRun with mismatched grids and toeplitz selected
    must raise (no silent fallback)."""
    saved_cr = cfg.cosmic_ray_grid
    saved_ph = cfg.photon_grid
    saved_method = cfg.kernel_method
    saved_max_mass = cfg.max_mass
    try:
        cfg.cosmic_ray_grid = (3, 14, 8)
        cfg.photon_grid = (-15, -6, 4)  # different bins/decade
        cfg.kernel_method = "toeplitz"
        cfg.max_mass = 1
        from prince_cr import cross_sections as _cs

        local_cs = _cs.CompositeCrossSection(
            [
                (0.0, _cs.TabulatedCrossSection, ("CRP2_TALYS",)),
                (0.14, _cs.SophiaSuperposition, ()),
            ]
        )
        with pytest.raises(RuntimeError, match="bins-per-decade"):
            core.PriNCeRun(max_mass=1, photon_field=pf, cross_sections=local_cs)
    finally:
        cfg.cosmic_ray_grid = saved_cr
        cfg.photon_grid = saved_ph
        cfg.kernel_method = saved_method
        cfg.max_mass = saved_max_mass


def test_invalid_kernel_method_raises(pf, cs):
    """An unrecognised `kernel_method` value must raise ValueError."""
    saved = cfg.kernel_method
    try:
        cfg.kernel_method = "fancy"
        with pytest.raises(ValueError, match="kernel_method"):
            core.PriNCeRun(max_mass=1, photon_field=pf, cross_sections=cs)
    finally:
        cfg.kernel_method = saved
