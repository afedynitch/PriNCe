"""Toeplitz kernel construction: grid-compatibility guard.

The legacy dense-tensor path was deleted; the toeplitz path is the only
kernel construction now. The CR-grid centers, CR-grid bin edges, and
photon-grid bin edges must all share the same log-step, otherwise the
integer-index lookups would map to the wrong corners and silently
produce wrong rates. `_assert_log_grids_compatible` enforces this.
"""

from __future__ import annotations

import pytest

import prince_cr.config as cfg
from prince_cr import core, interaction_rates


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
    """Top-level: building a PriNCeRun with mismatched grids must raise."""
    saved_cr = cfg.cosmic_ray_grid
    saved_ph = cfg.photon_grid
    saved_max_mass = cfg.max_mass
    try:
        cfg.cosmic_ray_grid = (3, 14, 8)
        cfg.photon_grid = (-15, -6, 4)  # different bins/decade
        cfg.max_mass = 1
        from prince_cr import cross_sections as _cs

        local_cs = _cs.FlukaPhotoNuclear()
        with pytest.raises(RuntimeError, match="bins-per-decade"):
            core.PriNCeRun(max_mass=1, photon_field=pf, cross_sections=local_cs)
    finally:
        cfg.cosmic_ray_grid = saved_cr
        cfg.photon_grid = saved_ph
        cfg.max_mass = saved_max_mass
