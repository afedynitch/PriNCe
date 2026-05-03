"""Unit tests for the FLUKA photo-nuclear loader and FlukaPhotoNuclear class.

The v0 db lives at /Users/anatoli/devel_mac/prince-fluka-utils/prince_db_v0.h5;
conftest.py points config.fluka_db_path at the prince-fluka-utils repo root.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_fluka_photo_nuclear_db_returns_expected_shape():
    """db_handler.fluka_photo_nuclear_db('FLUKA_2025') returns the expected dict."""
    from prince_cr.data import db_handler

    tables = db_handler.fluka_photo_nuclear_db("FLUKA_2025")

    # Required keys
    for key in (
        "energy_grid",
        "xbins",
        "inel_mothers",
        "mothers_daughters",
        "elementary_daughters",
        "inelastic_cross_sctions",
        "fragment_yields",
        "elementary_yields",
    ):
        assert key in tables, f"missing key {key}"

    # Shapes per spec (v0 db: 150 E pts, 200 x bins, 78 mothers)
    n_E = tables["energy_grid"].shape[0]
    n_x = tables["xbins"].shape[0] - 1
    assert tables["xbins"].shape[0] == n_x + 1
    assert tables["inelastic_cross_sctions"].shape[1] == n_E
    assert tables["fragment_yields"].shape[1] == n_E
    assert tables["elementary_yields"].shape[1] == n_E
    assert tables["elementary_yields"].shape[2] == n_x


def test_fluka_photo_nuclear_db_e_range_slices():
    """e_range arg slices the energy axis on every cross-section array."""
    from prince_cr.data import db_handler

    full = db_handler.fluka_photo_nuclear_db("FLUKA_2025")
    e0 = full["energy_grid"][0]
    e1 = full["energy_grid"][50]

    sliced = db_handler.fluka_photo_nuclear_db("FLUKA_2025", e_range=(e0, e1))

    # Sliced energy grid is shorter
    assert sliced["energy_grid"].shape[0] < full["energy_grid"].shape[0]
    n_E_sliced = sliced["energy_grid"].shape[0]
    assert sliced["inelastic_cross_sctions"].shape[1] == n_E_sliced
    assert sliced["fragment_yields"].shape[1] == n_E_sliced
    assert sliced["elementary_yields"].shape[1] == n_E_sliced


def test_fluka_photo_nuclear_db_missing_file_raises():
    """A missing FLUKA db raises a FileNotFoundError with a helpful message."""
    import prince_cr.config as cfg
    from prince_cr.data import db_handler

    saved_path = cfg.fluka_db_path
    saved_fname = cfg.fluka_db_fname
    try:
        cfg.fluka_db_path = "/nonexistent/path"
        cfg.fluka_db_fname = "missing.h5"
        with pytest.raises(FileNotFoundError, match="FLUKA db not found"):
            db_handler.fluka_photo_nuclear_db("FLUKA_2025")
    finally:
        cfg.fluka_db_path = saved_path
        cfg.fluka_db_fname = saved_fname
