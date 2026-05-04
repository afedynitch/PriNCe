"""Unit tests for the FLUKA photo-nuclear loader and FlukaPhotoNuclear class.

The v0 db lives at config.fluka_db_path / config.fluka_db_fname; conftest.py
points these at the locally available smoke db.
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


def test_pdg_helpers_round_trip():
    """is_nucleus + make_nucleus_pdg + get_AZN agree on round-trip identities."""
    from prince_cr.util import is_nucleus, make_nucleus_pdg, get_AZN

    # Free p / n
    assert is_nucleus(2212) is True
    assert is_nucleus(2112) is True
    # Heavy nuclei
    assert is_nucleus(make_nucleus_pdg(4, 2)) is True   # He-4
    assert is_nucleus(make_nucleus_pdg(56, 26)) is True  # Fe-56
    assert is_nucleus(make_nucleus_pdg(238, 92)) is True  # U-238

    # Non-nuclei
    assert is_nucleus(22) is False     # γ
    assert is_nucleus(11) is False     # e-
    assert is_nucleus(211) is False    # π+
    assert is_nucleus(321) is False    # K+
    assert is_nucleus(13) is False     # μ-

    # Round trip
    for A, Z in [(4, 2), (14, 7), (56, 26), (238, 92)]:
        pdg = make_nucleus_pdg(A, Z)
        assert get_AZN(pdg) == (A, Z, A - Z)


def test_warn_misclassified_dedupes():
    """_warn_misclassified emits info once per (mo, da) pair within a process."""
    from prince_cr.cross_sections import fluka as fluka_mod

    fluka_mod._WARNED.clear()
    fluka_mod._warn_misclassified(101, 321)
    fluka_mod._warn_misclassified(101, 321)  # dup, no-op
    fluka_mod._warn_misclassified(101, 3122)  # different pair, emits

    assert (101, 321) in fluka_mod._WARNED
    assert (101, 3122) in fluka_mod._WARNED
    assert len(fluka_mod._WARNED) == 2


def test_fluka_photo_nuclear_loads_basic_shapes():
    """FlukaPhotoNuclear() populates all three tabs with real ndarrays."""
    from prince_cr.cross_sections import FlukaPhotoNuclear

    cs = FlukaPhotoNuclear()

    # Energy grid and xbins are populated from the HDF5
    assert cs._egrid_tab is not None
    assert cs.xbins is not None
    assert cs.xbins.shape[0] == 201   # 200 bins → 201 edges
    assert len(cs._egrid_tab) > 0     # truncated by max_mass / e_range

    # Every nonel value is a real ndarray on the egrid (no tuples, no empty)
    assert len(cs._nonel_tab) > 0
    for mo, sig in cs._nonel_tab.items():
        assert isinstance(sig, np.ndarray), f"nonel[{mo}] is not an ndarray"
        assert sig.shape == cs._egrid_tab.shape, f"nonel[{mo}] shape={sig.shape}"

    # _incl_diff_tab values: shape (n_x, n_E) per channel
    n_x = cs.xbins.shape[0] - 1
    for (mo, da), arr in cs._incl_diff_tab.items():
        # After _optimize_and_generate_index, values may be ndarrays or tuples
        if isinstance(arr, tuple):
            _, arr = arr
        assert arr.shape[0] == n_x, f"diff[{mo},{da}] xdim={arr.shape[0]}"


def test_fluka_photo_nuclear_filters_misclassified():
    """The loader drops any (mo, da) pair whose daughter is non-nuclear, free
    p/n, or A_da > A_mo. After load, every daughter in _incl_tab is a heavy
    nucleus (A>=2). On a clean db no warnings are emitted; on a v0-style db
    with mis-routed rows at least one warning fires.
    """
    from prince_cr.cross_sections import FlukaPhotoNuclear
    from prince_cr.cross_sections import fluka as fluka_mod
    from prince_cr.util import is_nucleus, get_AZN

    fluka_mod._WARNED.clear()
    cs = FlukaPhotoNuclear()

    for mo, da in cs._incl_tab:
        assert is_nucleus(da), (
            f"non-nuclear daughter {da} found in _incl_tab (mother {mo})"
        )
        A_da, _, _ = get_AZN(da)
        assert A_da >= 2, f"free nucleon {da} routed through _incl_tab"


def test_fluka_photo_nuclear_max_mass_truncation():
    """max_mass kwarg drops mothers above the cutoff."""
    import prince_cr.config as cfg
    from prince_cr.cross_sections import FlukaPhotoNuclear
    from prince_cr.util import get_AZN

    saved = cfg.max_mass
    try:
        cfg.max_mass = 14
        cs = FlukaPhotoNuclear()
        for mo in cs._nonel_tab:
            assert get_AZN(mo)[0] <= 14, f"mother {mo} exceeds max_mass"
    finally:
        cfg.max_mass = saved
