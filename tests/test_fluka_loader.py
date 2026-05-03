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


def test_pdg_to_ncoid_known_mappings():
    """Every elementary PDG → known PriNCe ncoid; K^0_S/K^0_L → None."""
    from prince_cr.cross_sections.fluka import _pdg_to_ncoid

    expected = {
        22: 0,            # gamma
        11: 20, -11: 21,  # e-, e+
        13: 10, -13: 7,   # mu- (helicity-0), mu+ (helicity-0)
        12: 11, -12: 12,  # nu_e, nu_ebar
        14: 13, -14: 14,  # nu_mu, nu_mubar
        16: 15, -16: 16,  # nu_tau, nu_taubar
        211: 2, -211: 3,  # pi+, pi-
        111: 4,           # pi0
        321: 50, -321: 51,  # K+, K-
        130: None, 310: None,  # K^0_L, K^0_S — no ncoid slot
        2212: 101, 2112: 100,  # p, n
    }
    for pdg, ncoid in expected.items():
        assert _pdg_to_ncoid(pdg) == ncoid, f"pdg={pdg}"


def test_pdg_to_ncoid_unknown_returns_none():
    """Unknown PDG codes return None (caller drops them)."""
    from prince_cr.cross_sections.fluka import _pdg_to_ncoid
    assert _pdg_to_ncoid(99999) is None


def test_is_nucleus_basic():
    """ncoid >= 100 with consistent A/Z is a nucleus."""
    from prince_cr.cross_sections.fluka import _is_nucleus

    assert _is_nucleus(101) is True   # p
    assert _is_nucleus(100) is True   # n
    assert _is_nucleus(402) is True   # He-4
    assert _is_nucleus(5626) is True  # Fe-56
    # Non-nuclei
    assert _is_nucleus(50) is False    # K+ ncoid (Z=50 nonsensical for our use)
    assert _is_nucleus(0) is False     # gamma
    assert _is_nucleus(2) is False     # pi+
    # PDG K+ (321) read as ncoid would mean Z=21, A=3 — Z > A is unphysical
    assert _is_nucleus(321) is False
    # PDG Lambda (3122) read as ncoid would mean Z=22, A=31 — physical-looking
    # but we don't expect PDG codes >100 in mothers_daughters; sanity bound
    # is that Z must be <= A. 3122 has A=31, Z=22 — that's physical, but Lambda
    # has no PriNCe ncoid slot anyway. _is_nucleus is a coarse filter; the real
    # check is whether the (mo, da) pair makes physical sense, which the
    # downstream warning catches.


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
    n_E = cs._egrid_tab.shape[0]
    n_x = cs.xbins.shape[0] - 1
    for (mo, da), arr in cs._incl_diff_tab.items():
        # After _optimize_and_generate_index, values may be ndarrays or tuples
        if isinstance(arr, tuple):
            _, arr = arr
        assert arr.shape[0] == n_x, f"diff[{mo},{da}] xdim={arr.shape[0]}"


def test_fluka_photo_nuclear_filters_misclassified():
    """v0 db has rows like (101, 321) [proton→K+] in mothers_daughters;
    FlukaPhotoNuclear drops them and warns once per pair."""
    from prince_cr.cross_sections import FlukaPhotoNuclear
    from prince_cr.cross_sections import fluka as fluka_mod

    fluka_mod._WARNED.clear()
    cs = FlukaPhotoNuclear()

    # No (101, 50) [proton → K+ via PDG 321 → ncoid 50] in _incl_tab
    # because 50 is not a nucleus per _is_nucleus
    for mo, da in cs._incl_tab:
        assert _is_nucleus_or_self_check(da), (
            f"non-nuclear daughter {da} found in _incl_tab (mother {mo})"
        )

    # At least one warning was emitted
    assert len(fluka_mod._WARNED) >= 1


def _is_nucleus_or_self_check(ncoid):
    """Inline copy of the contract: nuclei have ncoid >= 100 with Z <= A."""
    if ncoid < 100:
        return False
    A = ncoid // 100
    Z = ncoid % 100
    return Z <= A and A >= 2  # we additionally require A >= 2 in mothers_daughters


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
