"""FLUKA-derived photo-nuclear cross-section model.

Consumes ``photo_nuclear/FLUKA_2025/`` from the HDF5 database produced by
the sibling repo ``prince-fluka-utils``. Heavy daughters (A_d >= 2) are
boost-conserving and stored 2D (channel × E_γ); free nucleons and the
elementary species (γ, e±, ν, π, K, μ) are redistributed and stored 3D
(channel × E_γ × x).

This module replaces ``photo_meson.py`` (SOPHIA + EmpiricalModel) and
``disintegration.py`` (PEANUT_IAS / CRP2_TALYS / Composite). One class,
one HDF5 group.
"""

from __future__ import annotations

import numpy as np

from prince_cr.util import info, get_AZN, is_nucleus
import prince_cr.config as config

from .base import CrossSectionBase


# The FLUKA db (prince-fluka-utils v1+) stores nucleus mothers / daughters as
# PDG nuclear codes (``10LZZZAAAI``) and elementary daughters as standard PDG
# codes. The only normalisation needed is mapping the bound H-1 / n-1 codes
# (``1000010010`` / ``1000000010``) onto PriNCe's canonical free-nucleon
# codes (``2212`` / ``2112``); everything else passes through unchanged.
_BOUND_NUCLEON_TO_FREE = {
    1000010010: 2212,  # H-1 nucleus → free proton
    1000000010: 2112,  # n-1 nucleus → free neutron
}


def _normalize_pdg(pdg_id: int) -> int:
    """Collapse the bound-nucleon variants onto canonical free-nucleon codes."""
    pdg_id = int(pdg_id)
    return _BOUND_NUCLEON_TO_FREE.get(pdg_id, pdg_id)


# K^0_S (310) and K^0_L (130) have no spec_data entry on the PriNCe side,
# and the FLUKA db routes them through elementary_daughters. We drop them
# at load time with a warning so the mass numbers/charge bookkeeping stays
# consistent. Once we add neutral kaons to spec_data this set can shrink.
_PDG_DROPPED_ELEMENTARY = frozenset({130, 310})


# Tracks (mo, da) pairs already warned about so repeated loads in the same
# process don't re-spam. Module-level so it survives across class instances.
_WARNED: set = set()


def _warn_misclassified(mo: int, da: int) -> None:
    """Log once per (mo, da) that this row was dropped from the
    boost-conserving bucket as anomalous. Catches two v0 db artefacts:

    1. **Non-nuclear daughter** (K^±, Λ, free p/n routed to
       ``mothers_daughters`` instead of ``elementary_daughters``).
    2. **Daughter heavier than mother** (e.g. He-3 → He-4) — physically
       impossible from γ + (Z,A); a known v0 generator bug.

    Both belong as open questions against prince-fluka-utils;
    FlukaPhotoNuclear simply skips them.
    """
    key = (int(mo), int(da))
    if key in _WARNED:
        return
    _WARNED.add(key)
    info(
        0,
        "FlukaPhotoNuclear: dropping anomalous ({0}, {1}) — either "
        "daughter is not a nucleus or A_da > A_mo (v0 db artefact)".format(mo, da),
    )


class FlukaPhotoNuclear(CrossSectionBase):
    """γ + (Z,A) cross sections from FLUKA via prince-fluka-utils.

    Replaces the old SOPHIA + PEANUT_IAS / CRP2_TALYS pair as PriNCe's
    photo-nuclear model. One HDF5 group provides every channel: heavy
    daughters (A_d >= 2) go into ``_incl_tab`` (boost-conserving), light
    daughters (free nucleons + elementary species) go into
    ``_incl_diff_tab`` (3D, redistributed in x = E_secondary / E_γ).

    The HANDOVER_photo_meson.md placeholder-trick failure mode does not
    apply here: every declared channel has a real ndarray, no
    ``()`` / ``np.array([])`` sentinels.
    """

    def __init__(self, *args, **kwargs):
        # Tells the interpolator we have differential channels
        self.supports_redistributions = True
        config.max_mass = kwargs.pop("max_mass", config.max_mass)
        model_tag = kwargs.pop("model_tag", "FLUKA_2025")
        CrossSectionBase.__init__(self)
        self._load(model_tag)
        self._optimize_and_generate_index()

    def _load(self, model_tag):
        from prince_cr.data import db_handler

        info(2, "Loading FLUKA photo-nuclear cross sections.")
        tables = db_handler.fluka_photo_nuclear_db(
            model_tag, e_range=config.cross_section_e_range
        )

        self._egrid_tab = tables["energy_grid"]    # GeV
        self.xbins = tables["xbins"]               # log-spaced [1e-5, 3], 200 bins
        # FLUKA db stores elementary yields as σ_inel · P(x_bin) (count-per-bin
        # convention); the response builder expects dσ/dx. Divide by bin width
        # on load so _incl_diff_tab matches SOPHIA's per-density convention.
        xwidths = self.xbins[1:] - self.xbins[:-1]

        # σ_inel: (n_m, n_E). Mothers are PDG nucleus codes; bound H-1 / n-1
        # codes are folded onto canonical free p / n.
        for raw_mo, sig in zip(
            tables["inel_mothers"], tables["inelastic_cross_sctions"]
        ):
            mo = _normalize_pdg(raw_mo)
            A, _, _ = get_AZN(mo)
            if A > config.max_mass:
                continue
            self._nonel_tab[mo] = sig              # cm^2

        # Boost-conserving: (mother_PDG, daughter_PDG); 2D yields (n_ch, n_E)
        for (raw_mo, raw_da), yld in zip(
            tables["mothers_daughters"], tables["fragment_yields"]
        ):
            mo = _normalize_pdg(raw_mo)
            da = _normalize_pdg(raw_da)
            # Generator artefacts: K^±, Λ, and free p/n sometimes land here.
            # Drop them; they belong in elementary_daughters. Also drop any
            # row where daughter mass > mother mass (unphysical; cross-isobar
            # mis-routing is a known v0/v1 db artefact).
            A_mo, _, _ = get_AZN(mo)
            A_da, _, _ = get_AZN(da)
            if not is_nucleus(da) or A_da < 2 or A_da > A_mo:
                _warn_misclassified(int(raw_mo), int(raw_da))
                continue
            if A_mo > config.max_mass:
                continue
            self._incl_tab[mo, da] = yld

        # Redistributed: (mother_PDG, daughter_PDG); 3D yields.
        # FLUKA stores (n_E, n_x) per-bin counts; transpose to PriNCe (n_x, n_E)
        # and divide by xwidths so the stored value is dσ/dx (cm^2 per unit x).
        for (raw_mo, da_pdg), yld_3d in zip(
            tables["elementary_daughters"], tables["elementary_yields"]
        ):
            da_pdg = int(da_pdg)
            if da_pdg in _PDG_DROPPED_ELEMENTARY:   # K^0_S / K^0_L
                continue
            mo = _normalize_pdg(raw_mo)
            A_mo, _, _ = get_AZN(mo)
            if A_mo > config.max_mass:
                continue
            self._incl_diff_tab[mo, da_pdg] = yld_3d.T / xwidths[:, None]

        # Initial range = full egrid
        self.set_range()
        info(
            2,
            "FlukaPhotoNuclear loaded: {0} mothers, {1} bc channels, "
            "{2} diff channels".format(
                len(self._nonel_tab), len(self._incl_tab), len(self._incl_diff_tab)
            ),
        )
