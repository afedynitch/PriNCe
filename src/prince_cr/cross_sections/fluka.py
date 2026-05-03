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

from prince_cr.util import info, get_AZN
import prince_cr.config as config

from .base import CrossSectionBase


# -- PDG → PriNCe ncoid mapping (TEMPORARY SHIM) -----------------------------
#
# The FLUKA db's elementary_daughters column stores PDG codes. PriNCe's
# internal indexing uses Neucosma ncoids. This table bridges the two.
#
# **Remove this entire block in the next session when PriNCe migrates to
# PDG natively across data.py / base.py / spec_data.** The mapping is
# load-only — once the rest of PriNCe speaks PDG, FlukaPhotoNuclear can
# pass elementary_daughters through unchanged.
#
# Muon helicity: chromo's Pythia post-decay produces helicity-mixed muons,
# so PDG ±13 maps to PriNCe's helicity-0 slots (7 for μ+, 10 for μ-).
# Helicity-resolved analytical decays remain available through decays.py
# for any test that needs them.
#
# K^0_S (310) and K^0_L (130) have no ncoid slot in particle_data.ppo;
# return None and let the caller drop these channels with a warning.
_PDG_TO_NCOID = {
    22:    0,    # gamma
    11:    20,   # e-
    -11:   21,   # e+
    13:    10,   # mu- (helicity 0)
    -13:   7,    # mu+ (helicity 0)
    12:    11,   # nu_e
    -12:   12,   # nu_ebar
    14:    13,   # nu_mu
    -14:   14,   # nu_mubar
    16:    15,   # nu_tau
    -16:   16,   # nu_taubar
    211:   2,    # pi+
    -211:  3,    # pi-
    111:   4,    # pi0
    321:   50,   # K+
    -321:  51,   # K-
    130:   None, # K^0_L — no PriNCe slot
    310:   None, # K^0_S — no PriNCe slot
    2212:  101,  # p
    2112:  100,  # n
}


def _pdg_to_ncoid(pdg: int):
    """Convert PDG code → PriNCe ncoid. Returns None for unmapped codes
    (K^0_S, K^0_L, plus anything outside the elementary-species set).
    """
    return _PDG_TO_NCOID.get(int(pdg))


def _is_nucleus(ncoid: int) -> bool:
    """True iff ``ncoid`` is plausibly a nucleus in PriNCe's ncoid scheme.

    PriNCe encodes nuclei as ``100 * A + Z`` with ``Z <= A``. Anything
    below 100, or with ``Z > A``, is not a nucleus.
    """
    if ncoid < 100:
        return False
    A = ncoid // 100
    Z = ncoid % 100
    return Z <= A


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

        # σ_inel: (n_m, n_E)
        for ncoid, sig in zip(
            tables["inel_mothers"], tables["inelastic_cross_sctions"]
        ):
            ncoid = int(ncoid)
            A, _, _ = get_AZN(ncoid)
            if A > config.max_mass:
                continue
            self._nonel_tab[ncoid] = sig           # cm^2

        # Boost-conserving: (mother_ncoid, daughter_ncoid), 2D yields (n_ch, n_E)
        for (mo, da), yld in zip(
            tables["mothers_daughters"], tables["fragment_yields"]
        ):
            mo, da = int(mo), int(da)
            # v0 generator inconsistency: K^±, Λ, and free p/n sometimes land
            # here. Drop them; they belong in elementary_daughters. Also drop
            # any row where daughter mass > mother mass (unphysical; a known
            # v0 db artefact from misrouted PDG codes and cross-isobar entries).
            A_mo, _, _ = get_AZN(mo)
            A_da, _, _ = get_AZN(da)
            if not _is_nucleus(da) or A_da < 2 or A_da > A_mo:
                _warn_misclassified(mo, da)
                continue
            if A_mo > config.max_mass:
                continue
            self._incl_tab[mo, da] = yld

        # Redistributed: (mother_ncoid, daughter_pdg→ncoid), 3D yields
        # FLUKA stores (n_E, n_x); transpose to PriNCe convention (n_x, n_E).
        for (mo, da_pdg), yld_3d in zip(
            tables["elementary_daughters"], tables["elementary_yields"]
        ):
            mo, da_pdg = int(mo), int(da_pdg)
            da_ncoid = _pdg_to_ncoid(da_pdg)
            if da_ncoid is None:                   # K^0_S / K^0_L drop
                continue
            A_mo, _, _ = get_AZN(mo)
            if A_mo > config.max_mass:
                continue
            self._incl_diff_tab[mo, da_ncoid] = yld_3d.T

        # Initial range = full egrid
        self.set_range()
        info(
            2,
            "FlukaPhotoNuclear loaded: {0} mothers, {1} bc channels, "
            "{2} diff channels".format(
                len(self._nonel_tab), len(self._incl_tab), len(self._incl_diff_tab)
            ),
        )
