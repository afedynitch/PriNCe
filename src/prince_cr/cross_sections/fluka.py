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
        self.max_mass = kwargs.pop("max_mass", config.max_mass)
        # Explicit db location overrides the module globals — pass when
        # running multiple builds from the same process or in tests.
        # ``None`` falls back to ``config.fluka_db_path`` /
        # ``config.fluka_db_fname`` at load time.
        self.db_path = kwargs.pop("db_path", None)
        self.db_fname = kwargs.pop("db_fname", None)
        model_tag = kwargs.pop("model_tag", "FLUKA_2025")
        CrossSectionBase.__init__(self)
        self._load(model_tag)
        self._optimize_and_generate_index()

    def _load(self, model_tag):
        from prince_cr.data import db_handler

        info(2, "Loading FLUKA photo-nuclear cross sections.")
        tables = db_handler.fluka_photo_nuclear_db(
            model_tag,
            e_range=config.cross_section_e_range,
            max_mass=self.max_mass,
            db_path=self.db_path,
            db_fname=self.db_fname,
        )

        self._egrid_tab = tables["energy_grid"]    # GeV
        self.xbins = tables["xbins"]               # log-spaced [1e-5, 3], 200 bins
        # Schema version: v4+ stores elementary_yields in per-nucleon x
        # convention (rescale applied at write time in prince-fluka-utils'
        # schema.py::_rescale_ey_to_per_nucleon). v3 and earlier need the
        # load-time per-mother log-bin shift below. Default v3 if the attr
        # is absent (older dbs).
        self._schema_version = int(tables.get("schema_version", 3))
        # FLUKA db stores elementary yields as σ_inel · P(x_bin) (count-per-bin
        # convention); the response builder expects dσ/dx. Divide by bin width
        # on load so _incl_diff_tab matches SOPHIA's per-density convention.
        xwidths = self.xbins[1:] - self.xbins[:-1]
        # FLUKA stores x = E_d_lab / (E_γ + m_target), so boost-conserved free
        # nucleons from a heavy mother peak at x_FLUKA = m_p / m_target ≈ 1/A_mo.
        # PriNCe's kernel runs on per-nucleon energies (see cr_sources.py:
        # AugerFitSource.injection_spectrum uses e_k = A * energy), where boost
        # conservation maps the same channel onto x_kernel = A_mo · x_FLUKA = 1.
        # We rescale per mother on load: log10(x_kernel) = log10(A_mo) +
        # log10(x_FLUKA), with conservation dσ/dx_kernel = (dσ/dx_FLUKA)/A_mo.
        # The rescaling is a fractional shift in log-bin index (same log-step
        # in source and destination grids). Bins falling above the upper xbin
        # edge after rescaling (rare back-emission tails, x_kernel > 3) drop.
        # Folded only into elementary daughters; boost-conserving heavy-daughter
        # channels stay on i_mo == i_da and don't query the x grid.
        log_step_x = np.log10(self.xbins[1] / self.xbins[0])
        n_x = len(xwidths)

        # σ_inel: (n_m, n_E). Mothers are PDG nucleus codes; bound H-1 / n-1
        # codes are folded onto canonical free p / n.
        for raw_mo, sig in zip(
            tables["inel_mothers"], tables["inelastic_cross_sctions"]
        ):
            mo = _normalize_pdg(raw_mo)
            A, _, _ = get_AZN(mo)
            if A > self.max_mass:
                continue
            self._nonel_tab[mo] = sig              # cm^2

        # Boost-conserving: (mother_PDG, daughter_PDG); 2D yields (n_ch, n_E).
        #
        # Daughter-only species — a fragment daughter that has no σ_inel of its
        # own (e.g. a tier-2/3 isotope absent from `inel_mothers` under a
        # lifetime-tier or max_mass cut) — are KEPT here on purpose. The chain
        # reducer in `_reduce_channels` (`_DecayChainReducer.follow`) recurses on
        # each unstable daughter, convolving its decay distribution (branching
        # ratios, cycle/depth-capped) until the flux lands on a stable/tracked
        # species; daughters that are genuinely undecayable (absent from
        # `spec_data`) are dropped there with a single aggregated warning. In
        # explicit-decay mode the reducer is skipped and daughter-only nuclei are
        # made matrix-safe by `_optimize_and_generate_index` (reactions[sp]=[])
        # and the daughter-only branch in `_estimate_batch_matrix`.
        #
        # (Earlier this loop dropped ALL daughter-only rows as a stop-gap to dodge
        # a `KeyError` in `_estimate_batch_matrix`. That KeyError is now handled
        # downstream, and the blanket drop silently discarded reducible
        # fragmentation flux — 1747 daughters / ~3e5 rows under the v6 tier-1
        # load, every one of which has a decay distribution in the db.)
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
            if A_mo > self.max_mass:
                continue
            self._incl_tab[mo, da] = yld

        # Redistributed: (mother_PDG, daughter_PDG); 3D yields.
        # FLUKA stores (n_E, n_x) per-bin counts; transpose to PriNCe (n_x, n_E),
        # divide by xwidths to get dσ/dx_FLUKA, then rescale per-mother by A_mo
        # to land on PriNCe's per-nucleon x_kernel = A_mo · x_FLUKA convention.
        for (raw_mo, da_pdg), yld_3d in zip(
            tables["elementary_daughters"], tables["elementary_yields"]
        ):
            da_pdg = int(da_pdg)
            if da_pdg in _PDG_DROPPED_ELEMENTARY:   # K^0_S / K^0_L
                continue
            mo = _normalize_pdg(raw_mo)
            A_mo, _, _ = get_AZN(mo)
            if A_mo > self.max_mass:
                continue
            dsig_dx = yld_3d.T / xwidths[:, None]
            if A_mo > 1 and self._schema_version < 4:
                # v3 and earlier dbs store x in FLUKA convention. Apply the
                # per-mother log-bin shift + A_mo Jacobian here at load time.
                # v4+ has this baked in at write time (see
                # prince-fluka-utils/schema.py::_rescale_ey_to_per_nucleon)
                # so we skip the rescale entirely.
                shift = np.log10(float(A_mo)) / log_step_x
                src_idx = np.arange(n_x) - shift
                i_lo = np.floor(src_idx).astype(int)
                frac = (src_idx - i_lo)[:, None]
                i_hi = i_lo + 1
                valid_lo = (i_lo >= 0) & (i_lo < n_x)
                valid_hi = (i_hi >= 0) & (i_hi < n_x)
                lo_safe = np.clip(i_lo, 0, n_x - 1)
                hi_safe = np.clip(i_hi, 0, n_x - 1)
                rescaled = (
                    np.where(valid_lo[:, None], dsig_dx[lo_safe, :], 0.0)
                    * (1.0 - frac)
                    + np.where(valid_hi[:, None], dsig_dx[hi_safe, :], 0.0)
                    * frac
                ) / A_mo
                self._incl_diff_tab[mo, da_pdg] = rescaled
            else:
                # v4+ or A_mo == 1: data is already in per-nucleon convention.
                self._incl_diff_tab[mo, da_pdg] = dsig_dx

        # Initial range = full egrid
        self.set_range()
        info(
            2,
            "FlukaPhotoNuclear loaded: {0} mothers, {1} bc channels, "
            "{2} diff channels".format(
                len(self._nonel_tab), len(self._incl_tab), len(self._incl_diff_tab)
            ),
        )
