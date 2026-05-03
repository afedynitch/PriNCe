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
    """Log once per (mo, da) that the daughter is not a nucleus and so was
    skipped from the boost-conserving bucket.

    The v0 prince-fluka-utils generator routes some non-nuclear daughters
    (K^±, Λ, free p/n) into ``mothers_daughters`` instead of
    ``elementary_daughters``. Filing this as an open question on the
    generator side; FlukaPhotoNuclear simply skips them.
    """
    key = (int(mo), int(da))
    if key in _WARNED:
        return
    _WARNED.add(key)
    info(
        0,
        "FlukaPhotoNuclear: dropping misclassified ({0}, {1}) — daughter "
        "is not a nucleus; expected in elementary_daughters bucket".format(mo, da),
    )
