"""prince_cr.source — multi-messenger source physics for PriNCe.

This subpackage hosts the in-source physics that complements PriNCe's
propagation solver: fluid-frame photon fields, single-zone cooling /
acceleration rates, and (later) a time-domain evolution solver that
produces escape spectra and source-frame neutrino / photon yields. The
propagation half of PriNCe consumes the escape spectrum as injection.

Phase 1 (current) covers only the cooling-rate side, enough to reproduce
Fig. 2 of Guo, Qian, Wu 2025 (Phys. Rev. D 112, 063022) for a single
nucleus species in a given jet zone. See the module docstrings in
`photonfields.py` and `rates.py` for the conventions and the formulas.
"""

from prince_cr.source.photonfields import SourceBrokenPowerLaw
from prince_cr.source.rates import (
    photonuclear_rate_inv,
    photonuclear_cool_inv,
    bethe_heitler_cool_inv,
    synchrotron_cool_inv,
    adiabatic_source_inv,
    hadronic_Ap_inv,
    acceleration_inv,
    secondary_yield_per_x_inv,
    secondary_yield_dN_dE_inv,
    jet_magnetic_field,
    jet_proton_density,
)

__all__ = [
    "SourceBrokenPowerLaw",
    "photonuclear_rate_inv",
    "photonuclear_cool_inv",
    "bethe_heitler_cool_inv",
    "synchrotron_cool_inv",
    "adiabatic_source_inv",
    "hadronic_Ap_inv",
    "acceleration_inv",
    "secondary_yield_per_x_inv",
    "secondary_yield_dN_dE_inv",
    "jet_magnetic_field",
    "jet_proton_density",
]
