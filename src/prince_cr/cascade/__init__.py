"""Electromagnetic cascade physics for gamma rays in the intergalactic medium.

Phase 5 of the source-physics roadmap: gamma-ray absorption (and, later, the
full electromagnetic cascade) on the CMB + EBL. See
``wiki/methods/em-cascade`` in the bookkeeping repo.

Implemented:
  - :mod:`prince_cr.cascade.opacity` — gamma-gamma pair-production optical
    depth ``tau(E, z)`` and the ``exp(-tau)`` attenuation (Phase A).
  - :mod:`prince_cr.cascade.kernels` — IC (Blumenthal-Gould) + gamma-gamma
    pair-injection microphysics kernels (Phase B).
  - :mod:`prince_cr.cascade.cascade` — saturated-generation 1D cascade
    producing the reprocessed (EGB) photon spectrum (Phase B).
"""

from prince_cr.cascade.opacity import attenuation, sigma_gg, tau_gg
from prince_cr.cascade.cascade import run_cascade, absorption_energy

__all__ = [
    "sigma_gg",
    "tau_gg",
    "attenuation",
    "run_cascade",
    "absorption_energy",
]
