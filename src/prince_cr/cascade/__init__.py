"""Electromagnetic cascade physics for gamma rays in the intergalactic medium.

Phase 5 of the source-physics roadmap: gamma-ray absorption (and, later, the
full electromagnetic cascade) on the CMB + EBL. See
``wiki/methods/em-cascade`` in the bookkeeping repo.

Currently implemented:
  - :mod:`prince_cr.cascade.opacity` — gamma-gamma pair-production optical
    depth ``tau(E, z)`` and the ``exp(-tau)`` attenuation.
"""

from prince_cr.cascade.opacity import attenuation, sigma_gg, tau_gg

__all__ = ["sigma_gg", "tau_gg", "attenuation"]
