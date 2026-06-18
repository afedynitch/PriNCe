"""Pair-production (gamma-gamma) optical depth for gamma rays propagating
through the intergalactic radiation fields (CMB + EBL).

This is the *absorption* half of the electromagnetic-cascade physics
(Phase 5 of the source-physics roadmap, ``wiki/methods/source-physics-in-prince``):
a gamma ray of observed energy ``E0`` emitted at redshift ``z_s`` is
attenuated by ``exp(-tau)`` through

.. math::

    \\tau(E_0, z_s) = \\int_0^{z_s}\\!\\! \\frac{c\\,dz}{(1+z)H(z)}
        \\int_0^\\infty\\!\\! d\\epsilon\\, n(\\epsilon, z)
        \\int_{-1}^{1}\\!\\! \\frac{d\\mu}{2}\\,(1-\\mu)\\,
        \\sigma_{\\gamma\\gamma}(s),

with the photon energy at redshift ``z`` being ``E_gamma(z) = E0 (1+z)``,
the target photon proper energy ``eps``, the collision-angle cosine ``mu``,
and the squared CM energy ``s = 2 E_gamma(z) eps (1 - mu)``. The factor
``(1-mu)`` is the relative-velocity / flux factor (the speed of light cancels
against ``dl = c\\,dt``); ``dmu/2`` is the isotropic angular average.

Units follow the PriNCe convention (cm, s, GeV). ``n(eps, z)`` is the proper
number density from :mod:`prince_cr.photonfields` in ``GeV^-1 cm^-3``.

References
----------
Gould & Schréder, Phys. Rev. 155, 1404 (1967) — Breit-Wheeler cross section.
Standard EBL-absorption formulation, e.g. Dwek & Krennrich (2013).
"""

import numpy as np
from scipy.integrate import trapezoid as trapz

from prince_cr.cosmology import H
from prince_cr.data import PRINCE_UNITS

#: Thomson cross section [cm^2], from the classical electron radius.
SIGMA_THOMSON = 8.0 / 3.0 * np.pi * PRINCE_UNITS.r_electron**2

#: Electron rest energy [GeV].
M_E = PRINCE_UNITS.m_electron


def sigma_gg(s):
    """Breit-Wheeler pair-production cross section :math:`\\gamma\\gamma\\to e^+e^-`.

    Args:
        s (float or ndarray): squared center-of-mass energy [GeV^2].

    Returns:
        ndarray: cross section [cm^2]; zero below threshold ``s < 4 m_e^2``.
    """
    s = np.atleast_1d(np.asarray(s, dtype="double"))
    out = np.zeros_like(s)
    thr = s > 4.0 * M_E**2
    if np.any(thr):
        # w = 1 - beta^2 = 4 m_e^2 / s. At s >> 4 m_e^2, beta -> 1 and the naive
        # (1+beta)/(1-beta) divides by an underflowed (1-beta)=0 -> inf/NaN. Use
        # the stable identity 1-beta = w/(1+beta) ⇒ (1+beta)/(1-beta) =
        # (1+beta)^2 / w. The cross section then -> 0 as w -> 0 (high-s tail),
        # as it should, with no division by zero.
        w = 4.0 * M_E**2 / s[thr]
        beta = np.sqrt(1.0 - w)
        b2 = 1.0 - w
        out[thr] = (
            3.0
            / 16.0
            * SIGMA_THOMSON
            * w
            * (
                (3.0 - b2 * b2) * np.log((1.0 + beta) ** 2 / w)
                - 2.0 * beta * (2.0 - b2)
            )
        )
    return out


def _kernel_per_length(E_gamma, z, photon_field, eps, mu):
    """Differential opacity per unit proper length ``dtau/dl`` [cm^-1].

    Integrates the angular and target-energy parts at a single redshift.

    Args:
        E_gamma (float): gamma-ray energy *at redshift z* [GeV].
        z (float): redshift (for the photon-field evaluation).
        photon_field: object with ``get_photon_density(eps, z)`` [GeV^-1 cm^-3].
        eps (ndarray): target photon energy grid [GeV] (log-spaced).
        mu (ndarray): collision-angle cosine grid in [-1, 1].

    Returns:
        float: ``dtau/dl`` [cm^-1].
    """
    n_eps = np.asarray(photon_field.get_photon_density(eps, z), dtype="double")
    n_eps = np.where(np.isfinite(n_eps) & (n_eps > 0.0), n_eps, 0.0)

    # s[i, j] = 2 E_gamma eps_i (1 - mu_j)
    one_minus_mu = 1.0 - mu
    s = 2.0 * E_gamma * np.outer(eps, one_minus_mu)
    sig = sigma_gg(s.ravel()).reshape(s.shape)

    # angular integral: (1/2) int_{-1}^{1} (1-mu) sigma dmu
    ang = 0.5 * trapz(one_minus_mu[None, :] * sig, mu, axis=1)  # per eps [cm^2]
    # target-energy integral: int n(eps) * ang deps
    return trapz(n_eps * ang, eps)  # [cm^-1]


def tau_gg(
    E0,
    z_s,
    photon_field,
    n_z=64,
    n_eps=256,
    n_mu=128,
    eps_min=1e-12,
    eps_max=1e-7,
):
    """Pair-production optical depth for a gamma ray observed at ``E0``.

    Args:
        E0 (float or ndarray): observed gamma-ray energy at z=0 [GeV].
        z_s (float): source / emission redshift.
        photon_field: target field with ``get_photon_density(eps, z)``
            (e.g. :class:`prince_cr.photonfields.CMBPhotonSpectrum`,
            ``CIBDominguez2D``, or a ``CombinedPhotonField``).
        n_z, n_eps, n_mu (int): integration grid sizes (redshift, target
            energy, angle).
        eps_min, eps_max (float): target-photon energy integration band [GeV].
            Default ~1 micro-eV .. 100 eV covers CMB through UV/EBL.

    Returns:
        ndarray: optical depth tau(E0), same shape as ``E0``.
    """
    E0 = np.atleast_1d(np.asarray(E0, dtype="double"))
    if z_s <= 0.0:
        return np.zeros_like(E0)

    z_grid = np.linspace(0.0, z_s, n_z)
    eps = np.logspace(np.log10(eps_min), np.log10(eps_max), n_eps)
    mu = np.linspace(-1.0, 1.0, n_mu)

    c_cm = PRINCE_UNITS.c  # cm/s

    tau = np.zeros_like(E0)
    for k, e0 in enumerate(E0):
        integrand_z = np.empty_like(z_grid)
        for i, z in enumerate(z_grid):
            E_gamma = e0 * (1.0 + z)  # gamma energy at redshift z
            dtau_dl = _kernel_per_length(E_gamma, z, photon_field, eps, mu)
            dl_dz = c_cm / ((1.0 + z) * H(z))  # proper length element [cm]
            integrand_z[i] = dtau_dl * dl_dz
        tau[k] = trapz(integrand_z, z_grid)
    return tau


def attenuation(E0, z_s, photon_field, **kwargs):
    """Survival fraction ``exp(-tau)`` of gamma rays. See :func:`tau_gg`."""
    return np.exp(-tau_gg(E0, z_s, photon_field, **kwargs))
