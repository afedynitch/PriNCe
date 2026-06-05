"""Microphysics kernels for the electromagnetic cascade (Phase B).

Native PriNCe implementations (GeV units throughout) of the two processes
that drive a 1D EM cascade on an isotropic photon field n(eps) [GeV^-1 cm^-3]:

  - Inverse Compton: the scattered-photon emission spectrum from an electron
    of Lorentz factor gamma (Blumenthal & Gould 1970, full Compton kernel,
    valid Thomson -> Klein-Nishina).
  - gamma-gamma pair production: the e+/e- injection spectrum from a photon
    of energy E0 (Aharonian, Atoyan & Nagapetyan 1983).

Each kernel is validated against an analytic limit and/or AM3's `ampy`
(see runs/2026-06-06_em-cascade-cascade/inputs/validate_kernels.py).
"""

import numpy as np
from scipy.integrate import trapezoid as trapz

from prince_cr.data import PRINCE_UNITS

SIGMA_THOMSON = 8.0 / 3.0 * np.pi * PRINCE_UNITS.r_electron**2  # cm^2
M_E = PRINCE_UNITS.m_electron  # GeV
C_CM = PRINCE_UNITS.c  # cm/s


# ---------------------------------------------------------------------------
# Inverse Compton — Blumenthal & Gould (1970), Rev. Mod. Phys. 42, 237, Eq. 2.48
# ---------------------------------------------------------------------------
def _ic_f(q, Gam):
    """BG70 spectral function f(q, Gamma) for isotropic IC. q, Gam arrays."""
    return (
        2.0 * q * np.log(q)
        + (1.0 + 2.0 * q) * (1.0 - q)
        + 0.5 * (Gam * q) ** 2 * (1.0 - q) / (1.0 + Gam * q)
    )


def ic_emission_spectrum(E1, E_e, eps, n_eps):
    """Inverse-Compton scattered-photon emission spectrum dN/(dt dE1).

    Args:
        E1 (ndarray): scattered (outgoing) photon energies [GeV].
        E_e (float): electron total energy [GeV].
        eps (ndarray): target photon energy grid [GeV] (log-spaced).
        n_eps (ndarray): target photon density n(eps) [GeV^-1 cm^-3].

    Returns:
        ndarray: dN/(dt dE1) [s^-1 GeV^-1], same shape as E1.
    """
    gamma = E_e / M_E
    E1 = np.atleast_1d(np.asarray(E1, dtype="double"))
    out = np.zeros_like(E1)

    pref = 0.75 * SIGMA_THOMSON * C_CM / gamma**2  # [cm^2 cm/s] = cm^3/s
    for i, e1 in enumerate(E1):
        if e1 >= E_e:
            continue
        Gam = 4.0 * eps * gamma / M_E  # = 4 eps gamma / (m_e c^2)
        q = e1 / (Gam * (E_e - e1))
        # BG70 kinematic range: 1/(4 gamma^2) <= q <= 1
        ok = (q >= 1.0 / (4.0 * gamma**2)) & (q <= 1.0) & (eps > 0) & (n_eps > 0)
        if not np.any(ok):
            continue
        integrand = np.zeros_like(eps)
        integrand[ok] = n_eps[ok] / eps[ok] * _ic_f(q[ok], Gam[ok])
        out[i] = pref * trapz(integrand, eps)
    return out


def ic_energy_loss_rate(E_e, eps, n_eps, n_E1=400):
    """Total IC energy-loss rate -dE/dt [GeV/s] by integrating the emission
    spectrum: int (E1 - eps_mean) dN/dt ~ int E1 dN/dt for upscattering."""
    gamma = E_e / M_E
    # scattered photons span up to ~ E_e; sample below it
    E1 = np.logspace(np.log10(eps.min()), np.log10(0.9999 * E_e), n_E1)
    dN = ic_emission_spectrum(E1, E_e, eps, n_eps)
    power_out = trapz(E1 * dN, E1)  # GeV/s radiated
    # subtract power removed from the field (number rate x mean target energy)
    rate = trapz(dN, E1)  # scatterings/s
    eps_mean = trapz(eps * n_eps, eps) / trapz(n_eps, eps)
    return power_out - rate * eps_mean


# ---------------------------------------------------------------------------
# gamma-gamma pair production e+/e- injection spectrum
#   Aharonian, Atoyan & Nagapetyan (1983); see Boettcher & Schlickeiser 1997.
# ---------------------------------------------------------------------------
def pair_injection_spectrum(E_e, E0, eps, n_eps):
    """Spectrum of produced electrons (and positrons) dN/(dt dE_e).

    A photon of energy E0 pair-produces on the isotropic field n(eps); this
    returns the production rate of e-/e+ at energy E_e.

    Args:
        E_e (ndarray): produced electron/positron energies [GeV].
        E0 (float): incident high-energy photon energy [GeV].
        eps (ndarray): target photon energy grid [GeV].
        n_eps (ndarray): target photon density [GeV^-1 cm^-3].

    Returns:
        ndarray: dN/(dt dE_e) [s^-1 GeV^-1] (sum of e- and e+).
    """
    E_e = np.atleast_1d(np.asarray(E_e, dtype="double"))
    out = np.zeros_like(E_e)
    # dimensionless energies in units of m_e c^2
    eg = E0 / M_E
    pref = 0.75 * SIGMA_THOMSON * C_CM / eg**3  # [cm^3/s], per BS97 Eq.

    for i, e_e in enumerate(E_e):
        ge = e_e / M_E
        if ge <= 1.0 or ge >= eg - 1.0:
            continue
        we = eps / M_E  # soft photon energy in m_e c^2 units
        # threshold on soft photon: eps_min for given (eg, ge)
        # BS97: integrand defined for w >= eg/(4 ge (eg-ge))
        wmin = eg / (4.0 * ge * (eg - ge))
        ok = (we > wmin) & (n_eps > 0)
        if not np.any(ok):
            continue
        w = we[ok]
        # BS97 (1997) Eq. (15) bracket, with x = eg/(4 w ge (eg-ge))... use
        # the standard kernel F(ge, eg, w):
        r = 0.25 * eg / (w * ge * (eg - ge))  # = wmin/w  in [0,1]
        F = (
            4.0 * eg**2 / (ge * (eg - ge)) * np.log(4.0 * w * ge * (eg - ge) / eg)
            - 8.0 * eg * w
            + (2.0 * (2.0 * eg * w - 1.0) * eg**2) / (ge * (eg - ge))
            - (1.0 - 1.0 / (eg * w)) * eg**4 / (ge**2 * (eg - ge) ** 2)
        )
        F = np.where(r < 1.0, F, 0.0)
        integrand = np.zeros_like(eps)
        integrand[ok] = n_eps[ok] / (eps[ok] / M_E) ** 2 * F / M_E
        # divide by M_E to convert d/d(ge) -> d/dE_e
        out[i] = pref * trapz(integrand / M_E, eps) * M_E  # bookkeeping below
    return out
