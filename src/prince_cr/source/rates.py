"""Source-frame cooling and acceleration rates.

All `*_inv` functions return the inverse timescale :math:`t^{-1}`
(s\\ :math:`^{-1}`) for the named process at a given fluid-rest-frame
cosmic-ray energy. The naming convention mirrors Guo, Qian, Wu 2025
(Phys. Rev. D 112, 063022), Eqs. 2–10.

The photonuclear integrators take a ``PhotonField`` instance plus a
callable ``sigma_eps_GeV`` that returns :math:`\\sigma(\\epsilon_\\gamma)`
in cm\\ :math:`^2` at the *nucleus rest frame* photon energy
:math:`\\epsilon_\\gamma` in GeV. The simplest source for that callable is
a PriNCe cross-section object: ``cs.nonel(pdg)`` returns
``(eps_GeV_grid, sigma_cm2_array)`` which is wrapped into a
log-log interpolator by ``make_sigma_callable``.

An optional ``eps_mask`` callable can zero the integrand outside a
photon-energy window — this mirrors the energy-axis mask used by
PriNCe's tracking-species feature, so a sub-channel split (e.g. GDR
vs. photomeson) is a one-call extension when needed.

Units throughout: PRINCE_UNITS (GeV for energies, cm/s for c). The
closed-form expressions for synchrotron and acceleration use CGS
Gaussian internally (B in Gauss) because that's the convention of the
underlying formulas.
"""

import numpy as np
from scipy.integrate import trapezoid as trapz
from scipy.interpolate import interp1d

from prince_cr.data import PRINCE_UNITS


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

SIGMA_THOMSON_CM2 = 6.6524587e-25
"""Thomson cross section in cm^2 (CODATA)."""

E_CHARGE_ESU = 4.8032047e-10
"""Elementary charge in CGS-Gaussian units (statC / esu)."""

MB_TO_CM2 = 1.0e-27
"""Millibarn to cm^2."""


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def make_sigma_callable(eps_GeV, sigma_cm2, clip_floor_cm2=1e-50):
    """Wrap a tabulated cross section as a log-log interpolator.

    Returns a callable ``sigma(eps_GeV)`` valid on ``[eps_GeV[0],
    eps_GeV[-1]]``; out-of-range queries return zero. Use the
    ``(eps_grid, sigma_grid)`` tuple returned by
    ``CrossSectionBase.nonel(mother)`` (PriNCe FLUKA xs object) as
    input.
    """
    eps_GeV = np.asarray(eps_GeV, dtype=float)
    sigma_cm2 = np.asarray(sigma_cm2, dtype=float)
    sigma_safe = np.clip(sigma_cm2, clip_floor_cm2, None)
    log_interp = interp1d(
        np.log(eps_GeV),
        np.log(sigma_safe),
        kind="linear",
        bounds_error=False,
        fill_value=np.log(clip_floor_cm2),
    )
    eps_lo, eps_hi = eps_GeV[0], eps_GeV[-1]

    def sigma_func(eps):
        eps = np.atleast_1d(np.asarray(eps, dtype=float))
        out = np.exp(log_interp(np.log(np.clip(eps, eps_lo, eps_hi))))
        out = np.where((eps < eps_lo) | (eps > eps_hi), 0.0, out)
        return out

    return sigma_func


def jet_magnetic_field(L_gamma_iso_erg_s, Gamma, R_cm, xi_B_over_e):
    """Fluid-frame magnetic-field strength (Gauss).

    :math:`B = \\sqrt{8\\pi\\, \\xi_{B/e}\\, U_\\gamma}` with
    :math:`U_\\gamma = L_{\\gamma,\\rm iso} / (4\\pi \\Gamma^2 R^2 c)`,
    the energy partition between magnetic field and radiation in the
    fluid frame.
    """
    U_gamma_erg_cm3 = L_gamma_iso_erg_s / (
        4.0 * np.pi * Gamma**2 * R_cm**2 * PRINCE_UNITS.c
    )
    U_B_erg_cm3 = xi_B_over_e * U_gamma_erg_cm3
    return np.sqrt(8.0 * np.pi * U_B_erg_cm3)


def jet_proton_density(L_gamma_iso_erg_s, Gamma, R_cm, L_k_ratio=10.0):
    """Thermal-proton number density in the jet (cm^-3).

    :math:`n_p \\approx L_k / (4\\pi \\Gamma^2 R^2 m_p c^2 c)` with
    :math:`L_k = L_{k,\\rm iso}` set to ``L_k_ratio * L_gamma_iso`` as
    in Guo+ 2025 (Sec. III.A): "we take L_{k,iso} = 10 L_{γ,iso}".
    """
    L_k_erg_s = L_k_ratio * L_gamma_iso_erg_s
    m_p_rest_erg = PRINCE_UNITS.m_proton * PRINCE_UNITS.GeV2erg
    return L_k_erg_s / (
        4.0 * np.pi * Gamma**2 * R_cm**2 * PRINCE_UNITS.c * m_p_rest_erg
    )


# ---------------------------------------------------------------------
# Chodorowski phi function for Bethe-Heitler (mirrors PriNCe's _phi)
# ---------------------------------------------------------------------

def _chodorowski_phi(xi):
    """Chodorowski et al. 1992 phi function — same form as PriNCe
    ``interaction_rates.ContinuousPairProductionLossRate._phi``."""
    bltal_ultrarel = np.poly1d([2.667, -14.45, 50.95, -86.07])

    def phi_simple(x):
        return x * bltal_ultrarel(np.log(x))

    c1, c2, c3, c4 = 0.8048, 0.1459, 1.137e-3, -3.879e-6
    f1, f2, f3 = 2.91, 78.35, 1837.0

    xi = np.asarray(xi, dtype=float)
    out = np.zeros_like(xi)
    le = xi < 25.0
    he = xi >= 25.0

    if np.any(le):
        x = xi[le]
        out[le] = (
            np.pi / 12.0 * (x - 2.0) ** 4
            / (
                c1 * (x - 2.0) ** 1
                + c2 * (x - 2.0) ** 2
                + c3 * (x - 2.0) ** 3
                + c4 * (x - 2.0) ** 4
            )
        )
    if np.any(he):
        x = xi[he]
        out[he] = phi_simple(x) / (
            1.0 - f1 * x ** -1 - f2 * x ** -2 - f3 * x ** -3
        )
    return out


# ---------------------------------------------------------------------
# Photonuclear rates (Eqs. 2-3)
# ---------------------------------------------------------------------

def _photonuclear_inv(
    E_A_GeV,
    A_mass_GeV,
    sigma_eps_GeV,
    photon_field,
    kappa_eps_GeV=1.0,
    eps_th_GeV=None,
    eps_mask=None,
    n_eps=400,
    n_Eg=300,
):
    """Generic photonuclear inverse timescale, Eq. 2 / Eq. 3 of Guo+ 2025.

    Computes

    .. math::

       t^{-1}(E_A) = \\frac{c}{2 \\Gamma_A^2}
            \\int_{\\epsilon_{th}}^{2 \\Gamma_A E_\\gamma^{\\max}}
                d\\epsilon\\, \\epsilon\\, \\sigma(\\epsilon)\\, \\kappa(\\epsilon)\\,
                m(\\epsilon)
            \\int_{\\max(E_\\gamma^{\\min}, \\epsilon/2\\Gamma_A)}^{E_\\gamma^{\\max}}
                dE_\\gamma\\, \\frac{n(E_\\gamma)}{E_\\gamma^2}

    with ``kappa = 1`` recovering the reaction rate (Eq. 2) and a non-
    trivial ``kappa(eps)`` giving the cooling rate (Eq. 3). The optional
    ``eps_mask(eps)`` is the photon-rest-frame energy mask used by
    PriNCe's tracking-species feature for sub-channel resolution.
    """
    E_A_GeV = float(E_A_GeV)
    gamma_A = E_A_GeV / A_mass_GeV
    E_g_min = photon_field.E_min_GeV
    E_g_max = photon_field.E_max_GeV
    if eps_th_GeV is None:
        eps_th_GeV = 1.0e-3  # 1 MeV — well below any photonuclear threshold
    eps_max_GeV = 2.0 * gamma_A * E_g_max
    if eps_max_GeV <= eps_th_GeV:
        return 0.0

    eps = np.logspace(np.log10(eps_th_GeV), np.log10(eps_max_GeV), n_eps)
    sigma_vals = sigma_eps_GeV(eps)
    if eps_mask is not None:
        sigma_vals = sigma_vals * np.asarray(eps_mask(eps), dtype=float)
    if callable(kappa_eps_GeV):
        kappa_vals = np.asarray(kappa_eps_GeV(eps), dtype=float)
    else:
        kappa_vals = np.full_like(eps, float(kappa_eps_GeV))

    inner = np.zeros_like(eps)
    for i, eps_i in enumerate(eps):
        E_lo = max(E_g_min, eps_i / (2.0 * gamma_A))
        if E_lo >= E_g_max:
            continue
        Eg = np.logspace(np.log10(E_lo), np.log10(E_g_max), n_Eg)
        n_vals = photon_field.get_photon_density(Eg, 0.0)
        inner[i] = trapz(n_vals / Eg**2, Eg)

    integrand = eps * sigma_vals * kappa_vals * inner
    outer = trapz(integrand, eps)
    return PRINCE_UNITS.c / (2.0 * gamma_A**2) * outer


def photonuclear_rate_inv(
    E_A_GeV,
    A_mass_GeV,
    sigma_eps_GeV,
    photon_field,
    eps_th_GeV=None,
    eps_mask=None,
    n_eps=400,
    n_Eg=300,
):
    """Eq. 2 — photonuclear reaction rate inverse time (s^-1).

    Returns the rate at which a nucleus of energy ``E_A_GeV`` and mass
    ``A_mass_GeV`` (mass-energy in GeV; e.g. ``A * m_p``) interacts
    with the photons of ``photon_field`` through a process whose cross
    section is ``sigma_eps_GeV(eps_GeV)`` (cm^2, as a function of the
    photon energy in the nucleus rest frame).
    """
    return _photonuclear_inv(
        E_A_GeV, A_mass_GeV, sigma_eps_GeV, photon_field,
        kappa_eps_GeV=1.0, eps_th_GeV=eps_th_GeV,
        eps_mask=eps_mask, n_eps=n_eps, n_Eg=n_Eg,
    )


def photonuclear_cool_inv(
    E_A_GeV,
    A_mass_GeV,
    sigma_eps_GeV,
    photon_field,
    kappa_eps_GeV=1.0,
    eps_th_GeV=None,
    eps_mask=None,
    n_eps=400,
    n_Eg=300,
):
    """Eq. 3 — photonuclear cooling rate inverse time (s^-1).

    Same form as Eq. 2 plus the inelasticity factor ``kappa(eps)``.
    ``kappa`` may be a constant (e.g. ``1/A`` for photodisintegration
    under single-nucleon emission) or a callable.
    """
    return _photonuclear_inv(
        E_A_GeV, A_mass_GeV, sigma_eps_GeV, photon_field,
        kappa_eps_GeV=kappa_eps_GeV, eps_th_GeV=eps_th_GeV,
        eps_mask=eps_mask, n_eps=n_eps, n_Eg=n_Eg,
    )


# ---------------------------------------------------------------------
# Secondary differential yield — neutrinos, photons, e±, free p/n
# ---------------------------------------------------------------------

def secondary_yield_per_x_inv(
    E_A_GeV,
    A,
    x_centers,
    dsig_dx_eps,
    eps_grid_GeV,
    photon_field,
    eps_mask=None,
    eps_th_GeV=None,
    n_Eg=300,
):
    """Per-x secondary-yield rate (s^-1 per x bin) for one daughter species.

    Generalises Eq. 2 of Guo+ 2025 to a *differential* cross section
    :math:`d\\sigma/dx` where the daughter PriNCe-x is
    :math:`x = E_d^{\\rm rest}/m_p \\approx A \\, E_d^{\\rm fluid}/E_A`.

    Returns ``dN_d/dx/dt(x; E_A)`` for each ``x`` in ``x_centers``.
    The fluid-frame daughter energy at each ``x`` is
    :math:`E_d = x \\cdot E_A / A`, so the spectrum
    :math:`dN_d/dE_d = (A/E_A)\\, dN_d/dx`.

    Parameters
    ----------
    E_A_GeV : float
        Fluid-frame total nucleus energy (GeV).
    A : int
        Mass number of the parent nucleus.
    x_centers : array_like, shape (n_x,)
        Daughter-x bin centres (PriNCe convention, log-spaced).
    dsig_dx_eps : ndarray, shape (n_x, n_eps)
        Differential cross section ``d sigma/dx`` (cm^2), tabulated on
        the daughter-x and nucleus-rest-frame photon-energy grids.
    eps_grid_GeV : array_like, shape (n_eps,)
        Photon-rest-frame energy grid (GeV) on which ``dsig_dx_eps`` is
        defined (the second axis).
    photon_field : PhotonField
        Source photon field; must expose ``get_photon_density(E, z)``
        in PriNCe units (E in GeV, density in :math:`{\\rm GeV}^{-1}\\,
        {\\rm cm}^{-3}`) and ``E_min_GeV`` / ``E_max_GeV`` bounds.
    eps_mask : callable, optional
        Optional mask ``mask(eps_GeV)`` applied to the integrand, same
        convention as `_photonuclear_inv`. Use to restrict to a
        sub-channel (e.g. eps >= 0.14 GeV for photomeson-only).
    eps_th_GeV : float, optional
        Lower bound of the outer integral. Defaults to the first
        positive-σ bin of ``eps_grid_GeV`` for the given x.
    n_Eg : int, optional
        Inner-integral resolution in fluid-frame photon energy.
    """
    E_A_GeV = float(E_A_GeV)
    gamma_A = E_A_GeV / (A * PRINCE_UNITS.m_proton)
    E_g_min = photon_field.E_min_GeV
    E_g_max = photon_field.E_max_GeV
    eps = np.asarray(eps_grid_GeV, dtype=float)
    if eps_th_GeV is not None:
        eps = eps[eps >= eps_th_GeV]
        dsig_dx_eps = dsig_dx_eps[:, -len(eps):]
    # Mask the eps-axis if requested.
    if eps_mask is not None:
        m = np.asarray(eps_mask(eps), dtype=float)
        dsig_dx_eps = dsig_dx_eps * m[None, :]
    # Inner integral over E_gamma for each eps bin: I(eps) = ∫ n/E^2 dE.
    inner = np.zeros_like(eps)
    for j, eps_j in enumerate(eps):
        E_lo = max(E_g_min, eps_j / (2.0 * gamma_A))
        if E_lo >= E_g_max:
            continue
        Eg = np.logspace(np.log10(E_lo), np.log10(E_g_max), n_Eg)
        n_vals = photon_field.get_photon_density(Eg, 0.0)
        inner[j] = trapz(n_vals / Eg**2, Eg)
    # Outer integral over eps for each x: yield(x) = (c/2g^2) * ∫ dε ε σ_x(ε) I(ε).
    integrand = eps[None, :] * dsig_dx_eps * inner[None, :]
    rate_per_x = PRINCE_UNITS.c / (2.0 * gamma_A**2) * trapz(integrand, eps, axis=1)
    return rate_per_x


def secondary_yield_dN_dE_inv(
    E_A_GeV,
    A,
    x_centers,
    dsig_dx_eps,
    eps_grid_GeV,
    photon_field,
    eps_mask=None,
    eps_th_GeV=None,
    n_Eg=300,
):
    """Daughter spectrum :math:`dN_d/dE_d/dt` (per GeV per s per parent).

    Returns ``(E_d_fluid_GeV, dN/dE/dt)`` where ``E_d = x * E_A / A`` is
    the daughter fluid-frame energy on the PriNCe-x grid. Convenience
    wrapper around `secondary_yield_per_x_inv`.
    """
    rate_per_x = secondary_yield_per_x_inv(
        E_A_GeV, A, x_centers, dsig_dx_eps, eps_grid_GeV, photon_field,
        eps_mask=eps_mask, eps_th_GeV=eps_th_GeV, n_Eg=n_Eg,
    )
    x = np.asarray(x_centers, dtype=float)
    E_d_GeV = x * E_A_GeV / A
    # dN/dE = dN/dx * dx/dE = dN/dx / (E_A/A).
    dN_dE = rate_per_x / (E_A_GeV / A)
    return E_d_GeV, dN_dE


# ---------------------------------------------------------------------
# Bethe-Heitler (Chodorowski parametrisation)
# ---------------------------------------------------------------------

def bethe_heitler_cool_inv(E_A_GeV, A_mass_GeV, Z, photon_field, n_xi=600):
    """Eq. 3 BH cooling rate, evaluated via the Chodorowski et al. 1992
    parametrisation.

    .. math::

       t_{\\rm BH}^{-1} = \\frac{c\\, \\alpha_{\\rm em} r_e^2 (m_e c^2)^2 Z^2}
                 {(m_A c^2) \\gamma_A}
            \\int_2^\\infty d\\xi\\, \\frac{\\phi(\\xi)}{\\xi^2}\\,
            n(m_e c^2 \\xi / (2 \\gamma_A))

    with :math:`\\xi = 2 \\gamma_A \\epsilon_\\gamma / m_e c^2`. The
    photon field is queried at fluid-frame energies
    :math:`m_e c^2 \\xi / (2 \\gamma_A)` — typically in the eV–keV range
    for the relevant CR energies and well inside the
    ``SourceBrokenPowerLaw`` window. Returns ``s^-1``.
    """
    gamma_A = E_A_GeV / A_mass_GeV
    xi = np.logspace(np.log10(2.0 + 1e-8), 16.0, n_xi)
    phi_xi2 = _chodorowski_phi(xi) / xi**2
    # Photon energies in the fluid frame at which to evaluate n_gamma.
    E_gamma_GeV = PRINCE_UNITS.m_electron * xi / (2.0 * gamma_A)
    n_vals = photon_field.get_photon_density(E_gamma_GeV, 0.0)
    integral = trapz(n_vals * phi_xi2, xi)
    prefactor = (
        PRINCE_UNITS.c
        * PRINCE_UNITS.fine_structure
        * PRINCE_UNITS.r_electron**2
        * PRINCE_UNITS.m_electron**2
        * Z**2
        / (A_mass_GeV * gamma_A)
    )
    return prefactor * integral


# ---------------------------------------------------------------------
# Closed-form continuous losses (Eqs. 7-10)
# ---------------------------------------------------------------------

def synchrotron_cool_inv(E_A_GeV, A_mass_GeV, Z, B_Gauss):
    """Eq. 8 — synchrotron cooling rate (s^-1).

    .. math::

       t_{\\rm syn}^{-1}(E_A) =
         \\frac{Z^4 \\sigma_T (m_e c^2)^2 E_A B^2 c}{6 \\pi (m_A c^2)^4}

    All masses passed as rest-mass energies in GeV; B in Gauss; result
    in s\\ :math:`^{-1}`. Equivalent to ``E_A / (m_A c^2 t_{\\rm syn})``
    when comparing against Eq. 8's energy-loss-timescale form.

    NB the charge enters as **Z⁴** (synchrotron is Larmor, P∝q⁴ ⇒ t⁻¹=P/E∝q⁴),
    not Z². The transcribed Guo Eq. 8 had Z² — corrected 2026-06-16 (invisible
    for protons Z=1; ×Z² error for nuclei, e.g. Fe ×676). See open-questions.
    """
    E_A_erg = E_A_GeV * PRINCE_UNITS.GeV2erg
    m_A_c2_erg = A_mass_GeV * PRINCE_UNITS.GeV2erg
    m_e_c2_erg = PRINCE_UNITS.m_electron * PRINCE_UNITS.GeV2erg
    return (
        Z**4
        * SIGMA_THOMSON_CM2
        * m_e_c2_erg**2
        * E_A_erg
        * B_Gauss**2
        * PRINCE_UNITS.c
        / (6.0 * np.pi * m_A_c2_erg**4)
    )


def adiabatic_source_inv(Gamma, R_cm):
    """Eq. 9 — source-frame adiabatic-expansion rate (s^-1).

    :math:`t_{\\rm ad} = R/(\\Gamma c)`; energy-independent.
    """
    return Gamma * PRINCE_UNITS.c / R_cm


def hadronic_Ap_inv(E_A_GeV, A, n_p_cm3):
    """Eq. 7 — hadronic A-p collision rate (s^-1).

    Uses :math:`\\sigma_{Ap}(s) \\approx A^{2/3} \\sigma_{pp}(s_{pp})`
    with :math:`s_{pp} \\approx s_{Ap}/A \\approx 2 (E_A/A) m_p` and the
    polynomial-in-:math:`\\ln s` parametrisation
    :math:`\\sigma_{pp}(s) \\approx [32.4 - 1.2 \\ln s + 0.21 \\ln^2 s]` mb
    (s in GeV\\ :math:`^2`) from Guo+ 2025 Sec. III.A.
    """
    m_p = PRINCE_UNITS.m_proton
    E_per_nuc = E_A_GeV / A
    s_pp = 2.0 * E_per_nuc * m_p + m_p**2  # GeV^2
    log_s = np.log(np.maximum(s_pp, 1e-10))
    sigma_pp_mb = 32.4 - 1.2 * log_s + 0.21 * log_s**2
    sigma_Ap_cm2 = A ** (2.0 / 3.0) * sigma_pp_mb * MB_TO_CM2
    return sigma_Ap_cm2 * n_p_cm3 * PRINCE_UNITS.c


def acceleration_inv(E_A_GeV, Z, B_Gauss, kappa_acc=10.0):
    """Eq. 10 — acceleration rate (s^-1).

    :math:`t_{\\rm acc} = \\kappa_{\\rm acc} E_A / (Z e c B)`. With
    :math:`\\kappa_{\\rm acc} = 1` recovers the Bohm limit; Guo+ adopt
    :math:`\\kappa_{\\rm acc} \\sim 10` as a fiducial.
    """
    E_A_erg = E_A_GeV * PRINCE_UNITS.GeV2erg
    t_acc = kappa_acc * E_A_erg / (Z * E_CHARGE_ESU * B_Gauss * PRINCE_UNITS.c)
    return 1.0 / t_acc
