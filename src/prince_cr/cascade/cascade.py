"""1D electromagnetic cascade on the CMB(+EBL) via the saturated-generation
method (Phase B).

A high-energy photon injected at redshift ``z`` develops a pair-production /
inverse-Compton cascade that reprocesses its energy down to the energy
``E_abs`` where the universe becomes transparent (gamma-gamma optical depth
~1). The escaping photons below ``E_abs`` form the cosmogenic / EGB
contribution.

Method (fully-developed / saturated limit, valid when the propagation length
greatly exceeds every interaction length above ``E_abs`` — true for the
cosmogenic cascade over cosmological distances):

  1. photons above ``E_abs`` pair-produce -> e+/e-  (energy-conserving split,
     :func:`prince_cr.cascade.kernels.pair_injection_spectrum`);
  2. each e+/e- fully IC-cools, radiating its entire energy as a photon
     spectrum (the cooled-IC kernel built here from the BG70 emission spectrum
     and the IC loss rate);
  3. new photons below ``E_abs`` escape; those above repeat from step 1.

Energy is conserved by construction (every transfer matrix preserves the
column energy), so the *spectral shape* is the physical observable to validate
against the universal cascade spectrum (Berezinsky & Smirnov 1975:
E^2 dN/dE roughly flat below the absorption break).
"""

import numpy as np
from scipy.integrate import trapezoid as trapz

from prince_cr.cascade.kernels import (
    M_E,
    ic_emission_spectrum,
    ic_energy_loss_rate,
    pair_injection_spectrum,
)
from prince_cr.cascade.opacity import tau_gg


def absorption_energy(z, photon_field, e_lo=10.0, e_hi=1e8, n=60, **tau_kw):
    """Energy where tau_gg(E, z) crosses 1 (the cascade escape energy) [GeV]."""
    E = np.logspace(np.log10(e_lo), np.log10(e_hi), n)
    tau = tau_gg(E, z, photon_field, **tau_kw)
    above = np.where(tau >= 1.0)[0]
    if len(above) == 0:
        return e_hi
    i = above[0]
    if i == 0:
        return e_lo
    # log-interpolate tau=1 between i-1 and i
    lt0, lt1 = np.log(tau[i - 1]), np.log(tau[i])
    le0, le1 = np.log(E[i - 1]), np.log(E[i])
    return float(np.exp(le0 + (0.0 - lt0) * (le1 - le0) / (lt1 - lt0)))


def _energy_conserving_matrix(P, E_in, E_out):
    """Rescale each input column so the transferred energy equals the input
    energy (enforces exact energy conservation per interaction)."""
    P = np.array(P, dtype="double")
    for j in range(P.shape[1]):
        out_E = trapz(E_out * P[:, j], E_out) if P.shape[0] > 1 else 0.0
        if out_E > 0:
            P[:, j] *= E_in[j] / out_E
    return P


def cooled_ic_photon_matrix(E_gamma, E_e_grid, eps, n_eps):
    """Photon spectrum radiated by an electron of energy E_e as it fully cools
    by IC: integrate the emission spectrum over the cooling history,
    dN/dE_gamma = int_{E_gamma}^{E_e} emission(E_gamma, E') / |dE/dt|(E') dE'.

    Returns matrix M[i, j] = dN_gamma(E_gamma_i) per electron at E_e_grid[j].
    """
    ne, ng = len(E_e_grid), len(E_gamma)
    loss = np.array([ic_energy_loss_rate(Ee, eps, n_eps) for Ee in E_e_grid])
    # emission[i, j] = dN/(dt dE_gamma_i) for electron E_e_grid[j]
    emis = np.zeros((ng, ne))
    for j, Ee in enumerate(E_e_grid):
        emis[:, j] = ic_emission_spectrum(E_gamma, Ee, eps, n_eps)
    M = np.zeros((ng, ne))
    for j in range(ne):
        # integrate emission(E_gamma, E') / loss(E') over E' from E_gamma up to E_e[j]
        for i in range(ng):
            sel = (E_e_grid >= E_gamma[i]) & (E_e_grid <= E_e_grid[j])
            if np.count_nonzero(sel) >= 2:
                integrand = emis[i, sel] / loss[sel]
                M[i, j] = trapz(integrand, E_e_grid[sel])
    return M


def pair_matrix(E_e, E_gamma_grid, eps, n_eps):
    """Matrix P[i, j] = dN_e(E_e_i) per photon at E_gamma_grid[j] (e+ + e-)."""
    P = np.zeros((len(E_e), len(E_gamma_grid)))
    for j, Eg in enumerate(E_gamma_grid):
        P[:, j] = pair_injection_spectrum(E_e, Eg, eps, n_eps)
    return P


def run_cascade(
    E_inj,
    z,
    photon_field,
    n_grid=120,
    e_min=1.0,
    eps=None,
    max_generations=40,
    tol=1e-3,
):
    """Develop a saturated EM cascade from a mono-energetic photon injection.

    Args:
        E_inj (float): injected photon energy [GeV].
        z (float): redshift of the cascade environment.
        photon_field: CMB(+EBL) target with ``get_photon_density``.
        n_grid (int): energy bins per decade-spanning log grid.
        e_min (float): low end of the spectrum grid [GeV].
        eps (ndarray): target photon energy grid; default brackets CMB.
        max_generations (int): cap on cascade generations.
        tol (float): stop when fractional energy above E_abs drops below this.

    Returns:
        dict with grid ``E`` [GeV], escaped photon spectrum ``dNdE``
        [1/GeV per injected photon], ``E_abs`` [GeV], ``E_inj``,
        energy-conservation ``energy_ratio`` (escaped/injected).
    """
    if eps is None:
        eps = np.logspace(-15, -9, 500)
    n_eps = np.asarray(photon_field.get_photon_density(eps, z), dtype="double")
    n_eps = np.where(np.isfinite(n_eps) & (n_eps > 0), n_eps, 0.0)

    E = np.logspace(np.log10(e_min), np.log10(E_inj), n_grid)
    E_abs = absorption_energy(z, photon_field)

    # transfer matrices (energy-conserving)
    P = _energy_conserving_matrix(pair_matrix(E, E, eps, n_eps), E, E)
    Mic = _energy_conserving_matrix(cooled_ic_photon_matrix(E, E, eps, n_eps), E, E)

    # photon spectrum (number per bin via dN/dE on grid E)
    gamma_spec = np.zeros_like(E)
    # inject delta at the bin closest to E_inj (as a narrow dN/dE)
    inj_idx = np.argmin(np.abs(E - E_inj))
    dE = np.gradient(E)
    gamma_spec[inj_idx] = 1.0 / dE[inj_idx]  # one photon

    escaped = np.zeros_like(E)
    above = E > E_abs
    below = ~above
    E_inj_tot = E_inj  # total injected energy

    for _gen in range(max_generations):
        # photons below E_abs escape
        escaped[below] += gamma_spec[below]
        gamma_spec[below] = 0.0
        # remaining (above E_abs) pair-produce -> electrons
        e_spec = P @ (gamma_spec * dE)  # dN_e/dE on grid E
        # electrons fully cool -> photons
        gamma_spec = Mic @ (e_spec * dE)
        # energy remaining above threshold
        E_above = trapz((E * gamma_spec)[above], E[above]) if np.any(above) else 0.0
        if E_above / E_inj_tot < tol:
            # only sub-threshold photons escape; discard the (<tol) above-E_abs
            # residual rather than counting absorbed photons as escaped.
            escaped[below] += gamma_spec[below]
            break

    escaped_E = trapz(E * escaped, E)
    return {
        "E": E,
        "dNdE": escaped,
        "E_abs": E_abs,
        "E_inj": E_inj,
        "energy_ratio": escaped_E / E_inj_tot,
    }
