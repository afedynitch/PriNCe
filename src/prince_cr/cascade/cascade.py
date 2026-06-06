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


def cascade_transfer_matrix(E, z, photon_field, eps=None, max_generations=40,
                            tol=1e-3):
    """Linear transfer matrix T for the locally-saturated EM cascade at z.

    ``T[i, j]`` = escaping-photon dN/dE in bin i per unit injected dN/dE in
    bin j. Sharp escape at ``E_abs(z) = absorption_energy(z)``: EM injected
    above E_abs cascades down to the universal pile-up below E_abs (the stiff
    part, solved here to convergence — *semi-analytic*); injection already
    below E_abs passes through unchanged (T is the identity there). Energy is
    conserved column-wise.

    This is the per-step EM operator for the co-evolved transport
    (``UHECRPropagationSolverETD2(enable_em_cascade=True)``): each z-step the
    EM particles produced by the nuclei are reprocessed by T, while ETD2
    carries the redshift transport of the sub-E_abs photons. The cascade is
    local (develops over ≪ a z-step), so per-step application is exact; ETD2's
    exact diagonal would otherwise mishandle the stiff off-diagonal production.

    Args:
        E (ndarray): photon/electron energy grid [GeV] (the species grid).
        z (float): redshift.
        photon_field: CMB(+EBL) target.
        eps, max_generations, tol: cascade integration controls.

    Returns:
        ndarray (n, n): the transfer matrix T.
    """
    if eps is None:
        eps = np.logspace(-15, -9, 400)
    n_eps = np.asarray(photon_field.get_photon_density(eps, z), dtype="double")
    n_eps = np.where(np.isfinite(n_eps) & (n_eps > 0), n_eps, 0.0)
    n = E.size
    dE = np.gradient(E)
    E_abs = absorption_energy(z, photon_field)

    P = _energy_conserving_matrix(pair_matrix(E, E, eps, n_eps), E, E)
    Mic = _energy_conserving_matrix(cooled_ic_photon_matrix(E, E, eps, n_eps), E, E)
    P_esc = np.where(E > E_abs, 0.0, 1.0)[:, None]  # escape below E_abs

    # Run the (linear) generation cascade on the identity → response columns.
    G = np.eye(n)                       # column j: unit dN/dE at bin j
    escaped = np.zeros((n, n))
    E_in = (E * dE)                     # energy weight per bin
    tot0 = E_in @ G  # = E (per column); used for the stop test
    for _ in range(max_generations):
        esc_now = P_esc * G
        escaped += esc_now
        interacting = G - esc_now
        e_spec = P @ (interacting * dE[:, None])
        G = Mic @ (e_spec * dE[:, None])
        if np.max((E_in @ G) / tot0) < tol:
            escaped += P_esc * G
            break
    return escaped


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
    escape_mode="smooth",
    inject_dNdE=None,
):
    """Develop a saturated EM cascade and return the escaping (observed-at-z=0)
    photon spectrum.

    The cascade is linear, so a single pass handles either a mono-energetic
    injection (default) or an arbitrary injection spectrum (``inject_dNdE``).
    With ``escape_mode="smooth"`` the escape probability uses the full-path
    ``tau_gg(E, z)``, so the output is already the spectrum observed at Earth
    from a source at redshift ``z``.

    Args:
        E_inj (float): top of the energy grid [GeV] (and the mono-energetic
            injection energy when ``inject_dNdE`` is None).
        z (float): source / cascade-environment redshift.
        photon_field: CMB(+EBL) target with ``get_photon_density``.
        n_grid (int): grid points (log-spaced ``e_min``..``E_inj``).
        e_min (float): low end of the spectrum grid [GeV].
        eps (ndarray): target photon energy grid; default brackets CMB.
        max_generations (int): cap on cascade generations.
        tol (float): stop when fractional energy above E_abs drops below this.
        inject_dNdE (callable or ndarray, optional): γ injection spectrum
            dN/dE [1/GeV]. A callable is evaluated on the internal grid; an
            ndarray must match ``n_grid``. Default: one photon at ``E_inj``.

    Returns:
        dict with grid ``E`` [GeV], escaped photon spectrum ``dNdE``
        [1/GeV], ``E_abs`` [GeV], ``E_inj``, energy-conservation
        ``energy_ratio`` (escaped/injected).
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

    dE = np.gradient(E)
    # initial photon spectrum (dN/dE on grid E)
    gamma_spec = np.zeros_like(E)
    if inject_dNdE is None:
        inj_idx = np.argmin(np.abs(E - E_inj))
        gamma_spec[inj_idx] = 1.0 / dE[inj_idx]  # one photon at E_inj
    elif callable(inject_dNdE):
        gamma_spec = np.asarray(inject_dNdE(E), dtype="double")
    else:
        gamma_spec = np.asarray(inject_dNdE, dtype="double")
        if gamma_spec.shape != E.shape:
            raise ValueError("inject_dNdE array must match the grid (n_grid).")
    E_inj_energy = trapz(E * gamma_spec, E)  # total injected energy

    escaped = np.zeros_like(E)
    E_inj_tot = E_inj_energy  # total injected energy

    if escape_mode == "smooth":
        # per-photon escape probability exp(-tau_gg(E, z)); the complement
        # interacts. Physical, smooth rollover (matches CRPropa).
        tau_E = tau_gg(E, z, photon_field, eps_min=1e-14, eps_max=1e-7)
        P_esc = np.exp(-tau_E)
    else:  # "sharp": step at E_abs (legacy)
        P_esc = np.where(E > E_abs, 0.0, 1.0)

    for _gen in range(max_generations):
        esc_now = gamma_spec * P_esc
        escaped += esc_now
        interacting = gamma_spec - esc_now  # = gamma_spec * (1 - P_esc)
        # interacting photons pair-produce -> e±, which fully IC-cool -> photons
        e_spec = P @ (interacting * dE)
        gamma_spec = Mic @ (e_spec * dE)
        if trapz(E * gamma_spec, E) / E_inj_tot < tol:
            escaped += gamma_spec * P_esc  # flush the small low-E remainder
            break

    escaped_E = trapz(E * escaped, E)
    return {
        "E": E,
        "dNdE": escaped,
        "E_abs": E_abs,
        "E_inj": E_inj,
        "energy_ratio": escaped_E / E_inj_tot,
    }
