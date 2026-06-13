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
    C_CM,
    M_E,
    SIGMA_THOMSON,
    _ic_f,
    ic_emission_spectrum,
    ic_energy_loss_rate,
    pair_injection_spectrum,
)
from prince_cr.cascade.opacity import tau_gg

# ---------------------------------------------------------------------------
# AM3-style photon-field factorization of the IC / γγ kernels.
#
# The IC emission, IC energy-loss, and γγ pair cross-section kernels are all
# field-FREE functions of (energies). The scalar builders above rebuild them
# per z via Python loops; these cached builders compute the field-free tensor
# ONCE (vectorized) and the per-z matrices are then a contraction with n(eps)
# (a matvec) — the same factorization PriNCe uses for the nuclear rates and
# AM3 for its cascade kernels. Validated bit-exact (≤1.6e-14) against the
# scalar builders. Cache keyed by the (E_out, E_in, eps) log-grid signature;
# _em_transfer_at calls these once per z-node with identical grids, so the
# tensors build once and every z reuses them.
# ---------------------------------------------------------------------------
_PREF_IC = 0.75 * SIGMA_THOMSON * C_CM
_IC_KERNEL_CACHE = {}


def cascade_kernel_cache_clear():
    _IC_KERNEL_CACHE.clear()


def _trapz_weights(x):
    """Per-point weights w with sum(w*y) == trapz(y, x) (exact)."""
    w = np.empty_like(x)
    w[1:-1] = (x[2:] - x[:-2]) / 2.0
    w[0] = (x[1] - x[0]) / 2.0
    w[-1] = (x[-1] - x[-2]) / 2.0
    return w


def _grid_key(*arrs):
    return tuple((a.size, float(a[0]), float(a[-1])) for a in arrs)


def _ic_emis_kernel(E_out, E_e, eps):
    """Field-free IC emission kernel A[i,j,k] with emis = pref[j]·(A @ n_eps);
    A folds the eps trapz weights so the contraction equals the scalar
    ic_emission_spectrum trapz exactly. Plus the per-column prefactor."""
    key = ("emis",) + _grid_key(E_out, E_e, eps)
    cached = _IC_KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    g = (E_e / M_E)[None, :, None]
    Eo = E_out[:, None, None]
    Ee = E_e[None, :, None]
    ep = eps[None, None, :]
    wln = _trapz_weights(eps)[None, None, :]
    with np.errstate(all="ignore"):
        Gam = 4.0 * ep * g / M_E
        q = Eo / (Gam * (Ee - Eo))
        ok = (q >= 1.0 / (4.0 * g**2)) & (q <= 1.0) & (Eo < Ee) & (ep > 0) & (g > 1.0)
        A = np.where(ok, _ic_f(q, Gam) / ep, 0.0) * wln
    pref = np.where((E_e / M_E) > 1.0, _PREF_IC / (E_e / M_E) ** 2, 0.0)
    _IC_KERNEL_CACHE[key] = (A, pref)
    return A, pref


def _ic_loss_kernel(E_e, eps, n_E1=400):
    """Field-free IC emission kernel on the per-E_e loss grid E1_j (logspace to
    0.9999·E_e[j], n_E1 pts) used by the energy-loss rate. Returns (B[j,m,k],
    E1[j,m], pref[j]); loss = power_out − rate·eps_mean, both from B @ n_eps."""
    key = ("loss",) + _grid_key(E_e, eps) + (n_E1,)
    cached = _IC_KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    n = E_e.size
    lo = np.log10(eps.min())
    E1 = np.empty((n, n_E1))
    for j in range(n):
        E1[j] = np.logspace(lo, np.log10(0.9999 * E_e[j]), n_E1)
    g = (E_e / M_E)[:, None, None]
    Eo = E1[:, :, None]
    Ee = E_e[:, None, None]
    ep = eps[None, None, :]
    wln = _trapz_weights(eps)[None, None, :]
    with np.errstate(all="ignore"):
        Gam = 4.0 * ep * g / M_E
        q = Eo / (Gam * (Ee - Eo))
        ok = (q >= 1.0 / (4.0 * g**2)) & (q <= 1.0) & (Eo < Ee) & (ep > 0) & (g > 1.0)
        B = np.where(ok, _ic_f(q, Gam) / ep, 0.0) * wln
    pref = np.where((E_e / M_E) > 1.0, _PREF_IC / (E_e / M_E) ** 2, 0.0)
    _IC_KERNEL_CACHE[key] = (B, E1, pref)
    return B, E1, pref


def _pair_kernel(E_e, E0, eps):
    """Field-free γγ pair-injection kernel A[i,j,k] with P = pref[j]·(A @ n_eps)
    (eps trapz weights folded in). Replicates pair_injection_spectrum."""
    key = ("pair",) + _grid_key(E_e, E0, eps)
    cached = _IC_KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    ge = (E_e / M_E)[:, None, None]
    eg = (E0 / M_E)[None, :, None]
    w = (eps / M_E)[None, None, :]
    wln = _trapz_weights(eps)[None, None, :]
    denom = ge * (eg - ge)
    valid = (ge > 1.0) & (ge < eg - 1.0) & (denom > 0)
    with np.errstate(all="ignore"):
        wmin = eg / (4.0 * denom)
        r = wmin / w
        F = (4.0 * eg**2 / denom * np.log(4.0 * w * denom / eg)
             - 8.0 * eg * w
             + 2.0 * (2.0 * eg * w - 1.0) * eg**2 / denom
             - (1.0 - 1.0 / (eg * w)) * eg**4 / denom**2)
        ok = valid & (w > wmin) & (r < 1.0)
        A = np.where(ok, F / (w**2) / M_E, 0.0) * wln
    pref = _PREF_IC / (E0 / M_E) ** 3
    _IC_KERNEL_CACHE[key] = (A, pref)
    return A, pref


def _ic_emission_matrix(E_out, E_e, eps, n_eps):
    """emis[i,j] via the cached field-free kernel contracted with n_eps."""
    A, pref = _ic_emis_kernel(E_out, E_e, eps)
    return pref[None, :] * np.tensordot(A, n_eps, axes=([2], [0]))


def _ic_loss_vector(E_e, eps, n_eps, n_E1=400):
    """IC energy-loss rate [GeV/s] per E_e via the cached loss kernel."""
    B, E1, pref = _ic_loss_kernel(E_e, eps, n_E1)
    dN = pref[:, None] * np.tensordot(B, n_eps, axes=([2], [0]))   # (j, n_E1)
    power_out = trapz(E1 * dN, E1, axis=1)
    rate = trapz(dN, E1, axis=1)
    eps_mean = trapz(eps * n_eps, eps) / trapz(n_eps, eps)
    return power_out - rate * eps_mean


def cascade_transfer_matrix(E, z, photon_field, eps=None, max_generations=40,
                            tol=1e-3, dz=None):
    """Linear EM-cascade transfer matrix T at redshift z (stiffness-split).

    ``T[i, j]`` = post-step-photon dN/dE in bin i per unit injected dN/dE in
    bin j. Energy is conserved column-wise (nothing leaves the EM pool here;
    photons escape the *system* only at z=0, via the integration ending).

    **Stiffness boundary (``dz`` given — the co-evolved transport mode).**
    The escape probability is the *per-z-step* survival ``exp(-Δτ_step(E))``,
    with ``Δτ_step(E) = (dτ_gg/dl)(E,z)·Δl_step`` and ``Δl_step = c·|dz|/((1+z)
    H(z))``. So only the **stiff** EM (E ≫ E_abs, mfp ≪ one z-step,
    Δτ_step ≫ 1) interacts and cascades down *within this step*; the
    **non-stiff** EM near/below E_abs (mfp ~ Hubble length, Δτ_step ≪ 1) passes
    through (T ≈ identity there) and is carried by ETD2 to the next step, where
    it redshifts and is gradually absorbed. Over the integration the surviving
    photons pile up at the *evolving* (z→0) γγ horizon — not the high-z
    injection horizon. This fixes the saturated-per-z bug that pinned each
    shell's pile-up at E_abs(z_inject) (see lessons/em-cascade-transfer-*).

    **Saturated mode (``dz`` None — standalone single-shot).** Falls back to the
    full-path smooth escape ``exp(-τ_gg(E,z))``: the fully-developed cascade
    observed at z=0 from a source at z (the `run_cascade` semantics). Use for
    single-source spectra, not for per-step transport.

    Args:
        E (ndarray): photon/electron energy grid [GeV] (the species grid).
        z (float): redshift.
        photon_field: CMB(+EBL) target.
        eps, max_generations, tol: cascade integration controls.
        dz (float, optional): integration step. If given, T is the per-step
            (stiffness-split) operator; if None, the saturated full-path one.

    Returns:
        tuple(ndarray, ndarray): ``(T_gamma, T_electron)``, each (n, n).
        ``T_gamma`` maps an injected *photon* dN/dE to the escaping-photon
        dN/dE; ``T_electron`` maps an injected *electron/positron* dN/dE to the
        escaping-photon dN/dE. An electron does NOT γγ-pair-produce — it first
        fully IC-cools to a photon spectrum (``Mic``), which then enters the
        same photon cascade. So ``T_electron = T_gamma @ Mic @ diag(dE)``.
        Treating e± as photons (the old single-matrix path) inserts a spurious
        pair-production step; for the *saturated* high-E cascade the escaped
        shape converges either way, but the channels differ for low-E e±
        injection (e.g. Bethe-Heitler / π±→μ→e photo-hadronic electrons that
        are already near or below E_abs), so we keep them separate.
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
    if dz is None:
        # Saturated single-shot: full-path smooth escape exp(-tau_gg(E,z)).
        tau_E = tau_gg(E, z, photon_field, eps_min=1e-14, eps_max=1e-7)
        P_esc = np.exp(-tau_E)[:, None]
    else:
        # Per-z-step (stiffness-split) escape: Δτ_step = (dτ_gg/dl)·Δl_step.
        # Stiff (Δτ_step≫1) cascades this step; non-stiff (Δτ_step≪1) is carried
        # by ETD2 to the next step → pile-up forms at the evolving z→0 horizon.
        from prince_cr.cascade.opacity import _kernel_per_length
        from prince_cr.cosmology import H

        eps_o = np.logspace(-12, -7, 256)
        mu = np.linspace(-1.0, 1.0, 128)
        dtau_dl = np.array(
            [_kernel_per_length(float(Ei), z, photon_field, eps_o, mu) for Ei in E]
        )
        dl_step = C_CM * abs(dz) / ((1.0 + z) * H(z))  # proper length per step [cm]
        P_esc = np.exp(-dtau_dl * dl_step)[:, None]

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
    # Electron channel: cool to photons (Mic @ ·dE) then run the photon cascade.
    T_electron = escaped @ (Mic * dE[None, :])
    return escaped, T_electron


def _ic_single_scatter_matrices(E, eps, n_eps):
    """Build the per-scatter inverse-Compton operators (KINETIC cascade).

    Returns (Sg, De, R) where, for an electron at ``E[j]`` that makes ONE IC
    scatter:
      * ``Sg[i, j]`` = emitted-photon spectrum dN/dE_gamma at ``E[i]`` per
        scatter (normalized to exactly one photon: ``sum_i Sg[i,j] dE_i = 1``),
      * ``De[i, j]`` = degraded-electron spectrum dN/dE_e' at ``E[i]`` per
        scatter (the electron recoils to ``E_e' = E[j] - E_gamma``; one
        electron out per scatter),
      * ``R[j]`` = IC scattering rate [s^-1] (for reference; the generation
        iteration is per-scatter so the absolute rate cancels).

    Unlike :func:`cooled_ic_photon_matrix` (continuous full-cooling in one
    step), these let the cascade follow the electron through discrete scatters
    so the hard (high-E) IC photons are emitted *before* the electron cools and
    can pair-produce again — building the multi-generation E_X..E_abs plateau.
    """
    dE = np.gradient(E)
    n = E.size
    # emis[i,j] = dN/(dt dE_i) for e at E[j] — field-factored cached kernel.
    emis = _ic_emission_matrix(E, E, eps, n_eps)
    R = (emis * dE[:, None]).sum(axis=0)          # scatters/s per electron
    Sg = np.where(R[None, :] > 0, emis / np.where(R > 0, R, 1.0)[None, :], 0.0)
    # Degraded electron: each emitted photon E_gamma=E[k] (number weight
    # w_k = Sg[k,j]*dE_k) recoils the electron to E_e' = E[j] - E[k]. Deposit
    # w_k onto the grid with an energy-conserving cloud-in-cell split between
    # the two bracketing bins (conserves both electron NUMBER and ENERGY; a
    # plain interpolation leaks ~8% per scatter near E_e'~E[j] where the log
    # grid is coarse).
    De = np.zeros((n, n))
    logE = np.log(E)
    for j in range(n):
        if R[j] <= 0:
            continue
        for k in range(n):
            w = Sg[k, j] * dE[k]
            if w <= 0:
                continue
            Eep = E[j] - E[k]                     # degraded electron energy
            if Eep <= E[0]:
                continue
            b = int(np.searchsorted(E, Eep) - 1)
            b = min(max(b, 0), n - 2)
            # linear CIC: f*E[b] + (1-f)*E[b+1] = Eep  -> conserves number AND energy
            f = (E[b + 1] - Eep) / (E[b + 1] - E[b])
            f = min(max(f, 0.0), 1.0)
            De[b, j] += w * f / dE[b]
            De[b + 1, j] += w * (1.0 - f) / dE[b + 1]
    return Sg, De, R


def kinetic_cascade_transfer(E, z, photon_field, eps=None, max_scatter=4000,
                             tol=1e-4, dz=None, cool_floor_frac=0.1):
    """KINETIC single-scatter EM-cascade transfer matrix (photon channel).

    Same interface/return as :func:`cascade_transfer_matrix` (``T_gamma``,
    ``T_electron``), but the inverse-Compton step is **per-scatter** with
    explicit electron degradation, rather than the cooled single-step
    (``cooled_ic_photon_matrix``). This reproduces the multi-generation
    development that fills the universal E_X..E_abs plateau (the cooled path
    undershoots it — see the Kalashev Fig 2 comparison).

    Each iteration: photons escape (``exp(-tau)``) or pair-produce -> electrons;
    every electron makes ONE IC scatter -> one photon (``Sg``) + one degraded
    electron (``De``); emitted photons re-enter the photon pool, degraded
    electrons scatter again. Iterates until the in-box electron+photon energy
    falls below ``tol`` of the injected energy. ``cool_floor_frac``: electrons
    whose mean IC photon is below ``cool_floor_frac * E`` no longer pump the
    >E_abs cascade meaningfully and are dumped via the cooled integral to keep
    the iteration count bounded (exact for the soft, escaping tail).
    """
    if eps is None:
        eps = np.logspace(-15, -9, 400)
    n_eps = np.asarray(photon_field.get_photon_density(eps, z), dtype="double")
    n_eps = np.where(np.isfinite(n_eps) & (n_eps > 0), n_eps, 0.0)
    n = E.size
    dE = np.gradient(E)

    P = _energy_conserving_matrix(pair_matrix(E, E, eps, n_eps), E, E)
    Sg, De, R = _ic_single_scatter_matrices(E, eps, n_eps)
    Mic = _energy_conserving_matrix(cooled_ic_photon_matrix(E, E, eps, n_eps), E, E)

    if dz is None:
        tau_E = tau_gg(E, z, photon_field, eps_min=1e-14, eps_max=1e-7)
        P_esc = np.exp(-tau_E)[:, None]
    else:
        from prince_cr.cascade.opacity import _kernel_per_length
        from prince_cr.cosmology import H
        eps_o = np.logspace(-12, -7, 256)
        mu = np.linspace(-1.0, 1.0, 128)
        dtau_dl = np.array(
            [_kernel_per_length(float(Ei), z, photon_field, eps_o, mu) for Ei in E]
        )
        dl_step = C_CM * abs(dz) / ((1.0 + z) * H(z))
        P_esc = np.exp(-dtau_dl * dl_step)[:, None]

    # mean emitted-photon energy per electron (for the cool-floor dump test)
    Egam_mean = (E[:, None] * Sg * dE[:, None]).sum(axis=0)  # <E_gamma>(E_j)

    G = np.eye(n)                  # injected photon delta per column
    e = np.zeros((n, n))           # electron pool
    escaped = np.zeros((n, n))
    E_in = E * dE
    tot0 = E_in @ G
    for _ in range(max_scatter):
        # photons: escape or interact (pair-produce)
        escaped += P_esc * G
        interacting = G - P_esc * G
        e = e + P @ (interacting * dE[:, None])
        # split electrons: "hot" (still pump the >E_abs cascade) vs "cold" (dump)
        cold = (Egam_mean < cool_floor_frac * E) | (R <= 0)
        e_cold = e * cold[:, None]
        e_hot = e * (~cold)[:, None]
        # cold electrons fully cool -> escaping photons (exact soft tail)
        G_cold = Mic @ (e_cold * dE[:, None])
        # hot electrons make one scatter -> one photon + degraded electron
        G_hot = Sg @ (e_hot * dE[:, None])
        e = De @ (e_hot * dE[:, None])
        G = G_hot + G_cold
        # cold-dumped photons escape immediately (they are below the horizon)
        escaped += P_esc * G_cold
        G = G - P_esc * G_cold     # avoid double-counting cold escape next round
        in_box = (E_in @ G) + (E_in @ e)
        if np.max(in_box / tot0) < tol:
            escaped += P_esc * G
            break
    T_electron = escaped @ (Mic * dE[None, :])
    return escaped, T_electron


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
    # Field-factored emission matrix + loss vector (cached field-free kernels
    # contracted with n_eps); bit-exact vs the per-column scalar builders.
    loss = _ic_loss_vector(E_e_grid, eps, n_eps)
    emis = _ic_emission_matrix(E_gamma, E_e_grid, eps, n_eps)   # (ng, ne)
    M = np.zeros((ng, ne))
    for j in range(ne):
        # integrate emission(E_gamma, E') / loss(E') over E' from E_gamma up to E_e[j]
        for i in range(ng):
            # γ=1 clamp: electrons at/below m_e have loss<=0 (ic_energy_loss_rate
            # returns 0 there); mask them out of the cooling integral so the
            # 1/loss division stays finite (no NaN when the grid reaches m_e).
            sel = (E_e_grid >= E_gamma[i]) & (E_e_grid <= E_e_grid[j]) & (loss > 0)
            if np.count_nonzero(sel) >= 2:
                integrand = emis[i, sel] / loss[sel]
                M[i, j] = trapz(integrand, E_e_grid[sel])
    return M


def pair_matrix(E_e, E_gamma_grid, eps, n_eps):
    """Matrix P[i, j] = dN_e(E_e_i) per photon at E_gamma_grid[j] (e+ + e-).

    Field-factored: contracts the cached field-free pair kernel with n_eps
    (matches the per-column pair_injection_spectrum to ~1.6e-14)."""
    A, pref = _pair_kernel(E_e, E_gamma_grid, eps)
    return pref[None, :] * np.tensordot(A, n_eps, axes=([2], [0]))


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
    """DEPRECATED — DO NOT USE. Buggy/approximate EM-cascade helper.

    .. deprecated::
        ``run_cascade`` carries the energy_ratio≈2 normalization bug and an
        under-developed cascade (it undershoots the low-energy tail and cuts
        off too sharply below the gamma-gamma horizon — see the Kalashev Fig 2
        comparison). **Use** :func:`cascade_transfer_matrix` instead: build
        ``T_gamma`` and apply it to the injection (a mono-energetic source is
        the column of ``T_gamma`` at the injection energy). That path is the
        validated, energy-conserving cascade engine. This function is kept only
        to avoid breaking old imports and raises a warning on call.

    Develop a saturated EM cascade and return the escaping (observed-at-z=0)
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
    import warnings
    warnings.warn(
        "run_cascade is DEPRECATED and buggy (energy_ratio~2 normalization; "
        "under-developed cascade that undershoots wings / cuts off below the "
        "gamma-gamma horizon). Use cascade_transfer_matrix instead.",
        DeprecationWarning, stacklevel=2,
    )
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
