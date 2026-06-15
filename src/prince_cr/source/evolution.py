"""prince_cr.source.evolution — single-zone time-domain evolution (Phase 2).

A minimal, correctness-first ETD2 single-zone solver for in-source
multi-messenger physics (the AM3-equivalent). The independent variable is
**time t** (no cosmology / Hubble), the operator M carries units of s^-1, and
continuous energy losses (synchrotron, IC, adiabatic) enter as a
**conservative upwind advection** operator in M — validated in
`runs/2026-06-15_am3-insource` (iteration 1) to be positivity-preserving and
number-conserving for pure cooling, with Chang-Cooper reducing to upwind in
that limit; ETD2's exponential step handles the cooling-stiffness, so no
implicit tridiagonal solve is needed.

Phase-2 scope (this file, growing):
  it3  electrons: synchrotron + (external) IC cooling, injection, escape;
       steady state + ETD2 march. Validated vs the analytic cooling break.
  it4  synchrotron *emission* -> self-consistent photon field (SSC).
  it5  energy-dependent escape spectrum.

The solver reuses the ETD2 phi-function *formulation* (n_{k+1} = e^{Mh} n_k +
h phi1(Mh) (M n_k + Q) for the constant-coefficient step; phi1(z)=(e^z-1)/z)
but is implemented standalone here — the single-zone system is small (~few
species x energy grid), so we favour a clean dense/sparse expm over the
production propagation solver's z-coupled, MKL/cupy-optimised machinery.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply, spsolve
from scipy.special import kv

# --- physical constants (CGS) ---
_SIGMA_T = 6.6524587321e-25      # cm^2
_C = 2.99792458e10               # cm/s
_ME_C2_ERG = 8.1871057769e-7     # m_e c^2 in erg
_ME_G = 9.1093837015e-28         # g
_E_ESU = 4.80320471e-10          # elementary charge (esu)
_H_ERG_S = 6.62607015e-27        # Planck (erg s)
_NU_B_PER_G = _E_ESU / (2.0 * np.pi * _ME_G * _C)   # ν_B / B  [Hz/Gauss]


def _build_fsync_table(n_t=4000, x_lo=1e-5, x_hi=60.0):
    """Tabulate the synchrotron function F(x) = x ∫_x^∞ K_{5/3}(t) dt."""
    t = np.logspace(np.log10(x_lo), np.log10(x_hi), n_t)
    k = kv(5.0 / 3.0, t)
    dI = 0.5 * (k[1:] + k[:-1]) * np.diff(t)          # trapz per interval
    tail = np.concatenate([np.cumsum(dI[::-1])[::-1], [0.0]])  # ∫_{t_i}^∞ K dt
    return t, np.maximum(t * tail, 1e-300)            # x grid, F(x) (floored >0)


_FSYNC_X, _FSYNC_F = _build_fsync_table()


def synchrotron_F(x):
    """F(x) via log-interp of the K_{5/3} table; F≈2.149 x^{1/3} (x≪1), 0 (x≫1)."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    m = (x > _FSYNC_X[0]) & (x < _FSYNC_X[-1])
    out[m] = np.exp(np.interp(np.log(x[m]), np.log(_FSYNC_X), np.log(_FSYNC_F)))
    lo = x <= _FSYNC_X[0]
    out[lo] = 2.1495282 * x[lo] ** (1.0 / 3.0)        # small-x asymptote
    return out


def _trapz_grid(n_bins, lo, hi):
    """Log-spaced cell centres + interfaces + widths for a 1-D FV grid."""
    edges_ln = np.linspace(np.log(lo), np.log(hi), n_bins + 1)
    g_if = np.exp(edges_ln)
    g = np.sqrt(g_if[:-1] * g_if[1:])          # geometric cell centres
    dg = np.diff(g_if)                          # cell widths (volume)
    return g, g_if, dg


class SingleZoneSolver:
    """Single-zone electron (γ) evolution under continuous cooling + injection
    + escape.  γ = E/m_e c² (dimensionless Lorentz factor).

    Parameters
    ----------
    gamma_lo, gamma_hi, n_bins : energy grid (log γ).
    B_Gauss : tangled magnetic field (synchrotron).
    u_rad_erg_cm3 : external radiation energy density for IC cooling (Thomson).
        SSC (self-consistent) is wired in it4; here it is a fixed external term.
    t_esc_s : energy-independent escape time (None = no escape). it5 generalises.
    """

    def __init__(self, gamma_lo=1.0, gamma_hi=1e8, n_bins=256,
                 B_Gauss=1.0, u_rad_erg_cm3=0.0, t_esc_s=None):
        self.g, self.g_if, self.dg = _trapz_grid(n_bins, gamma_lo, gamma_hi)
        self.n_bins = n_bins
        self.B = B_Gauss
        self.u_rad = u_rad_erg_cm3
        self.t_esc = t_esc_s
        # cooling normalisation β:  γ̇ = -β γ²   [1/s],  β = (4/3) σ_T c U / (m_e c²)
        u_B = B_Gauss ** 2 / (8.0 * np.pi)                      # erg/cm^3
        self._beta_syn = (4.0 / 3.0) * _SIGMA_T * _C * u_B / _ME_C2_ERG
        self._beta_ic = (4.0 / 3.0) * _SIGMA_T * _C * u_rad_erg_cm3 / _ME_C2_ERG

    # --- cooling rate at interfaces (Thomson; KN suppression added in it4) ---
    def gdot_if(self):
        """γ̇ at cell interfaces [1/s], negative (energy loss)."""
        beta = self._beta_syn + self._beta_ic
        return -beta * self.g_if ** 2

    def t_cool(self, gamma):
        beta = self._beta_syn + self._beta_ic
        return 1.0 / (beta * np.asarray(gamma))

    # --- operator assembly: conservative upwind advection (cooling) + escape ---
    def cooling_operator(self):
        """Sparse M [1/s] with dn/dt = M n for the cooling advection (+escape).

        Conservative finite-volume, upwind flux (γ̇<0 ⇒ donor = upper cell),
        open outflow at the low-γ boundary. Positivity- and number-preserving
        (iteration-1 result)."""
        N = self.n_bins
        g, dg = self.g, self.dg
        gdot = self.gdot_if()
        rows, cols, vals = [], [], []

        def add(i, j, v):
            rows.append(i); cols.append(j); vals.append(v)

        for k in range(1, N):                  # interface k between cell k-1, k
            a = gdot[k]                         # <0
            # upwind: a<0 ⇒ flux carries the UPPER cell (k); coeff of n_k = a
            #   F_k = a * n_k  (a<0)  ;  donor is the higher-γ cell
            ch = a                              # coeff of n_k in F_k
            cl = 0.0                            # coeff of n_{k-1} in F_k
            # cell k: interface k is LOWER boundary -> dn_k/dt += F_k/dg[k]
            add(k, k, ch / dg[k])
            add(k, k - 1, cl / dg[k])
            # cell k-1: interface k is UPPER boundary -> dn_{k-1}/dt -= F_k/dg[k-1]
            add(k - 1, k, -ch / dg[k - 1])
            add(k - 1, k - 1, -cl / dg[k - 1])
        # outflow at lowest interface (k=0): F_0 = gdot[0]*n_0 leaves the domain
        add(0, 0, gdot[0] / dg[0])
        M = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
        if self.t_esc is not None:
            M = M - sp.identity(N, format="csr") / self.t_esc
        return M

    # --- injection vector from a (broken) power law ---
    def injection_powerlaw(self, Q0, p, gamma_min, gamma_max):
        """Q(γ) = Q0 γ^-p for γ_min<γ<γ_max, else 0  [particles / cm^3 / s / (unit γ)]."""
        Q = np.where((self.g >= gamma_min) & (self.g <= gamma_max),
                     Q0 * self.g ** (-p), 0.0)
        return Q

    # --- synchrotron emission (it4: gap item 1) ---
    def nu_c(self, gamma):
        """Synchrotron critical frequency ν_c(γ) = (3/2) γ² ν_B  [Hz] (sinα=1)."""
        return 1.5 * np.asarray(gamma) ** 2 * (_NU_B_PER_G * self.B)

    def synchrotron_sed(self, n_e, nu=None):
        """Volume synchrotron emissivity j(ν) [erg s^-1 cm^-3 Hz^-1] from the
        electron spectrum ``n_e`` (per unit γ, cm^-3, on self.g).

            j(ν) = ∫ dγ n(γ) P(ν,γ),
            P(ν,γ) = (√3 e³ B / m_e c²) F(ν/ν_c(γ)).

        Returns (nu, j). If ``nu`` is None a log grid spanning the band is used.
        """
        n_e = np.asarray(n_e, dtype=float)
        if nu is None:
            nu_lo = float(self.nu_c(self.g[0])) * 1e-3
            nu_hi = float(self.nu_c(self.g[-1])) * 10.0
            nu = np.logspace(np.log10(nu_lo), np.log10(nu_hi), 256)
        nu = np.asarray(nu, dtype=float)
        P0 = np.sqrt(3.0) * _E_ESU ** 3 * self.B / _ME_C2_ERG     # erg/s/Hz prefactor
        nuc = self.nu_c(self.g)                                   # (Ng,)
        x = nu[:, None] / nuc[None, :]                            # (Nnu, Ng)
        Fx = synchrotron_F(x)
        # ∫ dγ  ->  sum over cells with width dg
        j = P0 * (Fx * (n_e * self.dg)[None, :]).sum(axis=1)      # (Nnu,)
        return nu, j

    # --- inverse-Compton emission SED (it4c: the Compton hump, with KN) ---
    def ic_sed(self, n_e, eps_t, n_ph_t, eps_out):
        """Isotropic inverse-Compton photon production spectrum
        Q_IC(ε1) [photons cm^-3 s^-1 erg^-1] from electrons ``n_e`` (per γ, on
        self.g) up-scattering a target photon field (``eps_t`` [erg],
        ``n_ph_t`` [cm^-3 erg^-1]) onto output energies ``eps_out`` [erg].

        Full Jones(1968)/Blumenthal-Gould(1970) kernel (includes Klein-Nishina):
            dN/dt/dε1 = (3 σ_T c)/(4 γ²) (n_ph(ε)/ε) G(q,Γ),
            Γ = 4 ε γ / (m_e c²),  q = ε1 / (Γ (γ m_e c² − ε1)),
            G = 2q ln q + (1+2q)(1−q) + ½(Γq)²(1−q)/(1+Γq),  1/(4γ²) ≤ q ≤ 1.
        """
        g = self.g
        gm = g * _ME_C2_ERG                                   # electron energy [erg]
        # broadcast (out, γ, target): keep grids modest for the triple loop
        e1 = np.asarray(eps_out)[:, None, None]
        gg = g[None, :, None]
        gmc = gm[None, :, None]
        et = np.asarray(eps_t)[None, None, :]
        nph = np.asarray(n_ph_t)[None, None, :]
        Gam = 4.0 * et * gg / _ME_C2_ERG
        denom = Gam * (gmc - e1)
        with np.errstate(divide="ignore", invalid="ignore"):
            q = e1 / denom
            G = (2.0 * q * np.log(np.clip(q, 1e-300, None)) + (1.0 + 2.0 * q) * (1.0 - q)
                 + 0.5 * (Gam * q) ** 2 * (1.0 - q) / (1.0 + Gam * q))
            kern = (3.0 * _SIGMA_T * _C) / (4.0 * gg ** 2) * (nph / et) * G
        valid = (q > 1.0 / (4.0 * gg ** 2)) & (q <= 1.0) & (e1 < gmc) & np.isfinite(kern)
        kern = np.where(valid, kern, 0.0)
        # integrate over target ε (axis 2) then over γ (axis 1)
        de_t = np.gradient(np.asarray(eps_t))
        dg = self.dg
        per_gamma = np.sum(kern * de_t[None, None, :], axis=2)        # (out, γ)
        Q = np.sum(per_gamma * (n_e * dg)[None, :], axis=1)           # (out,)
        return Q

    def synchrotron_photon_density(self, n_e, R_cm, nu=None):
        """Synchrotron photon NUMBER density n_ph(ε) [cm^-3 erg^-1] and ε [erg]
        in the zone (residence ~R/c): n_ph(ε) = (R/c) j(ν)/(hν) / h."""
        nu, j = self.synchrotron_sed(n_e, nu)
        eps = _H_ERG_S * nu                                  # erg
        n_ph = (R_cm / _C) * j / (_H_ERG_S * nu) / _H_ERG_S  # cm^-3 erg^-1
        return eps, n_ph

    def synchrotron_energy_density(self, n_e, R_cm):
        """Synchrotron photon energy density in the zone [erg/cm^3].

        U_syn ≈ (R/c) ∫ j(ν) dν  (photon residence time ~ R/c for a zone of
        size R; the standard one-zone estimate)."""
        nu, j = self.synchrotron_sed(n_e)
        L_vol = float(np.trapezoid(j, nu))           # erg s^-1 cm^-3
        return (R_cm / _C) * L_vol

    def set_ic_target(self, u_rad_erg_cm3):
        """Set/replace the (Thomson) IC target radiation energy density."""
        self.u_rad = u_rad_erg_cm3
        self._beta_ic = (4.0 / 3.0) * _SIGMA_T * _C * u_rad_erg_cm3 / _ME_C2_ERG

    @property
    def u_B(self):
        return self.B ** 2 / (8.0 * np.pi)

    def solve_ssc(self, Q, R_cm, tol=1e-4, max_iter=60, relax=0.5):
        """Self-consistent synchrotron-self-Compton fixed point.

        Iterate: electrons (steady state under syn+IC cooling) → synchrotron
        U_syn → IC cooling → … until U_syn converges. IC is Thomson here
        (KN refinement is it4c). Returns (n_e, info)."""
        u = 0.0
        hist = []
        for it in range(max_iter):
            self.set_ic_target(u)
            n_e = self.steady_state(Q)
            u_new = self.synchrotron_energy_density(n_e, R_cm)
            hist.append(u_new)
            denom = max(u_new, 1e-300)
            if abs(u_new - u) < tol * denom:
                u = u_new
                self.set_ic_target(u)
                n_e = self.steady_state(Q)
                return n_e, {"u_syn": u, "iters": it + 1, "converged": True,
                             "compton_Y": u / self.u_B, "hist": hist}
            u = (1.0 - relax) * u + relax * u_new
        return n_e, {"u_syn": u, "iters": max_iter, "converged": False,
                     "compton_Y": u / self.u_B, "hist": hist}

    # --- solvers ---
    def steady_state(self, Q, M=None):
        """Solve M n = -Q  (continuous injection balanced by cooling+escape)."""
        if M is None:
            M = self.cooling_operator()
        return spsolve(M.tocsc(), -np.asarray(Q, dtype=float))

    def evolve(self, n0, Q, t_end, dt, M=None):
        """Exponential-Euler (ETD1) march of dn/dt = M n + Q to t_end.

        Splits M = Λ + N with Λ = diag(M) (the stiff cooling diagonal) treated
        EXACTLY via the scalar exp, and the off-diagonal N applied as one SpMV:

            n_{k+1} = e^{Λh} n_k + φ1(Λh) · h · (N n_k + Q),   φ1(z)=(e^z-1)/z.

        This is the cheap, stiffness-stable core (cost ~ one SpMV + vector ops
        per step, NOT a global matrix exponential) — the same exp(h·diag)+SpMV
        structure the production ETD2 propagation solver uses. Large dt is fine;
        the fast high-γ modes are handled by e^{Λh}, not resolved step-by-step.
        ETD2's second-order φ2 term is a later refinement; ETD1 suffices to
        relax to the steady state."""
        if M is None:
            M = self.cooling_operator()
        M = M.tocsr()
        n = np.asarray(n0, dtype=float).copy()
        Q = np.asarray(Q, dtype=float)
        lam = M.diagonal()
        Noff = (M - sp.diags(lam)).tocsr()           # off-diagonal part
        nsteps = max(1, int(round(t_end / dt)))
        h = t_end / nsteps
        eL = np.exp(lam * h)
        z = lam * h
        phi1 = np.where(np.abs(z) > 1e-8, np.expm1(z) / np.where(z == 0, 1.0, z), 1.0)
        for _ in range(nsteps):
            n = eL * n + phi1 * h * (Noff @ n + Q)
        return n
