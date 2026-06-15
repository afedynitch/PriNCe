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

# --- physical constants (CGS) ---
_SIGMA_T = 6.6524587321e-25      # cm^2
_C = 2.99792458e10               # cm/s
_ME_C2_ERG = 8.1871057769e-7     # m_e c^2 in erg
_ME_G = 9.1093837015e-28         # g


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
