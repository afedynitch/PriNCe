"""prince_cr.source.evolution — single-zone time-domain evolution (Phase 2).

Single-zone in-source multi-messenger solver (the AM3-equivalent). The
independent variable is **time t** (no cosmology / Hubble), the operator M
carries units of s^-1, and continuous energy losses (synchrotron, IC,
adiabatic) enter as a **conservative upwind advection** operator in M —
validated in `runs/2026-06-15_am3-insource` to be positivity-preserving and
number-conserving for pure cooling (Chang-Cooper reduces to upwind in that
limit).

Three ways to reach steady state are provided, in increasing fidelity to the
production stack:
  * :meth:`SingleZoneSolver.steady_state`        — direct sparse solve, ``spsolve(M, -Q)``.
  * :meth:`SingleZoneSolver.evolve`              — ETD1 (exponential Euler) t-march.
  * :meth:`SingleZoneSolver.steady_state_etd2`   — the GENUINE ETD2: marches with
    PriNCe's production stepper :func:`prince_cr.solvers.etd2.etd2_step`
    (Cox-Matthews exponential RK2, the same integrator the propagation solver
    uses). Validated to match ``steady_state`` to machine precision and the
    analytic cooling break.

(Historical note: earlier revisions of this docstring called the whole module an
"ETD2 solver" while only ETD1 + spsolve were implemented — corrected 2026-06-18
when the real ETD2 path was wired to `solvers.etd2`.)

Phase-2 scope (this file, growing):
  it3  electrons: synchrotron + (external) IC cooling, injection, escape.
  it4  synchrotron *emission* -> self-consistent photon field (SSC).
  it5  energy-dependent escape spectrum.
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


def _build_fsync_table(n_t=6000, x_lo=1e-5, x_hi=300.0):
    """Tabulate the synchrotron function F(x) = x ∫_x^∞ K_{5/3}(t) dt.

    ``x_hi`` reaches deep into the exp tail (F∝x^{1/2}e^{-x}; F(300)~1e-128) so
    the kernel does NOT hard-zero in any realistic SED. A short x_hi (was 60)
    truncated the tail and put a derivative DISCONTINUITY in synchrotron cutoffs
    (the emitter crossing x_hi drops to 0); extending it makes the rolloff smooth."""
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


def _build_gavg_table(n_y=600, y_lo=1e-4, y_hi=100.0, n_alpha=128):
    """Pitch-angle-averaged synchrotron kernel for an ISOTROPIC electron
    distribution:  G(y) = ∫_0^{π/2} sin²α F(y/sinα) dα,  y = ν/ν_c0,
    ν_c0 = (3/2)γ²ν_B (the sinα=1 critical frequency).  The extra sinα (from
    B_perp = B sinα) and the (½ sinα)dα solid-angle weight combine to sin²α.

    ``y_hi`` extended (was 20) so G(y) carries the smooth exp tail to where it is
    negligible (G(100)~1e-43) instead of hard-zeroing — removes the derivative
    discontinuity in synchrotron-cutoff SEDs (see _build_fsync_table)."""
    y = np.logspace(np.log10(y_lo), np.log10(y_hi), n_y)
    a = np.linspace(1e-3, np.pi / 2, n_alpha)
    sa = np.sin(a)
    G = np.trapezoid(sa[None, :] ** 2 * synchrotron_F(y[:, None] / sa[None, :]), a, axis=1)
    return y, np.maximum(G, 1e-300)


_GAVG_Y, _GAVG_G = _build_gavg_table()


def synchrotron_Favg(y):
    """Pitch-angle-averaged synchrotron kernel G(y), y=ν/ν_c0 (isotropic e±)."""
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y)
    m = (y > _GAVG_Y[0]) & (y < _GAVG_Y[-1])
    out[m] = np.exp(np.interp(np.log(y[m]), np.log(_GAVG_Y), np.log(_GAVG_G)))
    lo = y <= _GAVG_Y[0]
    out[lo] = _GAVG_G[0] * (y[lo] / _GAVG_Y[0]) ** (1.0 / 3.0)   # x^{1/3} tail
    return out


def _trapz_grid(n_bins, lo, hi):
    """Log-spaced cell centres + interfaces + widths for a 1-D FV grid."""
    edges_ln = np.linspace(np.log(lo), np.log(hi), n_bins + 1)
    g_if = np.exp(edges_ln)
    g = np.sqrt(g_if[:-1] * g_if[1:])          # geometric cell centres
    dg = np.diff(g_if)                          # cell widths (volume)
    return g, g_if, dg


class CompositePhotonField:
    """The ONE common in-zone photon field = external (fixed) + internal
    (self-consistently evolved), and the single target for every process
    (IC, pγ, γγ, SSA). Unifies the in-source and propagation cases:

    - ``feedback=False`` → only the external/fixed component is returned →
      recovers the classical fixed-target behaviour (e.g. CMB+EBL propagation,
      where the source's own photons are negligible). One mechanism, feedback
      switched off.
    - ``feedback=True``  → external + the evolved internal photons (synchrotron
      + IC + cascade + π⁰→γγ), updated each iteration via :meth:`set_internal`
      → the self-consistent in-source case (the source photons dominate).

    Exposes ``get_photon_density(E_GeV, z)`` [GeV⁻¹ cm⁻³] + ``E_min_GeV`` /
    ``E_max_GeV`` (the union span), so it is a drop-in for the cascade kernels
    and the source-rate primitives.
    """

    def __init__(self, external=None, feedback=True):
        self.external = external
        self.feedback = bool(feedback)
        self._lgE = self._lgn = None
        self._imin = self._imax = None
        ext_lo = getattr(external, "E_min_GeV", None)
        ext_hi = getattr(external, "E_max_GeV", None)
        self._ext_lo, self._ext_hi = ext_lo, ext_hi
        self._refresh_bounds()

    def _refresh_bounds(self):
        los = [b for b in (self._ext_lo, self._imin if self.feedback else None) if b]
        his = [b for b in (self._ext_hi, self._imax if self.feedback else None) if b]
        self.E_min_GeV = min(los) if los else 1e-20
        self.E_max_GeV = max(his) if his else 1e14

    def set_internal(self, eps_GeV, n_GeV):
        """Set/replace the evolved internal photon density n(E) [GeV⁻¹ cm⁻³]."""
        eps_GeV = np.asarray(eps_GeV, float); n_GeV = np.asarray(n_GeV, float)
        m = (n_GeV > 0) & np.isfinite(n_GeV) & (eps_GeV > 0)
        if m.sum() >= 2:
            o = np.argsort(eps_GeV[m])
            self._lgE = np.log(eps_GeV[m][o]); self._lgn = np.log(n_GeV[m][o])
            self._imin, self._imax = float(eps_GeV[m].min()), float(eps_GeV[m].max())
        self._refresh_bounds()

    def get_photon_density(self, E, z=0.0):
        E = np.atleast_1d(np.asarray(E, float)); out = np.zeros_like(E)
        if self.external is not None:
            out = out + np.asarray(self.external.get_photon_density(E, z), float)
        if self.feedback and self._lgE is not None:
            w = (E >= self._imin) & (E <= self._imax)
            out[w] = out[w] + np.exp(np.interp(np.log(E[w]), self._lgE, self._lgn))
        return out


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
                 B_Gauss=1.0, u_rad_erg_cm3=0.0, t_esc_s=None,
                 mass_g=_ME_G, charge=1.0):
        self.g, self.g_if, self.dg = _trapz_grid(n_bins, gamma_lo, gamma_hi)
        self.n_bins = n_bins
        self.B = B_Gauss
        self.u_rad = u_rad_erg_cm3
        self.t_esc = t_esc_s          # scalar, callable(γ)->t_esc, array, or None
        # generic charged particle (default electron); leptonic work unchanged.
        self.mass = mass_g
        self.charge = charge
        self._mc2 = mass_g * _C ** 2                            # rest energy [erg]
        # Thomson xsec ∝ (q²/m)²  ⇒  σ_T,p = σ_T q⁴ (m_e/m)²
        self._sigma_T = _SIGMA_T * charge ** 4 * (_ME_G / mass_g) ** 2
        # synchrotron/IC cooling  γ̇=-βγ²  [1/s], β=(4/3)σ_T,p c U/(m c²)
        # ⇒ β ∝ q⁴ (m_e/m)³ relative to the electron.
        u_B = B_Gauss ** 2 / (8.0 * np.pi)                      # erg/cm^3
        self._beta_syn = (4.0 / 3.0) * self._sigma_T * _C * u_B / self._mc2
        self._beta_ic = (4.0 / 3.0) * self._sigma_T * _C * u_rad_erg_cm3 / self._mc2
        # gyro/critical-frequency scale ν_B = q e B/(2π m c)  ∝ q/m
        self._nu_B = charge * _E_ESU * B_Gauss / (2.0 * np.pi * mass_g * _C)
        # extra (non-∝γ²) energy-loss terms at interfaces, e.g. pγ / BH (it6b)
        self._gdot_extra_if = np.zeros_like(self.g_if)

    @classmethod
    def on_grid(cls, g, g_if, dg, **kw):
        """Construct on an EXPLICIT cell-centre / interface / width grid instead
        of the internal log ``_trapz_grid``. Used to put the validated radiative
        kernels onto the PriNCe ``em_grid`` (γ_e = E_e/m_e c² on its bin centres)
        so they assemble into the native combined operator. ``g``/``g_if``/``dg``
        are the centres (len N), interfaces (len N+1), and widths (len N) in γ_e.
        All physics (cooling, synchrotron/IC emission, SSA) is grid-agnostic."""
        g = np.asarray(g, float); g_if = np.asarray(g_if, float); dg = np.asarray(dg, float)
        self = cls(gamma_lo=float(g[0]), gamma_hi=float(g[-1]), n_bins=g.size, **kw)
        self.g, self.g_if, self.dg = g, g_if, dg
        self.n_bins = g.size
        self._gdot_extra_if = np.zeros_like(self.g_if)
        return self

    # --- cooling rate at interfaces (Thomson syn+IC ∝γ² + extra terms) ---
    def gdot_if(self):
        """γ̇ at cell interfaces [1/s], negative (energy loss).
        = -(β_syn+β_ic)γ²  +  Σ extra γ̇ terms (pγ, BH, ...)."""
        beta = self._beta_syn + self._beta_ic
        return -beta * self.g_if ** 2 + self._gdot_extra_if

    def add_pgamma_cooling(self, photon_field, sigma_eps_GeV, A_mass_GeV=None):
        """Add proton/nucleus pγ photo-meson continuous cooling to the operator,
        computed from the validated `rates.photonuclear_cool_inv` on a source
        ``photon_field`` (any object with get_photon_density(eps,z)) and a
        cross-section callable ``sigma_eps_GeV``. γ̇_pγ(γ) = -γ · t_cool⁻¹(E_A)."""
        from prince_cr.source.rates import photonuclear_cool_inv
        if A_mass_GeV is None:
            A_mass_GeV = self._mc2 / (1.602176634e-3)        # erg -> GeV
        E_if = self.g_if * A_mass_GeV                        # E_A at interfaces [GeV]
        rate = np.array([photonuclear_cool_inv(E, A_mass_GeV, sigma_eps_GeV,
                                               photon_field) for E in E_if])
        self._gdot_extra_if = self._gdot_extra_if - self.g_if * rate   # dγ/dt = -γ/t
        return self

    def t_cool(self, gamma):
        beta = self._beta_syn + self._beta_ic
        return 1.0 / (beta * np.asarray(gamma))

    # --- escape (it5): energy-dependent t_esc(γ) + the escape spectrum ---
    def set_escape(self, t_esc):
        """Set the escape time: scalar (energy-independent), callable γ→t_esc(γ)
        (e.g. diffusive t_esc∝γ^-δ), an array on self.g, or None (no escape)."""
        self.t_esc = t_esc

    def t_esc_arr(self):
        """Per-γ escape time [s] on self.g (inf where no escape)."""
        te = self.t_esc
        if te is None:
            return np.full_like(self.g, np.inf)
        if callable(te):
            return np.asarray(te(self.g), dtype=float)
        return np.broadcast_to(np.asarray(te, dtype=float), self.g.shape).copy()

    def escape_spectrum(self, n_e):
        """Escaping-particle rate spectrum dṄ_esc/dγ/dV = n(γ)/t_esc(γ)
        [cm^-3 s^-1 (unit γ)^-1] — the source 'EscapeSpectrum' that the PRINCE
        propagation solver consumes as injection. Returns (γ, rate)."""
        return self.g, np.asarray(n_e, dtype=float) / self.t_esc_arr()

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
            M = M - sp.diags(1.0 / self.t_esc_arr())     # energy-dependent escape
        return M

    # --- injection vector from a (broken) power law ---
    def injection_powerlaw(self, Q0, p, gamma_min, gamma_max, cutoff_steepness=None):
        """Q(γ) = Q0 γ^-p  with a low edge at γ_min and a high-energy cutoff at
        γ_max  [particles cm^-3 s^-1 (unit γ)^-1].

        ``cutoff_steepness`` None → sharp top-hat (hard γ_max). A float ``s``
        → smooth exponential cutoff Q ∝ γ^-p exp(−(γ/γ_max)^s) above γ_min
        (matches AM3's `set_powerlaw_injection_parameters` cut-off, s=1)."""
        g = self.g
        if cutoff_steepness is None:
            return np.where((g >= gamma_min) & (g <= gamma_max), Q0 * g ** (-p), 0.0)
        Q = Q0 * g ** (-p) * np.exp(-(g / gamma_max) ** cutoff_steepness)
        return np.where(g >= gamma_min, Q, 0.0)

    # --- synchrotron emission (it4: gap item 1) ---
    def nu_c(self, gamma):
        """Synchrotron critical frequency ν_c(γ) = (3/2) γ² ν_B  [Hz] (sinα=1).
        ν_B ∝ q/m, so protons radiate at much lower ν than electrons."""
        return 1.5 * np.asarray(gamma) ** 2 * self._nu_B

    def synchrotron_sed(self, n_e, nu=None, pitch_avg=True):
        """Volume synchrotron emissivity j(ν) [erg s^-1 cm^-3 Hz^-1] from the
        electron spectrum ``n_e`` (per unit γ, cm^-3, on self.g).

            j(ν) = ∫ dγ n(γ) P(ν,γ),
            P(ν,γ) = (√3 e³ B / m_e c²) K(ν/ν_c0(γ)),  ν_c0=(3/2)γ²ν_B.

        ``pitch_avg`` (default): K = the isotropic pitch-angle-averaged kernel
        ⟨sin²α F(x/sinα)⟩ (correct for isotropic e±, lowers the effective peak);
        else K = F(x) (single sinα=1 electron). Returns (nu, j).
        """
        n_e = np.asarray(n_e, dtype=float)
        if nu is None:
            nu_lo = float(self.nu_c(self.g[0])) * 1e-3
            nu_hi = float(self.nu_c(self.g[-1])) * 10.0
            nu = np.logspace(np.log10(nu_lo), np.log10(nu_hi), 256)
        nu = np.asarray(nu, dtype=float)
        P0 = np.sqrt(3.0) * self.charge ** 3 * _E_ESU ** 3 * self.B / self._mc2  # erg/s/Hz
        nuc = self.nu_c(self.g)                                   # ν_c0 (sinα=1)
        x = nu[:, None] / nuc[None, :]                            # (Nnu, Ng)
        K = synchrotron_Favg(x) if pitch_avg else synchrotron_F(x)
        j = P0 * (K * (n_e * self.dg)[None, :]).sum(axis=1)       # (Nnu,)
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

    def synchrotron_absorption(self, n_e, nu):
        """Synchrotron self-absorption coefficient α_ν [cm^-1] (isotropic e±):

            α_ν = -(1/(8π m_e ν²)) ∫ dγ P(ν,γ) γ² ∂_γ[n(γ)/γ²],

        with P(ν,γ)=√3 e³ B/(m_e c²)·K(ν/ν_c0) the angle-averaged single-electron
        power. Drives the I_ν∝ν^{5/2} optically-thick turnover."""
        n_e = np.asarray(n_e, dtype=float)
        g = self.g
        P0 = np.sqrt(3.0) * self.charge ** 3 * _E_ESU ** 3 * self.B / self._mc2
        nuc = self.nu_c(g)
        K = synchrotron_Favg(np.asarray(nu)[:, None] / nuc[None, :])      # (Nnu,Ng)
        # ∂_γ(n/γ²) on the log grid
        ratio = n_e / g ** 2
        dratio = np.gradient(ratio, g)
        integ = K * (g ** 2 * dratio)[None, :]
        alpha = -(1.0 / (8.0 * np.pi * self.mass * np.asarray(nu) ** 2)) * P0 * \
            (integ * self.dg[None, :]).sum(axis=1)
        return np.clip(alpha, 0.0, None)               # absorption ≥ 0

    def synchrotron_photon_density(self, n_e, R_cm, nu=None, ssa=True):
        if ssa:
            nu_, j = self.synchrotron_sed(n_e, nu)
            alpha = self.synchrotron_absorption(n_e, nu_)
            tau = alpha * R_cm
            # uniform-sphere escape probability: (1-e^{-τ})/τ  (→1 thin, →1/τ thick)
            with np.errstate(divide="ignore", invalid="ignore"):
                esc = np.where(tau > 1e-6, (1.0 - np.exp(-tau)) / tau, 1.0)
            eps = _H_ERG_S * nu_
            n_ph = (R_cm / _C) * (j * esc) / (_H_ERG_S * nu_) / _H_ERG_S
            return eps, n_ph
        return self._syn_photon_density_thin(n_e, R_cm, nu)

    def _syn_photon_density_thin(self, n_e, R_cm, nu=None):
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

    def steady_state_etd2(self, Q, M=None, n0=None, dt=None, n_steps=600,
                          rtol=1e-7, check_every=5, return_info=False):
        """Steady state via PriNCe's PRODUCTION ETD2 stepper
        (:func:`prince_cr.solvers.etd2.etd2_step`, Cox-Matthews exponential
        RK2 — the SAME integrator the propagation solver uses), marching
        ``dn/dt = M n + Q`` to its fixed point.

        This is the genuine ETD2 path (the module's earlier `evolve` is only
        ETD1, `steady_state` is a direct spsolve). The stiff cooling+escape
        diagonal ``D = diag(M)`` is treated EXACTLY via ``exp(h·D)``; the upwind
        off-diagonal advection is the explicit SpMV (2 per stage). Because the
        fast high-γ modes are damped exactly, a step ``dt`` of order the slowest
        relaxation time (≈ escape time) is stable and reaches steady state in a
        handful of steps.

        Returns ``n`` (or ``(n, info)`` with the residual history if
        ``return_info``). Validated to match :meth:`steady_state` (spsolve) and
        the analytic cooling break (see tests)."""
        from prince_cr.solvers.etd2 import etd2_step, _step_buffers, split_operator

        if M is None:
            M = self.cooling_operator()
        M = M.tocsr()
        d, L_off = split_operator(M)
        Q = np.asarray(Q, dtype=float)
        n = (np.zeros_like(self.g) if n0 is None
             else np.asarray(n0, dtype=float).copy())
        bufs = _step_buffers(n.size, np)

        def apply_F(x, out):
            out[:] = L_off @ x
            out += Q

        if dt is None:
            # slowest relaxation ≈ 1/min|diag| (the escape floor at low γ); ETD
            # handles the faster diagonal modes exactly, so this step is stable.
            nz = np.abs(d[d != 0.0])
            dt = (1.0 / nz.min()) if nz.size else 1.0

        hist = []
        n_prev = n.copy()
        n_iter = 0
        for k in range(n_steps):
            etd2_step(n, dt, d, apply_F, bufs, np)
            n_iter = k + 1
            if (k % check_every) == 0 or k == n_steps - 1:
                denom = max(float(np.abs(n).max()), 1e-300)
                res = float(np.abs(n - n_prev).max()) / denom
                hist.append(res)
                if res < rtol and k >= check_every:
                    break
                n_prev = n.copy()
        if return_info:
            return n, {"residuals": hist, "dt": dt, "n_steps": n_iter}
        return n
