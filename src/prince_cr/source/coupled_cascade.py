"""In-source coupled lepto-hadronic single-zone cascade — the keystone solver.

Solves the steady-state fixed point in which ONE common photon field is at once
the *target* for every process (inverse Compton, γγ pair production, synchrotron
self-absorption, pγ, Bethe-Heitler) AND the *sum* of all emission (synchrotron
+ IC + π⁰→γγ + proton synchrotron + the γγ-regenerated cascade). This is the
in-source analogue of what AM3 marches in time to; here it is solved as a
relaxation fixed point.

The full **γγ-regenerating EM cascade** is captured by the fixed-point iteration
itself, with no separate generation bookkeeping: each pass

    absorbed photons → e± pairs (energy-conserving ``pair_matrix``)
        → injected into the lepton zone → synchrotron + IC photons
        → some re-absorbed → e± pairs → …

advances the cascade by one generation, and iterating to convergence sums all
generations — i.e. the multi-generation cascade that redistributes the GeV–PeV
photons (π⁰→γγ, IC) down into the X-ray–MeV band (the it6e gap), on the
self-consistent field.

Reuse map (everything below is already validated against AM3):
  * leptons → :class:`prince_cr.source.evolution.SingleZoneSolver`
    (upwind synchrotron+IC cooling, escape, KN IC emission, SSA);
  * the common field → :class:`~.evolution.CompositePhotonField` (feedback=True);
  * γγ absorption → :func:`prince_cr.cascade.opacity._kernel_per_length` × c;
  * γγ pair injection → energy-conserving :func:`prince_cr.cascade.pair_matrix`.

The *hadronic* secondary injections (BH e±, π±→μ→e±, π⁰→γγ, proton synchrotron)
are supplied by the caller through two callbacks — they require the FLUKA
cross-section DB and the explicit-decay tables that live outside ``prince_cr``.
The callbacks receive the current field so the nonlinear amplification
(more photons → more pγ/BH → more secondaries → more photons) is captured.
"""
from __future__ import annotations

import numpy as np

from prince_cr.cascade.cascade import _energy_conserving_matrix, pair_matrix
from prince_cr.cascade.opacity import _kernel_per_length
from prince_cr.source.evolution import CompositePhotonField, SingleZoneSolver

_C = 2.99792458e10                 # cm/s
_ERG_GeV = 1.602176634e-3          # GeV per erg
_ME_C2_GeV = 0.51099895e-3         # m_e c^2 [GeV]
_ME_G = 9.1093837015e-28           # g


def gamma_gamma_abs_inv(E_GeV, field, eps_GeV, n_mu=64, z=0.0):
    """In-source γγ pair-production absorption rate t_γγ⁻¹(E) [s⁻¹].

    t_γγ⁻¹(E) = c · (dτ_gg/dl)(E) on the isotropic field ``field`` (the common
    in-zone field), with the soft-photon target integrated over ``eps_GeV``.
    Wraps :func:`prince_cr.cascade.opacity._kernel_per_length` (which returns
    dτ/dl [cm⁻¹]) so the photon-escape competition in :meth:`CoupledCascadeSolver
    .solve` is c/R (escape) vs c·dτ/dl (absorption)."""
    E_GeV = np.atleast_1d(np.asarray(E_GeV, float))
    mu = np.linspace(-1.0, 1.0, n_mu)
    out = np.array([_kernel_per_length(float(Ei), z, field, eps_GeV, mu)
                    for Ei in E_GeV])
    return _C * out                                   # cm⁻¹ · cm/s = s⁻¹


class CoupledCascadeSolver:
    """Single-zone coupled lepto-hadronic cascade, solved at steady state.

    Parameters
    ----------
    R_cm, B_Gauss, t_esc_s : zone size, tangled field, escape time (R/c default).
    external_field : optional fixed external photon field (CMB/EBL/accretion);
        ``None`` → pure in-source. Becomes the ``external`` of the common
        :class:`~.evolution.CompositePhotonField`.
    gamma_e : (lo, hi, n_bins) for the lepton (γ_e) grid. Default reaches 2e9 to
        hold the γγ pairs / BH e± from PeV photons.
    E_ph_GeV : (lo, hi, n_bins) for the common photon-field grid [GeV].
    """

    def __init__(self, R_cm, B_Gauss, t_esc_s=None,
                 external_field=None,
                 gamma_e=(1.0, 2e9, 320),
                 E_ph_GeV=(1e-13, 1e6, 280)):
        self.R = float(R_cm)
        self.B = float(B_Gauss)
        self.t_esc = (R_cm / _C) if t_esc_s is None else float(t_esc_s)
        glo, ghi, gn = gamma_e
        self.sze = SingleZoneSolver(gamma_lo=glo, gamma_hi=ghi, n_bins=gn,
                                    B_Gauss=B_Gauss, t_esc_s=self.t_esc)
        elo, ehi, en = E_ph_GeV
        self.E_ph = np.logspace(np.log10(elo), np.log10(ehi), en)
        self.dE_ph = np.gradient(self.E_ph)
        self.field = CompositePhotonField(external=external_field, feedback=True)
        # soft-photon target sub-grid for γγ / IC (where the field actually lives).
        # Span the full field floor (far-IR/radio) up — these low-energy photons are
        # the γγ targets that absorb the highest-energy (PeV–EeV) photons.
        self._eps_soft = np.logspace(np.log10(elo), np.log10(1e2),
                                     max(200, int(14 * np.log10(1e2 / elo))))

    # ---- field energy density (erg/cm³) for the Thomson IC cooling term ----
    def _u_rad(self):
        n = self.field.get_photon_density(self.E_ph)
        return float(np.trapezoid(self.E_ph * n, self.E_ph)) * _ERG_GeV

    # ---- sum production-rate components onto the common photon grid ----
    def _sum_on_Eph(self, comps):
        """comps = [(E_GeV, Q[GeV⁻¹cm⁻³s⁻¹]), ...] → summed Q on self.E_ph."""
        Q = np.zeros_like(self.E_ph)
        lnE = np.log(self.E_ph)
        for E, q in comps:
            E = np.asarray(E, float); q = np.asarray(q, float)
            m = (q > 0) & np.isfinite(q) & (E > 0)
            if m.sum() < 2:
                continue
            o = np.argsort(E[m])
            lo, hi = E[m][o][0], E[m][o][-1]
            v = np.exp(np.interp(lnE, np.log(E[m][o]), np.log(q[m][o]),
                                 left=-np.inf, right=-np.inf))
            Q += np.where((self.E_ph >= lo) & (self.E_ph <= hi), v, 0.0)
        return Q

    # ---- lepton synchrotron + IC photon-production rate on E_ph ----
    def _lepton_photon_production(self, n_e):
        """Q_γ [GeV⁻¹ cm⁻³ s⁻¹] from the lepton spectrum: synchrotron (with SSA)
        + inverse Compton on the *full common field* (KN). Pure production rate
        (no R/c residence — escape is the 1/t_esc term in the photon balance)."""
        # synchrotron (in-zone density already carries R/c + SSA) → rate = ÷t_esc
        eps_s, nph_s = self.sze.synchrotron_photon_density(n_e, self.R, ssa=True)
        E_s = eps_s / _ERG_GeV
        Q_s = nph_s * _ERG_GeV / self.t_esc            # cm⁻³ erg⁻¹→GeV⁻¹, ÷t_esc
        # IC on the common field as target (soft photons up to ~MeV)
        eps_t = self._eps_soft * _ERG_GeV              # erg
        n_t = self.field.get_photon_density(self._eps_soft) / _ERG_GeV  # cm⁻³erg⁻¹
        sel = n_t > 0
        if sel.sum() >= 2:
            eps_out = np.logspace(np.log10(self.E_ph[self.E_ph > 0][0] * _ERG_GeV),
                                  np.log10(self.E_ph[-1] * _ERG_GeV), 120)
            Q_ic_e = self.sze.ic_sed(n_e, eps_t[sel], n_t[sel], eps_out)
            E_ic = eps_out / _ERG_GeV
            Q_ic = Q_ic_e * _ERG_GeV                   # per-erg → per-GeV (rate)
        else:
            E_ic, Q_ic = E_s, np.zeros_like(E_s)
        return [(E_s, Q_s), (E_ic, Q_ic)]

    # ---- γγ pair injection from the absorbed photons (→ dN/dγ_e) ----
    def _pair_injection(self, n_gamma, tgg_inv):
        """e± injection dN/dγ_e on the lepton grid from γγ-absorbed photons.

        Absorbed-photon rate Ṅ_abs(E_γ) = n_γ(E_γ)·t_γγ⁻¹(E_γ); each absorbed
        photon → an e⁺e⁻ pair carrying its full energy (energy-conserving
        ``pair_matrix``). Returns dN/dγ_e on ``self.sze.g``."""
        ndot_abs = n_gamma * tgg_inv                   # GeV⁻¹ cm⁻³ s⁻¹
        if not np.any(ndot_abs > 0):
            return np.zeros_like(self.sze.g)
        E = self.E_ph
        eps = self._eps_soft
        n_eps = self.field.get_photon_density(eps)
        P = _energy_conserving_matrix(pair_matrix(E, E, eps, n_eps), E, E)
        Q_e_E = P @ (ndot_abs * self.dE_ph)            # cm⁻³ s⁻¹ GeV⁻¹ (in E_e=E)
        # map E_e[GeV] → γ_e and dN/dE→dN/dγ (×m_e c²); interpolate to lepton grid
        gam = E / _ME_C2_GeV
        Q_g = Q_e_E * _ME_C2_GeV
        m = (Q_g > 0) & np.isfinite(Q_g)
        if m.sum() < 2:
            return np.zeros_like(self.sze.g)
        g = self.sze.g
        out = np.exp(np.interp(np.log(g), np.log(gam[m]), np.log(Q_g[m]),
                               left=-np.inf, right=-np.inf))
        return np.where((g >= gam[m].min()) & (g <= gam[m].max()), out, 0.0)

    def solve(self, lepton_injection, photon_injection=None, n_iter=40,
              tol=2e-3, relax=0.5, verbose=True, hadronic_every=1):
        """Relax to the self-consistent coupled-cascade fixed point.

        Parameters
        ----------
        lepton_injection : callable(field) -> dN/dγ_e on ``self.sze.g``
            The field-dependent NON-γγ lepton injection: primary e⁻ + BH e± +
            π±→μ→e±. (γγ pairs are added internally each pass.)
        photon_injection : callable(field) -> (E_GeV, Q[GeV⁻¹cm⁻³s⁻¹]) or None
            The NON-lepton-EM photon injection: π⁰→γγ + proton synchrotron.
            (Lepton synchrotron + IC are added internally.)
        n_iter, tol, relax : fixed-point controls (under-relaxation on the
            internal field energy density).
        hadronic_every : recompute the (expensive) injection callbacks every N
            iterations; the cheap EM-cascade regeneration runs every iteration.

        Returns dict: ``field`` (the converged CompositePhotonField), ``n_e``,
        ``E_ph``/``n_gamma``, the production ``components``, and ``history``.
        """
        # seed the internal field from a first lepton pass (no γγ pairs yet)
        Q_e0 = np.asarray(lepton_injection(self.field), float)
        self.sze.set_ic_target(0.0)
        n_e = self.sze.steady_state(Q_e0)
        comps0 = self._lepton_photon_production(n_e)
        if photon_injection is not None:
            comps0 = comps0 + [photon_injection(self.field)]
        Q_prod = self._sum_on_Eph(comps0)
        n_gamma = Q_prod * self.t_esc                  # escape-only seed
        self.field.set_internal(self.E_ph, n_gamma)

        hist = []
        Q_e_had = Q_e0
        for it in range(n_iter):
            if (it % hadronic_every) == 0:
                Q_e_had = np.asarray(lepton_injection(self.field), float)
                phot_extra = (photon_injection(self.field)
                              if photon_injection is not None else None)
            tgg_inv = gamma_gamma_abs_inv(self.E_ph, self.field, self._eps_soft)
            Q_e_gg = self._pair_injection(n_gamma, tgg_inv)
            Q_e = Q_e_had + Q_e_gg

            self.sze.set_ic_target(self._u_rad())
            n_e = self.sze.steady_state(Q_e)

            comps = self._lepton_photon_production(n_e)
            if phot_extra is not None:
                comps = comps + [phot_extra]
            Q_prod = self._sum_on_Eph(comps)
            # photon steady state: production balanced by escape + γγ absorption
            n_gamma_new = Q_prod / (1.0 / self.t_esc + tgg_inv)

            U_old = float(np.trapezoid(self.E_ph * n_gamma, self.E_ph))
            n_gamma = (1.0 - relax) * n_gamma + relax * n_gamma_new
            self.field.set_internal(self.E_ph, n_gamma)
            U_new = float(np.trapezoid(self.E_ph * n_gamma, self.E_ph))
            res = abs(U_new - U_old) / max(U_new, 1e-300)
            hist.append({"iter": it, "U_gamma_GeV_cm3": U_new, "residual": res})
            if verbose:
                print(f"  it{it:02d}  U_γ={U_new:.4e} GeV/cm³  "
                      f"resid={res:.3e}  τγγ,max={float(np.max(tgg_inv))*self.t_esc:.2e}")
            if res < tol and it >= 2:
                break

        return {"field": self.field, "n_e": n_e, "gamma_e": self.sze.g,
                "E_ph": self.E_ph, "n_gamma": n_gamma,
                "components": comps, "tgg_inv": tgg_inv, "history": hist}

    # --- coupled NONLINEAR ETD2 march (no Picard) -------------------------
    def solve_etd2(self, lepton_injection, photon_injection=None, dt=None,
                   n_steps=4000, rtol=1e-6, check_every=10, n0=None,
                   verbose=True):
        """Solve the coupled e±↔γ in-source cascade by marching the joint state
        ``[n_e(γ_e), n_γ(E_ph)]`` to steady state with PriNCe's production ETD2
        stepper (``solvers.etd2.etd2_step``), with the **nonlinear photon-field
        feedback handled inside ETD2** — no outer Picard loop.

        ETD2 (Cox-Matthews) is an exponential integrator for the SEMILINEAR
        system ``dn/dt = D·n + F(n)``: the stiff linear **diagonal** D (lepton
        cooling self-loss + escape; photon escape + γγ absorption) is treated
        exactly via ``exp(h·D)``, and the **nonlinear remainder** F(n) — lepton
        advection gain, primary+γγ e± injection, and the e±→γ synchrotron+IC
        EMISSION + γγ pair production, all **convolved with the CURRENT in-source
        photon field** ``n_γ`` (the nonlinearity) — is evaluated per stage. So
        the IC target and the γγ kernels see the self-consistent field as it
        evolves, exactly as in a time-dependent code.

        The field-dependent diagonal (IC cooling rate, τ_γγ) is refrozen at the
        start of each step from the current state; the field convolutions in F
        use each stage's field. ``lepton_injection(field)`` / optional
        ``photon_injection(field)`` are the same callbacks as :meth:`solve`."""
        from prince_cr.solvers.etd2 import etd2_step, _step_buffers, split_operator

        Ng = self.sze.g.size
        NE = self.E_ph.size
        dim = Ng + NE
        bufs = _step_buffers(dim, np)
        state = (np.zeros(dim) if n0 is None else np.asarray(n0, float).copy())

        def _set_field(n_g):
            self.field.set_internal(self.E_ph, np.clip(n_g, 0.0, None))

        def _field_ops(n_g):
            """(M_lep, tgg) at the photon field ``n_g`` — the field-dependent
            operators. ``M_lep`` is the lepton cooling+escape operator WITH IC
            at this field; ``tgg`` the γγ absorption rate. Recomputed per stage
            so IC cooling and IC emission always see the SAME field (the fix for
            the frozen-field IC inconsistency)."""
            _set_field(n_g)
            self.sze.set_ic_target(self._u_rad())
            M = self.sze.cooling_operator().tocsr()
            tgg = gamma_gamma_abs_inv(self.E_ph, self.field, self._eps_soft)
            return M, tgg

        def _full_rhs(x):
            """The complete nonlinear RHS  dn/dt = [lepton ; photon]  at the
            field carried by ``x`` (stage-consistent)."""
            n_e = x[:Ng]
            n_g = x[Ng:]
            M, tgg = _field_ops(n_g)
            Q_e = np.asarray(lepton_injection(self.field), float)
            Q_e = Q_e + self._pair_injection(n_g, tgg)        # γγ→e± at stage field
            rhs_e = M @ n_e + Q_e                             # IC cooling: stage field
            Q_g = self._sum_on_Eph(self._lepton_photon_production(n_e))  # IC emission: stage field
            if photon_injection is not None:
                extra = photon_injection(self.field)
                if extra is not None:
                    Q_g = Q_g + self._sum_on_Eph([extra])
            rhs_g = Q_g - (1.0 / self.t_esc + tgg) * n_g      # escape + γγ sink
            return np.concatenate([rhs_e, rhs_g])

        # ETD2 semilinear split: D = frozen diagonal (per step), F(x) = full
        # nonlinear RHS(x) − D·x. Because F carries the WHOLE field-dependent RHS
        # at the stage field, IC cooling (in M) and IC emission (in Q_g) are
        # always mutually consistent → no frozen-field IC offset.
        if dt is None:
            dt = self.t_esc                                   # ETD handles faster modes exactly
        hist = []
        prev = state.copy()
        n_done = 0
        for k in range(n_steps):
            M0, tgg0 = _field_ops(state[Ng:])                 # frozen diagonal @ step start
            d = np.concatenate([np.asarray(M0.diagonal()),
                                -(1.0 / self.t_esc + tgg0)])

            def apply_F(x, out, _d=d):
                out[:] = _full_rhs(x)
                out -= _d * x

            etd2_step(state, dt, d, apply_F, bufs, np)
            np.clip(state, 0.0, None, out=state)              # guard tiny negative undershoot
            n_done = k + 1
            if (k % check_every) == 0:
                denom = max(float(np.abs(state).max()), 1e-300)
                res = float(np.abs(state - prev).max()) / denom
                hist.append(res)
                if verbose:
                    Ug = float(np.trapezoid(self.E_ph * state[Ng:], self.E_ph))
                    print(f"  etd2 it{k:04d}  U_γ={Ug:.4e} GeV/cm³  resid={res:.3e}")
                if res < rtol and k >= check_every:
                    break
                prev = state.copy()

        n_e = state[:Ng]
        n_gamma = state[Ng:]
        _set_field(n_gamma)
        tgg_inv = gamma_gamma_abs_inv(self.E_ph, self.field, self._eps_soft)
        return {"field": self.field, "n_e": n_e, "gamma_e": self.sze.g,
                "E_ph": self.E_ph, "n_gamma": n_gamma, "tgg_inv": tgg_inv,
                "history": hist, "n_steps": n_done}
