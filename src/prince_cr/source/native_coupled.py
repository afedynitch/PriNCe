"""Native unified-state in-source cascade — the time-dependent lepto-hadronic
single-zone solver on PriNCe's OWN state vector (plan §5.1-3).

This is the architecture the audit (lesson `in-source-bypassed-prince-sparse-
machinery`) pointed to: ONE PriNCe state vector

    state = [ p(+nuclei) on cr_grid | e± on em_grid | γ on em_grid | ν on cr_grid ]

is marched in **time t** by the production ETD2 stepper
(:func:`prince_cr.solvers.etd2.etd2_step`, Cox-Matthews). The march *is* the
light curve; steady state is ``t→∞``. The combined operator is

    dn/dt = L(state)·n + Q_inj
    L = c·int_rates.get_hadr_jacobian(pfield)            # photo-hadronic + (chain-reduced) decay
      ⊕ radiative em-grid blocks on the e±/γ state:
          synchrotron (SSA) + IC cooling (e±)            # advection on em_grid
          synchrotron + IC EMISSION (e±→γ)
          γγ absorption (γ sink) + pair production (γ→e±)
      ⊖ escape (1/t_esc, all species)

with **all field-dependent couplings folded with the CURRENT in-source photon
field each ETD2 stage** (genuine non-linear, time-dependent feedback — no
Picard), exactly as in :meth:`CoupledCascadeSolver.solve_etd2`, but now over the
whole native state vector and with the hadronic secondaries supplied by the
native sparse ``PhotoNuclearInteractionRate`` coupling instead of a Python fold.

The em_grid floors at m_e c² (0.5 MeV); the sub-MeV synchrotron photons
(radio→X-ray hump) are the IC/γγ/pγ *target* but are NOT a cascade species, so
they are reconstructed from the e± each stage and carried as a target-only
component of the common :class:`~.evolution.CompositePhotonField` (they don't
pair-produce — physically a reservoir, not a state). Photons that land ≥0.5 MeV
(IC + high-E synchrotron + π⁰→γγ) ARE the evolved γ state.
"""
from __future__ import annotations

import numpy as np

from prince_cr.cascade.cascade import _energy_conserving_matrix, pair_matrix
from prince_cr.data import PRINCE_UNITS
from prince_cr.source.coupled_cascade import gamma_gamma_abs_inv
from prince_cr.source.evolution import CompositePhotonField, SingleZoneSolver

_C = 2.99792458e10                 # cm/s
_ERG_GeV = 1.602176634e-3          # erg per GeV
_ME_C2_GeV = 0.51099895e-3         # m_e c^2 [GeV]
_ME_G = 9.1093837015e-28           # g
_MP_G = 1.67262192369e-24          # g
_H_ERG_S = 6.62607015e-27          # Planck [erg s]
_SIGMA_T = 6.6524587321e-25        # cm^2
# BH e± injection prefactor (AM3 BetheHeitler.cc Calc_BH_Rad): c·(3/16π)·σ_T·α
_BH_KINJ = _C * 3.0 / (16.0 * np.pi) * _SIGMA_T / 137.035999

_LEPTON_PDGS = (11, -11)
_NU_PDGS = (12, -12, 14, -14, 16, -16)


class NativeCoupledSolver:
    """Unified-state in-source cascade marched by the production ETD2 stepper.

    Parameters
    ----------
    run : PriNCeRun
        A run built with ``enable_em_cascade=True, enable_em_decoupled_grid=True``
        (γ, e± live on ``run.em_grid``; p, ν on ``run.cr_grid``). Provides the
        species layout (``run.spec_man``) and the native photo-hadronic coupling
        (``run.int_rates.get_hadr_jacobian(pfield=…)``).
    R_cm, B_Gauss : zone radius and tangled magnetic field.
    t_esc_s : escape time (default R/c).
    external_field : optional fixed external photon field (CMB/EBL/disc); becomes
        the ``external`` of the common field. ``None`` → pure in-source.
    hadronic : if True (default) include the native ``c·J(pfield)`` photo-hadronic
        + decay coupling; if False the solver is purely leptonic (e±↔γ SSC),
        used for the closure test against :meth:`CoupledCascadeSolver.solve_etd2`.
    """

    def __init__(self, run, R_cm, B_Gauss, t_esc_s=None, external_field=None,
                 hadronic=True, bethe_heitler=True, loss_stencil="fv2",
                 gg_pairs=True, gg_reservoir_sink=False):
        self.run = run
        # gg_pairs: inject γγ→e± pairs (the cascade source). Photons are still
        # γγ-ABSORBED (γ sink) regardless; gg_pairs=False only suppresses the e±
        # re-injection, isolating the cascade-feedback contribution (diagnostic).
        self.gg_pairs = gg_pairs
        # gg_reservoir_sink (DEFAULT OFF): also deplete the SUB-m_e synchrotron
        # reservoir by γγ pair-production against the hard (≥0.5 MeV) γ-state.
        # The reservoir (radio→X-ray synch below the em floor) is normally a passive
        # target+escape population — its γγ loss is negligible because n_soft ≫ n_hard,
        # and each γγ event's pair energy is already booked from the absorbed HARD
        # photon (the soft photon only adds E_soft ≪ E_hard). For γγ-THICK / pair-
        # runaway regimes where the reprocessed hard field approaches n_soft this is no
        # longer negligible; enabling this applies the soft-photon γγ sink
        #   n_soft(ε) → n_soft(ε) / (1 + t_esc · t_γγ⁻¹(ε))
        # using the SAME gamma_gamma_abs_inv kernel (evaluated at the reservoir energies
        # against the γ-state). Pairs are NOT re-injected here (already counted from the
        # hard side). A guard warns when the depletion is non-negligible. See
        # lessons/em-cascade-reservoir-gg-sink.
        self.gg_reservoir_sink = bool(gg_reservoir_sink)
        self._gg_sink_warned = False
        # Cooling-loss stencil for the e± advection.
        #   "fv2"     → conservative 2nd-order MUSCL FV (van Leer TVD limiter) — the
        #               DEFAULT. ETD2-safe (stiff loss stays on the diagonal, bounded
        #               limited off-diagonal, CFL cap reverts ultra-stiff high-γ cells
        #               to 1st-order), positivity-preserving (0 neg bins). Kills the
        #               1st-order upwind numerical diffusion WITHOUT more bins: PS
        #               leptonic synch 1.14×→1.02× vs AM3 at em16 (was the em16→em32
        #               workaround). See lesson cooling-stencil-diffusion-vs-etd2.
        #   None      → conservative FV 1st-order upwind (legacy default): ALL the
        #               stiff cooling self-loss is on the diagonal → ETD2-stable for
        #               the full lepto-hadronic system (pairs to γ~1e9). But
        #               O(h)-diffusive → inflates the leptonic synchrotron SED
        #               ~12-40% at em16 (it48); the HADRONIC observables (field/ν)
        #               are barely affected (it46 field 0.99×, ν 1.30× vs AM3).
        #   "upwind2" → MCEq 2nd-order (it49: syn 0.99/SSC 1.04 at em16). ETD2-STABLE
        #               ONLY when high-γ is UNPOPULATED (pure leptonic, e± ≤ γ_max):
        #               its i+2 off-diagonal carries stiff cooling → with hadronic
        #               pairs at γ~1e9 the explicit ETD2 stage diverges (h·|L_off|≫1).
        #   "upwind"  → MCEq 1st-order FD (bidiagonal; syn 1.01 leptonic). Like FV in
        #               structure → may be hadronic-stable; less diffusive than FV.
        #   "expfit"  → most accurate (spsolve 0.975) but ETD2-unstable (7-pt).
        # A robust higher-order fix for the ETD2 march needs a CONSERVATIVE 2nd-order
        # FV flux (stiff loss stays on the diagonal). See lesson cooling-stencil-diffusion.
        self.loss_stencil = loss_stencil
        self.sm = run.spec_man
        self.em = run.em_grid
        self.cr = run.cr_grid
        self.R = float(R_cm)
        self.B = float(B_Gauss)
        self.t_esc = (R_cm / _C) if t_esc_s is None else float(t_esc_s)
        self.hadronic = bool(hadronic)
        self.bethe_heitler = bool(bethe_heitler)
        self.dim = self.sm.dim_states

        if self.em is None:
            raise ValueError("NativeCoupledSolver needs an em_grid "
                             "(enable_em_decoupled_grid=True).")

        # --- species slices on the native state vector ---
        sref = self.sm.pdgid2sref
        self.sl = {pdg: sref[pdg].sl for pdg in self.sm.known_species}
        self.sl_g = self.sl[22]
        self.sl_ep = self.sl[-11]
        self.sl_em = self.sl[11]
        self.sl_p = self.sl.get(2212)
        self.nu_sls = [self.sl[p] for p in _NU_PDGS if p in self.sl]
        # All cr-grid CR species FREE-STREAM / escape at R/c (p, p̄, n, n̄, ν…) —
        # everything except the radiatively-handled em species (e±, γ). With a
        # finite config.tau_dec_threshold the NEUTRON (2112) is tracked (not
        # chain-reduced) so it ESCAPES here instead of β-decaying in-zone — fixing
        # the spurious low-E e⁻/ν̄_e (n→p e⁻ ν̄_e); high-γ neutrons physically leave
        # (decay length γcτ_n ≫ R). See lesson in-source-neutron-beta-decay.
        _em_pdgs = (11, -11, 22)
        self.sl_cr_escape = [self.sl[pdg] for pdg in self.sm.known_species
                             if pdg not in _em_pdgs]

        # --- e± radiative solver ON the em_grid (γ_e = E_e/m_e c²) ---
        gC = self.em.grid / _ME_C2_GeV
        gIF = self.em.bins / _ME_C2_GeV
        dG = self.em.widths / _ME_C2_GeV
        self.sze = SingleZoneSolver.on_grid(gC, gIF, dG, B_Gauss=B_Gauss,
                                            t_esc_s=self.t_esc)
        # --- proton synchrotron solver on the proton γ_p grid (cr_grid/m_p) ---
        # blazar_simple's X-ray–MeV plateau is PROTON SYNCHROTRON (see
        # results/am3-insource-coupled-cascade); γ_p~1e6–1e8 protons radiate there.
        _MP_GeV = PRINCE_UNITS.m_proton
        self._mp_GeV = _MP_GeV
        if self.sl_p is not None:
            gpC = self.cr.grid / _MP_GeV
            gpIF = self.cr.bins / _MP_GeV
            gpDG = self.cr.widths / _MP_GeV
            self.szp = SingleZoneSolver.on_grid(gpC, gpIF, gpDG, B_Gauss=B_Gauss,
                                                mass_g=_MP_G, charge=1.0,
                                                t_esc_s=self.t_esc)
        else:
            self.szp = None
        self.Eg = self.em.grid               # γ photon energies [GeV]
        self.dEg = self.em.widths
        self.lnEg = np.log(self.Eg)

        # --- common photon field ---
        self.field = CompositePhotonField(external=external_field, feedback=True)
        # soft-photon target sub-grid: must reach BELOW the em floor (0.5 MeV)
        # down to the radio/IR synchrotron that is the γγ target for the PeV–EeV
        # photons and the IC seed. Up to ~100 GeV.
        self._eps_soft = np.logspace(-16.0, 2.0, 320)
        # synchrotron reconstruction grid (for the sub-MeV target + the ≥0.5 MeV
        # emission onto the γ state). Photon-energy [GeV] → ν [Hz].
        E_syn = np.logspace(-18.0, np.log10(self.Eg[-1]), 420)
        self._syn_nu = (E_syn * _ERG_GeV) / _H_ERG_S   # ν[Hz] = E[GeV]·erg/GeV / h
        # fixed BH grids (keep _bh_kernel_cached warm across stages): e± energy
        # output grid + proton grid (cr_grid) + target sub-grid (_eps_soft).
        self._bh_Ee = np.logspace(-3.0, 9.0, 130)      # e± energy [GeV]
        self._bh_gam = self._bh_Ee / _ME_C2_GeV        # → γ_e for the lepton grid
        self._bh_eps = np.logspace(-12.0, 1.0, 110)    # BH target band [GeV] (eV–10 GeV)

        self._J_cache = None                 # last hadronic jacobian (debug)
        self._L_had_frozen = None             # c·J frozen for the current ETD2 step
        self._photon_grid_warned = False      # one-shot pγ target-grid coverage guard
        self._n_e_frozen = None               # step-start e± total for the fv2 limiter
        self._dt = self.t_esc                 # ETD2 step (for the fv2 CFL stiffness cap)

    # ---------------- field bookkeeping ----------------
    def _proton_synch_density(self, n_p):
        """Proton-synchrotron in-zone photon density (E_GeV, n_GeV), full range.
        ``n_p`` = dN/dE_p on cr_grid → dN/dγ_p on the szp grid (×m_p). Optically
        thin (ssa=False), as in it6g. This is the blazar_simple X-ray–MeV plateau."""
        if self.szp is None:
            return np.empty(0), np.empty(0)
        n_p_g = np.clip(n_p, 0.0, None) * self._mp_GeV          # dN/dγ_p
        eps_p, nph_p = self.szp.synchrotron_photon_density(n_p_g, self.R,
                                                           nu=self._syn_nu, ssa=False)
        return eps_p / _ERG_GeV, nph_p * _ERG_GeV               # GeV, cm^-3 GeV^-1

    def _synch_target(self, n_e_tot, n_p=None):
        """Sub-em-floor synchrotron photon density (radio→X-ray hump) reconstructed
        from the e± (SSA) AND proton (thin) spectra — the IC/γγ/pγ target reservoir
        that lives BELOW the em_grid floor (0.5 MeV). Returns (E_GeV, n_GeV)."""
        eps_s, nph_s = self.sze.synchrotron_photon_density(n_e_tot, self.R,
                                                           nu=self._syn_nu, ssa=True)
        E_e = eps_s / _ERG_GeV; n_e = nph_s * _ERG_GeV
        comps = [(E_e, n_e)]
        if n_p is not None:
            comps.append(self._proton_synch_density(n_p))
        # sum the components below the em floor onto a common (e±) energy grid
        below = (E_e < self.Eg[0])
        Eb = E_e[below]
        ntot = np.zeros_like(Eb)
        lnEb = np.log(np.clip(Eb, 1e-300, None))
        for E, n in comps:
            m = (n > 0) & np.isfinite(n) & (E > 0)
            if m.sum() < 2:
                continue
            o = np.argsort(E[m])
            v = np.exp(np.interp(lnEb, np.log(E[m][o]), np.log(n[m][o]),
                                 left=-np.inf, right=-np.inf))
            ntot += np.where((Eb >= E[m][o][0]) & (Eb <= E[m][o][-1]), v, 0.0)
        keep = ntot > 0
        return Eb[keep], ntot[keep]

    def _ic_subMeV(self, n_e_tot):
        """Sub-em-floor inverse-Compton (SSC) photon density (E_GeV, n_GeV),
        reconstructed from the e± on the CURRENT field as IC target. The em_grid
        γ-state holds only the ≥0.5 MeV IC; for low-γ_max leptonic SSC the whole
        IC hump sits below the floor (e.g. ~600 eV for γ_max=1e3) — this recovers
        it for the OUTPUT field/SED. (Not used as a solve target — 2nd-order IC is
        negligible at the relevant Compton dominance.)"""
        eps_t = self._eps_soft * _ERG_GeV
        n_t = self.field.get_photon_density(self._eps_soft) / _ERG_GeV
        sel = n_t > 0
        if sel.sum() < 3:
            return np.empty(0), np.empty(0)
        eps_out = np.logspace(np.log10(1e-10 * _ERG_GeV),
                              np.log10(self.Eg[-1] * _ERG_GeV), 220)
        Q_ic = self.sze.ic_sed(n_e_tot, eps_t[sel], n_t[sel], eps_out) * _ERG_GeV
        E_ic = eps_out / _ERG_GeV
        n_ic = Q_ic * self.t_esc                       # in-zone density (sub-MeV: γγ≈0)
        below = (E_ic < self.Eg[0]) & (n_ic > 0) & np.isfinite(n_ic)
        return E_ic[below], n_ic[below]

    def _set_field(self, n_e_tot, n_g, n_p=None, include_ic=False):
        """Common field = external + [sub-MeV e±+proton synchrotron target]
        + [γ-state ≥0.5 MeV]. ``include_ic`` (OUTPUT only): also fold the sub-MeV
        IC/SSC into the field so the emergent SED carries the full Compton hump
        (the solve leaves it off — synchrotron is the dominant IC/γγ target)."""
        Esub, nsub = self._synch_target(n_e_tot, n_p)
        ng = np.clip(n_g, 0.0, None)
        self.field.set_internal(np.concatenate([Esub, self.Eg]),
                                np.concatenate([nsub, ng]))
        # OPTIONAL γγ depletion of the sub-m_e synch reservoir (γγ-thick regimes).
        # Needs the field just set (to read the hard γ-state as the γγ target).
        if self.gg_reservoir_sink and Esub.size:
            nsub = self._reservoir_gg_sink(Esub, nsub)
            self.field.set_internal(np.concatenate([Esub, self.Eg]),
                                    np.concatenate([nsub, ng]))
        if include_ic and Esub.size:
            E_ic, n_ic = self._ic_subMeV(n_e_tot)      # uses the synchrotron target just set
            if E_ic.size >= 2:
                add = np.exp(np.interp(np.log(Esub), np.log(E_ic),
                                       np.log(np.clip(n_ic, 1e-300, None)),
                                       left=-np.inf, right=-np.inf))
                add = np.where((Esub >= E_ic.min()) & (Esub <= E_ic.max()), add, 0.0)
                self.field.set_internal(np.concatenate([Esub, self.Eg]),
                                        np.concatenate([nsub + add, ng]))

    def _reservoir_gg_sink(self, Esub, nsub):
        """Deplete the sub-m_e synchrotron reservoir by γγ pair-production against the
        hard (≥0.5 MeV) γ-state, for γγ-thick / pair-runaway regimes (opt-in via
        gg_reservoir_sink). Steady state with escape + γγ loss:
            n_soft(ε) → n_soft(ε) · (1 + t_esc·t_γγ⁻¹(ε))⁻¹ ,
        where t_γγ⁻¹(ε) is the SAME `gamma_gamma_abs_inv` kernel evaluated at the soft
        reservoir energies ``Esub`` with the γ-state grid ``self.Eg`` as the (hard)
        target. Pairs from these events are already injected from the absorbed hard
        photon (the soft side only contributes E_soft ≪ E_hard), so they are NOT
        re-injected here. Emits a one-shot guard warning when the depletion is
        non-negligible (the n_soft ≫ n_hard assumption being stressed). Returns the
        depleted reservoir density."""
        rate = np.asarray(gamma_gamma_abs_inv(Esub, self.field, self.Eg), dtype=float)
        rate = np.where(np.isfinite(rate) & (rate > 0), rate, 0.0)
        supp = 1.0 / (1.0 + self.t_esc * rate)             # escape / (escape + γγ)
        max_depl = float(1.0 - np.min(supp)) if supp.size else 0.0
        if max_depl > 0.1 and not self._gg_sink_warned:    # >10% reservoir depletion
            import warnings
            i = int(np.argmin(supp))
            warnings.warn(
                "NativeCoupledSolver: γγ depletion of the sub-m_e synchrotron reservoir "
                f"reaches {max_depl:.0%} (at ε≈{Esub[i]*1e9:.2e} eV). The reservoir "
                "(n_soft ≫ n_hard) approximation is being stressed — results in this "
                "regime carry an extra systematic; consider a full sub-m_e photon "
                "treatment. (gg_reservoir_sink partially corrects the soft-photon loss.)",
                RuntimeWarning, stacklevel=2)
            self._gg_sink_warned = True
        return nsub * supp

    def _u_rad(self):
        """Field energy density [erg/cm³] over the full target span (for IC cooling)."""
        e = self._eps_soft
        n = self.field.get_photon_density(e)
        return float(np.trapezoid(e * n, e)) * _ERG_GeV

    def _set_ic_cooling_target(self):
        """Set the e± IC-cooling target to the current field SPECTRUM so the cooling is
        Klein-Nishina-corrected (γ-dependent), not Thomson. eps[erg], n_ph[cm^-3 erg^-1]."""
        eps_t = self._eps_soft * _ERG_GeV                                  # erg
        n_t = self.field.get_photon_density(self._eps_soft) / _ERG_GeV     # cm^-3 erg^-1
        self.sze.set_ic_target_spectrum(eps_t, n_t)

    # ---------------- rebin a production component onto em.grid ----------------
    def _sum_on_Eg(self, comps):
        """comps = [(E_GeV, Q[GeV⁻¹cm⁻³s⁻¹]), …] → summed Q on the em γ-grid."""
        Q = np.zeros_like(self.Eg)
        for E, q in comps:
            E = np.asarray(E, float); q = np.asarray(q, float)
            m = (q > 0) & np.isfinite(q) & (E > 0)
            if m.sum() < 2:
                continue
            o = np.argsort(E[m])
            lo, hi = E[m][o][0], E[m][o][-1]
            v = np.exp(np.interp(self.lnEg, np.log(E[m][o]), np.log(q[m][o]),
                                 left=-np.inf, right=-np.inf))
            Q += np.where((self.Eg >= lo) & (self.Eg <= hi), v, 0.0)
        return Q

    # ---------------- e±→γ emission (synchrotron ≥floor + IC) + p-syn ----------------
    def _emission_onto_emgrid(self, n_e_tot, n_p=None):
        """Photon-production rate Q_γ [GeV⁻¹cm⁻³s⁻¹] on the em γ-grid: e±
        synchrotron (SSA) + inverse Compton (KN) on the common field, plus
        PROTON synchrotron — the parts landing ≥0.5 MeV. Production rate =
        in-zone density / t_esc (escape competes in the photon balance), as in
        solve_etd2. (The sub-floor parts go to the target field via _set_field.)"""
        eps_s, nph_s = self.sze.synchrotron_photon_density(n_e_tot, self.R,
                                                           nu=self._syn_nu, ssa=True)
        E_s = eps_s / _ERG_GeV
        Q_s = nph_s * _ERG_GeV / self.t_esc
        comps = [(E_s, Q_s)]
        # IC on the common field as target (soft photons, KN kernel)
        eps_t = self._eps_soft * _ERG_GeV
        n_t = self.field.get_photon_density(self._eps_soft) / _ERG_GeV
        sel = n_t > 0
        if sel.sum() >= 2:
            eps_out = np.logspace(np.log10(self.Eg[0] * _ERG_GeV),
                                  np.log10(self.Eg[-1] * _ERG_GeV), 120)
            Q_ic = self.sze.ic_sed(n_e_tot, eps_t[sel], n_t[sel], eps_out) * _ERG_GeV
            comps.append((eps_out / _ERG_GeV, Q_ic))
        # proton synchrotron (the blazar X-ray–MeV plateau), ≥floor part
        if n_p is not None and self.szp is not None:
            E_ps, n_ps = self._proton_synch_density(n_p)
            comps.append((E_ps, n_ps / self.t_esc))
        return self._sum_on_Eg(comps)

    # ---------------- γγ pair injection (γ→e±) ----------------
    def _pair_injection(self, n_g, tgg_inv):
        """e± injection dN/dγ_e on the em lepton grid from γγ-absorbed photons
        (energy-conserving pair_matrix). Returns dN/dγ_e on ``self.sze.g``."""
        ndot_abs = np.clip(n_g, 0.0, None) * tgg_inv
        if not np.any(ndot_abs > 0):
            return np.zeros_like(self.sze.g)
        E = self.Eg
        eps = self._eps_soft
        n_eps = self.field.get_photon_density(eps)
        P = _energy_conserving_matrix(pair_matrix(E, E, eps, n_eps), E, E)
        Q_e_E = P @ (ndot_abs * self.dEg)            # cm⁻³ s⁻¹ GeV⁻¹ at E_e=E
        gam = E / _ME_C2_GeV
        Q_g = Q_e_E * _ME_C2_GeV                      # dN/dE→dN/dγ
        m = (Q_g > 0) & np.isfinite(Q_g)
        if m.sum() < 2:
            return np.zeros_like(self.sze.g)
        g = self.sze.g
        out = np.exp(np.interp(np.log(g), np.log(gam[m]), np.log(Q_g[m]),
                               left=-np.inf, right=-np.inf))
        return np.where((g >= gam[m].min()) & (g <= gam[m].max()), out, 0.0)

    # ---------------- Bethe-Heitler pair injection (p γ → p e±) ----------------
    def _bh_pair_injection(self, n_p):
        """BH e± injection dN/dγ_e on the em lepton grid from the proton spectrum
        ``n_p`` (dN/dE_p on cr_grid) folded with the common field, via AM3's exact
        absolute assembly (``BetheHeitler.cc::Calc_BH_Rad``):

            Q(E_e) = k_inj · Σⱼ Σₖ K[i,j,k]·ΔN_ph[k]·ΔN_p[j],
            k_inj = c·(3/16π)·σ_T·α,

        with the field-free Born kernel ``K`` cached on the FIXED grids
        (``_bh_Ee`` / ``cr_grid`` / ``_eps_soft``) so the per-stage cost is the
        einsum, not the kernel build. BH is NOT in the photo-nuclear ``c·J``;
        this is the separate pair-production e± source (the proton continuous
        energy LOSS side is deferred — small for these parameters)."""
        from prince_cr.cascade.bethe_heitler import _bh_kernel_cached
        E_p = self.cr.grid
        m = (n_p > 0) & np.isfinite(n_p) & (E_p >= 1e3)
        if m.sum() < 2:
            return np.zeros_like(self.sze.g)
        dN_p = np.where(m, np.clip(n_p, 0.0, None) * self.cr.widths, 0.0)  # protons/bin
        eps_f = self._bh_eps                                              # GeV
        n_f = self.field.get_photon_density(eps_f)
        dN_ph = n_f * eps_f * np.gradient(np.log(eps_f))                  # photons/bin
        K = _bh_kernel_cached(self._bh_Ee, E_p, eps_f)                    # (i,j,k) cached
        Q_E = _BH_KINJ * np.einsum("ijk,k,j->i", K, dN_ph, dN_p)          # E·dN/dE (e±)
        Q_g = Q_E / self._bh_gam                                          # → dN/dγ_e
        m2 = (Q_g > 0) & np.isfinite(Q_g)
        if m2.sum() < 2:
            return np.zeros_like(self.sze.g)
        g = self.sze.g
        out = np.exp(np.interp(np.log(g), np.log(self._bh_gam[m2]), np.log(Q_g[m2]),
                               left=-np.inf, right=-np.inf))
        return np.where((g >= self._bh_gam[m2].min()) & (g <= self._bh_gam[m2].max()),
                        out, 0.0)

    def _check_photon_grid_covers_field(self):
        """One-shot guard: the pγ TARGET grid (``config.photon_grid`` →
        ``int_rates.e_photon``) must cover the in-source photon field.

        The PriNCe default ``photon_grid=(-15,-6,8)`` caps targets at ~875 eV
        (the cosmological CMB+EBL range). An in-source blazar field carries
        keV–GeV flux, and because the pγ (Δ) threshold target energy scales as
        ε_th ∝ 1/γ_p, the *lower*-energy protons need those higher-energy
        targets to interact at all. If the field has significant flux above the
        grid's upper edge, the pγ fold silently drops those targets and starves
        the low-E secondary (π⁰→γγ + ν) LEFT FLANK — energy that then shows up
        in the γγ cascade instead. Widen ``config.photon_grid`` (before building
        the PriNCeRun) so its upper edge covers the field."""
        if self._photon_grid_warned:
            return
        self._photon_grid_warned = True
        try:
            self._photon_grid_guard_body()
        except Exception:
            pass  # the guard is purely advisory — never let it break a solve

    def _photon_grid_guard_body(self):
        eph = self.run.int_rates.e_photon.grid          # GeV
        sed = eph ** 2 * self.field.get_photon_density(eph)
        peak = float(np.max(sed)) if sed.size else 0.0
        if peak <= 0:
            return
        # The crucial high-energy targets (keV–MeV) are tiny relative to the
        # synchrotron SED PEAK (~1e-7), so a peak-fraction test misses them.
        # The right discriminator is whether the field is still ALIVE and
        # CONTINUING past the grid edge (a truncated power-law) rather than
        # already dead (the cosmological CMB+EBL case, where the field has
        # fallen ≳10 decades below peak well before 875 eV).
        sed_edge = float(sed[-1])
        if sed_edge < 1e-10 * peak:
            return                                       # field already negligible at the edge
        e_above = eph[-1] * 10.0 ** np.array([1.0, 2.0, 3.0])
        sed_above = float(np.max(e_above ** 2
                                 * self.field.get_photon_density(e_above)))
        if sed_above > 1e-4 * sed_edge:                  # field continues past the edge → targets dropped
            import warnings
            warnings.warn(
                "NativeCoupledSolver: the pγ target-photon grid upper edge "
                "({0:.3g} GeV) sits inside a live photon field (SED there is "
                "{1:.1e}× the field peak and still falling as a power law above "
                "it). The photo-hadronic fold drops every target above the edge, "
                "so lower-energy protons (which need higher-energy targets, "
                "ε_th∝1/γ_p) cannot interact — this starves the low-E secondary "
                "(π⁰→γγ + ν) LEFT FLANK and over-feeds the γγ cascade. Widen "
                "config.photon_grid (e.g. (-15, 0, 8)) BEFORE building the "
                "PriNCeRun so its upper edge covers the field.".format(
                    eph[-1], sed_edge / peak),
                RuntimeWarning,
                stacklevel=2,
            )

    # ---------------- hadronic native coupling ----------------
    def _hadr_matrix(self):
        """c·J(pfield) [1/s] over the full state — photo-hadronic + chain-reduced
        decay, secondaries already on the right grids (γ/e± on em, ν on cr).

        Frozen per ETD2 step: the photo-hadronic rates vary slowly, so we rebuild
        the (expensive) pfield fold once at step start (in :meth:`_diagonal`) and
        reuse it across the step's two stages — standard ETD2 rate-cache semantics,
        the same the production propagation solver uses. ``_L_had_frozen`` holds
        the current step's matrix; cleared between steps."""
        if self._L_had_frozen is not None:
            return self._L_had_frozen
        self._check_photon_grid_covers_field()
        J = self.run.int_rates.get_hadr_jacobian(0.0, 1.0, force_update=True,
                                                 pfield=self.field)
        self._J_cache = J
        return _C * J

    # ---------------- the full nonlinear RHS ----------------
    def _full_rhs(self, state, Qp, Qe_prim):
        """Complete dn/dt at the field carried by ``state`` (stage-consistent)."""
        n_ep = state[self.sl_ep]
        n_em = state[self.sl_em]
        n_g = state[self.sl_g]
        n_e_tot = n_ep + n_em
        n_p = state[self.sl_p] if self.sl_p is not None else None

        # refresh the common field from this stage's state (e±+proton synch target)
        self._set_field(n_e_tot, n_g, n_p)
        self._set_ic_cooling_target()

        rhs = np.zeros_like(state)

        # --- hadronic: c·J @ state (proton loss + secondary injection) ---
        if self.hadronic:
            L_had = self._hadr_matrix()
            inj_had = L_had @ state
            # UNITS: the c·J secondaries come out as dN/dE on each species' energy
            # grid (PriNCe's transport convention, like p/ν/γ). The e± state lives
            # on the em_grid in dN/dγ_e (γ_e=E_e/m_ec², the SingleZoneSolver/cooling
            # convention used by M_e, Q_pair, Qe_prim). Convert the e± rows:
            # dN/dE_e → dN/dγ_e = ×(dE/dγ)=×m_ec². WITHOUT this the photo-pion e±
            # were over-injected by 1/m_ec²≈1957× — the high-γ pair over-production
            # / 39× high-E synchrotron (γ/p/ν stay dN/dE, consistent with their
            # states → field 0.99× and ν 1.30× were unaffected).
            inj_had[self.sl_ep] *= _ME_C2_GeV
            inj_had[self.sl_em] *= _ME_C2_GeV
            rhs += inj_had

        # --- e± radiative cooling + escape (advection on em_grid) ---
        M_e = self.sze.cooling_operator(loss_stencil=self.loss_stencil,
                                        n_state=self._n_e_frozen).tocsr()
        rhs[self.sl_ep] += M_e @ n_ep
        rhs[self.sl_em] += M_e @ n_em

        # --- γγ absorption rate + pair injection (γ→e±) ---
        tgg = gamma_gamma_abs_inv(self.Eg, self.field, self._eps_soft)
        # γγ pair injection is the TOTAL pair spectrum (both leptons) → split 0.5/0.5.
        Q_pair = (self._pair_injection(n_g, tgg) if self.gg_pairs
                  else np.zeros_like(self.sze.g))      # dN/dγ_e, shared e+/e-
        rhs[self.sl_ep] += 0.5 * Q_pair
        rhs[self.sl_em] += 0.5 * Q_pair
        # Bethe-Heitler: _bh_pair_injection returns the SINGLE-lepton kernel_BH_elec spectrum, so
        # the e+ AND e- EACH follow it (a BH event makes a pair). Inject it fully into both species
        # — do NOT lump it into the γγ Q_pair and 0.5-split, which would drop the second lepton and
        # halve the BH pair power (energy-balance: P_inj/P_loss 0.47→~0.94). See
        # wiki lessons/native-bh-single-lepton-half.
        if self.bethe_heitler and self.sl_p is not None:
            Q_bh = self._bh_pair_injection(state[self.sl_p])
            rhs[self.sl_ep] += Q_bh
            rhs[self.sl_em] += Q_bh

        # --- primary lepton injection (into e-) ---
        if Qe_prim is not None:
            rhs[self.sl_em] += Qe_prim

        # --- e±→γ emission (+ proton synchrotron) + γ escape + γγ sink ---
        Q_g_emit = self._emission_onto_emgrid(n_e_tot, n_p)
        rhs[self.sl_g] += Q_g_emit - (1.0 / self.t_esc + tgg) * n_g

        # --- proton injection (pγ loss + secondary coupling already in c·J) ---
        if self.sl_p is not None and Qp is not None:
            rhs[self.sl_p] += Qp

        # --- escape of ALL cr-grid CR species (p, p̄, n, n̄, ν): free-stream R/c.
        # Neutron escapes here (tracked under finite tau_dec_threshold) instead of
        # β-decaying in-zone — the n→p e⁻ ν̄_e fix.
        for sl in self.sl_cr_escape:
            rhs[sl] -= state[sl] / self.t_esc

        return rhs, tgg

    def _diagonal(self, state):
        """Frozen ETD2 diagonal D = the stiff exactly-integrated part:
        e± cooling self-loss + escape; γ escape + γγ sink; proton pγ loss +
        escape; ν escape. Field-dependent pieces frozen at step start."""
        n_ep = state[self.sl_ep]; n_em = state[self.sl_em]; n_g = state[self.sl_g]
        n_p = state[self.sl_p] if self.sl_p is not None else None
        self._n_e_frozen = n_ep + n_em        # freeze fv2 limiter weights @ step start
        self.sze._fv2_dt = self._dt           # CFL stiffness cap for the fv2 limiter
        self._set_field(n_ep + n_em, n_g, n_p)
        self._set_ic_cooling_target()
        d = np.zeros_like(state)

        # rebuild + freeze the (expensive) hadronic pfield fold once per step
        self._L_had_frozen = None
        if self.hadronic:
            self._L_had_frozen = self._hadr_matrix()

        M_e = self.sze.cooling_operator(loss_stencil=self.loss_stencil,
                                        n_state=self._n_e_frozen).tocsr()
        diag_e = np.asarray(M_e.diagonal())
        d[self.sl_ep] += diag_e
        d[self.sl_em] += diag_e

        tgg = gamma_gamma_abs_inv(self.Eg, self.field, self._eps_soft)
        d[self.sl_g] += -(1.0 / self.t_esc + tgg)

        # cr-grid species: c·J self-loss diagonal (pγ/nγ loss; ν≈0) + escape R/c
        dh = (np.asarray(self._L_had_frozen.diagonal()) if self.hadronic
              else np.zeros_like(state))
        for sl in self.sl_cr_escape:
            d[sl] += dh[sl] - 1.0 / self.t_esc
        return d

    # ---------------- the ETD2 march ----------------
    def solve_etd2(self, Qp=None, Qe_prim=None, dt=None, n_steps=4000,
                   rtol=1e-6, check_every=10, n0=None, verbose=True,
                   record_times=None):
        """March the unified state to steady state (or record a light curve).

        Parameters
        ----------
        Qp : proton injection dN/dE_p [GeV⁻¹cm⁻³s⁻¹] on ``run.cr_grid`` (or None).
        Qe_prim : primary e⁻ injection dN/dγ_e on ``self.sze.g`` (or None).
        dt : ETD2 step [s] (default t_esc; ETD handles faster modes exactly).
        record_times : optional sorted list of times [s] at which to snapshot the
            full state → the light curve. If given, also returns ``snapshots``.

        Returns dict with the converged ``state`` and per-block views.
        """
        from prince_cr.solvers.etd2 import etd2_step, _step_buffers

        bufs = _step_buffers(self.dim, np)
        state = (np.zeros(self.dim) if n0 is None else np.asarray(n0, float).copy())
        if dt is None:
            dt = self.t_esc
        self._dt = dt

        snapshots = []
        rec = sorted(record_times) if record_times else []
        hist = []
        prev = state.copy()
        t = 0.0
        n_done = 0
        for k in range(n_steps):
            d = self._diagonal(state)
            # Every diagonal entry is physically a LOSS (cooling, escape, γγ, pγ);
            # production is off-diagonal. A positive d is a discretisation artefact
            # (e.g. the higher-order loss stencil's one-sided high-γ boundary row in
            # the empty region) and would make exp(dt·d) overflow → NaN. Clamp ≤0;
            # the (true_diag−0)·x for those cells stays explicit in apply_F (safe —
            # those cells carry ~no mass). ETD2 stays consistent: apply_F uses the
            # SAME clamped d.
            np.minimum(d, 0.0, out=d)

            def apply_F(x, out, _d=d):
                r, _ = self._full_rhs(x, Qp, Qe_prim)
                out[:] = r
                out -= _d * x

            etd2_step(state, dt, d, apply_F, bufs, np)
            np.clip(state, 0.0, None, out=state)
            t += dt
            n_done = k + 1
            while rec and t >= rec[0]:
                snapshots.append((rec.pop(0), state.copy()))
            if (k % check_every) == 0:
                denom = max(float(np.abs(state).max()), 1e-300)
                res = float(np.abs(state - prev).max()) / denom
                hist.append(res)
                if verbose:
                    Ug = float(np.trapezoid(self.Eg * state[self.sl_g], self.Eg))
                    print(f"  native-etd2 it{k:04d}  t={t:.3e}s  "
                          f"U_γ={Ug:.4e} GeV/cm³  resid={res:.3e}")
                if res < rtol and k >= check_every:
                    break
                prev = state.copy()

        out = self._unpack(state)
        out.update(history=hist, n_steps=n_done, t=t)
        if record_times:
            out["snapshots"] = snapshots
        return out

    def _unpack(self, state):
        """Per-block views of the state vector + the converged common field."""
        n_g = state[self.sl_g]
        n_p = state[self.sl_p] if self.sl_p is not None else None
        # OUTPUT field carries the full SSC (sub-MeV IC folded in); solve unaffected
        self._set_field(state[self.sl_ep] + state[self.sl_em], n_g, n_p, include_ic=True)
        nu_tot = sum(state[sl] for sl in self.nu_sls) if self.nu_sls else None
        return {
            "state": state,
            "field": self.field,
            "gamma_e": self.sze.g,
            "n_ep": state[self.sl_ep], "n_em": state[self.sl_em],
            "n_e": state[self.sl_ep] + state[self.sl_em],
            "E_ph": self.Eg, "n_gamma": n_g,
            "E_p": (self.cr.grid if self.sl_p is not None else None),
            "n_p": (state[self.sl_p] if self.sl_p is not None else None),
            "E_nu": self.cr.grid, "n_nu": nu_tot,
        }
