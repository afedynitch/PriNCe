"""Contains solvers, i.e. integrators, kernels, steppers, for PriNCe."""

import numpy as np
import scipy.sparse as sp

from prince_cr.cosmology import H
from prince_cr.data import PRINCE_UNITS, EnergyGrid
from prince_cr.util import PrinceProgressBar, info
import prince_cr.config as config

from .partial_diff import DifferentialOperator


# Sentinel for ``_build_mkl_handle(blocksize=...)`` so callers can ask
# for ``None`` (force CSR) without colliding with "use the config default".
_DEFAULT = object()


def _resolve_prince_run(args, kwargs):
    """Pull ``prince_run`` from positional/keyword args of the solver constructor.

    Both façade ``__new__`` dispatchers need this to pick a backend subclass
    before ``__init__`` runs.
    """
    if "prince_run" in kwargs:
        return kwargs["prince_run"]
    if len(args) >= 3:
        return args[2]
    raise TypeError(
        "ETD2 solver constructor requires `prince_run` (positional or keyword)."
    )


class UHECRPropagationResult(object):
    """Reduced version of solver class, that only holds the result vector and defined add and multiply"""

    def __init__(self, state, egrid, spec_man):
        self.spec_man = spec_man
        self.egrid = egrid
        self.state = state

    def to_dict(self):
        dic = {}
        dic["egrid"] = self.egrid
        dic["state"] = self.state
        dic["known_spec"] = self.known_species

        return dic

    @classmethod
    def from_dict(cls, dic):
        egrid = dic["egrid"]
        edim = egrid.size
        state = dic["state"]
        known_spec = dic["known_spec"]

        from ..data import SpeciesManager

        spec_man = SpeciesManager(known_spec, edim)

        return cls(state, egrid, spec_man)

    @property
    def known_species(self):
        return self.spec_man.known_species

    def __add__(self, other):
        cumstate = self.state + other.state

        if not np.array_equal(self.egrid, other.egrid):
            raise Exception(
                "Cannot add Propagation Results, they are defined in different energy grids!"
            )
        if not np.array_equal(self.known_species, other.known_species):
            raise Exception(
                "Cannot add Propagation Results, have different species managers!"
            )
        else:
            return UHECRPropagationResult(cumstate, self.egrid, self.spec_man)

    def __mul__(self, number):
        if not np.isscalar(number):
            raise Exception(
                "Can only multiply result by scalar number, got type {:} instead!".format(
                    type(number)
                )
            )
        else:
            newstate = self.state * number
            return UHECRPropagationResult(newstate, self.egrid, self.spec_man)

    def get_solution(self, pdg_id):
        """Returns the spectrum in energy per nucleon"""
        spec = self.spec_man.pdgid2sref[pdg_id]
        return self.egrid, self.state[spec.lidx() : spec.uidx()]

    def get_solution_scale(self, pdg_id, epow=0):
        """Returns the spectrum scaled back to total energy"""
        spec = self.spec_man.pdgid2sref[pdg_id]
        egrid = spec.A * self.egrid
        return egrid, egrid**epow * self.state[spec.lidx() : spec.uidx()] / spec.A

    def _check_id_grid(self, pdg_ids, egrid):
        # Take egrid from first id ( doesn't cover the range for iron for example)
        # create a common egrid or used supplied one
        if egrid is None:
            max_mass = max([s.A for s in self.spec_man.species_refs])
            emin_log, emax_log, nbins = list(config.cosmic_ray_grid)
            emax_log = np.log10(max_mass) + emax_log
            nbins *= 4
            com_egrid = EnergyGrid(emin_log, emax_log, nbins).grid
        else:
            com_egrid = egrid

        from prince_cr.util import is_nucleus

        if isinstance(pdg_ids, list):
            pass
        elif pdg_ids == "CR":
            pdg_ids = [s for s in self.known_species if is_nucleus(s)]
        elif pdg_ids == "nu":
            # all (anti-)neutrinos: ν_e (12), ν_μ (14), ν_τ (16) and their CP partners.
            pdg_ids = [
                s for s in self.known_species
                if s in (12, -12, 14, -14, 16, -16)
            ]
        elif pdg_ids == "all":
            pdg_ids = self.known_species
        elif isinstance(pdg_ids, tuple):
            select, vmin, vmax = pdg_ids
            pdg_ids = [s for s in self.known_species if vmin <= select(s) <= vmax]

        return pdg_ids, com_egrid

    def _collect_interpolated_spectra(self, pdg_ids, epow, egrid=None):
        """Collect interpolated spectra in a 2D array. Used by
        get_solution_group and get_lnA"""
        pdg_ids, com_egrid = self._check_id_grid(pdg_ids, egrid)

        # collect all the spectra in 2d array of dimension
        spectra = np.zeros((len(pdg_ids), com_egrid.size))
        for idx, pid in enumerate(pdg_ids):
            curr_egrid, curr_spec = self.get_solution_scale(pid, epow)
            mask = curr_spec > 0.0
            if np.count_nonzero(mask) > 0:
                res = np.exp(
                    np.interp(
                        np.log(com_egrid),
                        np.log(curr_egrid[mask]),
                        np.log(curr_spec[mask]),
                        left=np.nan,
                        right=np.nan,
                    )
                )
            else:
                res = np.zeros_like(com_egrid)
            spectra[idx] = np.nan_to_num(res)

        return pdg_ids, com_egrid, spectra

    def get_solution_group(self, pdg_ids, epow=3, egrid=None):
        """Return the summed spectrum (in total energy) for all elements in the range"""

        _, com_egrid, spectra = self._collect_interpolated_spectra(pdg_ids, epow, egrid)
        spectrum = spectra.sum(axis=0)

        return com_egrid, spectrum

    def get_lnA(self, pdg_ids, egrid=None):
        """Return the average ln(A) as a function of total energy for all
        elements in the range"""

        pdg_ids, com_egrid, spectra = self._collect_interpolated_spectra(
            pdg_ids, 0, egrid
        )

        # get the average and variance by using the spectra as weights
        lnA = np.array([np.log(self.spec_man.pdgid2sref[el].A) for el in pdg_ids])
        total = spectra.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            average = (lnA[:, np.newaxis] * spectra).sum(axis=0) / total
            variance = (lnA[:, np.newaxis] ** 2 * spectra).sum(
                axis=0
            ) / total - average**2

        return com_egrid, average, variance

    def get_energy_density(self, pdg_id):
        from scipy.integrate import trapezoid as trapz

        A = self.spec_man.pdgid2sref[pdg_id].A
        return trapz(A * self.egrid * self.get_solution(pdg_id), self.egrid)


class UHECRPropagationSolver(object):
    def __init__(
        self,
        initial_z,
        final_z,
        prince_run,
        enable_adiabatic_losses=True,
        enable_pairprod_losses=True,
        enable_photohad_losses=True,
        enable_injection_jacobian=True,
        enable_partial_diff_jacobian=True,
        enable_decay=False,
        enable_decay_jacobian=True,
        z_offset=0.0,
    ):
        self.initial_z = initial_z + z_offset
        self.final_z = final_z + z_offset
        self.z_offset = z_offset

        # Per-run backend dispatch handle. Solver code reads
        # ``self.backend.<knob>`` rather than ``config.<knob>`` so a
        # single process can run multiple PriNCeRun instances with
        # different backend settings.
        self.prince_run = prince_run
        self.backend = prince_run.backend

        self.current_z_rates = None
        self.recomp_z_threshold = self.backend.update_rates_z_threshold

        self.spec_man = prince_run.spec_man
        self.egrid = prince_run.cr_grid.grid
        self.ebins = prince_run.cr_grid.bins
        self.widths = prince_run.cr_grid.widths
        # Flags to enable/disable different loss types
        self.enable_adiabatic_losses = enable_adiabatic_losses
        self.enable_pairprod_losses = enable_pairprod_losses
        self.enable_photohad_losses = enable_photohad_losses
        # Flag for Jacobian injection:
        # if True injection in jacobion
        # if False injection only per dz-step
        self.enable_injection_jacobian = enable_injection_jacobian
        self.enable_partial_diff_jacobian = enable_partial_diff_jacobian
        # Explicit decay: when True, the constant Λ operator is folded into
        # ``L(z)`` as a third-operator term ``dldz · Λ`` parallel to the
        # photo-hadronic and continuous-loss blocks. Effective only when
        # the cross-section chain reducer has been gated off so unstable
        # mothers reach the state vector (``config.enable_explicit_decay``;
        # see methods/explicit-decay-kernel-design.md). Silently a no-op
        # otherwise — Λ_diag stays empty when no species in the state
        # vector is unstable, so the per-step operator skips the add.
        self.enable_decay = enable_decay
        self.enable_decay_jacobian = enable_decay_jacobian

        self.had_int_rates = prince_run.int_rates
        # EM cascade Jacobian (γ/e± couplings), summed into M(z) if enabled.
        self.em_int_rates = getattr(prince_run, "em_int_rates", None)
        # Co-evolved EM cascade (operator-split): each z-step the EM particles
        # produced by the nuclei are reprocessed by the per-z saturated cascade
        # transfer T(z) (semi-analytic for the stiff γγ/IC part), while ETD2
        # carries the redshift transport. See cascade.cascade_transfer_matrix
        # and methods/em-cascade-in-transport.
        self.enable_em_cascade = getattr(prince_run, "enable_em_cascade", False)
        self._em_T = None
        self._em_T_cache = None  # (z_nodes, [T(z)]) built once per solve
        self._em_dz = 1e-2  # per-step EM transfer step; set in solve()
        # Bethe-Heitler e± source: per-z energy-normalized pair shape R(z).
        self._em_bh = None
        self._em_bh_cache = None   # (z_nodes, [R_BH(z)]) built once per solve
        self._em_z = None          # current redshift (for dl_step in BH inject)
        # Tier 3 cross-grid coupling regrid; stays None unless decoupled (set
        # in the enable_em_cascade block below). Referenced unconditionally in
        # _refresh_z_caches, so default it here for the nuclear-only path.
        self._em_regrid_R = None
        self._em_native = False
        self._em_grid_obj = None
        if self.enable_em_cascade:
            sm = prince_run.spec_man
            self._em_gamma_sl = sm.pdgid2sref[22].sl
            self._em_lep_sl = [sm.pdgid2sref[p].sl for p in (11, -11) if p in sm.pdgid2sref]
            # γ/e± live on their own grid when decoupled (Tier 3); else they
            # share the cr grid. A=1 so the grid array IS the energy array.
            em_grid = getattr(prince_run, "em_grid", None)
            self._em_E = (em_grid.grid if em_grid is not None
                          else prince_run.cr_grid.grid)
            # Native cross-grid coupling: builders write γ/e± daughter rows
            # directly on the EM grid → no runtime regrid R. The Λ_off decay
            # builder needs the EM bin edges for its rectangular EM blocks.
            self._em_native = bool(
                getattr(prince_run, "enable_em_native_coupling", False)
            )
            self._em_grid_obj = em_grid
            # proton slice + grid for the BH pair source. Every nucleus lives on
            # the nuclear (cr) grid even when the EM sector is decoupled, so the
            # BH source maps cr-grid energies (input) onto the EM grid (output
            # e±) — see _em_bh_at / _inject_bh_pairs.
            self._em_proton_sl = sm.pdgid2sref[2212].sl if 2212 in sm.pdgid2sref else None
            self._em_proton_E = prince_run.cr_grid.grid
            # All charged nuclei drive Bethe-Heitler, not just protons. The cr
            # grid is per-NUCLEON energy, so γ = E_cr/m_p is identical across
            # species → the e± SHAPE matrix R_BH (built once for the proton in
            # _em_bh_at) is reused for every nucleus. The Z² enhancement and the
            # 1/A per-nucleon factor already live in the pair-loss vector
            # (interaction_rates.ContinuousPairProductionLossRate.scale_vec =
            # units·Z²/A); _inject_bh_pairs multiplies each species' weight by A
            # so the deposited e± energy equals the nucleus' TOTAL BH loss
            # (∝ Z²). See lessons/em-cascade-bh-nuclei.
            self._em_bh_species = [
                (s.sl, float(s.A))
                for s in sm.species_refs
                if getattr(s, "is_nucleus", False) and abs(s.charge) >= 1
            ]
            # Tier 3: cross-grid coupling regrid. The photo-nuclear response and
            # decay Λ_off emit γ/e± daughter rows at cr-grid indices; left-
            # multiplying the assembled coupling by R remaps those rows onto the
            # EM grid (energy-conserving). Identity on nuclear blocks → None when
            # not decoupled (no-op). See methods/em-grid-boost-tier3-plan.md §3.
            self._em_regrid_R = (
                self._build_em_regrid_operator(prince_run)
                if (
                    getattr(prince_run, "enable_em_decoupled_grid", False)
                    and not self._em_native
                )
                else None
            )
        self.adia_loss_rates_grid = prince_run.adia_loss_rates_grid
        self.pair_loss_rates_grid = prince_run.pair_loss_rates_grid
        self.adia_loss_rates_bins = prince_run.adia_loss_rates_bins
        self.pair_loss_rates_bins = prince_run.pair_loss_rates_bins
        self.intp = None

        self.state = np.zeros(prince_run.dim_states)
        self.result = None
        self.dim_states = prince_run.dim_states

        self.list_of_sources = []

        self.diff_operator = DifferentialOperator(
            prince_run.cr_grid,
            prince_run.spec_man.nspec,
            spec_man=prince_run.spec_man,
            grids=getattr(prince_run, "grids", None),
        ).operator

    @property
    def known_species(self):
        return self.spec_man.known_species

    @property
    def res(self):
        if self.result is None:
            self.result = UHECRPropagationResult(self.state, self.egrid, self.spec_man)
        return self.result

    def pre_step_hook(self, t):
        """This function is called after initializing the solver
        but before the first step."""
        pass

    def post_step_hook(self, t):
        """This call-back like function is called after each successful step"""
        pass

    def add_source_class(self, source_instance):
        self.list_of_sources.append(source_instance)

    def dldz(self, z):
        return -1.0 / ((1.0 + z) * H(z) * PRINCE_UNITS.cm2sec)

    def injection(self, dz, z):
        """This needs to return the injection rate
        at each redshift value z"""
        f = self.dldz(z) * dz * PRINCE_UNITS.cm2sec
        if len(self.list_of_sources) > 1:
            return f * np.sum([s.injection_rate(z) for s in self.list_of_sources], axis=0)
        else:
            return f * self.list_of_sources[0].injection_rate(z)


class UHECRPropagationSolverETD2(UHECRPropagationSolver):
    """Exponential time-differencing RK2 (Cox-Matthews) integrator.

    This is the algorithm parent. Construct it directly and ``__new__``
    dispatches to the appropriate backend subclass:

    * ``ETD2SolverCPU`` for ``linear_algebra_backend in {"scipy", "mkl"}``
    * ``ETD2SolverCUPY`` for ``linear_algebra_backend == "cupy"`` (eager
      fused kernels by default; CUDA Graph capture/replay when
      ``backend.use_cuda_graphs`` is True).

    The ETD2 method treats the diagonal of
    ``L(z) = J(z) + dl/dz · D · diag(κ(z))`` exactly via
    ``exp(h · diag(L))`` and the off-diagonal block with two SpMVs per
    stage (4 SpMVs / step). Source term ``b(z) = injection(z)`` enters via
    the same φ₁/φ₂ machinery, frozen at step start to preserve 2nd order.

    Caching: the expensive z-dependent pieces — the photo-hadronic rate
    matrix ``M_raw(z)`` and the pair-production loss vector ``κ_pair(z)``
    (CIB interpolated at ``dim_cr × xi_steps`` points) — are refreshed
    together at ``recomp_z_threshold`` resolution. Truly cheap pieces
    (``dl/dz(z)``, ``κ_adia(z)`` closed form, source ``b(z)``) are
    recomputed every step.
    """

    def __new__(cls, *args, **kwargs):
        if cls is UHECRPropagationSolverETD2:
            prince_run = _resolve_prince_run(args, kwargs)
            backend = prince_run.backend.linear_algebra_backend.lower()
            target = ETD2SolverCUPY if backend == "cupy" else ETD2SolverCPU
            return object.__new__(target)
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pieces refreshed at ``recomp_z_threshold`` resolution (z tracked by
        # ``self.current_z_rates`` from the base class).
        self._etd2_M_raw_diag = None
        self._etd2_M_raw_off = None
        # κ_pair(z) is expensive (CIB interpolated at dim_cr × xi_steps points).
        # κ_adia(z) is closed-form and trivially cheap, recomputed per step.
        self._etd2_kappa_pair_cached = None
        # Constant (or one-shot per solve) FD operator split.
        self._etd2_D_diag = None
        self._etd2_D_off = None
        # Host scratch for κ accumulation. Used by both backends — the cupy
        # path uploads from this buffer to a device counterpart.
        self._etd2_kappa_buf = None
        # Persistent ``apply_F`` closure. Identity is stable for a solve.
        self._etd2_apply_F = None

        self._backend = self.backend.linear_algebra_backend.lower()
        self._is_cupy_backend = self._backend == "cupy"

    # ------------------------------------------------------------------
    # Abstract API — implemented by backend subclasses.
    # ------------------------------------------------------------------
    def _refresh_z_caches(self, z):
        raise NotImplementedError

    def _operator_at(self, z):
        raise NotImplementedError

    def _ensure_apply_F(self):
        raise NotImplementedError

    def _ensure_D_split(self):
        raise NotImplementedError

    def _ensure_Lambda_split(self):
        """Backend-specific build of the constant decay operator Λ.

        Default no-op: backends that haven't implemented explicit decay
        yet (cupy paths until Step E lands) silently skip the build, so
        ``enable_decay=True`` simply has no effect on those backends.
        """
        pass

    def _init_state(self):
        raise NotImplementedError

    def _reset_for_solve(self):
        """Backend-specific cleanup at the start of each ``solve()`` call."""
        pass

    def _run_integration(self, state, z_grid, step_hook=None):
        raise NotImplementedError

    def _em_transfer_at(self, z):
        """Cascade transfer T(z) (the per-z-step stiffness-split operator,
        passed ``dz=self._em_dz`` so only the stiff EM part cascades within a
        step; near-E_abs photons are carried by ETD2 between steps).

        Now that the IC/γγ cross-section kernels are field-free and cached
        (built once, in-memory + optional disk), building T at the EXACT
        requested z is just a contraction + vectorized assembly + the cascade
        iteration. So the default is exact per-z (``config.em_transfer_z_nodes
        == 0``): T is built at each solver refresh and memoised by z — no
        coarse-grid interpolation. Set ``em_transfer_z_nodes = N > 0`` to
        restore the legacy N-node log(1+z) grid + nearest-neighbour select
        (cheaper, slightly approximate in z)."""
        import prince_cr.config as config
        from prince_cr.cascade.cascade import (
            cascade_transfer_matrix, kinetic_cascade_transfer,
        )
        # kinetic single-scatter cascade by default (fills the E_X..E_abs
        # plateau; matches Kalashev Fig 2). Legacy cooled path kept via flag.
        transfer = (
            kinetic_cascade_transfer
            if getattr(config, "em_kinetic_cascade", True)
            else cascade_transfer_matrix
        )
        field = self.prince_run.photon_field
        n_nodes = int(getattr(config, "em_transfer_z_nodes", 0))

        if n_nodes > 0:
            # Legacy coarse-grid interpolation (nearest of N nodes).
            if not isinstance(self._em_T_cache, tuple):
                zlo, zhi = sorted((float(self.final_z), float(self.initial_z)))
                zc = np.expm1(np.linspace(np.log1p(zlo), np.log1p(zhi), n_nodes))
                Ts = [transfer(self._em_E, zz, field, dz=self._em_dz) for zz in zc]
                self._em_T_cache = (zc, Ts)
            zc, Ts = self._em_T_cache
            return Ts[int(np.argmin(np.abs(zc - z)))]

        # Exact per-z (default): memoise by z so a window's repeated z reuses.
        if not isinstance(self._em_T_cache, dict):
            self._em_T_cache = {}
        key = round(float(z), 7)
        T = self._em_T_cache.get(key)
        if T is None:
            T = transfer(self._em_E, float(z), field, dz=self._em_dz)
            self._em_T_cache[key] = T
        return T

    def _apply_em_cascade(self, state):
        """Operator-split EM step (co-evolved cascade): reprocess the EM
        particles (γ + e±) through the per-z saturated cascade transfer
        ``self._em_T`` and deposit the escaping photons back into γ. The stiff
        above-E_abs γγ/IC cascade is handled semi-analytically by ``T``;
        sub-E_abs photons pass through unchanged; e± are fully converted to
        photons. Energy-conserving. ``state`` is mutated in place."""
        if self._em_T is None:
            return
        T_gamma, T_electron = self._em_T
        ph = np.array(state[self._em_gamma_sl])
        lep = np.zeros_like(ph)
        for sl in self._em_lep_sl:
            lep = lep + state[sl]
            state[sl] = 0.0
        # photons γγ-cascade (T_gamma); e± IC-cool first then cascade (T_electron)
        state[self._em_gamma_sl] = T_gamma @ ph + T_electron @ lep

    def _em_bh_at(self, z):
        """Nearest precomputed Bethe-Heitler e± pair-shape matrix R_BH(z).

        ``R_BH[i, j]`` is the energy-normalized e± spectrum dN/dE_e (one lepton
        sign) from a proton at ``E[j]`` on the photon field at z, with columns
        scaled so the *pair* carries E[j] (``2·∫E_e R[:,j] dE_e = E[j]``). The
        per-step injection (``_inject_bh_pairs``) rescales each column by the
        proton's actual BH energy-loss rate, so the deposited pair energy equals
        the energy the proton sink removes — exact conservation. Built once on a
        the BH cross-section kernel is field-free and cached (built once), the
        per-z R_BH is just a contraction + the energy-pinning normalization —
        so the default is exact per-z (``config.em_transfer_z_nodes == 0``),
        memoised by z. Set ``> 0`` for the legacy N-node nearest select."""
        import prince_cr.config as config
        from prince_cr.cascade.bethe_heitler import bh_pair_shape_matrix

        field = self.prince_run.photon_field
        # Output e± energies on the EM grid; input proton energies on the
        # nuclear grid. When the grids coincide (shared-grid path) this is the
        # original square R_BH; when decoupled R_BH is (d_em, d_cr).
        E_out = self._em_E
        E_in = self._em_proton_E
        eps = np.logspace(-12.5, -7.0, 40)  # CMB+EBL band [GeV]

        def _build(zz):
            n_eps = np.asarray(field.get_photon_density(eps, zz), dtype="double")
            n_eps = np.where(np.isfinite(n_eps) & (n_eps > 0), n_eps, 0.0)
            return bh_pair_shape_matrix(E_out, E_in, eps, n_eps)

        n_nodes = int(getattr(config, "em_transfer_z_nodes", 0))
        if n_nodes > 0:
            if not isinstance(self._em_bh_cache, tuple):
                zlo, zhi = sorted((float(self.final_z), float(self.initial_z)))
                zc = np.expm1(np.linspace(np.log1p(zlo), np.log1p(zhi), n_nodes))
                self._em_bh_cache = (zc, [_build(zz) for zz in zc])
            zc, Rs = self._em_bh_cache
            return Rs[int(np.argmin(np.abs(zc - z)))]

        if not isinstance(self._em_bh_cache, dict):
            self._em_bh_cache = {}
        key = round(float(z), 7)
        R = self._em_bh_cache.get(key)
        if R is None:
            R = _build(float(z))
            self._em_bh_cache[key] = R
        return R

    def _inject_bh_pairs(self, state):
        """Inject Bethe-Heitler e⁺e⁻ into the EM species for this z-step.

        Every charged nucleus loses BH energy via the continuous pair-loss term;
        here we deposit that energy as e± (per sign) with the W-kernel shape
        ``self._em_bh`` (R_BH), scaled by each species' BH energy-loss rate so
        the total injected pair energy = the nucleus energy lost
        (energy-conserving). Mutates ``state`` in place; call BEFORE
        ``_apply_em_cascade``.

        The cr grid is per-NUCLEON energy, so γ = E_cr/m_p is identical for all
        species and the proton-built R_BH gives the correct e± SHAPE for every
        nucleus (the pair lab energy is set by γ, i.e. the per-nucleon scale).
        The pair-loss vector already carries units·Z²/A per species; multiplying
        the weight by A makes the deposited e± energy equal the nucleus' TOTAL
        BH loss (∝ Z²) — Z² more pairs than a proton at the same γ, as expected.
        """
        if self._em_bh is None or not self._em_bh_species:
            return
        from prince_cr.cosmology import H
        from prince_cr.data import PRINCE_UNITS

        z = self._em_z
        # The BH source integrates over the (cr) per-nucleon grid; R_BH maps it
        # to the EM grid. E_p drives the per-nucleus energy-loss fraction.
        E_p = self._em_proton_E
        dE = np.gradient(E_p)
        loss_full = self.pair_loss_rates_grid.loss_vector(z)             # GeV/cm, dim_states
        dl_step = PRINCE_UNITS.c * abs(self._em_dz) / ((1.0 + z) * H(z))  # cm
        # Accumulate the per-nucleon e± source over all charged species.
        # column weight_s = A·(energy lost per nucleon this step)/E_cr · n_s
        w = np.zeros_like(E_p)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_E = np.where(E_p > 0, 1.0 / E_p, 0.0)
            for sl, A in self._em_bh_species:
                n_s = np.asarray(state[sl], dtype="double")
                w += A * loss_full[sl] * dl_step * inv_E * n_s * dE
        e_src = self._em_bh @ w  # dN/dE_e per lepton sign (on the EM grid)
        for sl in self._em_lep_sl:
            state[sl] = state[sl] + e_src

    def _build_em_regrid_operator(self, prince_run):
        """Build the constant ``(dim_states, dim_states)`` daughter-row regrid R
        for the decoupled EM grid (Tier 3 step 3).

        Block-diagonal in princeidx order: identity on every nuclear block; on
        each EM species' block the cr→em overlap matrix ``P`` (``d_em × d_cr``)
        padded with zero columns to ``d_em × d_em``. Left-multiplying the
        assembled coupling by R remaps γ/e± daughter rows — written by the
        response builder / decay Λ_off at cr indices ``da.lidx() + 0..d_cr-1`` —
        onto the EM grid, conserving daughter number.
        """
        from scipy.sparse import block_diag, identity, hstack, csr_matrix
        from prince_cr.data import energy_regrid_matrix

        sm = prince_run.spec_man
        cr, em = prince_run.cr_grid, prince_run.em_grid
        if em.d < cr.d:
            raise ValueError(
                "EM grid ({0} bins) is coarser than the nuclear grid ({1}); the "
                "response builder writes d_cr daughter indices, which would "
                "overflow the EM block. Raise config.em_grid_bins_dec.".format(
                    em.d, cr.d
                )
            )
        P = energy_regrid_matrix(cr, em)                       # (d_em, d_cr)
        P_pad = hstack([P, csr_matrix((em.d, em.d - cr.d))], format="csr")
        blocks = [
            P_pad if s.grid_tag != "default" else identity(s._tr_dim, format="csr")
            for s in sorted(sm.species_refs, key=lambda x: x.princeidx)
        ]
        return block_diag(blocks, format="csr")

    def _finalize_state(self, state):
        """Backend-specific post-solve transform (e.g. cupy → host array)."""
        return state

    def _coerce_state(self, state):
        """Backend-specific coercion of the initial state (identity on the
        host backends; the cupy backend uploads host arrays to device)."""
        return state

    def close(self):
        """Release backend resources. Idempotent. Safe to skip."""
        pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Shared utility: index-map for CPU in-place M_off refresh.
    # ------------------------------------------------------------------
    @staticmethod
    def _build_M_off_index_map(M, M_off):
        """Pre-compute index maps from M.data into M_off.data and diag.

        Returns ``(off_to_M_idx, diag_to_M_idx)``:

        * ``off_to_M_idx`` (length ``M_off.nnz``): for each entry of
          ``M_off.data``, the index in ``M.data`` it came from.
        * ``diag_to_M_idx`` (length ``M.shape[0]``): for each row r, the
          index in ``M.data`` of the (r, r) entry, or -1 if absent.

        Built once at the first cache window in O(nnz). On subsequent
        windows the per-window cost drops to a couple of ``data[idx]``
        gathers — typically <0.5 ms vs ~3 ms for the full
        ``mkl_sparse_d_create_csr + mkl_sparse_set_mv_hint +
        mkl_sparse_optimize`` rebuild.

        Vectorized via ``numpy.searchsorted`` over flat (row, col) keys —
        at production grid (M.nnz ≈ 1.2 M, M_off.nnz ≈ 895 k) the
        original per-row two-pointer Python loop showed up as ~0.9 s of
        one-shot setup time in cProfile of a 4.6 s solve. The
        searchsorted version runs in ~5 ms while preserving the same
        contract.

        Robust to: unsorted CSR column indices (scipy doesn't enforce
        sorting after ``+/-/eliminate_zeros``); and to ``M_off`` being a
        strict subset of ``M`` minus the diagonal (``eliminate_zeros``
        may also drop off-diagonal entries that happened to be zero in
        ``M``, though this is unusual for the photo-hadronic Jacobian).
        """
        if not (sp.isspmatrix_csr(M) and sp.isspmatrix_csr(M_off)):
            raise TypeError("Index map requires CSR M and CSR M_off.")
        if M.shape != M_off.shape:
            raise ValueError("M and M_off shapes disagree.")

        # scipy's binary ops don't guarantee per-row sorted column
        # indices; sort both in place once so the (row, col) keys
        # below come out monotonically increasing.
        if not getattr(M, "has_sorted_indices", False):
            M.sort_indices()
        if not getattr(M_off, "has_sorted_indices", False):
            M_off.sort_indices()

        n = M.shape[0]

        # Flat (row, col) keys — unique per entry, monotone non-decreasing
        # along data because CSR is row-major and the call above sorted
        # the within-row column indices.
        M_rows = np.repeat(
            np.arange(n, dtype=np.int64), np.diff(M.indptr)
        )
        M_keys = M_rows * n + M.indices.astype(np.int64, copy=False)

        # Diagonal: every k where M.indices[k] == M_rows[k] is an (r, r) entry.
        diag_mask = M.indices == M_rows
        diag_rows = M_rows[diag_mask]
        diag_to_M = np.full(n, -1, dtype=np.int64)
        # Each row appears at most once in CSR with sorted indices, so the
        # scatter is unambiguous.
        diag_to_M[diag_rows] = np.flatnonzero(diag_mask).astype(np.int64)

        # Off-diagonal: build M_off keys the same way and locate each one
        # in M_keys via searchsorted. ``M_off`` is by construction a
        # subset of M's off-diagonal entries (split_operator zeroed the
        # diagonal then eliminate_zeros may drop more), so every M_off
        # key is present in M_keys. searchsorted on a sorted array
        # returns the insertion index, which equals the matching M.data
        # index when the key is present.
        if M_off.nnz:
            Moff_rows = np.repeat(
                np.arange(n, dtype=np.int64), np.diff(M_off.indptr)
            )
            Moff_keys = Moff_rows * n + M_off.indices.astype(np.int64, copy=False)
            off_to_M = np.searchsorted(M_keys, Moff_keys).astype(np.int64)
            # Sanity: every searchsorted hit must land on a matching key.
            # Cheap O(M_off.nnz) verification — catches the (split_operator
            # contract violated) case loudly.
            if not np.array_equal(M_keys[off_to_M], Moff_keys):
                bad = np.flatnonzero(M_keys[off_to_M] != Moff_keys)
                r_bad = int(Moff_keys[bad[0]] // n)
                raise RuntimeError(
                    f"Row {r_bad}: M_off has columns with no source in M."
                )
        else:
            off_to_M = np.empty(0, dtype=np.int64)

        return off_to_M, diag_to_M

    # ------------------------------------------------------------------
    # Outer driver. Backend subclass plugs in via _init_state /
    # _reset_for_solve / _run_integration / _finalize_state hooks.
    # ------------------------------------------------------------------
    def solve(
        self,
        dz=1e-3,
        verbose=False,
        summary=False,
        progressbar=False,
    ):
        from time import time

        start_time = time()
        info(2, "ETD2: setting up integration")

        # Sign convention: integrate from initial_z (high) to final_z (low),
        # so step h = -dz < 0. Build a uniform grid of size dz, with the
        # final step truncated to land exactly on final_z.
        dz_step = -float(abs(dz))
        n_full = int(np.floor((self.final_z - self.initial_z) / dz_step))
        z_grid = self.initial_z + np.arange(n_full + 1) * dz_step
        if abs(z_grid[-1] - self.final_z) > 1e-12:
            z_grid = np.concatenate([z_grid, [self.final_z]])

        # Force first-step rebuild of the cached photo-hadronic matrix
        # and the apply_F closure pinned to the previous solve's M_off.
        self.current_z_rates = None
        self._etd2_M_raw_off = None
        self._etd2_apply_F = None
        self._em_T_cache = None
        self._em_bh_cache = None
        # Step size for the per-step (stiffness-split) EM cascade transfer.
        self._em_dz = float(abs(dz))
        self._reset_for_solve()
        self._ensure_D_split()
        self._ensure_Lambda_split()

        state = self._coerce_state(self._init_state())
        self.pre_step_hook(self.initial_z)

        nsteps = len(z_grid) - 1
        info(2, f"ETD2: integrating with {nsteps} steps of dz≈{dz_step:.2e}")
        with PrinceProgressBar(bar_type=progressbar, nsteps=nsteps) as pbar:
            _pbar_update = pbar.update if pbar.pbar is not None else None
            if self.enable_em_cascade:
                # Operator-split: after each ETD2 step, reprocess the EM
                # particles through the per-z saturated cascade transfer T(z).
                # T(z) is refreshed in _refresh_z_caches; ``state`` is updated
                # in place by the integrator, so the closure sees the latest.
                def step_hook():
                    self._inject_bh_pairs(state)   # BH e± source (before cascade)
                    self._apply_em_cascade(state)
                    if _pbar_update is not None:
                        _pbar_update()
            else:
                step_hook = _pbar_update
            self._run_integration(state, z_grid, step_hook=step_hook)

        self.post_step_hook(self.final_z)
        self.state = self._finalize_state(state)
        end_time = time()
        info(2, "ETD2: integration completed in {0:.2f} s".format(end_time - start_time))

        if summary or verbose:
            print("ETD2 summary:")
            print(f"  steps: {len(z_grid) - 1}")
            print(f"  initial z: {self.initial_z} → final z: {self.final_z}")
            print(f"  wall time: {end_time - start_time:.3f} s")


class ETD2SolverCPU(UHECRPropagationSolverETD2):
    """ETD2 backend for the scipy / MKL Sparse BLAS path.

    Per-step body uses scipy's CSR ``@`` (single-threaded BLAS-2) for the
    SpMV unless ``backend.linear_algebra_backend == "mkl"``, in which
    case the M_off / D_off are wrapped in :class:`MklSparseMatrix`
    handles for the ``mkl_sparse_d_mv`` fast path. The photo-hadronic
    M_off has its values refreshed in place across cache windows via a
    pre-computed index map (see :meth:`_build_M_off_index_map`); the
    constant FD operator D_off is built once with ``optimize=True`` and
    held for the whole solve.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # MKL handle wrapping `_etd2_M_raw_off`. Built once at the first
        # cache window and reused across subsequent windows via in-place
        # value updates — the photo-hadronic sparsity pattern is fixed at
        # init (see CLAUDE.md "Matrix Updates"), so MKL's optimised layout
        # remains valid as the data values change.
        self._etd2_M_raw_off_mkl = None
        # Index map: for each entry in M_off.data, the position in
        # M_raw.data it came from. Populated once at the first cache
        # window so subsequent windows can do
        # ``M_off.data[:] = M_raw.data[off_to_M_idx]`` without re-running
        # split_operator + mkl_sparse_optimize.
        self._etd2_M_off_to_M_idx = None
        self._etd2_M_diag_to_M_idx = None
        # MKL handle wrapping `_etd2_D_off`. One-shot for the whole solve.
        self._etd2_D_off_mkl = None
        # Persistent per-step host scratch. Sized to ``dim_states`` at
        # solve() start. Reused across all steps.
        self._etd2_d_buf = None
        self._etd2_kx_buf = None
        # Mutable per-step parameter holder, captured by the persistent
        # apply_F closure. ``dldz`` is a python float scalar; ``b`` is
        # either None or a per-step injection vector.
        self._etd2_apply_F_params = {"dldz": 1.0, "kappa": None, "b": None}
        # Λ_diag: per-bin diagonal of the constant explicit-decay operator,
        # ``-1/(c · γ · τ_rest)`` in cm⁻¹ per (species, bin). Populated
        # lazily by :meth:`_ensure_Lambda_split` on first solve when
        # ``enable_decay`` is True. Stays None when no species in the
        # state vector is unstable (e.g. when the cross-section chain
        # reducer has already folded them out — the default), in which
        # case the per-step operator skips the add.
        self._etd2_Lambda_diag = None
        # Λ_off: off-diagonal daughter-redistribution block (CSR), built
        # alongside Λ_diag. Each unstable mother contributes one block per
        # (mother, daughter) branching, with the redistribution kernel
        # pulled from ``decays.get_decay_matrix_bin_average`` and scaled
        # by ``cm2sec / (γ_mo · τ_mo) · branching_ratio``. Stays None when
        # no surviving branching has its daughter in the state vector.
        self._etd2_Lambda_off = None

    # ------------------------------------------------------------------
    # Outer-driver hooks
    # ------------------------------------------------------------------
    def _init_state(self):
        return np.zeros(self.dim_states)

    def _reset_for_solve(self):
        # Drop the M_off cache + index maps; a fresh ``_refresh_z_caches``
        # rebuilds them from the current rate matrix on the first window.
        self._etd2_M_off_to_M_idx = None
        self._etd2_M_diag_to_M_idx = None
        if self._etd2_M_raw_off_mkl is not None:
            self._etd2_M_raw_off_mkl.close()
            self._etd2_M_raw_off_mkl = None

    def _run_integration(self, state, z_grid, step_hook=None):
        from .etd2 import integrate
        integrate(
            state, z_grid,
            operator_at=self._operator_at,
            step_hook=step_hook,
        )

    def close(self):
        for attr in ("_etd2_M_raw_off_mkl", "_etd2_D_off_mkl"):
            h = getattr(self, attr, None)
            if h is not None:
                try:
                    h.close()
                except Exception:
                    pass
                setattr(self, attr, None)

    # ------------------------------------------------------------------
    # MKL handle factory
    # ------------------------------------------------------------------
    def _build_mkl_handle(self, off, optimize=True, blocksize=_DEFAULT):
        """Wrap a scipy CSR/COO/etc. off-diagonal into an MklSparseMatrix.

        Returns ``None`` if the matrix has zero nnz — the kernel skips
        the SpMV in that case rather than feeding MKL an empty handle
        (some MKL versions are squirrelly about zero-nnz CSRs).

        ``optimize=True`` chooses the inspector-executor fast path —
        ~2× faster per gemv but caches data values internally, so
        :meth:`MklSparseMatrix.update_data` cannot refresh them. Use
        ``False`` for matrices whose values change across cache windows
        (PriNCe's photo-hadronic ``M_off``); ``True`` for matrices that
        are constant for the whole solve (the FD operator ``D_off``).

        ``blocksize`` defaults to ``self.backend.mkl_bsr_blocksize``.
        Pass ``None`` to force CSR regardless of the run setting — used
        for matrices like the FD operator where the BSR sweet spot
        wasn't measured and CSR opt=True is already fast.
        """
        from .. import mkl_sparse

        if not sp.isspmatrix_csr(off):
            off = off.tocsr()
        if off.dtype != np.float64:
            off = off.astype(np.float64)
        if off.nnz == 0:
            return None
        if blocksize is _DEFAULT:
            blocksize = self.backend.mkl_bsr_blocksize
        return mkl_sparse.MklSparseMatrix(off, blocksize=blocksize, optimize=optimize)

    # ------------------------------------------------------------------
    # Cache-window refresh + per-step operator
    # ------------------------------------------------------------------
    def _refresh_z_caches(self, z):
        """Refresh all rate-cache-window-bound pieces at z.

        Bundles the photo-hadronic matrix and the pair-production loss
        vector — both expensive and both naturally tied to the same z
        resolution. Cheap pieces (adiabatic κ, dl/dz, source b(z)) stay
        per-step.
        """
        from .etd2 import split_operator

        # scale_fac=1.0 → raw rate matrix (no dldz factor); we apply dldz later.
        M = self.had_int_rates.get_hadr_jacobian(z, 1.0, force_update=True)
        if not self.enable_photohad_losses:
            # Zero matrix with full sparsity preserved.
            M = M.copy()
            M.data[:] = 0

        # Sum the EM cascade Jacobian (γ/e± couplings, 1/cm) into M. The union
        # sparsity pattern is z-stable, so the first-window index maps stay valid.
        if self.em_int_rates is not None:
            M = M + self.em_int_rates.get_hadr_jacobian(z, 1.0, force_update=True)
            M.sort_indices()

        # Tier 3 note: the γ/e± daughter-row regrid (cr→em) is NOT applied to M
        # here — doing so perturbs the matmul sparsity pattern and breaks the
        # z-stable in-place index-map fast path below. Instead R is applied at
        # the vector level in apply_F (R·(coupling·x), associativity), where it
        # only touches the off-diagonal coupling rows. See _make_apply_F_*.

        first_window = self._etd2_M_raw_off is None
        if first_window:
            d_M, M_off = split_operator(M)
            self._etd2_M_raw_diag = d_M
            self._etd2_M_raw_off = M_off
            self._etd2_M_off_to_M_idx, self._etd2_M_diag_to_M_idx = (
                self._build_M_off_index_map(M, M_off)
            )
            if self._backend == "mkl" and config.has_mkl:
                # optimize=False so update_data() can refresh values
                # across cache windows without re-running mkl_sparse_optimize.
                self._etd2_M_raw_off_mkl = self._build_mkl_handle(M_off, optimize=False)
        else:
            # Same sparsity pattern: refresh the values in place. The MKL
            # handle's optimised layout is invariant under value updates.
            M_off = self._etd2_M_raw_off
            if not getattr(M, "has_sorted_indices", False):
                # The index map was built against sorted M.indices, so
                # subsequent windows must keep that invariant.
                M.sort_indices()
            if M_off.data.size:
                M_off.data[:] = M.data[self._etd2_M_off_to_M_idx]
            diag_idx = self._etd2_M_diag_to_M_idx
            d_M = np.zeros(M.shape[0], dtype=np.float64)
            present = diag_idx >= 0
            d_M[present] = M.data[diag_idx[present]]
            self._etd2_M_raw_diag = d_M
            if self._etd2_M_raw_off_mkl is not None:
                # Push the same data into MKL's pinned buffer (no-op if
                # the wrapper happens to share M_off.data — keeps
                # semantics explicit either way).
                self._etd2_M_raw_off_mkl.update_data(M_off.data)

        if self.enable_pairprod_losses:
            self._etd2_kappa_pair_cached = self.pair_loss_rates_grid.loss_vector(z)
        else:
            self._etd2_kappa_pair_cached = None

        if self.enable_em_cascade:
            self._em_T = self._em_transfer_at(z)
            self._em_bh = self._em_bh_at(z)
            self._em_z = float(z)

        self.current_z_rates = z

    def _ensure_D_split(self):
        """Compute and cache the FD operator's diagonal/off-diagonal split."""
        if self._etd2_D_diag is not None:
            return
        from .etd2 import split_operator

        D = self.diff_operator
        if not hasattr(D, "indices"):
            D = D.tocsr()
        d_D, D_off = split_operator(D)
        self._etd2_D_diag = d_D
        self._etd2_D_off = D_off

        if self._backend == "mkl" and config.has_mkl:
            # D is constant for the whole solve, so the MKL handle is
            # one-shot. optimize=True picks the inspector-executor fast
            # path; the data values never change so the internal cache
            # is fine.
            if self._etd2_D_off_mkl is None:
                # Force CSR for D_off: it's small, constant for the whole
                # solve, and the BSR sweet spot wasn't measured for this
                # sparsity. CSR opt=True is already fast for it.
                self._etd2_D_off_mkl = self._build_mkl_handle(
                    D_off, optimize=True, blocksize=None
                )

    def _ensure_Lambda_split(self):
        """Build the constant explicit-decay operator Λ.

        Two outputs:

        * ``self._etd2_Lambda_diag``: per-(species, bin) diagonal,
          ``-1/(c · γ · τ_rest)`` in cm⁻¹. Population loss term.
        * ``self._etd2_Lambda_off``: off-diagonal CSR block, daughter
          production from each (mother, daughter) branching with the
          redistribution kernel from ``decays.get_decay_matrix_bin_average``
          (same machinery the chain reducer uses, so explicit-decay mode
          and chain-reduced mode use identical decay distributions).

        One-shot for the whole solve. Stays a no-op when
        ``enable_decay=False`` or when no species in the state vector is
        unstable. The latter is the reachable state without
        ``config.enable_explicit_decay`` (Step C): the chain reducer folds
        unstable mothers into their stable daughters at cross-section
        build time, so by the time the solver is built the state vector
        contains only stable species and Λ would be identically zero — we
        skip the buffers in that case so the per-step operator avoids a
        useless add.
        """
        if self._etd2_Lambda_diag is not None or not self.enable_decay:
            return

        import scipy.sparse as sp
        from ..data import spec_data, PRINCE_UNITS
        from ..util import is_nucleus
        from ..decays import get_decay_matrix_bin_average

        e_grid = self.egrid  # per-bin energies, ascending log-spaced
        m_proton = PRINCE_UNITS.m_proton
        cm2sec = PRINCE_UNITS.cm2sec  # 1/c in cm units

        # Build the (n_E, n_E) redistribution kernel on the cosmic-ray
        # energy grid using the same form the chain reducer uses on its
        # cross-section x-grid: ``int_scale × P(x)`` with
        # ``int_scale[i, j] = Δlog E_j``. Proton daughter (boost-conserving,
        # x ≈ 1) agrees with chain mode at <1 % rtol; differential daughters
        # (β-decay neutrinos at x ≈ 10⁻³) inherit a ~factor-2 spectral-shape
        # difference vs chain mode that traces to the discretization mismatch
        # — the chain reducer evaluates the kernel on a 200-bin x-grid (36
        # bins/dec) before folding into M via the photon-field convolution,
        # while Λ_off acts directly on the 88-bin energy grid (8 bins/dec).
        # Same physics in the limit of fine bins; tracked as a known
        # discretization effect in the test's relaxed ν̄_e tolerance and
        # in `wiki/open-questions.md`.
        e_bins = self.ebins
        e_centers = self.egrid
        e_widths = e_bins[1:] - e_bins[:-1]
        n_E = len(e_centers)
        int_scale = np.tile(e_widths / e_centers, (n_E, 1))
        dec_bins = np.outer(e_bins, 1.0 / e_centers)
        dec_bins_lower = dec_bins[:-1]
        dec_bins_upper = dec_bins[1:]

        # Native EM coupling: decay daughters homed on the decoupled EM grid
        # get a rectangular (d_em × n_E) block evaluated at EM-grid daughter
        # energies. x[i, j] = em_bins[i]/e_centers[j] depends only on
        # (i − r·j) when the EM log-step divides the cr one (r = δ_cr/δ_em,
        # integer — guaranteed by the core.py grid snap), so the kernel is
        # evaluated once on a 1D x-supercolumn at δ_em steps and scattered
        # with row-stride r. Bypasses get_decay_matrix_bin_average's square
        # fill_diagonal path, which assumes equal steps on both axes.
        em_native_tags = ()
        if self._em_native and self._em_grid_obj is not None:
            em_native_tags = ("em",)
            _emg = self._em_grid_obj
            _em_bins = _emg.bins
            _d_em = _emg.d
            _step_cr = np.log10(e_centers[1] / e_centers[0])
            _step_em = np.log10(_emg.grid[1] / _emg.grid[0])
            _r = int(round(_step_cr / _step_em))
            _k0 = _r * (n_E - 1)
            _x00 = _em_bins[0] / e_centers[0]
            _x_lo_1d = _x00 * 10 ** (
                (np.arange(_d_em + _k0) - _k0) * _step_em
            )
            _x_up_1d = _x_lo_1d * 10 ** _step_em
            # K[i, j] = i − r·j + k0 → index into the supercolumn result.
            _K_em = (
                np.arange(_d_em)[:, None] - _r * np.arange(n_E)[None, :] + _k0
            )
            _int_scale_em = np.tile(e_widths / e_centers, (_d_em, 1))

        Lambda_diag = np.zeros(self.dim_states, dtype=np.float64)
        # COO accumulator for Λ_off; CSR-converted at the end.
        off_rows: list = []
        off_cols: list = []
        off_vals: list = []
        # Per-(mo, da) cache so repeated lookups (e.g. multiple branching
        # entries pointing at the same daughter) hit the same redistribution
        # array. Mirrors `_DecayChainReducer._decay_cache`.
        decay_cache: dict = {}
        n_unstable = 0

        # Cache the daughter-indexed tracked-species map (decay-routed
        # variants only) for the duration of this build. Empty when no
        # tracked species were registered or all are photo-nuclear-only.
        tracked_for_da = {}
        for trk in self.spec_man.species_refs:
            if not getattr(trk, "is_tracking", False):
                continue
            if trk.process_class == "photo-nuclear":
                continue
            tracked_for_da.setdefault(trk.real_pdgid, []).append(trk)

        for s in self.spec_man.species_refs:
            tau = getattr(s, "lifetime", np.inf)
            if not np.isfinite(tau) or tau <= 0.0:
                continue
            n_unstable += 1
            # Per-nucleon convention for nuclei: γ = E_pn / m_p, A-independent.
            # Light/hadronic species carry their total energy in the per-bin
            # state, so γ = E_total / m_species. For tracked species the
            # ``is_nucleus`` check on the synthetic PDG returns False; we
            # dispatch on ``real_pdgid`` instead so tracked nuclei get the
            # per-nucleon γ that matches their real counterpart.
            eff_pdgid = getattr(s, "real_pdgid", None) or s.pdgid
            if is_nucleus(eff_pdgid):
                gamma = e_grid / m_proton
            else:
                # Fall back to the species' own rest mass; spec_data carries
                # masses in GeV alongside lifetimes (data.py:417, 429, 494).
                m_sp = spec_data.get(eff_pdgid, {}).get("mass", m_proton)
                gamma = e_grid / m_sp
            # Λ_diag = -1/(c · γ · τ) in cm⁻¹. cm2sec = 1/c (sec/cm), so
            #   -1/(c · γτ) = -cm2sec/(γτ).
            rate_per_cm = cm2sec / (gamma * tau)  # (n_E,), positive
            Lambda_diag[s.lidx() : s.uidx()] = -rate_per_cm

            # Passive-observer guard: tracked species participate in Λ_diag
            # (they lose flux to their own decay) but do NOT seed Λ_off
            # productions — their decay daughters do not feed back into the
            # network. See methods/tracking-species-design.md § Loss term.
            if getattr(s, "is_tracking", False):
                continue

            # Off-diagonal: walk the mother's branchings and place a
            # ``rate · branching · redistribution`` block at each surviving
            # (mother -> daughter) pair.
            branchings = spec_data.get(s.pdgid, {}).get("branchings", [])
            mo_lidx = s.lidx()
            for br, daughter_pdgs in branchings:
                for da_pdg in daughter_pdgs:
                    if da_pdg not in self.spec_man.pdgid2sref:
                        # Daughter not in the state vector — flux leaks out
                        # silently. Same behaviour as the chain reducer's
                        # ``if da not in spec_data: return`` (base.py:646).
                        continue
                    da_ref = self.spec_man.pdgid2sref[da_pdg]
                    key = (s.pdgid, da_pdg)
                    dec_dist = decay_cache.get(key)
                    if dec_dist is None:
                        if getattr(da_ref, "grid_tag", "default") in em_native_tags:
                            # Rectangular EM block: 1D supercolumn at δ_em
                            # steps, scattered with row-stride r.
                            res_1d = get_decay_matrix_bin_average(
                                s.pdgid, da_pdg, _x_lo_1d, _x_up_1d
                            )
                            dec_dist = _int_scale_em * res_1d[_K_em]
                        else:
                            dec_dist = int_scale * get_decay_matrix_bin_average(
                                s.pdgid, da_pdg, dec_bins_lower, dec_bins_upper
                            )
                        decay_cache[key] = dec_dist
                    if not np.any(dec_dist):
                        continue
                    # block[i, j] = dec_dist[i, j] · br · rate_per_cm[j]
                    # Per-mother-bin scaling broadcasts column-wise.
                    block = dec_dist * (br * rate_per_cm)[None, :]
                    nz = np.nonzero(block)
                    if nz[0].size == 0:
                        continue
                    off_rows.append(da_ref.lidx() + nz[0])
                    off_cols.append(mo_lidx + nz[1])
                    off_vals.append(block[nz])
                    # Tracking hook: duplicate the same block into each
                    # tracked species whose ``real_pdgid`` matches this
                    # daughter and whose ``parent_pdgs`` admits this
                    # mother. ``process_class`` was filtered when building
                    # ``tracked_for_da`` (photo-nuclear-only trackers are
                    # excluded). No e_gamma_range applies on the decay
                    # side — the kernel has no photon-energy axis here.
                    for trk in tracked_for_da.get(da_pdg, ()):
                        if s.pdgid not in trk.parent_pdgs:
                            continue
                        off_rows.append(trk.lidx() + nz[0])
                        off_cols.append(mo_lidx + nz[1])
                        off_vals.append(block[nz])

        if n_unstable == 0:
            # No unstable species reached the state vector — leave the
            # buffers ``None`` so ``_operator_at`` / ``apply_F`` skip the add.
            return

        self._etd2_Lambda_diag = Lambda_diag
        if off_rows:
            rows = np.concatenate(off_rows)
            cols = np.concatenate(off_cols)
            vals = np.concatenate(off_vals)
            self._etd2_Lambda_off = sp.coo_matrix(
                (vals, (rows, cols)),
                shape=(self.dim_states, self.dim_states),
                dtype=np.float64,
            ).tocsr()
            # Tier 3: decay also emits γ/e± daughter rows. Native coupling
            # (_em_native) writes them at EM-grid indices above; the legacy R
            # path writes cr indices and regrids at the vector level in
            # apply_F (R·(Λ_off·x)), not here.

    def _operator_at(self, z):
        """Return ``(d, apply_F)`` for the ETD2 step at redshift ``z``.

        d = dldz(z) · (M_raw_diag + κ(z) ⊙ D_diag)
        apply_F(x, out) = dldz(z) · M_raw_off · x
                          + dldz(z) · D_off · (κ(z) ⊙ x)
                          + b(z)

        κ(z) = κ_adia(z) (per-step, closed form) + κ_pair(z) (cached at
        ``recomp_z_threshold`` together with the photo-hadronic matrix).

        Hot-path bookkeeping notes:

        * ``d``, ``kappa`` and the ``apply_F`` closure are built once
          per ``solve()`` (in :meth:`_ensure_apply_F` / :meth:`solve`)
          and refreshed in place per step. At m=56 / dz=1e-3 that's
          ~50 µs/step of saved alloc + closure-build overhead vs
          rebuilding both per call.
        * ``b`` is per-step (the injection rate depends on z) and is
          pulled from ``injection(1.0, z)`` directly each step. We keep
          a reference into the closure's mutable parameter dict so the
          closure body can read the fresh value without being rebuilt.
        """
        if (
            self.current_z_rates is None
            or abs(z - self.current_z_rates) > self.recomp_z_threshold
        ):
            self._refresh_z_caches(z)

        if self._etd2_apply_F is None:
            self._ensure_apply_F()

        dldz = self.dldz(z)
        if self.enable_partial_diff_jacobian:
            kappa = self._etd2_kappa_buf
            kappa.fill(0.0)
            if self.enable_adiabatic_losses:
                kappa += self.adia_loss_rates_grid.loss_vector(z)
            if self._etd2_kappa_pair_cached is not None:
                kappa += self._etd2_kappa_pair_cached
        else:
            kappa = None

        if self.enable_injection_jacobian and self.list_of_sources:
            b = self.injection(1.0, z)
        else:
            b = None

        # Diagonal d = dldz · (M_raw_diag + κ ⊙ D_diag) — written into a
        # persistent buffer; ``etd2_step`` does not retain ``d`` past the
        # step body, so reusing the buffer is safe.
        d = self._etd2_d_buf
        np.multiply(self._etd2_M_raw_diag, dldz, out=d)
        if kappa is not None:
            # d += dldz * (kappa * D_diag) without a fresh temporary.
            np.multiply(kappa, self._etd2_D_diag, out=self._etd2_kx_buf)
            np.multiply(self._etd2_kx_buf, dldz, out=self._etd2_kx_buf)
            np.add(d, self._etd2_kx_buf, out=d)
        if self._etd2_Lambda_diag is not None:
            # d += dldz · Λ_diag — Λ_diag is constant across the whole
            # solve so it just rides the per-step ``dldz`` scalar.
            np.multiply(self._etd2_Lambda_diag, dldz, out=self._etd2_kx_buf)
            np.add(d, self._etd2_kx_buf, out=d)

        # Refresh the persistent apply_F closure's per-step params.
        params = self._etd2_apply_F_params
        params["dldz"] = dldz
        params["kappa"] = kappa
        params["b"] = b
        return d, self._etd2_apply_F

    def _ensure_apply_F(self):
        """Lazily build the persistent apply_F closure.

        Called from :meth:`_operator_at` on first invocation per solve.
        The closure captures ``self._etd2_apply_F_params`` (mutable
        dict) so per-step refreshes happen via dict assignment, not by
        rebuilding the closure.
        """
        if self._etd2_apply_F is not None:
            return
        # Persistent host scratch sized to dim_states.
        self._etd2_d_buf = np.empty(self.dim_states, dtype=np.float64)
        self._etd2_kx_buf = np.empty(self.dim_states, dtype=np.float64)
        # ``kappa`` is sized to the loss-vector grid (== dim_states) but
        # takes its dtype from the loss grid.
        self._etd2_kappa_buf = np.zeros_like(
            self.adia_loss_rates_grid.energy_vector
        )

        if self._backend == "mkl" and self._etd2_M_raw_off_mkl is not None:
            self._etd2_apply_F = self._make_apply_F_mkl()
        else:
            self._etd2_apply_F = self._make_apply_F_scipy()

    def _make_apply_F_scipy(self):
        """Build the persistent scipy-backed ``apply_F(x, out)`` closure.

        ``M_off`` and ``D_off`` are pinned at closure-build time; both
        are stable ndarrays for the whole solve (M_off has its data
        refreshed in place by :meth:`_refresh_z_caches`; D_off is
        constant). ``dldz`` / ``kappa`` / ``b`` come from the mutable
        ``self._etd2_apply_F_params`` dict — :meth:`_operator_at`
        refreshes the values per step.
        """
        M_off = self._etd2_M_raw_off
        D_off = self._etd2_D_off
        Lambda_off = self._etd2_Lambda_off  # CSR or None; constant for the solve.
        R = self._em_regrid_R  # Tier 3 cr→em daughter-row regrid; None if shared.
        params = self._etd2_apply_F_params
        kx_buf = self._etd2_kx_buf

        if R is None:
            def apply_F(x, out):
                kappa = params["kappa"]
                np.copyto(out, M_off.dot(x))
                if kappa is not None:
                    np.multiply(kappa, x, out=kx_buf)
                    np.add(out, D_off.dot(kx_buf), out=out)
                if Lambda_off is not None:
                    # Λ_off rides the same dldz scalar as M_off / D_off (see
                    # methods/explicit-decay-kernel-design.md). Constant in z.
                    np.add(out, Lambda_off.dot(x), out=out)
                np.multiply(out, params["dldz"], out=out)
                b = params["b"]
                if b is not None:
                    np.add(out, b, out=out)
        else:
            # Tier 3 decoupled EM grid: the coupling (photo-nuclear M_off +
            # decay Λ_off) writes γ/e± daughter rows at cr indices; regrid them
            # onto the EM grid via R before the transport term. R·(coupling·x)
            # = (R·coupling)·x (associativity) and R is identity on nuclear
            # rows, so only the EM-daughter rows move. D_off (per-grid
            # transport) is already on the correct grid — NOT regridded.
            def apply_F(x, out):
                kappa = params["kappa"]
                coup = M_off.dot(x)
                if Lambda_off is not None:
                    coup = coup + Lambda_off.dot(x)
                np.copyto(out, R.dot(coup))
                if kappa is not None:
                    np.multiply(kappa, x, out=kx_buf)
                    np.add(out, D_off.dot(kx_buf), out=out)
                np.multiply(out, params["dldz"], out=out)
                b = params["b"]
                if b is not None:
                    np.add(out, b, out=out)

        return apply_F

    def _make_apply_F_mkl(self):
        """Build the persistent MKL-backed ``apply_F(x, out)`` closure.

        Per call:

          out = M_off · x                                (mkl gemv α=1, β=0)
          if kappa: kx_buf = κ ⊙ x; out += D_off · kx_buf (mkl gemv α=1, β=1)
          out *= dldz; if b: out += b

        ``etd2_step`` calls ``apply_F`` twice per ETD2 step against four
        persistent buffers (state, F_phi, a, F_a) plus kx_buf. We
        memoise the ctypes pointer for each ndarray by ``id`` so we
        don't redo ``arr.ctypes.data_as`` on every step. The buffers
        come from ``etd2._step_buffers`` and live for the whole solve,
        so the cached pointers stay valid across the integration loop.
        """
        from ctypes import POINTER, c_double

        mkl_M = self._etd2_M_raw_off_mkl
        mkl_D = self._etd2_D_off_mkl

        # Pre-box the only two (alpha, beta) constants we ever use,
        # avoiding ~5 µs of ``c_double(...)`` per gemv in the hot loop.
        alpha_box = c_double(1.0)
        beta_zero = c_double(0.0)
        beta_one = c_double(1.0)
        mkl_M_op = mkl_M.gemv_preboxed
        mkl_D_op = mkl_D.gemv_preboxed if mkl_D is not None else None

        ptr_cache = {}

        def get_p(arr):
            key = id(arr)
            p = ptr_cache.get(key)
            if p is None:
                p = arr.ctypes.data_as(POINTER(c_double))
                ptr_cache[key] = p
            return p

        kx_buf = self._etd2_kx_buf
        kx_p = get_p(kx_buf)
        params = self._etd2_apply_F_params
        # Λ_off MKL-handle wrap deferred to Step E; for now, fall back to a
        # scipy SpMV when explicit decay is enabled with the MKL backend.
        Lambda_off = self._etd2_Lambda_off
        R = self._em_regrid_R  # Tier 3 cr→em daughter-row regrid; None if shared.

        if R is None:
            def apply_F(x, out):
                x_p = get_p(x)
                out_p = get_p(out)
                mkl_M_op(alpha_box, x_p, beta_zero, out_p)
                kappa = params["kappa"]
                if kappa is not None:
                    np.multiply(kappa, x, out=kx_buf)
                    mkl_D_op(alpha_box, kx_p, beta_one, out_p)
                if Lambda_off is not None:
                    np.add(out, Lambda_off.dot(x), out=out)
                np.multiply(out, params["dldz"], out=out)
                b = params["b"]
                if b is not None:
                    np.add(out, b, out=out)
        else:
            # Tier 3 decoupled EM grid: regrid the coupling rows (M_off + Λ_off)
            # cr→em via R BEFORE adding the per-grid transport D_off. Compute
            # the coupling into out first, apply R in place, then fuse D_off.
            def apply_F(x, out):
                x_p = get_p(x)
                out_p = get_p(out)
                mkl_M_op(alpha_box, x_p, beta_zero, out_p)   # out = M_off·x
                if Lambda_off is not None:
                    np.add(out, Lambda_off.dot(x), out=out)  # + Λ_off·x
                out[:] = R.dot(out)                          # regrid EM rows
                kappa = params["kappa"]
                if kappa is not None:
                    np.multiply(kappa, x, out=kx_buf)
                    mkl_D_op(alpha_box, kx_p, beta_one, out_p)  # + D_off·(κx)
                np.multiply(out, params["dldz"], out=out)
                b = params["b"]
                if b is not None:
                    np.add(out, b, out=out)

        return apply_F


class ETD2SolverCUPY(UHECRPropagationSolverETD2):
    """ETD2 backend for the cupy / cuSPARSE path.

    Per-step body uses fused cupy ``ElementwiseKernel``s
    (:func:`prince_cr.solvers.etd2.cupy_kernels`) and a custom warp-per-
    row CSR SpMV (:func:`prince_cr.solvers.etd2.csr_spmv`). When
    ``backend.use_cuda_graphs`` is True the per-cache-window step body
    is recorded into a ``cupy.cuda.Graph`` after a warmup + capture pass
    and replayed for the rest of the window; re-capture triggers on
    each ``_refresh_z_caches`` because the M_off / D_off cuSPARSE
    descriptors close over device pointers that change at each window.

    The eager (non-graph) path is the bit-exact reference for the graph
    path at fp64 (max abs diff = 0). The eager path is also what the
    multi-RHS subclass uses, since CUDA Graph capture is not currently
    implemented for cuSPARSE SpMM under cupy 14.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The Tier-3 runtime regrid R was only ever wired into the scipy/MKL
        # apply_F closures — the device step body would silently skip it
        # (γ/e± daughter rows left at cr indices = wrong energies). Refuse
        # the combination instead; the native coupling (default) needs no R.
        if self._em_regrid_R is not None:
            raise NotImplementedError(
                "The decoupled-grid runtime regrid R is not implemented on "
                "the cupy backend. Use enable_em_native_coupling=True (the "
                "default) or a host backend (scipy/MKL) for the legacy R "
                "path."
            )
        # cupy persistent device buffers. Allocated once per solve (in
        # :meth:`_ensure_apply_F`) and held for the whole integration so
        # the per-step body is alloc-free — required for kernel fusion
        # to pay off at the per-step scale and for stable buffer
        # addresses inside captured CUDA Graphs. Per-step values
        # (kappa, b, dldz, d) are refreshed in place; cache-window-bound
        # values (M_diag, D_diag) are referenced by the apply_F closure
        # and rebound on cache window refresh.
        self._etd2_d_buf_cp = None
        self._etd2_kappa_buf_cp = None
        self._etd2_b_buf_cp = None
        self._etd2_kx_buf_cp = None
        self._etd2_dldz_buf_cp = None
        # Mutable closure-state for the cupy apply_F. Rebound on each
        # ``_refresh_z_caches`` because ``split_operator`` returns fresh
        # cupy CSR arrays at every cache window.
        self._etd2_apply_F_state_cp = None
        # CUDA Graph machinery. Allocated lazily when
        # ``backend.use_cuda_graphs`` is True. Re-captured on each cache
        # window refresh because the cuSPARSE descriptors close over
        # ``M_off`` / ``D_off`` device pointers that change at each
        # window.
        self._etd2_graph_exec = None
        self._etd2_graph_stream = None
        self._etd2_graph_state_buf = None
        self._etd2_graph_needs_capture = True
        # Device mirrors of the co-evolved EM cascade pieces (see
        # _em_upload_device_caches); refreshed per cache window.
        self._em_T_dev = None
        self._em_bh_dev = None
        self._em_bh_coeff_dev = None

    # ------------------------------------------------------------------
    # Outer-driver hooks
    # ------------------------------------------------------------------
    def _init_state(self):
        import cupy as cp
        dt = np.dtype(self.backend.cupy_dtype)
        return cp.zeros(self.dim_states, dtype=dt)

    def _coerce_state(self, state):
        """Device-coerce an externally supplied initial state (tests and
        diagnostics commonly monkey-patch ``_init_state`` with a host
        ndarray, which the device SpMV cannot consume)."""
        import cupy as cp
        dt = np.dtype(self.backend.cupy_dtype)
        return cp.asarray(state, dtype=dt)

    def _finalize_state(self, state):
        # D2H + upcast back to fp64 host so downstream consumers
        # (UHECRPropagationResult, plotting, et al.) see the canonical
        # dtype.
        import cupy as cp
        return cp.asnumpy(state).astype(np.float64)

    def _reset_for_solve(self):
        # The GPU D split is not constant across solves: a backend flip
        # (cupy_dtype change, etc.) requires rebuilding it. ``_ensure_D_split``
        # repopulates it from the (host) ``self.diff_operator``.
        self._etd2_D_diag = None
        self._etd2_D_off = None
        self._etd2_d_buf_cp = None
        self._etd2_kappa_buf_cp = None
        self._etd2_b_buf_cp = None
        self._etd2_kx_buf_cp = None
        self._etd2_dldz_buf_cp = None
        self._etd2_apply_F_state_cp = None
        # Drop any captured CUDA Graph; the previous solve's buffer
        # addresses are no longer valid.
        self._etd2_graph_exec = None
        self._etd2_graph_stream = None
        self._etd2_graph_state_buf = None
        self._etd2_graph_needs_capture = True

    def _run_integration(self, state, z_grid, step_hook=None):
        if self.backend.use_cuda_graphs and config.has_cupy:
            self._solve_loop_graphs(state, z_grid, step_hook=step_hook)
        else:
            from .etd2 import integrate
            integrate(
                state, z_grid,
                operator_at=self._operator_at,
                step_hook=step_hook,
            )

    def close(self):
        for attr in (
            "_etd2_M_raw_off",
            "_etd2_M_raw_diag",
            "_etd2_D_off",
            "_etd2_D_diag",
            "_etd2_d_buf_cp",
            "_etd2_kappa_buf_cp",
            "_etd2_b_buf_cp",
            "_etd2_kx_buf_cp",
            "_etd2_dldz_buf_cp",
            "_etd2_apply_F_state_cp",
            "_etd2_graph_exec",
            "_etd2_graph_stream",
            "_etd2_graph_state_buf",
        ):
            setattr(self, attr, None)
        self._etd2_graph_needs_capture = True

    # ------------------------------------------------------------------
    # Cache-window refresh + per-step operator
    # ------------------------------------------------------------------
    def _refresh_z_caches(self, z):
        """GPU variant: split_operator returns cupy CSR arrays directly."""
        from .etd2 import split_operator

        M = self.had_int_rates.get_hadr_jacobian(z, 1.0, force_update=True)
        if not self.enable_photohad_losses:
            M = M.copy()
            M.data[:] = 0

        # Sum the EM cascade Jacobian. NOTE: untested on the cupy backend —
        # the EM rate is built host-side (scipy); convert to device CSR before
        # the add. (CPU/scipy path is the validated one for the EM cascade.)
        if self.em_int_rates is not None:
            import cupyx.scipy.sparse as _cpsp

            M = M + _cpsp.csr_matrix(
                self.em_int_rates.get_hadr_jacobian(z, 1.0, force_update=True)
            )

        # Re-splitting each cache window is a few ms on the GPU at
        # production grid; we skip a host-style index-map gather (the
        # data already lives on-device, and ``M`` is rebuilt fresh by
        # the cupy ``_update_rates_gpu`` path each window).
        d_M, M_off = split_operator(M)
        self._etd2_M_raw_diag = d_M
        self._etd2_M_raw_off = M_off
        # The apply_F closure dereferences M_off / D_off through a
        # mutable container so we can swap pointers here without
        # rebuilding it. Triggers re-capture of the CUDA graph on the
        # next per-step call.
        self._rebind_apply_F()

        if self.enable_pairprod_losses:
            self._etd2_kappa_pair_cached = self.pair_loss_rates_grid.loss_vector(z)
        else:
            self._etd2_kappa_pair_cached = None

        if self.enable_em_cascade:
            self._em_T = self._em_transfer_at(z)
            self._em_bh = self._em_bh_at(z)
            self._em_z = float(z)
            self._em_upload_device_caches()

        self.current_z_rates = z

    # ------------------------------------------------------------------
    # Co-evolved EM cascade on device. The host base-class step methods
    # (`_apply_em_cascade` / `_inject_bh_pairs`) mix numpy matrices with
    # the cupy state — that was the long-standing host-array crash in
    # test_em_absorption_in_transport. Mirror the window-constant pieces
    # (T(z), R_BH(z), BH weight coefficients) to device at each cache
    # refresh and run the per-step EM ops fully on device.
    # ------------------------------------------------------------------
    def _em_upload_device_caches(self):
        import cupy as cp

        dt = np.dtype(self.backend.cupy_dtype)
        T_gamma, T_electron = self._em_T
        self._em_T_dev = (
            cp.asarray(np.asarray(T_gamma), dtype=dt),
            cp.asarray(np.asarray(T_electron), dtype=dt),
        )
        self._em_bh_dev = None
        self._em_bh_coeff_dev = None
        if self._em_bh is not None and self._em_bh_species:
            from prince_cr.cosmology import H
            from prince_cr.data import PRINCE_UNITS

            self._em_bh_dev = cp.asarray(np.asarray(self._em_bh), dtype=dt)
            # Window-constant per-species weight coefficients (host math,
            # no state dependence): w_s(E) = A·loss_s(E)·dl_step/E·dE.
            z = self._em_z
            E_p = self._em_proton_E
            dE = np.gradient(E_p)
            loss_full = self.pair_loss_rates_grid.loss_vector(z)
            dl_step = PRINCE_UNITS.c * abs(self._em_dz) / ((1.0 + z) * H(z))
            with np.errstate(divide="ignore", invalid="ignore"):
                inv_E = np.where(E_p > 0, 1.0 / E_p, 0.0)
            self._em_bh_coeff_dev = [
                (sl, cp.asarray(A * loss_full[sl] * dl_step * inv_E * dE, dtype=dt))
                for sl, A in self._em_bh_species
            ]
        # Host-sync so the mirrors are complete before any other stream
        # (in particular the non-blocking CUDA-graphs capture stream, which
        # does not implicitly order against the default stream) reads them.
        # Once per cache window — negligible.
        cp.cuda.get_current_stream().synchronize()

    def _apply_em_cascade(self, state):
        if self._em_T is None:
            return
        import cupy as cp

        T_gamma, T_electron = self._em_T_dev
        ph = state[self._em_gamma_sl].copy()
        lep = cp.zeros_like(ph)
        for sl in self._em_lep_sl:
            lep += state[sl]
            state[sl] = 0.0
        state[self._em_gamma_sl] = T_gamma @ ph + T_electron @ lep

    def _inject_bh_pairs(self, state):
        if self._em_bh_dev is None or not self._em_bh_coeff_dev:
            return
        w = None
        for sl, coeff in self._em_bh_coeff_dev:
            term = coeff * state[sl]
            w = term if w is None else w + term
        e_src = self._em_bh_dev @ w
        for sl in self._em_lep_sl:
            state[sl] = state[sl] + e_src

    def _ensure_D_split(self):
        """Compute the FD operator split and upload to GPU once per solve."""
        if self._etd2_D_diag is not None:
            return
        from .etd2 import split_operator

        D = self.diff_operator
        if not hasattr(D, "indices"):
            D = D.tocsr()
        d_D, D_off = split_operator(D)

        import cupy as cp
        import cupyx.scipy.sparse as csp

        dt = np.dtype(self.backend.cupy_dtype)
        self._etd2_D_diag = cp.asarray(d_D, dtype=dt)
        D_off_cast = D_off.astype(dt)
        self._etd2_D_off = csp.csr_matrix(
            (
                cp.asarray(D_off_cast.data, dtype=dt),
                cp.asarray(D_off_cast.indices, dtype=cp.int32),
                cp.asarray(D_off_cast.indptr, dtype=cp.int32),
            ),
            shape=D_off_cast.shape,
        )

    def _operator_at(self, z):
        """Return ``(d_buf_cp, apply_F)``. Uploads per-step values to device."""
        if (
            self.current_z_rates is None
            or abs(z - self.current_z_rates) > self.recomp_z_threshold
        ):
            self._refresh_z_caches(z)

        if self._etd2_apply_F is None:
            self._ensure_apply_F()

        dldz = self.dldz(z)
        if self.enable_partial_diff_jacobian:
            kappa = self._etd2_kappa_buf
            kappa.fill(0.0)
            if self.enable_adiabatic_losses:
                kappa += self.adia_loss_rates_grid.loss_vector(z)
            if self._etd2_kappa_pair_cached is not None:
                kappa += self._etd2_kappa_pair_cached
        else:
            kappa = None

        if self.enable_injection_jacobian and self.list_of_sources:
            b = self.injection(1.0, z)
        else:
            b = None

        return self._operator_at_device(dldz, kappa, b)

    def _operator_at_device(self, dldz, kappa_host, b_host):
        """Refresh persistent device buffers and compute (n,) ``d`` on-device.

        Writes per-step values into pre-allocated device buffers, then
        evaluates ``d = dldz · (M_diag + κ ⊙ D_diag)`` via the fused
        ``compute_d`` ElementwiseKernel. No allocations on the hot path
        beyond a tiny intermediate cupy array for the host-to-device
        upload (mempool-reused). Returns the persistent ``d`` and
        ``apply_F`` references.
        """
        import cupy as cp
        from .etd2 import cupy_kernels

        dt = self._etd2_dldz_buf_cp.dtype
        # Update per-step scalar via a 1-element device buffer. Stays
        # at the same address across the solve so a captured graph's
        # kernels (which take ``raw T dldz_buf`` and read ``[0]``)
        # always see the fresh value at replay time.
        self._etd2_dldz_buf_cp[0] = dt.type(dldz)

        if kappa_host is not None:
            cp.copyto(
                self._etd2_kappa_buf_cp,
                cp.asarray(kappa_host, dtype=dt),
            )
        # b is optional; when None we keep the b_buf zeroed (filled at
        # alloc time) and emit the no-b kernel variant from apply_F.
        if b_host is not None:
            cp.copyto(
                self._etd2_b_buf_cp,
                cp.asarray(b_host, dtype=dt),
            )

        K = cupy_kernels()
        if kappa_host is not None:
            K.compute_d(
                self._etd2_M_raw_diag,
                self._etd2_kappa_buf_cp,
                self._etd2_D_diag,
                self._etd2_dldz_buf_cp,
                self._etd2_d_buf_cp,
            )
        else:
            K.compute_d_no_kappa(
                self._etd2_M_raw_diag,
                self._etd2_dldz_buf_cp,
                self._etd2_d_buf_cp,
            )

        return self._etd2_d_buf_cp, self._etd2_apply_F

    def _ensure_apply_F(self):
        """Allocate persistent cupy buffers and build the apply_F closure.

        ``M_off`` / ``D_off`` are bound through the mutable
        ``self._etd2_apply_F_state_cp`` dict so :meth:`_rebind_apply_F`
        can swap them on cache-window refresh without rebuilding the
        closure object (which keeps a stable callable identity for
        :func:`etd2.integrate` and the CUDA Graph capture path).
        """
        if self._etd2_apply_F is not None:
            return
        import cupy as cp

        dt = np.dtype(self.backend.cupy_dtype)
        n = self.dim_states

        # Host scratch for kappa accumulation. Same dtype as the loss-
        # vector grid so the sum stays in fp64 host arithmetic
        # regardless of the active GPU dtype.
        if self._etd2_kappa_buf is None:
            self._etd2_kappa_buf = np.zeros_like(
                self.adia_loss_rates_grid.energy_vector
            )

        # Persistent per-step device buffers. Held for the whole solve;
        # released in :meth:`close`.
        self._etd2_d_buf_cp = cp.empty(n, dtype=dt)
        self._etd2_kappa_buf_cp = cp.zeros(n, dtype=dt)
        self._etd2_b_buf_cp = cp.zeros(n, dtype=dt)
        self._etd2_kx_buf_cp = cp.empty(n, dtype=dt)
        self._etd2_dldz_buf_cp = cp.empty(1, dtype=dt)

        has_kappa = bool(self.enable_partial_diff_jacobian)
        has_b = bool(self.enable_injection_jacobian) and bool(self.list_of_sources)

        self._etd2_apply_F_state_cp = {
            "M_off": self._etd2_M_raw_off,
            "D_off": self._etd2_D_off,
            "has_kappa": has_kappa,
            "has_b": has_b,
        }

        self._etd2_apply_F = self._make_apply_F()

    def _rebind_apply_F(self):
        """Swap ``M_off`` / ``D_off`` references on a cache window refresh.

        Also drops the captured CUDA Graph executable — its cuSPARSE
        descriptors hold device pointers from the previous window's
        M_off/D_off and would be dangling after we update the dict and
        Python frees the old cupy CSR objects.
        """
        if self._etd2_apply_F_state_cp is None:
            return
        # Drop the old graph BEFORE swapping the dict entries — once
        # the dict no longer references the previous window's M_off,
        # cupy's mempool may reclaim its data buffer, and any captured
        # spmv calls that pointed at it would fault on replay.
        self._etd2_graph_exec = None
        self._etd2_apply_F_state_cp["M_off"] = self._etd2_M_raw_off
        self._etd2_apply_F_state_cp["D_off"] = self._etd2_D_off
        self._etd2_graph_needs_capture = True

    def _make_apply_F(self):
        """Build the persistent cupy ``apply_F(x, out)`` closure.

        Per call:

          1. ``cusparse spmv`` writes ``M_off · x`` into ``out`` (alpha=1, beta=0).
          2. if κ active: ``kx = κ ⊙ x``; ``cusparse spmv`` adds
             ``D_off · kx`` into ``out`` (alpha=1, beta=1).
          3. fused ``scale_b``: ``out = out · dldz + b``  (or just
             ``out · dldz`` when sources are off).

        All scalars (``dldz``) come from the 1-element ``_etd2_dldz_buf_cp``
        device buffer the kernel reads via ``raw T``. ``M_off`` / ``D_off``
        are dereferenced through ``_etd2_apply_F_state_cp`` so the
        cache-window refresh path can swap them without rebuilding the
        closure.
        """
        import cupy as cp
        from .etd2 import cupy_kernels, csr_spmv

        K = cupy_kernels()
        kx_buf = self._etd2_kx_buf_cp
        kappa_buf = self._etd2_kappa_buf_cp
        b_buf = self._etd2_b_buf_cp
        dldz_buf = self._etd2_dldz_buf_cp
        state = self._etd2_apply_F_state_cp

        def apply_F(x, out):
            M_off = state["M_off"]
            csr_spmv(M_off, x, out, accumulate=False)
            if state["has_kappa"]:
                cp.multiply(kappa_buf, x, out=kx_buf)
                csr_spmv(state["D_off"], kx_buf, out, accumulate=True)
            if state["has_b"]:
                K.scale_b(out, b_buf, dldz_buf, out)
            else:
                K.scale_no_b(out, dldz_buf, out)

        return apply_F

    def _step_body_graph(self, state, bufs, K, has_kappa):
        """Run one ETD2 step's device-only kernel sequence.

        All inputs come from persistent device buffers; no Python
        scalars are baked into kernel launch params (per-step values
        live in ``raw T`` 1-element device buffers read with ``[0]``),
        so this body is safe to record into a CUDA Graph and replay
        across many steps within a cache window.

        Sequence (11 cupy ops at full feature load): compute_d →
        phi_compute → spmv_M → multiply κ⊙state → spmv_D → scale_b →
        post_apply1 → spmv_M → multiply κ⊙a → spmv_D → scale_b →
        post_apply2.
        """
        d_buf = self._etd2_d_buf_cp
        if has_kappa:
            K.compute_d(
                self._etd2_M_raw_diag,
                self._etd2_kappa_buf_cp,
                self._etd2_D_diag,
                self._etd2_dldz_buf_cp,
                d_buf,
            )
        else:
            K.compute_d_no_kappa(
                self._etd2_M_raw_diag,
                self._etd2_dldz_buf_cp,
                d_buf,
            )
        K.phi_compute(d_buf, bufs["h_buf"], bufs["eD"], bufs["phi1"], bufs["phi2"])
        self._etd2_apply_F(state, bufs["F_phi"])
        K.post_apply1(
            bufs["eD"], state, bufs["phi1"], bufs["F_phi"], bufs["h_buf"], bufs["a"]
        )
        self._etd2_apply_F(bufs["a"], bufs["F_a"])
        K.post_apply2(
            bufs["a"],
            bufs["F_a"],
            bufs["F_phi"],
            bufs["phi2"],
            bufs["h_buf"],
            state,
        )

    def _solve_loop_graphs(self, state, z_grid, step_hook=None):
        """CUDA Graph capture/replay integration loop.

        Replaces :func:`etd2.integrate` for the cupy-with-graphs path.
        Per cache window: warmup step (eager), capture step (records
        kernels into a ``cupy.cuda.Graph``), then replay for the
        remainder of the window. Re-capture triggers on the next
        ``_refresh_z_caches`` (signalled via
        :attr:`_etd2_graph_needs_capture`).

        All per-step host work (κ accumulation, b construction,
        dldz scalar) happens outside the captured region; the graph
        only contains the per-step kernel sequence built by
        :meth:`_step_body_graph`.

        Parameters
        ----------
        step_hook : callable, optional
            Invoked once per step after the device kernels are
            scheduled (graph launch / eager body); used by
            ``solve(progressbar=...)``.
        """
        import cupy as cp
        from .etd2 import _step_buffers, cupy_kernels

        K = cupy_kernels()
        bufs = _step_buffers(state.shape[0], xp=cp, dtype=state.dtype)

        # Capture stream — non-default; default stream forbids capture.
        stream = cp.cuda.Stream(non_blocking=True)
        self._etd2_graph_stream = stream

        nsteps = len(z_grid) - 1
        # Per-cache-window state machine: 0=needs warmup, 1=needs capture,
        # 2=replay. Resets to 0 whenever ``_etd2_graph_needs_capture``
        # flips back to True (cache-window refresh).
        win_step_count = 0
        last_h = None

        has_kappa = bool(self.enable_partial_diff_jacobian)
        has_b = bool(self.enable_injection_jacobian) and bool(self.list_of_sources)

        for k in range(nsteps):
            z0 = z_grid[k]
            z1 = z_grid[k + 1]
            h = z1 - z0

            # Cache-window refresh check (mirrors :meth:`_operator_at`).
            if (
                self.current_z_rates is None
                or abs(z0 - self.current_z_rates) > self.recomp_z_threshold
            ):
                self._refresh_z_caches(z0)
                # _rebind_apply_F drops _etd2_graph_exec and sets
                # _etd2_graph_needs_capture = True. Reset window state.
                win_step_count = 0

            # Lazy-init persistent buffers on first call after the first
            # _refresh_z_caches.
            if self._etd2_apply_F is None:
                self._ensure_apply_F()

            # Per-step host work: κ, b, dldz — exactly the same arithmetic
            # the eager :meth:`_operator_at_device` does, lifted here so we
            # can keep the captured region pure-device.
            dldz = self.dldz(z0)
            if has_kappa:
                kappa = self._etd2_kappa_buf
                kappa.fill(0.0)
                if self.enable_adiabatic_losses:
                    kappa += self.adia_loss_rates_grid.loss_vector(z0)
                if self._etd2_kappa_pair_cached is not None:
                    kappa += self._etd2_kappa_pair_cached
            else:
                kappa = None
            if has_b:
                b = self.injection(1.0, z0)
            else:
                b = None

            dt = self._etd2_dldz_buf_cp.dtype

            # All device work happens on the capture stream so that the
            # per-step uploads (dldz / κ / b / h) are correctly ordered
            # before the kernel launches that read them — without a
            # stream context the uploads would land on the per-thread
            # default stream and race against the non-default capture
            # stream's kernels.
            with stream:
                self._etd2_dldz_buf_cp[0] = dt.type(dldz)
                if kappa is not None:
                    cp.copyto(
                        self._etd2_kappa_buf_cp,
                        cp.asarray(kappa, dtype=dt),
                    )
                if b is not None:
                    cp.copyto(
                        self._etd2_b_buf_cp,
                        cp.asarray(b, dtype=dt),
                    )
                if last_h != h:
                    bufs["h_buf"][0] = dt.type(h)
                    last_h = h
                    # h changing within a cache window is the truncated
                    # final step. Force re-capture: although h_buf is
                    # read via ``raw T`` at replay time and would
                    # technically pick up the new value, the captured
                    # graph might have been planned around the previous
                    # step count.
                    if win_step_count == 2:
                        self._etd2_graph_exec = None
                        self._etd2_graph_needs_capture = True
                        win_step_count = 0

                # Per-window state machine.
                if self._etd2_graph_needs_capture and win_step_count == 0:
                    # Warmup pass — runs the body eagerly on the capture
                    # stream so cuSPARSE's per-stream handle and cupy's
                    # ElementwiseKernel JIT compile *outside* the captured
                    # region. Advances state like a normal step.
                    self._step_body_graph(state, bufs, K, has_kappa)
                    win_step_count = 1
                elif self._etd2_graph_needs_capture and win_step_count == 1:
                    # Capture pass. ``begin_capture`` puts the stream
                    # in record-only mode — the kernel launches are
                    # captured into the graph but do NOT execute. We
                    # launch the graph once after end_capture so this
                    # step advances state like a real step (otherwise
                    # we'd silently lose one step per cache window).
                    stream.begin_capture()
                    self._step_body_graph(state, bufs, K, has_kappa)
                    graph = stream.end_capture()
                    graph.upload(stream)
                    graph.launch(stream)
                    self._etd2_graph_exec = graph
                    self._etd2_graph_needs_capture = False
                    win_step_count = 2
                else:
                    # Replay — single graph.launch per step.
                    self._etd2_graph_exec.launch(stream)

                if step_hook is not None:
                    # Issue the hook INSIDE the stream context: the
                    # co-evolved EM step launches device kernels, and the
                    # capture stream is non-blocking — hook ops on the
                    # default stream would race the async graph launch.
                    # On-stream they are ordered after this step's kernels
                    # and before the next step's uploads.
                    step_hook()

        stream.synchronize()
        return state


# =====================================================================
# Multi-RHS ETD2: K independent solutions sharing the operator.
# =====================================================================

class _MultiRHSView:
    """Per-RHS view of a :class:`MultiRHSPropagationSolverETD2`.

    Each instance owns its own ``list_of_sources`` (the sources whose
    injection feeds the k-th column of the multi-RHS state matrix) and
    exposes ``.res`` returning a :class:`UHECRPropagationResult`
    backed by column k of the parent's state.
    """

    def __init__(self, parent, k):
        self.parent = parent
        self.k = k
        self.list_of_sources = []

    def add_source_class(self, source_instance):
        self.list_of_sources.append(source_instance)

    @property
    def state(self):
        s = self.parent.state
        if s.ndim == 2:
            return s[:, self.k]
        return s

    @property
    def res(self):
        return UHECRPropagationResult(
            self.state, self.parent.egrid, self.parent.spec_man
        )


class MultiRHSPropagationSolverETD2(UHECRPropagationSolverETD2):
    """ETD2 integrator that propagates K independent RHSs simultaneously.

    The operator ``L(z) = J(z) + dl/dz · D · diag(κ(z))``, the dense
    rate-cache rebuild, and all diagonal factors (κ, eD, phi1, phi2)
    depend only on z and are shared across all K solutions. Per-RHS
    state lives as columns of a ``(dim_states, K)`` array; per-RHS
    injection sources are attached via ``solver[k].add_source_class``.

    Compared with K back-to-back single-RHS solves, this replaces the
    four per-step CSR SpMVs with four CSR-SpMMs (M_off, D_off in each
    of the two ``apply_F`` calls). At max_mass=24, scipy SpMM gives
    ~3.5× per-RHS at K=64; cupy/cuSPARSE gives ~36×. Cache-window work
    is K-independent.

    This is the algorithm parent. Construct it directly and ``__new__``
    dispatches to a backend subclass:

    * ``_MultiRHSETD2SolverCPU`` — scipy SpMM (MKL Sparse SpMM not
      wrapped; the MKL backend setting falls through to scipy).
    * ``_MultiRHSETD2SolverCUPY`` — eager cuSPARSE SpMM. CUDA Graph
      capture is not currently supported on the multi-RHS path
      (cupy 14 blocks cuSPARSE during capture).
    """

    def __new__(cls, *args, **kwargs):
        if cls is MultiRHSPropagationSolverETD2:
            prince_run = _resolve_prince_run(args, kwargs)
            backend = prince_run.backend.linear_algebra_backend.lower()
            target = (
                _MultiRHSETD2SolverCUPY if backend == "cupy"
                else _MultiRHSETD2SolverCPU
            )
            return object.__new__(target)
        return object.__new__(cls)

    def __init__(self, *args, K, **kwargs):
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K!r}")
        super().__init__(*args, **kwargs)
        self._K = int(K)
        self._views = [_MultiRHSView(self, k) for k in range(self._K)]
        # Pre-allocate the host state shape so ``solver[k].state`` works
        # before solve() (returns zeros, like the parent).
        self.state = np.zeros((self.dim_states, self._K))
        # Multi-RHS state-shape scratch buffers, allocated lazily by
        # the concrete subclass's :meth:`_ensure_apply_F`.
        self._etd2_KX_buf = None
        self._etd2_KX_buf_cp = None
        self._etd2_B_buf_cp = None

    # ------------------------------------------------------------------
    # View access
    # ------------------------------------------------------------------
    def __len__(self):
        return self._K

    def __getitem__(self, k):
        return self._views[k]

    @property
    def K(self):
        return self._K

    def add_source_class(self, source_instance):
        raise TypeError(
            "MultiRHSPropagationSolverETD2 expects per-RHS sources via "
            "solver[k].add_source_class(...). The k-th view's sources drive "
            "the k-th column of the multi-RHS state."
        )

    @property
    def list_of_sources(self):
        # Parent's ``_operator_at`` checks this to decide whether to call
        # ``injection``. Returning any view's source list makes the
        # gate behave like "at least one RHS has a source".
        for v in self._views:
            if v.list_of_sources:
                return v.list_of_sources
        return []

    @list_of_sources.setter
    def list_of_sources(self, value):
        # Parent's __init__ writes ``self.list_of_sources = []``; the
        # setter is a no-op since per-view lists own that state.
        if value:
            raise TypeError(
                "Use solver[k].add_source_class(...) to attach sources to "
                "the k-th RHS view."
            )

    @property
    def res(self):
        raise TypeError(
            "MultiRHSPropagationSolverETD2.res is per-view: use "
            "solver[k].res for the k-th RHS."
        )

    # ------------------------------------------------------------------
    # Injection — column k = view k's per-source-class sum
    # ------------------------------------------------------------------
    def injection(self, dz, z):
        f = self.dldz(z) * dz * PRINCE_UNITS.cm2sec
        out = np.zeros((self.dim_states, self._K))
        for k, view in enumerate(self._views):
            srcs = view.list_of_sources
            if not srcs:
                continue
            if len(srcs) > 1:
                col = np.sum([s.injection_rate(z) for s in srcs], axis=0)
            else:
                col = srcs[0].injection_rate(z)
            out[:, k] = f * col
        return out

    # ------------------------------------------------------------------
    # Backend-agnostic multi-RHS integration loop. ``state`` shape is
    # (n, K); ``apply_F`` is the multi-RHS closure built by the concrete
    # subclass's :meth:`_ensure_apply_F`.
    # ------------------------------------------------------------------
    def _integrate_multi(self, state, z_grid, step_hook=None):
        from .etd2 import _array_module, _step_buffers

        xp = _array_module(state)
        n, K = state.shape
        dtype = state.dtype
        # (n,) diag-shape scratch — populated by the host
        # ``_compute_diag_factors`` or the cupy ``phi_compute`` kernel.
        bufs = _step_buffers(n, xp=xp, dtype=dtype)
        # (n, K) state-shape scratch — overrides the (n,) buffers
        # ``_step_buffers`` allocated for the single-RHS path.
        bufs["F_phi"] = xp.empty((n, K), dtype=dtype)
        bufs["F_a"] = xp.empty((n, K), dtype=dtype)
        bufs["a"] = xp.empty((n, K), dtype=dtype)
        bufs["scratch_NK"] = xp.empty((n, K), dtype=dtype)

        nsteps = len(z_grid) - 1
        for k in range(nsteps):
            z0 = z_grid[k]
            z1 = z_grid[k + 1]
            h = z1 - z0
            d, apply_F = self._operator_at(z0)
            self._etd2_step_multi(state, h, d, apply_F, bufs, xp)
            if step_hook is not None:
                step_hook()
        return state

    @staticmethod
    def _etd2_step_multi(state, h, d, apply_F, bufs, xp):
        """One multi-RHS ETD2 step, in place on (n, K) ``state``.

        Phi factors are (n,) functions of ``d`` and are broadcast over
        the K column axis. Otherwise identical to
        :func:`etd2._etd2_step_numpy` / ``_etd2_step_cupy``.
        """
        from .etd2 import _compute_diag_factors

        if xp is np:
            _compute_diag_factors(h, d, bufs, np)
        else:
            from .etd2 import cupy_kernels

            K = cupy_kernels()
            bufs["h_buf"][0] = h
            K.phi_compute(d, bufs["h_buf"], bufs["eD"], bufs["phi1"], bufs["phi2"])

        eD = bufs["eD"]
        phi1 = bufs["phi1"]
        phi2 = bufs["phi2"]
        F_phi = bufs["F_phi"]
        F_a = bufs["F_a"]
        a = bufs["a"]
        SCR = bufs["scratch_NK"]

        apply_F(state, F_phi)
        # a = eD * state + h * phi1 * F_phi
        xp.multiply(eD[:, None], state, out=a)
        xp.multiply(phi1[:, None], F_phi, out=SCR)
        SCR *= h
        xp.add(a, SCR, out=a)

        apply_F(a, F_a)
        # state <- a + h * phi2 * (F_a - F_phi)
        xp.subtract(F_a, F_phi, out=SCR)
        SCR *= h
        xp.multiply(phi2[:, None], SCR, out=SCR)
        xp.add(a, SCR, out=state)
        return state


class _MultiRHSETD2SolverCPU(MultiRHSPropagationSolverETD2, ETD2SolverCPU):
    """Multi-RHS CPU: scipy SpMM. ``M_off``-handle MKL ``mkl_sparse_d_mm``
    is not wrapped, so the MKL backend setting falls through to scipy
    SpMM here too.
    """

    def _init_state(self):
        return np.zeros((self.dim_states, self._K))

    def _run_integration(self, state, z_grid, step_hook=None):
        self._integrate_multi(state, z_grid, step_hook=step_hook)

    def _ensure_apply_F(self):
        if self._etd2_apply_F is not None:
            return
        # (n,) diag scratch (same shape as single-RHS).
        self._etd2_d_buf = np.empty(self.dim_states, dtype=np.float64)
        self._etd2_kx_buf = np.empty(self.dim_states, dtype=np.float64)
        self._etd2_kappa_buf = np.zeros_like(
            self.adia_loss_rates_grid.energy_vector
        )
        # (n, K) state-shape scratch.
        self._etd2_KX_buf = np.empty(
            (self.dim_states, self._K), dtype=np.float64
        )
        # MKL Sparse SpMM not wrapped; force scipy SpMM regardless of backend.
        self._etd2_apply_F = self._make_apply_F_scipy()

    def _make_apply_F_scipy(self):
        M_off = self._etd2_M_raw_off
        D_off = self._etd2_D_off
        params = self._etd2_apply_F_params
        KX_buf = self._etd2_KX_buf

        def apply_F(X, OUT):
            # SpMM: scipy CSR ``@`` 2D dense returns (n, K) ndarray; no
            # in-place option in scipy.sparse, so we copy into OUT.
            np.copyto(OUT, M_off.dot(X))
            kappa = params["kappa"]
            if kappa is not None:
                np.multiply(kappa[:, None], X, out=KX_buf)
                np.add(OUT, D_off.dot(KX_buf), out=OUT)
            np.multiply(OUT, params["dldz"], out=OUT)
            B = params["b"]
            if B is not None:
                np.add(OUT, B, out=OUT)

        return apply_F

    def _make_apply_F_mkl(self):
        # MKL Sparse SpMM not wrapped; fall back to scipy SpMM.
        return self._make_apply_F_scipy()


class _MultiRHSETD2SolverCUPY(MultiRHSPropagationSolverETD2, ETD2SolverCUPY):
    """Multi-RHS cupy: eager cuSPARSE SpMM. CUDA Graphs not supported."""

    def _init_state(self):
        import cupy as cp
        dt = np.dtype(self.backend.cupy_dtype)
        return cp.zeros((self.dim_states, self._K), dtype=dt)

    def _run_integration(self, state, z_grid, step_hook=None):
        # Multi-RHS does not support CUDA Graph capture (cupy 14 blocks
        # cuSPARSE during capture); the eager cuSPARSE SpMM is already
        # K-amortised so the graph win would be small.
        self._integrate_multi(state, z_grid, step_hook=step_hook)

    def _ensure_apply_F(self):
        if self._etd2_apply_F is not None:
            return
        import cupy as cp

        dt = np.dtype(self.backend.cupy_dtype)
        n = self.dim_states
        K = self._K
        if self._etd2_kappa_buf is None:
            self._etd2_kappa_buf = np.zeros_like(
                self.adia_loss_rates_grid.energy_vector
            )
        # (n,) device buffers.
        self._etd2_d_buf_cp = cp.empty(n, dtype=dt)
        self._etd2_kappa_buf_cp = cp.zeros(n, dtype=dt)
        self._etd2_dldz_buf_cp = cp.empty(1, dtype=dt)
        # (n, K) device buffers.
        self._etd2_KX_buf_cp = cp.empty((n, K), dtype=dt)
        self._etd2_B_buf_cp = cp.zeros((n, K), dtype=dt)

        has_kappa = bool(self.enable_partial_diff_jacobian)
        has_b = bool(self.enable_injection_jacobian) and any(
            v.list_of_sources for v in self._views
        )
        self._etd2_apply_F_state_cp = {
            "M_off": self._etd2_M_raw_off,
            "D_off": self._etd2_D_off,
            "has_kappa": has_kappa,
            "has_b": has_b,
        }
        self._etd2_apply_F = self._make_apply_F()

    def _make_apply_F(self):
        import cupy as cp

        kappa_buf = self._etd2_kappa_buf_cp
        KX_buf = self._etd2_KX_buf_cp
        B_buf = self._etd2_B_buf_cp
        dldz_buf = self._etd2_dldz_buf_cp
        state = self._etd2_apply_F_state_cp

        def apply_F(X, OUT):
            M_off = state["M_off"]
            # eager cuSPARSE SpMM through cupyx.scipy.sparse ``@``.
            cp.copyto(OUT, M_off @ X)
            if state["has_kappa"]:
                cp.multiply(kappa_buf[:, None], X, out=KX_buf)
                cp.add(OUT, state["D_off"] @ KX_buf, out=OUT)
            # Tail: out = out * dldz + B (broadcast scalar from
            # 1-element device buffer).
            cp.multiply(OUT, dldz_buf, out=OUT)
            if state["has_b"]:
                cp.add(OUT, B_buf, out=OUT)

        return apply_F

    def _operator_at_device(self, dldz, kappa_host, b_host):
        """Multi-RHS variant: upload K-shaped B instead of (n,) b."""
        import cupy as cp
        from .etd2 import cupy_kernels

        dt = self._etd2_dldz_buf_cp.dtype
        self._etd2_dldz_buf_cp[0] = dt.type(dldz)
        if kappa_host is not None:
            cp.copyto(
                self._etd2_kappa_buf_cp,
                cp.asarray(kappa_host, dtype=dt),
            )
        # B is (n, K) host; upload mirrors the single-RHS path and
        # stays within the cupy mempool's working set.
        if b_host is not None:
            cp.copyto(
                self._etd2_B_buf_cp,
                cp.asarray(b_host, dtype=dt),
            )

        K = cupy_kernels()
        if kappa_host is not None:
            K.compute_d(
                self._etd2_M_raw_diag,
                self._etd2_kappa_buf_cp,
                self._etd2_D_diag,
                self._etd2_dldz_buf_cp,
                self._etd2_d_buf_cp,
            )
        else:
            K.compute_d_no_kappa(
                self._etd2_M_raw_diag,
                self._etd2_dldz_buf_cp,
                self._etd2_d_buf_cp,
            )
        return self._etd2_d_buf_cp, self._etd2_apply_F
