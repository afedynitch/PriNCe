"""Contains solvers, i.e. integrators, kernels, steppers, for PriNCe."""

import numpy as np

from prince_cr.cosmology import H
from prince_cr.data import PRINCE_UNITS, EnergyGrid
from prince_cr.util import info
import prince_cr.config as config

from .partial_diff import DifferentialOperator


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

    def get_solution(self, nco_id):
        """Returns the spectrum in energy per nucleon"""
        spec = self.spec_man.ncoid2sref[nco_id]
        return self.egrid, self.state[spec.lidx() : spec.uidx()]

    def get_solution_scale(self, nco_id, epow=0):
        """Returns the spectrum scaled back to total energy"""
        spec = self.spec_man.ncoid2sref[nco_id]
        egrid = spec.A * self.egrid
        return egrid, egrid**epow * self.state[spec.lidx() : spec.uidx()] / spec.A

    def _check_id_grid(self, nco_ids, egrid):
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

        if isinstance(nco_ids, list):
            pass
        elif nco_ids == "CR":
            nco_ids = [s for s in self.known_species if s >= 100]
        elif nco_ids == "nu":
            nco_ids = [s for s in self.known_species if s in [11, 12, 13, 14, 15, 16]]
        elif nco_ids == "all":
            nco_ids = self.known_species
        elif isinstance(nco_ids, tuple):
            select, vmin, vmax = nco_ids
            nco_ids = [s for s in self.known_species if vmin <= select(s) <= vmax]

        return nco_ids, com_egrid

    def _collect_interpolated_spectra(self, nco_ids, epow, egrid=None):
        """Collect interpolated spectra in a 2D array. Used by
        get_solution_group and get_lnA"""
        nco_ids, com_egrid = self._check_id_grid(nco_ids, egrid)

        # collect all the spectra in 2d array of dimension
        spectra = np.zeros((len(nco_ids), com_egrid.size))
        for idx, pid in enumerate(nco_ids):
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

        return nco_ids, com_egrid, spectra

    def get_solution_group(self, nco_ids, epow=3, egrid=None):
        """Return the summed spectrum (in total energy) for all elements in the range"""

        _, com_egrid, spectra = self._collect_interpolated_spectra(nco_ids, epow, egrid)
        spectrum = spectra.sum(axis=0)

        return com_egrid, spectrum

    def get_lnA(self, nco_ids, egrid=None):
        """Return the average ln(A) as a function of total energy for all
        elements in the range"""

        nco_ids, com_egrid, spectra = self._collect_interpolated_spectra(
            nco_ids, 0, egrid
        )

        # get the average and variance by using the spectra as weights
        lnA = np.array([np.log(self.spec_man.ncoid2sref[el].A) for el in nco_ids])
        total = spectra.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            average = (lnA[:, np.newaxis] * spectra).sum(axis=0) / total
            variance = (lnA[:, np.newaxis] ** 2 * spectra).sum(
                axis=0
            ) / total - average**2

        return com_egrid, average, variance

    def get_energy_density(self, nco_id):
        from scipy.integrate import trapezoid as trapz

        A = self.spec_man.ncoid2sref[nco_id].A
        return trapz(A * self.egrid * self.get_solution(nco_id), self.egrid)


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
        z_offset=0.0,
    ):
        self.initial_z = initial_z + z_offset
        self.final_z = final_z + z_offset
        self.z_offset = z_offset

        self.current_z_rates = None
        self.recomp_z_threshold = config.update_rates_z_threshold

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

        self.had_int_rates = prince_run.int_rates
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
            prince_run.cr_grid, prince_run.spec_man.nspec
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

    Treats the diagonal of ``L(z) = J(z) + dl/dz · D · diag(κ(z))`` exactly
    via ``exp(h · diag(L))`` and the off-diagonal block with two SpMVs per
    stage (4 SpMVs / step). Source term ``b(z) = injection(z)`` enters via
    the same φ₁/φ₂ machinery, frozen at step start to preserve 2nd order.

    Caching: the expensive z-dependent pieces — the photo-hadronic rate
    matrix ``M_raw(z)`` and the pair-production loss vector ``κ_pair(z)``
    (CIB interpolated at ``dim_cr × xi_steps`` points) — are refreshed
    together at ``recomp_z_threshold`` resolution. Truly cheap pieces
    (``dl/dz(z)``, ``κ_adia(z)`` closed form, source ``b(z)``) are
    recomputed every step.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pieces refreshed at ``recomp_z_threshold`` resolution (z tracked by
        # ``self.current_z_rates`` from the base class).
        self._etd2_M_raw_diag = None
        self._etd2_M_raw_off = None
        # κ_pair(z) is expensive (CIB interpolated at dim_cr × xi_steps points).
        # κ_adia(z) is closed-form and trivially cheap, recomputed per step.
        self._etd2_kappa_pair_cached = None
        # Constant pieces, populated on first solve().
        self._etd2_D_diag = None
        self._etd2_D_off = None

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
            M = M.copy() if hasattr(M, "copy") else M
            M.data = np.zeros_like(M.data)
        d_M, M_off = split_operator(M)
        self._etd2_M_raw_diag = d_M
        self._etd2_M_raw_off = M_off

        if self.enable_pairprod_losses:
            self._etd2_kappa_pair_cached = self.pair_loss_rates_grid.loss_vector(z)
        else:
            self._etd2_kappa_pair_cached = None

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

    def _operator_at(self, z):
        """Return ``(d, apply_F)`` for the ETD2 step at redshift ``z``.

        d = dldz(z) · (M_raw_diag + κ(z) ⊙ D_diag)
        apply_F(x, out) = dldz(z) · M_raw_off · x
                          + dldz(z) · D_off · (κ(z) ⊙ x)
                          + b(z)

        κ(z) = κ_adia(z) (per-step, closed form) + κ_pair(z) (cached at
        ``recomp_z_threshold`` together with the photo-hadronic matrix).
        """
        if (
            self.current_z_rates is None
            or abs(z - self.current_z_rates) > self.recomp_z_threshold
        ):
            self._refresh_z_caches(z)

        dldz = self.dldz(z)
        if self.enable_partial_diff_jacobian:
            kappa = np.zeros_like(self.adia_loss_rates_grid.energy_vector)
            if self.enable_adiabatic_losses:
                kappa += self.adia_loss_rates_grid.loss_vector(z)
            if self._etd2_kappa_pair_cached is not None:
                kappa += self._etd2_kappa_pair_cached
        else:
            kappa = None

        # Diagonal d = dldz · (M_raw_diag + κ ⊙ D_diag)
        d = dldz * self._etd2_M_raw_diag.copy()
        if kappa is not None:
            d += dldz * (kappa * self._etd2_D_diag)

        if self.enable_injection_jacobian and self.list_of_sources:
            b = self.injection(1.0, z)
        else:
            b = None

        M_off = self._etd2_M_raw_off
        D_off = self._etd2_D_off if kappa is not None else None

        # Pre-allocate one scratch buffer for κ⊙x — reused inside apply_F.
        kx_buf = np.empty(self.dim_states) if kappa is not None else None

        def apply_F(x, out):
            np.copyto(out, M_off.dot(x))
            if kappa is not None:
                np.multiply(kappa, x, out=kx_buf)
                np.add(out, D_off.dot(kx_buf), out=out)
            np.multiply(out, dldz, out=out)
            if b is not None:
                np.add(out, b, out=out)

        return d, apply_F

    def solve(
        self,
        dz=1e-3,
        verbose=False,
        summary=False,
        progressbar=False,
    ):
        from time import time
        from .etd2 import integrate

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

        state = np.zeros(self.dim_states)
        # Force first-step rebuild of the cached photo-hadronic matrix.
        self.current_z_rates = None
        self._ensure_D_split()

        self.pre_step_hook(self.initial_z)

        info(2, f"ETD2: integrating with {len(z_grid) - 1} steps of dz≈{dz_step:.2e}")
        integrate(state, z_grid, operator_at=self._operator_at)

        self.post_step_hook(self.final_z)
        self.state = state
        end_time = time()
        info(2, "ETD2: integration completed in {0:.2f} s".format(end_time - start_time))

        if summary or verbose:
            print("ETD2 summary:")
            print(f"  steps: {len(z_grid) - 1}")
            print(f"  initial z: {self.initial_z} → final z: {self.final_z}")
            print(f"  wall time: {end_time - start_time:.3f} s")
