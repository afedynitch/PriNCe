"""Contains solvers, i.e. integrators, kernels, steppers, for PriNCe."""

import numpy as np
import scipy.sparse as sp

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
        # κ_pair(z) is expensive (CIB interpolated at dim_cr × xi_steps points).
        # κ_adia(z) is closed-form and trivially cheap, recomputed per step.
        self._etd2_kappa_pair_cached = None
        # Constant pieces, populated on first solve().
        self._etd2_D_diag = None
        self._etd2_D_off = None
        # MKL handle wrapping `_etd2_D_off`. One-shot for the whole solve.
        self._etd2_D_off_mkl = None
        self._backend = config.linear_algebra_backend.lower()

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
                self._etd2_D_off_mkl = self._build_mkl_handle(D_off, optimize=True)

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
        # indices; sort both in place once so the per-row two-pointer
        # walk below stays linear.
        if not getattr(M, "has_sorted_indices", False):
            M.sort_indices()
        if not getattr(M_off, "has_sorted_indices", False):
            M_off.sort_indices()

        n = M.shape[0]
        M_indptr = M.indptr
        M_indices = M.indices
        Moff_indptr = M_off.indptr
        Moff_indices = M_off.indices

        off_to_M = np.empty(M_off.nnz, dtype=np.int64)
        diag_to_M = np.full(n, -1, dtype=np.int64)

        k_off = 0
        for r in range(n):
            m_lo, m_hi = M_indptr[r], M_indptr[r + 1]
            o_lo, o_hi = Moff_indptr[r], Moff_indptr[r + 1]
            o = o_lo
            for k in range(m_lo, m_hi):
                col = M_indices[k]
                if col == r:
                    diag_to_M[r] = k
                    continue
                # Walk M_off forward until its column matches; entries
                # ``eliminate_zeros`` removed are skipped silently. Both
                # arrays are sorted ascending in column.
                while o < o_hi and Moff_indices[o] < col:
                    o += 1
                if o < o_hi and Moff_indices[o] == col:
                    off_to_M[k_off] = k
                    k_off += 1
                    o += 1
            if o != o_hi:
                # M_off has columns at this row that don't appear in M —
                # that contradicts split_operator's contract.
                raise RuntimeError(
                    f"Row {r}: M_off has columns with no source in M."
                )

        if k_off != M_off.nnz:
            raise RuntimeError(
                f"Index map covered {k_off} of {M_off.nnz} M_off entries."
            )
        return off_to_M, diag_to_M

    @staticmethod
    def _build_mkl_handle(off, optimize=True):
        """Wrap a scipy CSR/COO/etc. off-diagonal into an MklSparseMatrix.

        Returns ``None`` if the matrix has zero nnz — the kernel skips the
        SpMV in that case rather than feeding MKL an empty handle (some
        MKL versions are squirrelly about zero-nnz CSRs). Picks BSR if
        ``config.mkl_bsr_blocksize`` is set, else CSR.

        ``optimize=True`` chooses the inspector-executor fast path —
        ~2× faster per gemv but caches data values internally, so
        :meth:`MklSparseMatrix.update_data` cannot refresh them. Use
        ``False`` for matrices whose values change across cache windows
        (PriNCe's photo-hadronic ``M_off``); ``True`` for matrices that
        are constant for the whole solve (the FD operator ``D_off``).
        """
        from .. import mkl_sparse

        if not sp.isspmatrix_csr(off):
            off = off.tocsr()
        if off.dtype != np.float64:
            off = off.astype(np.float64)
        if off.nnz == 0:
            return None
        bs = getattr(config, "mkl_bsr_blocksize", None)
        return mkl_sparse.MklSparseMatrix(off, blocksize=bs, optimize=optimize)

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

        # Pre-allocate one scratch buffer for κ⊙x — reused inside apply_F.
        kx_buf = np.empty(self.dim_states) if kappa is not None else None

        if self._backend == "mkl" and self._etd2_M_raw_off_mkl is not None:
            apply_F = self._make_apply_F_mkl(kappa, dldz, b, kx_buf)
        else:
            apply_F = self._make_apply_F_scipy(kappa, dldz, b, kx_buf)

        return d, apply_F

    def _make_apply_F_scipy(self, kappa, dldz, b, kx_buf):
        M_off = self._etd2_M_raw_off
        D_off = self._etd2_D_off if kappa is not None else None

        def apply_F(x, out):
            np.copyto(out, M_off.dot(x))
            if kappa is not None:
                np.multiply(kappa, x, out=kx_buf)
                np.add(out, D_off.dot(kx_buf), out=out)
            np.multiply(out, dldz, out=out)
            if b is not None:
                np.add(out, b, out=out)

        return apply_F

    def _make_apply_F_mkl(self, kappa, dldz, b, kx_buf):
        """Build an MKL-backed ``apply_F(x, out)``.

        Per call:

          out = M_off · x                                (mkl gemv α=1, β=0)
          if kappa: kx_buf = κ ⊙ x; out += D_off · kx_buf (mkl gemv α=1, β=1)
          out *= dldz; if b: out += b

        ``etd2_step`` calls ``apply_F`` twice per ETD2 step against four
        persistent buffers (state, F_phi, a, F_a) plus kx_buf. We memoise
        the ctypes pointer for each ndarray by ``id`` so we don't redo
        ``arr.ctypes.data_as`` on every step. The buffers come from
        ``etd2._step_buffers`` and live for the whole solve, so the
        cached pointers stay valid across the integration loop.
        """
        from ctypes import POINTER, c_double

        mkl_M = self._etd2_M_raw_off_mkl
        mkl_D = self._etd2_D_off_mkl if kappa is not None else None

        # Pre-box the only two (alpha, beta) constants we ever use,
        # avoiding ~5 µs of ``c_double(...)`` per gemv in the hot loop.
        # Sanity-check the M_off SpMV via the gemv_ctargs path on the
        # first call only — if MKL is going to fail it'll do so loudly
        # there. Subsequent calls go through gemv_preboxed.
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

        kx_p = get_p(kx_buf) if kappa is not None else None

        def apply_F(x, out):
            x_p = get_p(x)
            out_p = get_p(out)
            mkl_M_op(alpha_box, x_p, beta_zero, out_p)
            if kappa is not None:
                np.multiply(kappa, x, out=kx_buf)
                mkl_D_op(alpha_box, kx_p, beta_one, out_p)
            np.multiply(out, dldz, out=out)
            if b is not None:
                np.add(out, b, out=out)

        return apply_F

    def close(self):
        """Release backend resources (currently: MKL sparse handles).

        Idempotent. Safe to call repeatedly and safe to skip — Python
        reference-counting eventually frees the handles via
        ``MklSparseMatrix.__del__``, but the underlying MKL-internal
        optimised-layout memory only drops on explicit
        ``mkl_sparse_destroy``. Calling ``close()`` between long-running
        runs avoids accumulating that memory.
        """
        for attr in ("_etd2_M_raw_off_mkl", "_etd2_D_off_mkl"):
            h = getattr(self, attr, None)
            if h is not None:
                try:
                    h.close()
                except Exception:
                    pass
                setattr(self, attr, None)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

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
        # Drop the M_off cache too: a fresh ``_refresh_z_caches`` will
        # rebuild it (and the MKL handle + index maps) from the current
        # rate matrix, then keep the same handle alive across all this
        # solve's cache windows via in-place data updates.
        self.current_z_rates = None
        self._etd2_M_raw_off = None
        self._etd2_M_off_to_M_idx = None
        self._etd2_M_diag_to_M_idx = None
        if self._etd2_M_raw_off_mkl is not None:
            self._etd2_M_raw_off_mkl.close()
            self._etd2_M_raw_off_mkl = None
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
