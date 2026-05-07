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
        # Persistent per-step host scratch buffers. Sized to ``dim_states``
        # at solve() start. Reused across all steps to keep the
        # ``_operator_at`` body alloc-free on the hot path.
        self._etd2_d_buf = None
        self._etd2_kappa_buf = None
        self._etd2_kx_buf = None
        # Mutable per-step parameter holder, captured by the persistent
        # apply_F closure. ``dldz`` is a python float scalar; ``b`` is
        # either None or a per-step injection vector that we copy into
        # ``self._etd2_b_buf`` so the closure can read a fixed buffer.
        self._etd2_apply_F = None
        self._etd2_apply_F_params = {"dldz": 1.0, "kappa": None, "b": None}
        self._etd2_b_buf = None
        self._backend = self.backend.linear_algebra_backend.lower()
        # The full-GPU cupy backend reuses the dense-matvec lookahead
        # plumbing from the CPU/GPU hybrid backend (handled in
        # ``interaction_rates``) but consumes the result on-device.
        self._is_cupy_backend = self._backend == "cupy"
        # CPU/GPU hybrid lookahead state. Populated at solve() start
        # when ``config.use_cupy_dense_lookahead`` is True; reset after.
        self._lookahead_anchors = None
        self._next_anchor_idx = 0
        # cupy persistent device buffers. Allocated once per solve (in
        # :meth:`_ensure_apply_F_cupy`) and held for the whole integration
        # so the per-step body is alloc-free — required for kernel fusion
        # to pay off at the per-step scale and for stable buffer addresses
        # inside captured CUDA Graphs. Per-step values (kappa, b, dldz, d)
        # are refreshed in place; cache-window-bound values (M_diag,
        # D_diag) are referenced by the apply_F closure and rebound on
        # cache window refresh.
        self._etd2_d_buf_cp = None
        self._etd2_kappa_buf_cp = None
        self._etd2_b_buf_cp = None
        self._etd2_kx_buf_cp = None
        self._etd2_dldz_buf_cp = None
        # Mutable closure-state for the cupy apply_F. Rebound on each
        # ``_refresh_z_caches`` because ``split_operator`` returns fresh
        # cupy CSR arrays at every cache window (the host path keeps
        # the same handles via in-place data updates; the GPU path
        # currently re-splits — see :meth:`_refresh_z_caches`).
        self._etd2_apply_F_state_cp = None
        # CUDA Graph machinery. Allocated lazily when
        # ``config.use_cuda_graphs`` is True. Re-captured on each cache
        # window refresh because the SpMV kernel calls inside
        # ``apply_F`` close over the current ``M_off`` / ``D_off``
        # cupy CSRs whose pointers go stale across windows.
        self._etd2_graph_exec = None
        self._etd2_graph_stream = None
        self._etd2_graph_state_buf = None
        self._etd2_graph_needs_capture = True

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
            # Zero matrix with full sparsity preserved. Works for either
            # scipy or cupy CSR — both support ``.copy()`` and slice
            # assignment on ``.data``.
            M = M.copy()
            M.data[:] = 0

        first_window = self._etd2_M_raw_off is None
        if self._is_cupy_backend:
            # Full-GPU path: ``M`` is a cupyx.scipy.sparse.csr_matrix
            # already on-device (see :meth:`PhotoNuclearInteractionRate.
            # _update_rates_cupy`). The xp-aware ``split_operator``
            # returns cupy arrays. Re-splitting each cache window is a
            # few ms on the GPU at production grid; we skip the host
            # index-map gather for now (deferred until profile shows it
            # matters).
            d_M, M_off = split_operator(M)
            self._etd2_M_raw_diag = d_M
            self._etd2_M_raw_off = M_off
            # The apply_F closure dereferences M_off / D_off through a
            # mutable container so we can swap pointers here without
            # rebuilding it. Triggers re-capture of the CUDA graph (if
            # active) on the next ``_operator_at_cupy``.
            self._rebind_apply_F_cupy()
        elif first_window:
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

        # CPU/GPU hybrid: dispatch the next anchor's GPU dense matvec in
        # the background while this cache window's ETD2 steps run on CPU.
        # The consume happens inside the next ``_refresh_z_caches`` call
        # via ``had_int_rates._try_consume_for(z)``.
        self._schedule_next_lookahead_anchor()

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

        if self._is_cupy_backend:
            # Full-GPU: upload D split to GPU once. Stored under the
            # same attribute names as the host path — ``_make_apply_F_cupy``
            # reads them as cupy arrays. Dtype follows ``config.cupy_dtype``.
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

        Vectorized via ``numpy.searchsorted`` over flat (row, col) keys
        — at production grid (M.nnz ≈ 1.2 M, M_off.nnz ≈ 895 k) the
        original per-row two-pointer Python loop showed up as ~0.9 s
        of one-shot setup time in cProfile of a 4.6 s solve. The
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

        # Off-diagonal: build M_off keys the same way and locate each
        # one in M_keys via searchsorted. ``M_off`` is by construction a
        # subset of M's off-diagonal entries (split_operator zeroed the
        # diagonal then eliminate_zeros may drop more), so every M_off
        # key is present in M_keys. searchsorted on a sorted array
        # returns the insertion index, which equals the matching
        # M.data index when the key is present.
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
    # Hybrid-backend deterministic-grid lookahead for GPU dense matvec
    # ------------------------------------------------------------------
    def _compute_lookahead_anchors(self, z_grid):
        """Predict the redshifts at which ``_refresh_z_caches`` will fire.

        ETD2 walks ``z_grid[:-1]`` with ``operator_at(z_k)``; the cache
        invalidates whenever ``|z - last_anchor| > recomp_z_threshold``.
        That contract is deterministic — anchor[0] is z_grid[0]; each
        subsequent anchor is the first grid point with abs(Δ) above
        threshold from the previous anchor. The list returned here
        feeds the GPU-dgemv lookahead.
        """
        anchors = []
        last = None
        thr = self.recomp_z_threshold
        for z in z_grid[:-1]:
            if last is None or abs(z - last) > thr:
                anchors.append(float(z))
                last = z
        return anchors

    def _lookahead_active(self):
        return (
            self.backend.use_cupy_dense_lookahead
            and config.has_cupy
            and self._lookahead_anchors is not None
        )

    def _schedule_next_lookahead_anchor(self):
        """Dispatch the upcoming anchor's GPU rebuild, if lookahead is on."""
        if not self._lookahead_active():
            return
        idx = self._next_anchor_idx
        if idx >= len(self._lookahead_anchors):
            return
        z_next = self._lookahead_anchors[idx]
        self.had_int_rates.schedule_rebuild(z_next)
        self._next_anchor_idx = idx + 1

    def _build_mkl_handle(self, off, optimize=True, blocksize=_DEFAULT):
        """Wrap a scipy CSR/COO/etc. off-diagonal into an MklSparseMatrix.

        Returns ``None`` if the matrix has zero nnz — the kernel skips the
        SpMV in that case rather than feeding MKL an empty handle (some
        MKL versions are squirrelly about zero-nnz CSRs).

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

    def _operator_at(self, z):
        """Return ``(d, apply_F)`` for the ETD2 step at redshift ``z``.

        d = dldz(z) · (M_raw_diag + κ(z) ⊙ D_diag)
        apply_F(x, out) = dldz(z) · M_raw_off · x
                          + dldz(z) · D_off · (κ(z) ⊙ x)
                          + b(z)

        κ(z) = κ_adia(z) (per-step, closed form) + κ_pair(z) (cached at
        ``recomp_z_threshold`` together with the photo-hadronic matrix).

        Hot-path bookkeeping notes (CPU paths only — cupy variant lives
        in :meth:`_operator_at_cupy`):

        * ``d``, ``kappa`` and the ``apply_F`` closure are built once
          per ``solve()`` (in :meth:`_ensure_apply_F` / :meth:`solve`)
          and refreshed in place per step. At m=56 / dz=1e-3 that's
          ~50 µs/step of saved alloc + closure-build overhead vs
          rebuilding both per call.
        * ``b`` is per-step (the injection rate depends on z) and is
          pulled from ``injection(1.0, z)`` directly each step. We
          keep a reference into the closure's mutable parameter dict
          so the closure body can read the fresh value without being
          rebuilt.
        """
        if (
            self.current_z_rates is None
            or abs(z - self.current_z_rates) > self.recomp_z_threshold
        ):
            self._refresh_z_caches(z)

        if self._etd2_apply_F is None:
            # cupy path needs host kappa scratch too —
            # ``_ensure_apply_F_cupy`` allocates it alongside the
            # persistent device buffers. Both branches require
            # ``_refresh_z_caches`` to have run first (M_off must exist),
            # which is guaranteed by the cache-window check above.
            if self._is_cupy_backend:
                self._ensure_apply_F_cupy()
            else:
                self._ensure_apply_F()

        dldz = self.dldz(z)
        if self.enable_partial_diff_jacobian:
            # Persistent kappa buffer; lazily initialized in
            # :meth:`_ensure_apply_F` against the host loss-vector dtype.
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

        if self._is_cupy_backend:
            return self._operator_at_cupy(z, dldz, kappa, b)

        # Diagonal d = dldz · (M_raw_diag + κ ⊙ D_diag) — written into a
        # persistent buffer; ``etd2_step`` does not retain ``d`` past
        # the step body, so reusing the buffer is safe.
        d = self._etd2_d_buf
        np.multiply(self._etd2_M_raw_diag, dldz, out=d)
        if kappa is not None:
            # d += dldz * (kappa * D_diag) without a fresh temporary.
            np.multiply(kappa, self._etd2_D_diag, out=self._etd2_kx_buf)
            np.multiply(self._etd2_kx_buf, dldz, out=self._etd2_kx_buf)
            np.add(d, self._etd2_kx_buf, out=d)

        # Refresh the persistent apply_F closure's per-step params.
        params = self._etd2_apply_F_params
        params["dldz"] = dldz
        params["kappa"] = kappa
        params["b"] = b
        return d, self._etd2_apply_F

    def _ensure_apply_F(self):
        """Lazily build the persistent CPU apply_F closure.

        Called from :meth:`solve` after :meth:`_ensure_D_split`. The
        closure captures ``self._etd2_apply_F_params`` (mutable dict)
        so per-step refreshes happen via dict assignment in
        :meth:`_operator_at`, not by rebuilding the closure.
        """
        if self._etd2_apply_F is not None:
            return
        # Persistent host scratch sized to dim_states.
        self._etd2_d_buf = np.empty(self.dim_states, dtype=np.float64)
        self._etd2_kx_buf = np.empty(self.dim_states, dtype=np.float64)
        # ``kappa`` is sized to the loss-vector grid, which is the same
        # ``dim_states`` length but takes its dtype from the loss grid.
        self._etd2_kappa_buf = np.zeros_like(
            self.adia_loss_rates_grid.energy_vector
        )

        if self._backend == "mkl" and self._etd2_M_raw_off_mkl is not None:
            self._etd2_apply_F = self._make_apply_F_mkl()
        else:
            self._etd2_apply_F = self._make_apply_F_scipy()

    def _operator_at_cupy(self, z, dldz, kappa_host, b_host):
        """cupy variant: refresh persistent device buffers.

        Writes per-step values into pre-allocated device buffers, then
        evaluates ``d = dldz · (M_diag + κ ⊙ D_diag)`` via the fused
        ``compute_d`` ElementwiseKernel. No allocations on the hot path
        beyond a tiny intermediate cupy array for the host-to-device
        upload (mempool-reused). Returns the persistent ``d`` and
        ``apply_F`` references — the closure was built once at first
        call and rebound to current ``M_off`` / ``D_off`` on each cache
        window refresh by :meth:`_rebind_apply_F_cupy`.
        """
        import cupy as cp
        from .etd2 import cupy_kernels

        if self._etd2_apply_F is None:
            self._ensure_apply_F_cupy()

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

        # Compute d on the device via the fused kernel. Writes into the
        # persistent ``_etd2_d_buf_cp`` so the address ``etd2_step``
        # eventually loads from is stable across the whole solve.
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

    def _ensure_apply_F_cupy(self):
        """Allocate persistent cupy buffers and build the apply_F closure.

        Called from :meth:`_operator_at_cupy` on first invocation per
        solve. ``M_off`` / ``D_off`` are bound through the mutable
        ``self._etd2_apply_F_state_cp`` dict so :meth:`_rebind_apply_F_cupy`
        can swap them on cache-window refresh without rebuilding the
        closure object (which keeps a stable callable identity for
        :func:`etd2.integrate` and the CUDA Graph capture path).
        """
        import cupy as cp

        dt = np.dtype(self.backend.cupy_dtype)
        n = self.dim_states

        # Host scratch for kappa accumulation (used by ``_operator_at``
        # to compute κ_adia + κ_pair before upload). Same dtype as the
        # loss-vector grid so the sum stays in fp64 host arithmetic
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

        # ``_etd2_M_raw_off`` / ``_etd2_D_off`` are referenced by the
        # closure through this dict so the cache-window-refresh path
        # can swap them in place via :meth:`_rebind_apply_F_cupy`.
        self._etd2_apply_F_state_cp = {
            "M_off": self._etd2_M_raw_off,
            "D_off": self._etd2_D_off,
            "has_kappa": has_kappa,
            "has_b": has_b,
        }

        self._etd2_apply_F = self._make_apply_F_cupy()

    def _rebind_apply_F_cupy(self):
        """Swap ``M_off`` / ``D_off`` references on a cache window refresh.

        Called from :meth:`_refresh_z_caches` cupy branch after
        ``split_operator`` returns fresh cupy CSR arrays. Also drops the
        captured CUDA Graph executable — its cuSPARSE descriptors hold
        device pointers from the previous window's M_off/D_off and would
        be dangling after we update the dict and Python frees the old
        cupy CSR objects.
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

    def _make_apply_F_cupy(self):
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

        Compared with the eager ``cp.copyto(out, M_off @ x)`` /
        ``out += D_off @ kx`` / ``out *= dldz`` / ``out += b`` chain,
        this saves ~3 launches per ``apply_F`` call (2 calls/step) plus
        the implicit allocations in cupy's sparse ``@``.
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

    def _make_apply_F_scipy(self):
        """Build the persistent scipy-backed ``apply_F(x, out)`` closure.

        The closure reads ``dldz`` / ``kappa`` / ``b`` from
        ``self._etd2_apply_F_params`` (a mutable dict refreshed by
        :meth:`_operator_at` per step). ``kx_buf`` is the persistent
        scratch allocated in :meth:`_ensure_apply_F`. ``M_off`` and
        ``D_off`` are pinned at closure-build time; both are stable
        ndarrays for the whole solve (M_off has its data refreshed in
        place by :meth:`_refresh_z_caches`; D_off is constant).
        """
        M_off = self._etd2_M_raw_off
        D_off = self._etd2_D_off
        params = self._etd2_apply_F_params
        kx_buf = self._etd2_kx_buf

        def apply_F(x, out):
            kappa = params["kappa"]
            np.copyto(out, M_off.dot(x))
            if kappa is not None:
                np.multiply(kappa, x, out=kx_buf)
                np.add(out, D_off.dot(kx_buf), out=out)
            np.multiply(out, params["dldz"], out=out)
            b = params["b"]
            if b is not None:
                np.add(out, b, out=out)

        return apply_F

    def _step_body_cupy_graph(self, state, bufs, K, has_kappa):
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
        :meth:`_step_body_cupy_graph`.

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
                # _rebind_apply_F_cupy drops _etd2_graph_exec and sets
                # _etd2_graph_needs_capture = True. Reset window state.
                win_step_count = 0

            # Lazy-init persistent buffers on first call after the first
            # _refresh_z_caches. Mirrors the cupy branch in :meth:`_operator_at`.
            if self._etd2_apply_F is None:
                self._ensure_apply_F_cupy()

            # Per-step host work: κ, b, dldz — exactly the same arithmetic
            # the eager :meth:`_operator_at_cupy` does, lifted here so we
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
                    self._step_body_cupy_graph(state, bufs, K, has_kappa)
                    win_step_count = 1
                elif self._etd2_graph_needs_capture and win_step_count == 1:
                    # Capture pass. ``begin_capture`` puts the stream
                    # in record-only mode — the kernel launches are
                    # captured into the graph but do NOT execute. We
                    # launch the graph once after end_capture so this
                    # step advances state like a real step (otherwise
                    # we'd silently lose one step per cache window).
                    stream.begin_capture()
                    self._step_body_cupy_graph(state, bufs, K, has_kappa)
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
                step_hook()

        stream.synchronize()
        return state

    def _make_apply_F_mkl(self):
        """Build the persistent MKL-backed ``apply_F(x, out)`` closure.

        Per call:

          out = M_off · x                                (mkl gemv α=1, β=0)
          if kappa: kx_buf = κ ⊙ x; out += D_off · kx_buf (mkl gemv α=1, β=1)
          out *= dldz; if b: out += b

        ``etd2_step`` calls ``apply_F`` twice per ETD2 step against four
        persistent buffers (state, F_phi, a, F_a) plus kx_buf. We memoise
        the ctypes pointer for each ndarray by ``id`` so we don't redo
        ``arr.ctypes.data_as`` on every step. The buffers come from
        ``etd2._step_buffers`` and live for the whole solve, so the
        cached pointers stay valid across the integration loop. Per-
        step ``dldz``/``kappa``/``b`` come from the mutable
        ``self._etd2_apply_F_params`` dict — :meth:`_operator_at`
        refreshes the values; the closure is built exactly once per
        solve.
        """
        from ctypes import POINTER, c_double

        mkl_M = self._etd2_M_raw_off_mkl
        mkl_D = self._etd2_D_off_mkl

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

        kx_buf = self._etd2_kx_buf
        kx_p = get_p(kx_buf)
        params = self._etd2_apply_F_params

        def apply_F(x, out):
            x_p = get_p(x)
            out_p = get_p(out)
            mkl_M_op(alpha_box, x_p, beta_zero, out_p)
            kappa = params["kappa"]
            if kappa is not None:
                np.multiply(kappa, x, out=kx_buf)
                mkl_D_op(alpha_box, kx_p, beta_one, out_p)
            np.multiply(out, params["dldz"], out=out)
            b = params["b"]
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
        # On the cupy backend the M/D split lives on the GPU. Drop the
        # cupy references so Python ref-counts them back to the
        # mempool. Explicit None release keeps long-running scripts
        # (mass scans) from pinning successive matrices on-device.
        if self._is_cupy_backend:
            for attr in (
                "_etd2_M_raw_off",
                "_etd2_M_raw_diag",
                "_etd2_D_off",
                "_etd2_D_diag",
                # cupy persistent device buffers + graph machinery.
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
        # Drop the persistent CPU apply_F closure so it gets rebuilt
        # against this solve's M_off / D_off (the M_off ndarray identity
        # changes when ``_refresh_z_caches`` runs split_operator on the
        # first window — the CPU closure pins ``M_off`` by reference).
        self._etd2_apply_F = None
        # On the full-GPU cupy backend, also drop the cupy D split so
        # ``_ensure_D_split`` rebuilds it for this solve. (The host
        # path keeps D — it's truly constant for the lifetime of the
        # PriNCeRun, but the GPU mirror may have come from a prior
        # backend selection.)
        if self._is_cupy_backend:
            self._etd2_D_diag = None
            self._etd2_D_off = None
            # Persistent cupy buffers + apply_F state. Dropped so this
            # solve starts fresh — they get reallocated by
            # :meth:`_ensure_apply_F_cupy` on the first
            # :meth:`_operator_at_cupy` call.
            self._etd2_d_buf_cp = None
            self._etd2_kappa_buf_cp = None
            self._etd2_b_buf_cp = None
            self._etd2_kx_buf_cp = None
            self._etd2_dldz_buf_cp = None
            self._etd2_apply_F_state_cp = None
            # Drop any captured CUDA Graph; the previous solve's
            # buffer addresses are no longer valid.
            self._etd2_graph_exec = None
            self._etd2_graph_stream = None
            self._etd2_graph_state_buf = None
            self._etd2_graph_needs_capture = True
        self._ensure_D_split()

        if self._is_cupy_backend:
            import cupy as cp

            dt = np.dtype(self.backend.cupy_dtype)
            state = cp.zeros(self.dim_states, dtype=dt)
        else:
            state = np.zeros(self.dim_states)

        # Hybrid-lookahead bookkeeping. Compute the cache-window anchor
        # schedule from the deterministic z_grid + threshold, drop any
        # stale GPU rebuild from a previous solve, and prime the first
        # anchor before entering the integrate loop — the first
        # ``_refresh_z_caches`` call will consume it. The full-GPU cupy
        # backend has its own GPU dense-matvec path and does not use
        # the lookahead machinery.
        if (
            not self._is_cupy_backend
            and self.backend.use_cupy_dense_lookahead
            and config.has_cupy
        ):
            self._lookahead_anchors = self._compute_lookahead_anchors(z_grid)
            self._next_anchor_idx = 0
            self.had_int_rates.drop_pending()
            self._schedule_next_lookahead_anchor()
            info(2, f"ETD2: scheduled {len(self._lookahead_anchors)} "
                    "GPU dense-matvec anchors for hybrid-backend lookahead.")
        else:
            self._lookahead_anchors = None
            self._next_anchor_idx = 0

        self.pre_step_hook(self.initial_z)

        nsteps = len(z_grid) - 1
        info(2, f"ETD2: integrating with {nsteps} steps of dz≈{dz_step:.2e}")
        with PrinceProgressBar(bar_type=progressbar, nsteps=nsteps) as pbar:
            step_hook = pbar.update if pbar.pbar is not None else None
            if (
                self._is_cupy_backend
                and self.backend.use_cuda_graphs
                and config.has_cupy
            ):
                self._solve_loop_graphs(state, z_grid, step_hook=step_hook)
            else:
                integrate(
                    state, z_grid,
                    operator_at=self._operator_at,
                    step_hook=step_hook,
                )

        # Done with the solve — clear lookahead state.
        if self._lookahead_anchors is not None:
            self.had_int_rates.drop_pending()
            self._lookahead_anchors = None
            self._next_anchor_idx = 0

        self.post_step_hook(self.final_z)
        if self._is_cupy_backend:
            # D2H + upcast back to fp64 host so downstream consumers
            # (UHECRPropagationResult, plotting, et al.) see the
            # canonical dtype.
            import cupy as cp

            self.state = cp.asnumpy(state).astype(np.float64)
        else:
            self.state = state
        end_time = time()
        info(2, "ETD2: integration completed in {0:.2f} s".format(end_time - start_time))

        if summary or verbose:
            print("ETD2 summary:")
            print(f"  steps: {len(z_grid) - 1}")
            print(f"  initial z: {self.initial_z} → final z: {self.final_z}")
            print(f"  wall time: {end_time - start_time:.3f} s")


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

    v1 limitations:

    * MKL backend falls through to scipy SpMM (we don't yet wrap
      ``mkl_sparse_d_mm``). Set ``linear_algebra_backend = "scipy"`` or
      ``"cupy"`` for explicit selection.
    * CUDA Graph capture is not implemented for the multi-RHS path
      (cupy 14 blocks cuSPARSE during capture; the eager cuSPARSE SpMM
      is already K-amortised so the graph win is small).
    """

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
        # :meth:`_ensure_apply_F` / :meth:`_ensure_apply_F_cupy`.
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
    # apply_F closures (multi-RHS)
    # ------------------------------------------------------------------
    def _ensure_apply_F(self):
        if self._etd2_apply_F is not None:
            return
        # (n,) diag scratch (same as parent).
        self._etd2_d_buf = np.empty(self.dim_states, dtype=np.float64)
        self._etd2_kx_buf = np.empty(self.dim_states, dtype=np.float64)
        self._etd2_kappa_buf = np.zeros_like(
            self.adia_loss_rates_grid.energy_vector
        )
        # (n, K) state-shape scratch.
        self._etd2_KX_buf = np.empty(
            (self.dim_states, self._K), dtype=np.float64
        )
        # MKL SpMM is not wrapped; force scipy SpMM for v1.
        self._etd2_apply_F = self._make_apply_F_scipy_multi()

    def _make_apply_F_scipy_multi(self):
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
        # MKL Sparse BLAS gemv is single-RHS; mkl_sparse_d_mm not wrapped.
        # Fall back to scipy SpMM.
        return self._make_apply_F_scipy_multi()

    def _ensure_apply_F_cupy(self):
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
        self._etd2_apply_F = self._make_apply_F_cupy_multi()

    def _make_apply_F_cupy_multi(self):
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

    def _operator_at_cupy(self, z, dldz, kappa_host, b_host):
        """Multi-RHS variant: refresh device buffers, compute (n,) ``d``."""
        import cupy as cp
        from .etd2 import cupy_kernels

        if self._etd2_apply_F is None:
            self._ensure_apply_F_cupy()

        dt = self._etd2_dldz_buf_cp.dtype
        self._etd2_dldz_buf_cp[0] = dt.type(dldz)
        if kappa_host is not None:
            cp.copyto(
                self._etd2_kappa_buf_cp,
                cp.asarray(kappa_host, dtype=dt),
            )
        # B is (n, K) host; per-cycle upload mirrors the single-RHS
        # path and stays within the cupy mempool's working set.
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

    def _solve_loop_graphs(self, *args, **kwargs):
        raise NotImplementedError(
            "CUDA Graph capture is not implemented for the multi-RHS "
            "solver; the eager cuSPARSE SpMM is already K-amortised."
        )

    # ------------------------------------------------------------------
    # solve() and integration loop
    # ------------------------------------------------------------------
    def solve(self, dz=1e-3, verbose=False, summary=False, progressbar=False):
        from time import time

        start_time = time()
        info(2, f"ETD2 multi-RHS (K={self._K}): setting up integration")

        dz_step = -float(abs(dz))
        n_full = int(np.floor((self.final_z - self.initial_z) / dz_step))
        z_grid = self.initial_z + np.arange(n_full + 1) * dz_step
        if abs(z_grid[-1] - self.final_z) > 1e-12:
            z_grid = np.concatenate([z_grid, [self.final_z]])

        # Reset cache-window state so this solve rebuilds the rate
        # matrix and the apply_F closure from scratch (mirrors parent).
        self.current_z_rates = None
        self._etd2_M_raw_off = None
        self._etd2_M_off_to_M_idx = None
        self._etd2_M_diag_to_M_idx = None
        if self._etd2_M_raw_off_mkl is not None:
            self._etd2_M_raw_off_mkl.close()
            self._etd2_M_raw_off_mkl = None
        self._etd2_apply_F = None
        if self._is_cupy_backend:
            self._etd2_D_diag = None
            self._etd2_D_off = None
            self._etd2_d_buf_cp = None
            self._etd2_kappa_buf_cp = None
            self._etd2_KX_buf_cp = None
            self._etd2_B_buf_cp = None
            self._etd2_dldz_buf_cp = None
            self._etd2_apply_F_state_cp = None

        self._ensure_D_split()

        n = self.dim_states
        K = self._K
        if self._is_cupy_backend:
            import cupy as cp

            dt = np.dtype(self.backend.cupy_dtype)
            STATE = cp.zeros((n, K), dtype=dt)
        else:
            STATE = np.zeros((n, K))

        # Multi-RHS does not use the hybrid-backend lookahead machinery
        # (which is single-RHS-only and tied to the dense matvec
        # caching pattern); explicitly drop any prior bookkeeping.
        self._lookahead_anchors = None
        self._next_anchor_idx = 0

        self.pre_step_hook(self.initial_z)
        nsteps = len(z_grid) - 1
        info(2, f"ETD2 multi-RHS: integrating with {nsteps} "
                f"steps of dz≈{dz_step:.2e}")
        with PrinceProgressBar(bar_type=progressbar, nsteps=nsteps) as pbar:
            step_hook = pbar.update if pbar.pbar is not None else None
            self._integrate_multi(STATE, z_grid, step_hook=step_hook)
        self.post_step_hook(self.final_z)

        if self._is_cupy_backend:
            import cupy as cp

            self.state = cp.asnumpy(STATE).astype(np.float64)
        else:
            self.state = STATE
        end_time = time()
        info(2, "ETD2 multi-RHS: integration completed in "
                f"{end_time - start_time:.2f} s")

        if summary or verbose:
            print(f"ETD2 multi-RHS summary (K={self._K}):")
            print(f"  steps: {len(z_grid) - 1}")
            print(f"  initial z: {self.initial_z} → final z: {self.final_z}")
            print(f"  wall time: {end_time - start_time:.3f} s")

    def _integrate_multi(self, state, z_grid, step_hook=None):
        """Multi-RHS ETD2 loop. ``state`` has shape ``(n, K)``.

        ``step_hook`` is called once per step (after the state advance);
        used by the ``solve(progressbar=...)`` path.
        """
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

    def _etd2_step_multi(self, state, h, d, apply_F, bufs, xp):
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
