"""The module contains classes for computations of interaction rates"""

import numpy as np
from scipy.integrate import trapezoid as trapz

from prince_cr.data import PRINCE_UNITS
from prince_cr.util import info, is_nucleus
import prince_cr.config as config


def _cupy_backend_active(prince_run):
    """Return True if the cupy backend is selected on this run.

    Checked at call time rather than module load — bench / test code
    flips the backend across solves on a single ``PriNCeRun``, so a
    cached flag goes stale. ``config.has_cupy`` gates the result on
    cupy actually being importable in this process.
    """
    return (
        config.has_cupy
        and prince_run.backend.linear_algebra_backend.lower() == "cupy"
    )


def _response_integral_operator(ygr, y_grid):
    """Fixed ``(len(y_grid), len(ygr))`` matrix M such that, for ANY response
    ``f`` sampled on ``ygr``,

        ``M @ f  ==  InterpolatedUnivariateSpline(ygr, f, k=1,
                      ext='zeros').antiderivative()(y_grid)``

    Because the k=1 interpolation + analytic antiderivative + resample is a
    linear operator and ``ygr``/``y_grid`` are shared across all channels, one
    matmul reproduces the per-channel loop bit-for-bit (including the
    ext='zeros' out-of-support extrapolation, which is baked into M). Built
    once per matrix build from the n=len(ygr) basis responses (~ms)."""
    from scipy.interpolate import InterpolatedUnivariateSpline
    n = len(ygr)
    M = np.empty((len(y_grid), n))
    e = np.zeros(n)
    for k in range(n):
        e[k] = 1.0
        M[:, k] = InterpolatedUnivariateSpline(
            ygr, e, k=1, ext="zeros"
        ).antiderivative()(y_grid)
        e[k] = 0.0
    return M


class PhotoNuclearInteractionRate(object):
    """Implementation of photo-hadronic/nuclear interaction rates.
    This Version directly writes the data into a CSC-matrix and only updates the data each time.
    """

    def __init__(self, prince_run=None, with_dense_jac=True, *args, **kwargs):
        info(3, "creating instance")
        self.with_dense_jac = with_dense_jac

        #: Owning PriNCeRun (source of `photon_field`, see property below)
        self.prince_run = prince_run

        #: Reference to CrossSection object
        self.cross_sections = prince_run.cross_sections

        #: Reference to species manager
        self.spec_man = prince_run.spec_man

        # Initialize grids
        self.e_photon = prince_run.ph_grid
        self.e_cosmicray = prince_run.cr_grid

        # Initialize cache of redshift value
        self._ratemat_zcache = None

        # Initialize the matrices for batch computation
        self._batch_rows = None
        self._batch_cols = None
        self._batch_matrix = None
        self._batch_vec = None
        self.coupling_mat = None
        self.dense_coupling_mat = None

        # GPU mirror of the dense rate kernel; populated by
        # :meth:`_ensure_coupling_mat_gpu` on the cupy backend.
        self._batch_matrix_gpu = None

        # xs-dtype mirror of the dense rate kernel for the host
        # (scipy / MKL) backends. Populated lazily by
        # :meth:`_ensure_xs_buffers` once the BackendConfig is visible
        # on the PriNCeRun. When ``backend.xs_dtype`` matches the host
        # solver dtype (fp64), aliases the existing buffers — no extra
        # memory, no per-window cast.
        self._batch_matrix_xs = None
        self._batch_vec_xs = None
        self._photon_buf_xs = None

        self._estimate_batch_matrix()
        self._init_matrices()
        self._init_coupling_mat()

    @property
    def photon_field(self):
        return self.prince_run.photon_field

    def photon_vector(self, z, pfield=None):
        """Returns photon vector at redshift `z` on photon grid.

        This vector is in fact a matrix of vectors of the interpolated
        photon field with dimensions (dim_cr, xi_steps).

        Args:
            z (float): redshift
            pfield: optional photon field override; defaults to ``self.photon_field``.
        """
        pf = pfield if pfield is not None else self.photon_field
        return pf.get_photon_density(self.e_photon.grid, z)

    def _native_em_enabled(self):
        """True when γ/e± daughter rows are built natively on the decoupled
        EM grid (``config.em_native_coupling``) instead of at cr indices +
        runtime regrid R."""
        return bool(
            getattr(self.prince_run, "enable_em_native_coupling", False)
        ) and getattr(self.prince_run, "em_grid", None) is not None

    def _em_native_pdgs(self):
        """PDG ids of state-vector species homed on the EM grid."""
        return frozenset(
            s.pdgid
            for s in self.spec_man.species_refs
            if getattr(s, "grid_tag", "default") == "em"
        )

    def _estimate_batch_matrix(self):
        """estimate dimension of the batch matrix"""
        dcr = self.e_cosmicray.d
        dph = self.e_photon.d

        # O(1) membership lookups; the list-form versions remain in
        # place for any external readers but the inner loop does not
        # touch them. See `wiki/results/prof-init-heavy-mass.md` Rec #2.
        bc_set = self.cross_sections.known_bc_channels_set
        diff_set = self.cross_sections.known_diff_channels_set

        # Native EM-grid coupling: diff channels with a γ/e± daughter write
        # their rows on the (finer) EM grid; the exact post-x-cut row count
        # is cheap to precompute from the two grids, so use it instead of
        # the dcr²/2 bound (the EM block would otherwise be over-allocated
        # by the bins/dec ratio).
        n_em_rows = None
        em_pdgs = ()
        if self._native_em_enabled():
            em_grid = self.prince_run.em_grid
            xl_em = em_grid.bins[:-1][None, :] / self.e_cosmicray.grid[:, None]
            n_em_rows = int(
                np.count_nonzero((xl_em >= config.x_cut) & (xl_em <= 1))
            )
            em_pdgs = self._em_native_pdgs()

        batch_dim = 0
        for specid in self.spec_man.known_species:
            if not is_nucleus(specid):
                continue
            # Add the main diagonal self-couplings (absoption)
            batch_dim += dcr
            for rtup in self.cross_sections.reactions[specid]:
                # Off main diagonal couplings (reinjection)
                if rtup in bc_set:
                    batch_dim += dcr
                elif rtup in diff_set:
                    if n_em_rows is not None and rtup[1] in em_pdgs:
                        batch_dim += n_em_rows
                    else:
                        # Only half of the elements can be non-zero (energy conservation)
                        batch_dim += int(dcr**2 / 2) + 1

        info(2, "Batch matrix dimensions are {0}x{1}".format(batch_dim, dph))
        # CSR build: accumulate only nonzeros as COO (data, batch-row, photon-col)
        # in `_emit_tile`, then assemble one CSR — no dense (batch_dim, dph) buffer.
        self._csr_build = getattr(config, "batch_matrix_format", "csr") == "csr"
        if self._csr_build:
            # Direct CSR assembly: per tile store values, column indices, and
            # per-row nnz counts (→ indptr via cumsum). No COO intermediate.
            self._csr_data, self._csr_indices, self._csr_rownnz = [], [], []
            self._batch_matrix = None
        else:
            self._batch_matrix = np.zeros((batch_dim, dph))
            info(3, "Memory usage: {0} MB".format(self._batch_matrix.nbytes / 1024**2))
        self._batch_rows = []
        self._batch_cols = []
        self._ibatch = 0

    def _emit_tile(self, block, rows, cols):
        """Deposit one channel's tile: `block` is (n_rows × dph), `rows`/`cols`
        are the coupling (daughter, mother) indices for its n_rows batch entries.
        CSR build extracts `block`'s nonzeros as COO (batch-row, photon-col, val)
        — the all-zero rows (mother energies where the response doesn't fire)
        cost nothing; dense build writes into the preallocated buffer."""
        n = block.shape[0]
        if self._csr_build:
            # np.nonzero is C-order → rows grouped, columns sorted within each
            # row: already CSR-canonical. Store values + column indices, and the
            # per-row nnz counts (incl. zeros for all-zero rows) so indptr is a
            # cumsum at the end — the finished (data, indices, indptr) arrays go
            # straight into csr_matrix with no COO copy.
            ii, jj = np.nonzero(block)
            if ii.size:
                self._csr_data.append(np.asarray(block[ii, jj], dtype=np.float32))
                self._csr_indices.append(jj.astype(np.int32))
            self._csr_rownnz.append(np.bincount(ii, minlength=n).astype(np.int32))
        else:
            self._batch_matrix[self._ibatch:self._ibatch + n, :] = block
        self._ibatch += n
        self._batch_rows.append(rows)
        self._batch_cols.append(cols)

    def _assert_log_grids_compatible(self):
        """Verify that CR-grid centers, CR-grid bin edges, and photon-grid bin
        edges all share the same log-step. The kernel construction relies on
        this assumption; mismatched grids would silently produce wrong rates.
        Raises RuntimeError on mismatch.
        """
        ecr = self.e_cosmicray.grid
        bcr = self.e_cosmicray.bins
        bph = self.e_photon.bins
        if ecr.size < 2 or bcr.size < 2 or bph.size < 2:
            raise RuntimeError(
                "Kernel construction requires at least 2 grid points in each axis "
                f"(got dcr={ecr.size}, bcr={bcr.size}, bph={bph.size})."
            )
        s_ecr = np.log10(ecr[1] / ecr[0])
        s_bcr = np.log10(bcr[1] / bcr[0])
        s_bph = np.log10(bph[1] / bph[0])
        if not (np.isclose(s_ecr, s_bcr) and np.isclose(s_ecr, s_bph)):
            cr_bpd = self.e_cosmicray.d / np.log10(
                self.e_cosmicray.bins[-1] / self.e_cosmicray.bins[0]
            )
            ph_bpd = self.e_photon.d / np.log10(
                self.e_photon.bins[-1] / self.e_photon.bins[0]
            )
            raise RuntimeError(
                "Kernel construction requires the cosmic-ray and photon grids "
                "to share the same bins-per-decade (log-step). "
                f"Got cr~{cr_bpd:.3f} bins/decade, ph~{ph_bpd:.3f} bins/decade "
                f"(log-steps: cr_centers={s_ecr:.6f}, cr_bins={s_bcr:.6f}, "
                f"ph_bins={s_bph:.6f}). "
                "Align config.cosmic_ray_grid and config.photon_grid on the "
                "same bins/decade."
            )

    def _init_matrices(self):
        """Toeplitz/log-grid kernel construction.

        For each channel `(mo, da)`:
          - bc:    B[i_mo, j] = (m_p/E_mo[i_mo]) · ΔR̂[i_mo + j]
          - diff:  B[i_mo, i_da, j] = (m_p/E_mo[i_mo]) · (Δec[i_mo]/Δec[i_da])
                                      · ΔΔR̂[i_mo + j, i_da - i_mo + (dcr - 1)]

        Each response spline is sampled once: 1D antiderivatives on a length
        (dcr + dph) log-y grid; 2D antiderivatives on a (dcr + dph) × (2 dcr)
        outer-product grid. The final dense tile is built by integer-index
        broadcasting and stuffed into `_batch_matrix`. Downstream
        (`_init_coupling_mat`) lex-sorts the rows/cols.

        Requires the CR and photon grids to share the same bins/decade
        (`_assert_log_grids_compatible`).
        """
        self._assert_log_grids_compatible()

        spec_man = self.spec_man
        sp_id_ref = spec_man.pdgid2sref
        resp = self.cross_sections.resp
        m_pr = PRINCE_UNITS.m_proton

        ecr = self.e_cosmicray.grid
        bcr = self.e_cosmicray.bins
        bph = self.e_photon.bins
        dcr = self.e_cosmicray.d
        dph = self.e_photon.d
        delta_ec = self.e_cosmicray.widths

        # ----- log-grid construction -----
        log_step = np.log10(ecr[1] / ecr[0])
        # 1D y-grid: y[k] = ecr[0] * bph[0] / m_p · 10^(k δ), k = 0..dcr+dph-1
        y0 = ecr[0] * bph[0] / m_pr
        y_grid = y0 * 10 ** (np.arange(dcr + dph) * log_step)
        # 1D x-grid for diff channels: covers (i_da - i_mo) ∈ [-(dcr-1), dcr-1]
        # plus +1 offset for x_u, so length 2*dcr. Index offset is (dcr-1).
        x0 = bcr[0] / ecr[0]
        x_grid = x0 * 10 ** ((np.arange(2 * dcr) - (dcr - 1)) * log_step)

        # ----- prefactors -----
        factor_mo = m_pr / ecr  # shape (dcr,) — used by bc and diagonal nonel
        # factor_diff[i_mo, i_da] = (m_p/ecr[i_mo]) · (delta_ec[i_mo]/delta_ec[i_da])
        factor_diff = factor_mo[:, None] * (delta_ec[:, None] / delta_ec[None, :])

        # ----- index arrays for tile assembly -----
        i_idx = np.arange(dcr)
        j_idx = np.arange(dph)
        # A[i_mo, j] = i_mo + j  ∈ [0, dcr+dph-2]   (index into ΔR̂ / a-axis of ΔΔR̂)
        A_2d = i_idx[:, None] + j_idx[None, :]
        # B_idx[i_mo, i_da] = (i_da - i_mo) + (dcr - 1)  ∈ [0, 2 dcr - 2]
        B_idx = (i_idx[None, :] - i_idx[:, None]) + (dcr - 1)

        # ----- pre-sample 1D antiderivatives once -----
        # Fast path: the per-channel "k=1 interp -> antiderivative -> sample on
        # y_grid" is a fixed linear operator M (shared ygrid/y_grid), so
        # ΔR̂ = diff(f_stack @ Mᵀ) reproduces the loop bit-for-bit in one matmul.
        # `response_ygrid`/`*_f_stack` are built in ResponseFunction.
        fast = getattr(config, "fast_response_build", True) and hasattr(resp, "response_ygrid")
        if fast:
            M = _response_integral_operator(resp.response_ygrid, y_grid)
            nonel_dR, incl_dR = {}, {}
            if resp.nonel_f_stack.size:
                dRn = np.diff(resp.nonel_f_stack @ M.T, axis=1)
                nonel_dR = {mo: dRn[i] for i, mo in enumerate(resp.nonel_keys)}
            if resp.incl_f_stack.size:
                dRi = np.diff(resp.incl_f_stack @ M.T, axis=1)
                incl_dR = {key: dRi[i] for i, key in enumerate(resp.incl_keys)}
        else:
            # Reference per-channel path (config.fast_response_build = False).
            nonel_dR = {}
            for mo, intp in resp.nonel_intp.items():
                nonel_dR[mo] = np.diff(intp.antiderivative()(y_grid))
            incl_dR = {}
            for key, intp in resp.incl_intp.items():
                incl_dR[key] = np.diff(intp.antiderivative()(y_grid))

        # ----- pre-sample 2D antiderivatives once -----
        # diff: dict[(mo,da)] → ΔΔR̂ of shape (dcr+dph-1, 2*dcr-1)
        # Rec #3: batch the per-channel `intp2d.ev(...)` — at m=245
        # this loop fired 2600 spline evals on the same fixed grid.
        # `ResponseFunction` precomputes a stacked
        # `BilinearGrid2D((nx, ny, n_ch))` for shared-grid channels;
        # one batched call replaces the per-channel loop. Falls back
        # to the per-channel path if the shared-grid invariant
        # somehow doesn't hold for this DB.
        diff_ddR = {}
        if resp.incl_diff_intp_integral:
            Y2d, X2d = np.meshgrid(y_grid, x_grid, indexing="ij")
            Yflat, Xflat = Y2d.ravel(), X2d.ravel()
            shape2d = Y2d.shape
            stack_intp = getattr(resp, "incl_diff_integral_stack_intp", None)
            stack_keys = getattr(resp, "incl_diff_integral_keys", None)
            if stack_intp is not None and stack_keys:
                # Batched path: one bilinear lookup, shape (n_pts, n_ch)
                # then reshape to (*shape2d, n_ch). The slot-3 axis is
                # the channel index from `stack_keys`.
                R_stack = stack_intp.ev(Xflat, Yflat).reshape(*shape2d, len(stack_keys))
                ddR_stack = (
                    R_stack[1:, 1:, :]
                    - R_stack[1:, :-1, :]
                    - R_stack[:-1, 1:, :]
                    + R_stack[:-1, :-1, :]
                )
                for i, key in enumerate(stack_keys):
                    diff_ddR[key] = ddR_stack[..., i]
                # Any channels left uncovered by the stack (shouldn't
                # happen in practice — every channel shares the FLUKA
                # ygrid — but the fallback keeps us correct). Use the
                # per-channel `.ev` for those keys only.
                if len(stack_keys) != len(resp.incl_diff_intp_integral):
                    covered = set(stack_keys)
                    for key, intp2d in resp.incl_diff_intp_integral.items():
                        if key in covered:
                            continue
                        R = intp2d.ev(Xflat, Yflat).reshape(shape2d)
                        diff_ddR[key] = (
                            R[1:, 1:] - R[1:, :-1] - R[:-1, 1:] + R[:-1, :-1]
                        )
            else:
                # No stacked tensor (shared-grid invariant violated, or
                # an older ResponseFunction without the stack attr).
                for key, intp2d in resp.incl_diff_intp_integral.items():
                    R = intp2d.ev(Xflat, Yflat).reshape(shape2d)
                    ddR = R[1:, 1:] - R[1:, :-1] - R[:-1, 1:] + R[:-1, :-1]
                    diff_ddR[key] = ddR

        # ----- iterate channels and fill _batch_matrix -----
        x_cut = config.x_cut
        x_cut_proton = config.x_cut_proton

        # x_l(i_mo, i_da) = bcr[i_da] / ecr[i_mo] depends on (i_da - i_mo) only;
        # precompute a (dcr, dcr) array once.
        xl_2d = bcr[:-1][None, :] / ecr[:, None]  # shape (dcr, dcr)

        # ----- native EM-grid daughter construction -----
        # Channels whose daughter is homed on the decoupled EM grid get their
        # rows built directly at EM-grid energies. The Toeplitz log-shift
        # generalizes to two grids when the EM log-step divides the cr one:
        # with r = δ_cr/δ_em (integer), x_l(i_mo, i_da) = bem[i_da]/ecr[i_mo]
        # = x0_em · 10^(δ_em·(i_da − r·i_mo)), so the ΔΔR̂ x-axis is sampled
        # once at δ_em steps and indexed by (i_da − r·i_mo). The y/photon axis
        # is mother-side and unchanged. See methods/em-grid-boost-tier3-plan.
        native_em = self._native_em_enabled()
        em_native_pdgs = self._em_native_pdgs() if native_em else frozenset()
        if native_em:
            em_grid = self.prince_run.em_grid
            bem = em_grid.bins
            dem = em_grid.d
            delta_em = em_grid.widths
            log_step_em = np.log10(em_grid.grid[1] / em_grid.grid[0])
            r_ratio = int(round(log_step / log_step_em))
            if not np.isclose(log_step, r_ratio * log_step_em, rtol=1e-12):
                raise RuntimeError(
                    "Native EM coupling requires the cr log-step to be an "
                    "integer multiple of the EM log-step (got cr={0:.8g}, "
                    "em={1:.8g}). Snap the EM grid (core.py) / align "
                    "em_grid_bins_dec.".format(log_step, log_step_em)
                )
            k0_em = r_ratio * (dcr - 1)
            # x edges at δ_em steps covering i_da − r·i_mo ∈ [−k0_em, dem−1],
            # +1 trailing edge for the upper bin bound.
            x0_em = bem[0] / ecr[0]
            x_grid_em = x0_em * 10 ** (
                (np.arange(dem + k0_em + 1) - k0_em) * log_step_em
            )
            factor_diff_em = factor_mo[:, None] * (
                delta_ec[:, None] / delta_em[None, :]
            )
            i_em_idx = np.arange(dem)
            B_idx_em = (i_em_idx[None, :] - r_ratio * i_idx[:, None]) + k0_em
            xl_2d_em = bem[:-1][None, :] / ecr[:, None]  # (dcr, dem)
            cuts2d_em = np.logical_and(
                xl_2d_em >= config.x_cut, xl_2d_em <= 1
            )
            imo_grid_em, ida_grid_em = np.meshgrid(
                np.arange(dcr), i_em_idx, indexing="ij"
            )
            rows_em_cut = ida_grid_em[cuts2d_em]
            cols_em_cut = imo_grid_em[cuts2d_em]
            # Sample points for the per-channel 2D antiderivative on the EM
            # x-grid (same y/mother grid as the shared path).
            Y2d_em, X2d_em = np.meshgrid(y_grid, x_grid_em, indexing="ij")
            Yflat_em, Xflat_em = Y2d_em.ravel(), X2d_em.ravel()
            shape2d_em = Y2d_em.shape

        # batch-row cursor lives on self (_emit_tile increments self._ibatch)
        emo_idcs = np.arange(dcr)
        eda_idcs = np.arange(dcr)
        # i_mo grid for diagonal-nonel handling in diff channels
        diag = np.arange(dcr)

        # Iteration order is irrelevant — `_init_coupling_mat` lex-sorts.
        known_species_rev = spec_man.known_species[::-1]
        import itertools

        # See `_estimate_batch_matrix` for the rationale behind the
        # set companions; same hot-loop justification.
        bc_set = self.cross_sections.known_bc_channels_set
        diff_set = self.cross_sections.known_diff_channels_set

        # Tracked-species e_gamma masks. Built once per kernel construction
        # and applied to the corresponding ``(mo, tracked_pdg)`` tile before
        # it lands in ``_batch_matrix``. The mask zeros out photon-energy
        # bins outside the species' ``e_gamma_range`` so only in-window
        # photon kinematics contribute to the tracked-species rows.
        # ``None`` value means no filtering. See
        # methods/tracking-species-design.md § "Energy-range filtering".
        ph_centers = self.e_photon.grid
        _tracked_pdg_mask = {}
        for s in spec_man.species_refs:
            if not getattr(s, "is_tracking", False):
                continue
            if s.e_gamma_range is None:
                _tracked_pdg_mask[s.pdgid] = None
            else:
                E_lo, E_hi = s.e_gamma_range
                _tracked_pdg_mask[s.pdgid] = (
                    (ph_centers >= E_lo) & (ph_centers < E_hi)
                )

        for moid, daid in itertools.product(known_species_rev, known_species_rev):
            if not is_nucleus(moid):
                continue

            # Diagonal absorption term only fires when the mother actually has
            # a non-elastic cross section. Default chain-reducer mode lands
            # here with the invariant ``(moid == daid) ⇒ moid in nonel_dR``;
            # explicit-decay mode can land us with daughter-only nuclei (e.g.
            # free n from He-4 photo-disintegration; neutron has no
            # photo-nuclear of its own) where the (n, n) pair has neither
            # in_bc/in_diff membership nor a nonel entry — so we skip below.
            has_nonel = (moid == daid) and (moid in nonel_dR)
            in_bc = (moid, daid) in bc_set
            in_diff = (moid, daid) in diff_set

            if in_bc or (has_nonel and not in_diff):
                # ---- bc channel branch ----
                has_incl = (moid, daid) in incl_dR
                if not (has_nonel or has_incl):
                    raise Exception("Channel without interactions:", (moid, daid))

                # Tile = factor_mo[i_mo] · ΔR̂[A_2d[i_mo, j]]
                tile = np.zeros((dcr, dph))
                if has_incl:
                    tile += factor_mo[:, None] * incl_dR[(moid, daid)][A_2d]
                if has_nonel:
                    tile -= factor_mo[:, None] * nonel_dR[moid][A_2d]
                # Energy-range filter for tracked species: zero out photon-
                # energy bins outside ``trk.e_gamma_range``. No-op for real
                # species and for tracked species with no window set.
                _msk = _tracked_pdg_mask.get(daid) if _tracked_pdg_mask else None
                if _msk is not None:
                    tile[:, ~_msk] = 0.0

                self._emit_tile(tile,
                                sp_id_ref[daid].lidx() + eda_idcs,
                                sp_id_ref[moid].lidx() + emo_idcs)

            elif in_diff and daid in em_native_pdgs:
                # ---- diff channel, EM-grid daughter (native coupling) ----
                # Same construction as the cr-grid diff branch below, with the
                # daughter axis on the EM grid: per-channel ΔΔR̂ sampled on the
                # δ_em x-grid, Toeplitz index (i_da − r·i_mo), and the bin-width
                # ratio against the EM daughter bins. No nonel diagonal here —
                # mothers are nuclei, the daughter is γ/e±.
                intp2d = resp.incl_diff_intp_integral.get((moid, daid))
                if intp2d is None:
                    raise Exception(
                        "incl_diff_intp_integral missing for", (moid, daid)
                    )
                R_em = intp2d.ev(Xflat_em, Yflat_em).reshape(shape2d_em)
                ddR_em = (
                    R_em[1:, 1:] - R_em[1:, :-1] - R_em[:-1, 1:] + R_em[:-1, :-1]
                )
                res = factor_diff_em[:, :, None] * ddR_em[
                    A_2d[:, None, :],      # (dcr, 1, dph) — i_mo + j
                    B_idx_em[:, :, None],  # (dcr, dem, 1) — i_da − r·i_mo + k0
                ]
                np.clip(res, 0.0, None, out=res)

                _msk = _tracked_pdg_mask.get(daid) if _tracked_pdg_mask else None
                if _msk is not None:
                    res[..., ~_msk] = 0.0

                kept = res[cuts2d_em]  # (n_kept, dph)
                n_kept = kept.shape[0]
                if n_kept == 0:
                    continue

                rows = sp_id_ref[daid].lidx() + rows_em_cut
                cols = sp_id_ref[moid].lidx() + cols_em_cut

                self._emit_tile(kept, rows, cols)

            elif in_diff:
                # ---- diff channel branch ----
                if (moid, daid) not in diff_ddR:
                    raise Exception("incl_diff_intp_integral missing for", (moid, daid))

                ddR = diff_ddR[(moid, daid)]  # shape (dcr+dph-1, 2*dcr-1)

                # Compute the (dcr, dcr, dph) tile via fancy indexing.
                # res[i_mo, i_da, j] = factor_diff[i_mo, i_da] · ddR[A_2d[i_mo,j], B_idx[i_mo, i_da]]
                res = factor_diff[:, :, None] * ddR[
                    A_2d[:, None, :],     # (dcr, 1, dph) — i_mo + j
                    B_idx[:, :, None],    # (dcr, dcr, 1) — i_da - i_mo + dcr-1
                ]

                # Clip negatives (spline-induced) BEFORE the nonel subtraction —
                # otherwise we'd erase the negative diagonal contribution.
                np.clip(res, 0.0, None, out=res)

                # Diagonal nonel subtraction (only when mo==da)
                if has_nonel:
                    nonel_tile = factor_mo[:, None] * nonel_dR[moid][A_2d]
                    # Subtract on the diagonal i_mo == i_da
                    res[diag, diag, :] -= nonel_tile

                # Energy-range filter on the photon-energy axis for tracked
                # species. Applied here so the cut still lands ahead of the
                # x-cut filter / extraction below — masked bins drop out of
                # both the kept-rows count and the kernel values.
                _msk = _tracked_pdg_mask.get(daid) if _tracked_pdg_mask else None
                if _msk is not None:
                    res[..., ~_msk] = 0.0

                # x-cut filter (depends only on (i_mo, i_da))
                cut_low = x_cut_proton if daid == 2212 else x_cut
                cuts2d = np.logical_and(xl_2d >= cut_low, xl_2d <= 1)

                # Extract kept rows and the corresponding (row, col) indices
                kept = res[cuts2d]  # shape (n_kept, dph)
                n_kept = kept.shape[0]
                if n_kept == 0:
                    continue

                # Row index = sp_id_ref[daid].lidx() + i_da; column = sp_id_ref[moid].lidx() + i_mo
                # Build the same (i_mo, i_da) grid used for the cut.
                imo_grid, ida_grid = np.meshgrid(emo_idcs, eda_idcs, indexing="ij")
                rows = sp_id_ref[daid].lidx() + ida_grid[cuts2d]
                cols = sp_id_ref[moid].lidx() + imo_grid[cuts2d]

                self._emit_tile(kept, rows, cols)

            else:
                info(20, "Species combination not included in model", moid, daid)

        ibatch = self._ibatch
        if self._csr_build:
            from scipy.sparse import csr_matrix
            data = (np.concatenate(self._csr_data) if self._csr_data
                    else np.zeros(0, np.float32))
            indices = (np.concatenate(self._csr_indices) if self._csr_indices
                       else np.zeros(0, np.int32))
            counts = (np.concatenate(self._csr_rownnz) if self._csr_rownnz
                      else np.zeros(ibatch, np.int32))
            self._csr_data = self._csr_indices = self._csr_rownnz = None
            # indptr = [0, cumsum(per-row nnz)]; int64 only if nnz overflows int32.
            idt = np.int64 if data.size >= 2**31 else np.int32
            if idt == np.int64:
                indices = indices.astype(np.int64)
            indptr = np.empty(ibatch + 1, dtype=idt)
            indptr[0] = 0
            np.cumsum(counts, dtype=idt, out=indptr[1:])
            del counts
            # Direct CSR: the arrays are already row-grouped + column-sorted
            # (np.nonzero C-order) with no duplicates, so this wraps them with
            # NO COO intermediate and no sort.
            self._batch_matrix = csr_matrix(
                (data, indices, indptr), shape=(ibatch, dph)
            )
            self._batch_matrix.has_canonical_format = True
            del data, indices
            mat_bytes = (self._batch_matrix.data.nbytes
                         + self._batch_matrix.indices.nbytes
                         + self._batch_matrix.indptr.nbytes)
        else:
            self._batch_matrix = self._batch_matrix[:ibatch, :]
            mat_bytes = self._batch_matrix.nbytes
        self._batch_rows = np.concatenate(self._batch_rows, axis=None)
        self._batch_cols = np.concatenate(self._batch_cols, axis=None)
        self._batch_vec = np.zeros(ibatch)

        info(2, f"Batch matrix shape: {self._batch_matrix.shape}"
                f"{' (CSR nnz=' + str(self._batch_matrix.nnz) + ')' if self._csr_build else ''}")
        info(2, f"Batch rows shape: {self._batch_rows.shape}")

        memory = (
            mat_bytes
            + self._batch_rows.nbytes
            + self._batch_cols.nbytes
            + self._batch_vec.nbytes
        ) / 1024**2
        info(3, "Memory usage after initialization: {:} MB".format(memory))

    def _init_coupling_mat(self):
        """Initialises the coupling matrix directly in sparse (csr) format.

        Always builds a host scipy CSR — the full-GPU cupy backend
        lazily mirrors it to GPU on first use via
        :meth:`_ensure_coupling_mat_gpu`. Decoupling the GPU upload
        from PriNCeRun construction lets bench / test code flip
        ``config.linear_algebra_backend`` between solves on a single
        run instance.
        """
        info(0, "Initiating coupling matrix in ({:}) format".format("CSR"))

        from scipy.sparse import csr_matrix

        # Pin the CSR shape to the full state-vector dimension so
        # daughter-only species (e.g. tracked rows whose mother column
        # never appears) don't shrink the column axis below
        # ``dim_states``. Without this, ``split_operator`` later sees a
        # rectangular matrix and raises on the diagonal subtraction.
        n = self.prince_run.dim_states
        self.coupling_mat = csr_matrix(
            (self._batch_vec, (self._batch_rows, self._batch_cols)),
            shape=(n, n),
            copy=True,
        )

        # create an index to sort by rows and then columns,
        # which is the same ordering CSR has internally
        # lexsort sorts by last argument first!!!
        self.sortidx = np.lexsort((self._batch_cols, self._batch_rows))

        self._batch_rows = self._batch_rows[self.sortidx]
        self._batch_cols = self._batch_cols[self.sortidx]
        self._batch_matrix = self._batch_matrix[self.sortidx, :]

        # Invalidate any xs-dtype mirror that was built before the sort
        # permutation (none should exist at this point, but guard anyway
        # in case the order ever changes).
        self._batch_matrix_xs = None
        self._batch_vec_xs = None
        self._photon_buf_xs = None

        # cupy mirror state (full-GPU backend). ``_coupling_mat_gpu``
        # shares its ``data`` buffer with ``_batch_vec_gpu`` when their
        # dtypes match, so the cupy.dot result inhabits the CSR data
        # slot directly — no scipy ↔ cupy round-trip per cache window.
        self._batch_matrix_gpu = None
        self._batch_vec_gpu = None
        self._coupling_mat_gpu = None
        self._ratemat_zcache_gpu = None

    # ------------------------------------------------------------------
    # Host xs-dtype buffers (scipy / MKL backends)
    # ------------------------------------------------------------------
    def _ensure_xs_buffers(self):
        """Prepare the xs-dtype mirror of ``_batch_matrix`` plus matching
        host scratch buffers, used by the scipy / MKL ``_update_rates``
        path to run the cache-rebuild dense matvec at ``backend.xs_dtype``.

        When ``backend.xs_dtype`` is ``"float64"`` or ``None``, the
        method aliases the existing fp64 buffers — no extra memory and
        no per-window cast (zero-overhead default for parity bench /
        legacy callers that explicitly request fp64).

        When ``backend.xs_dtype`` differs from the host solver dtype
        (the common case: fp32 SGEMV against fp64 solver state), a
        fp32 cast of ``_batch_matrix`` is allocated alongside a fp32
        ``_batch_vec_xs`` (size = ``nnz`` of the coupling matrix) and a
        fp32 ``_photon_buf_xs`` (size = ``dph``). The matvec writes
        into ``_batch_vec_xs`` in fp32; the upcast to the fp64 CSR
        data buffer happens in ``_update_coupling_mat``.

        Idempotent: re-runs only if no mirror exists yet or if a
        mid-process ``backend.xs_dtype`` flip changed the requested
        precision.
        """
        if getattr(self, "_csr_build", False):
            # CSR path folds the sparse kernel directly (already at build dtype);
            # no dense mirror. _batch_vec_xs is (re)bound by the SpMV fold.
            return
        xs_attr = self.prince_run.backend.xs_dtype
        dt_solver = np.dtype("float64")
        dt_xs = np.dtype(xs_attr) if xs_attr is not None else dt_solver
        if (
            self._batch_matrix_xs is not None
            and self._batch_matrix_xs.dtype == dt_xs
        ):
            return
        if dt_xs == dt_solver:
            self._batch_matrix_xs = self._batch_matrix
            self._batch_vec_xs = self._batch_vec
            self._photon_buf_xs = None
        else:
            self._batch_matrix_xs = np.ascontiguousarray(
                self._batch_matrix, dtype=dt_xs
            )
            self._batch_vec_xs = np.zeros(self._batch_vec.size, dtype=dt_xs)
            self._photon_buf_xs = np.zeros(
                self._batch_matrix.shape[1], dtype=dt_xs
            )

    # ------------------------------------------------------------------
    # MKL Sparse BLAS fold (host, CSR rate kernel)
    # ------------------------------------------------------------------
    def _ensure_batch_matrix_mkl(self):
        """Lazily build an optimized MKL Sparse handle for the CSR fold.

        The response kernel ``_batch_matrix`` (batch_dim × dph) is CONSTANT
        for the whole solve — only the photon vector changes — so the
        ``mkl_sparse_optimize`` path applies (unlike ``M_off``, which is
        refreshed in place and must stay un-optimized). Built at most once
        per run; ctypes handle, so it is NOT pickled — rebuilt on demand
        after a cache load. Threads are governed by the caller's
        ``config.set_thread_count`` / ``set_mkl_threads``.
        """
        if getattr(self, "_batch_matrix_mkl", None) is not None:
            return
        from ctypes import POINTER, c_double, c_float
        from .mkl_sparse import MklSparseMatrix

        A = self._batch_matrix                       # fp32 (default) CSR
        self._batch_matrix_mkl = MklSparseMatrix(A, optimize=True)
        n_rows, n_cols = A.shape
        fl = c_float if A.dtype == np.float32 else c_double
        self._fold_x_mkl = np.zeros(n_cols, dtype=A.dtype)   # photon (dph,)
        self._fold_y_mkl = np.zeros(n_rows, dtype=A.dtype)   # batch_vec (batch_dim,)
        self._fold_x_p = self._fold_x_mkl.ctypes.data_as(POINTER(fl))
        self._fold_y_p = self._fold_y_mkl.ctypes.data_as(POINTER(fl))

    # ------------------------------------------------------------------
    # Direct fold into the ETD2 split buffers (host, CSR rate kernel)
    # ------------------------------------------------------------------
    def build_split_fold(self, off_to_M, diag_to_M):
        """Build response sub-kernels that fold DIRECTLY into ``M_off.data`` and
        the diagonal, so the ETD2 refresh skips the full coupling.data fold + the
        per-window off/diag gather.

        ``M.data[k] == (_batch_matrix @ photon)[k]`` (the fold output, in coupling
        order — the sortidx applied in ``_init_coupling_mat``). ``off_to_M[j]`` is
        the ``M.data`` index feeding ``M_off.data[j]``, so
        ``M_off.data = _batch_matrix[off_to_M] @ photon`` — a row-subset fold,
        bit-identical to fold-then-gather. Same for the diagonal via ``diag_to_M``.

        One-time; caches the row-subset kernels on this instance (fixed for the
        run) and reuses them across solves. Costs ~1× the ``_batch_matrix`` memory
        for the off-diagonal subset.
        """
        bm = self._batch_matrix
        # Backend-independent row-subset kernels: build once.
        if getattr(self, "_sf_off_kernel", None) is None:
            self._sf_off_kernel = bm[np.asarray(off_to_M, dtype=np.int64)]
            present = np.asarray(diag_to_M) >= 0
            self._sf_diag_present = present
            self._sf_diag_kernel = bm[np.asarray(diag_to_M, dtype=np.int64)[present]]
            self._sf_off_mkl = None
        # MKL fold handle: (re)build iff the MKL backend is active and it is
        # absent — so flipping backends on one PriNCeRun rebuilds it rather than
        # silently reusing a scipy-only build (or vice versa).
        want_mkl = (
            self.prince_run.backend.linear_algebra_backend.lower() == "mkl"
            and config.has_mkl and config.mkl is not None
            and self._sf_off_kernel.nnz
        )
        if want_mkl and getattr(self, "_sf_off_mkl", None) is None:
            from ctypes import POINTER, c_double, c_float
            from .mkl_sparse import MklSparseMatrix
            self._sf_off_mkl = MklSparseMatrix(self._sf_off_kernel, optimize=True)
            dph = bm.shape[1]
            noff = self._sf_off_kernel.shape[0]
            fl = c_float if bm.dtype == np.float32 else c_double
            self._sf_ph = np.zeros(dph, dtype=bm.dtype)
            self._sf_off_y = np.zeros(noff, dtype=bm.dtype)
            self._sf_ph_p = self._sf_ph.ctypes.data_as(POINTER(fl))
            self._sf_off_y_p = self._sf_off_y.ctypes.data_as(POINTER(fl))
        self._split_fold_ready = True

    def fold_split(self, z, moff_data_out, dmg_out):
        """Fold the response kernel directly into ``moff_data_out`` (= M_off.data,
        fp64) and ``dmg_out`` (= diagonal vector, fp64) at redshift ``z``.

        Replaces the ETD2 refresh's [full fold → coupling.data cast → off/diag
        gather] with two row-subset SpMVs (off + diag) written straight into the
        split buffers. scale_fac is 1.0 (the refresh's convention; dldz is applied
        later per step). Bit-identical to the gather path modulo SpMV summation
        order (0 on scipy, ε on MKL)."""
        photon = self.photon_vector(z)
        ph = np.asarray(photon, dtype=self._batch_matrix.dtype)
        use_mkl = (
            getattr(self, "_sf_off_mkl", None) is not None
            and self.prince_run.backend.linear_algebra_backend.lower() == "mkl"
        )
        if use_mkl:
            np.copyto(self._sf_ph, ph)
            self._sf_off_mkl.gemv_ctargs(1.0, self._sf_ph_p, 0.0, self._sf_off_y_p)
            np.copyto(moff_data_out, self._sf_off_y)   # fp32 -> fp64
        elif moff_data_out.size:
            np.copyto(moff_data_out, self._sf_off_kernel.dot(ph))
        dmg_out.fill(0.0)
        if self._sf_diag_kernel.shape[0]:
            dmg_out[self._sf_diag_present] = self._sf_diag_kernel.dot(ph)

    # ------------------------------------------------------------------
    # Full-GPU cupy backend (GPU dense matvec + GPU SpMV)
    # ------------------------------------------------------------------
    def _ensure_coupling_mat_gpu(self):
        """Lazily upload ``_batch_matrix`` and the coupling sparsity to GPU.

        Runs at most once per PriNCeRun — subsequent calls are no-ops.
        Builds:

        * ``_batch_matrix_gpu``:  cupy mirror of the dense rate kernel
          (~80 MB at ``max_mass=56``), in ``backend.xs_dtype`` (fp32
          for the SGEMM win, fp64 for parity with scipy).
        * ``_batch_vec_gpu``:  cupy ndarray sized to the host coupling
          matrix's ``nnz``, in ``backend.xs_dtype``. Receives the
          result of each cache-window dense matvec.
        * ``_coupling_mat_gpu``:  cupyx.scipy.sparse.csr_matrix in
          ``cupy_dtype`` (the *solver* dtype). Its ``.data`` is what
          the ETD2 hot path SpMVs against, so per-step arithmetic stays
          at solver precision regardless of the SGEMM dtype.

        When ``backend.xs_dtype is None`` or matches ``cupy_dtype``,
        ``_batch_vec_gpu`` is aliased into ``_coupling_mat_gpu.data``
        (no per-window cast). When they differ, ``_update_rates_gpu``
        casts ``_batch_vec_gpu`` → ``_coupling_mat_gpu.data`` once per
        cache window — that cast is the only mixed-precision boundary
        in the pipeline.
        """
        dt_solver = np.dtype(self.prince_run.backend.cupy_dtype)
        xs_attr = self.prince_run.backend.xs_dtype
        dt_xs = np.dtype(xs_attr) if xs_attr is not None else dt_solver
        if (
            self._coupling_mat_gpu is not None
            and self._coupling_mat_gpu.dtype == dt_solver
            and self._batch_matrix_gpu is not None
            and self._batch_matrix_gpu.dtype == dt_xs
        ):
            return
        # Dtype switch (bench flipping fp32 ↔ fp64, or toggling the
        # mixed-precision pipeline mid-process): drop the stale mirror
        # so it rebuilds in the new dtype pair.
        self._coupling_mat_gpu = None
        self._batch_matrix_gpu = None
        self._batch_vec_gpu = None
        if not config.has_cupy:
            raise RuntimeError(
                "PhotoNuclearInteractionRate: cupy backend requested but "
                "cupy is not importable."
            )
        import cupy as cp
        import cupyx.scipy.sparse as csp

        if self._csr_build:
            # cupy CSR mirror of the sparse kernel; the fold is a cuSPARSE SpMV.
            self._batch_matrix_gpu = csp.csr_matrix(
                self._batch_matrix.astype(dt_xs, copy=False)
            )
        else:
            self._batch_matrix_gpu = cp.asarray(self._batch_matrix, dtype=dt_xs)

        host_mat = self.coupling_mat
        nnz = host_mat.data.size
        self._batch_vec_gpu = cp.zeros(nnz, dtype=dt_xs)
        # When dtypes match, alias data ≡ _batch_vec_gpu (no per-window
        # cast). When they differ, allocate a separate solver-dtype
        # buffer; the cast happens in ``_update_rates_gpu``.
        if dt_xs == dt_solver:
            coupling_data = self._batch_vec_gpu
        else:
            coupling_data = cp.zeros(nnz, dtype=dt_solver)
        self._coupling_mat_gpu = csp.csr_matrix(
            (
                coupling_data,
                cp.asarray(host_mat.indices, dtype=cp.int32),
                cp.asarray(host_mat.indptr, dtype=cp.int32),
            ),
            shape=host_mat.shape,
        )
        # Reset cache so the next ``_update_rates`` recomputes against
        # this freshly-allocated GPU buffer.
        self._ratemat_zcache_gpu = None

    def _update_rates_gpu(self, z, pfield=None):
        """GPU dense matvec → ``_batch_vec_gpu`` (xs dtype), then upcast
        (if needed) into ``_coupling_mat_gpu.data`` (solver dtype).

        Mirrors the host ``_update_rates`` semantics: returns True if a
        recompute happened, False if the (z, pfield) cache hit. When
        ``backend.xs_dtype == cupy_dtype`` (or unset), ``_batch_vec_gpu``
        IS the coupling-matrix data buffer (no cast). When they differ
        — the mixed-precision pipeline — the cast bridges fp32 SGEMM
        output to the fp64 ETD2 hot path.
        """
        import cupy as cp

        is_override = pfield is not None
        if not is_override and self._ratemat_zcache_gpu == z:
            return False
        info(5, "Updating batch rate vectors (GPU).")
        photon_host = self.photon_vector(z, pfield=pfield)
        photon_gpu = cp.asarray(photon_host, dtype=self._batch_matrix_gpu.dtype)
        if self._csr_build:
            # cuSPARSE SpMV; write into the pinned _batch_vec_gpu buffer so the
            # (aliased or cast) coupling-data path below is unchanged.
            self._batch_vec_gpu[:] = self._batch_matrix_gpu.dot(photon_gpu)
        else:
            cp.dot(self._batch_matrix_gpu, photon_gpu, out=self._batch_vec_gpu)
        # Mixed-precision: bridge xs-dtype matvec result into the
        # solver-dtype coupling-matrix data buffer. ``cp.copyto`` casts
        # element-wise on the device. No-op when buffers are aliased
        # (single-precision pipeline).
        if self._coupling_mat_gpu.data is not self._batch_vec_gpu:
            cp.copyto(self._coupling_mat_gpu.data, self._batch_vec_gpu)
        # Override path: don't latch the cache, so the next normal call
        # recomputes against the default photon field.
        self._ratemat_zcache_gpu = None if is_override else z
        return True

    def _update_rates(self, z, force_update=False, pfield=None):
        """Batch compute all nonel and inclusive rates if z changes.

        The result is always stored in the same vectors, since '_init_rate_matstruc'
        makes use of views to link ranges of the vector to locations in the matrix.

        Args:
            z (float): Redshift value at which the photon field is taken.
            pfield: optional photon field override (one-shot, bypasses cache).

        Returns:
            (bool): True if fields we indeed updated, False if nothing happened.
        """
        # Full-GPU cupy backend: bypass the host ``_batch_vec`` and
        # write directly into the GPU coupling matrix's data buffer.
        if _cupy_backend_active(self.prince_run):
            self._ensure_coupling_mat_gpu()
            if not (force_update or self._ratemat_zcache_gpu != z or pfield is not None):
                return False
            return self._update_rates_gpu(z, pfield=pfield)

        if pfield is not None or self._ratemat_zcache != z or force_update:
            info(5, "Updating batch rate vectors.")

            photon = self.photon_vector(z, pfield=pfield)
            if self._csr_build:
                # CSR path: sparse SpMV against the kernel (fp32). The result
                # is upcast into the fp64 coupling data by _update_coupling_mat
                # (it reads _batch_vec_xs when set). Flat in dph — the point.
                backend = self.prince_run.backend
                if (
                    backend.linear_algebra_backend.lower() == "mkl"
                    and config.has_mkl
                    and config.mkl is not None
                ):
                    # MKL Sparse BLAS fold: the response kernel is constant
                    # for the whole solve, so an optimized handle (built once)
                    # runs a multi-threaded SpMV ~13-36x faster than scipy's
                    # single-thread CSR loop. Writes into the pinned fp32
                    # output buffer, which _update_coupling_mat upcasts.
                    self._ensure_batch_matrix_mkl()
                    np.copyto(self._fold_x_mkl, photon)   # fp64 -> xs dtype
                    self._batch_matrix_mkl.gemv_ctargs(
                        1.0, self._fold_x_p, 0.0, self._fold_y_p
                    )
                    self._batch_vec_xs = self._fold_y_mkl
                else:
                    ph = np.asarray(photon, dtype=self._batch_matrix.dtype)
                    self._batch_vec_xs = self._batch_matrix.dot(ph)
                self._ratemat_zcache = None if pfield is not None else z
                return True
            backend = self.prince_run.backend
            self._ensure_xs_buffers()
            use_mixed = self._batch_matrix_xs is not self._batch_matrix
            if (
                backend.linear_algebra_backend.lower() == "mkl"
                and config.has_mkl
                and config.mkl is not None
                and backend.use_mkl_dense_matvec
                and not use_mixed
            ):
                # Route through MKL CBLAS DGEMV so the dense matvec
                # shares MKL's threadpool with the Sparse BLAS path,
                # avoiding the OpenBLAS-vs-MKL oversubscription that
                # otherwise caps the whole solve. Disabled by default
                # because Zen + MKL 2026 dispatches DGEMV serially
                # (see :data:`config.use_mkl_dense_matvec`). The MKL
                # binding is fp64-only; when ``backend.xs_dtype`` is
                # fp32 (the default) the mixed-precision branch below
                # takes over via numpy's BLAS dispatch (SGEMV under
                # OpenBLAS/Accelerate).
                from . import mkl_dense

                mkl_dense.dgemv_y_eq_Ax(
                    self._batch_matrix,
                    np.ascontiguousarray(photon, dtype=np.float64),
                    self._batch_vec,
                )
            elif use_mixed:
                # Mixed-precision: cast photon down to xs dtype, run
                # the matvec in xs dtype (SGEMV-fast on every BLAS at
                # this shape), leave the result in ``_batch_vec_xs``.
                # ``_update_coupling_mat`` does the fp32→fp64 cast on
                # the copy into the CSR data buffer.
                np.copyto(self._photon_buf_xs, photon)
                np.dot(
                    self._batch_matrix_xs,
                    self._photon_buf_xs,
                    out=self._batch_vec_xs,
                )
            else:
                np.dot(
                    self._batch_matrix,
                    photon,
                    out=self._batch_vec,
                )
            # Invalidate cache for one-shot pfield overrides so the next
            # normal-path call recomputes against `self.photon_field`.
            self._ratemat_zcache = None if pfield is not None else z
            return True
        else:
            return False

    def _update_coupling_mat(self, z, scale_fac, force_update=False, pfield=None):
        """Updates the sparse (csr) coupling matrix
        Only the data vector is updated to minimize computation
        """
        # Do not execute dot product if photon field didn't change
        if self._update_rates(z, force_update, pfield=pfield):
            if _cupy_backend_active(self.prince_run):
                # Full-GPU path: matvec already wrote (and cast, if
                # mixed-precision) into ``_coupling_mat_gpu.data``.
                # Apply the scale in place at solver precision so the
                # ``scale_fac`` factor never roundtrips through fp32.
                # Fast path for the common ``scale_fac == 1.0`` case.
                if scale_fac != 1.0:
                    data = self._coupling_mat_gpu.data
                    data *= data.dtype.type(scale_fac)
                return
            # Sparsity-pattern contract (see CLAUDE.md): only `data` is mutated;
            # `indices`/`indptr` were fixed by `_init_coupling_mat`. SpMV backends
            # (scipy/MKL/cupy) rely on the pattern being intact.
            #
            # In-place refresh into the existing ``coupling_mat.data`` buffer.
            # ``self.coupling_mat.data = scale_fac * self._batch_vec`` allocates
            # a fresh ndarray every cache window — at production grid that's
            # ~1.7 M float64 = ~13 MB allocated + freed ~100×/solve, which
            # showed up as ~23 ms/window in the line profile (the scipy
            # backend's per-window cost is ~150 ms of which 23 ms was this
            # rebind alone). Writing into the pinned buffer also keeps any
            # MKL Sparse BLAS handle that pinned ``coupling_mat.data`` valid
            # across windows (relevant if the MKL handle ever wraps the
            # downstream ``M`` directly rather than its ``M_off`` copy).
            #
            # ``_batch_vec_xs`` is the matvec output buffer at
            # ``backend.xs_dtype`` (fp32 by default); when that matches
            # the host solver dtype, it IS ``_batch_vec``. ``np.copyto`` /
            # ``np.multiply`` upcast fp32 → fp64 on the write into the
            # CSR data buffer — that cast is the only mixed-precision
            # boundary in the host pipeline (mirrors the cupy path).
            src = self._batch_vec_xs if self._batch_vec_xs is not None else self._batch_vec
            assert self.coupling_mat.data.size == src.size
            if scale_fac == 1.0:
                np.copyto(self.coupling_mat.data, src)
            else:
                np.multiply(
                    src, scale_fac, out=self.coupling_mat.data
                )

    def get_hadr_jacobian(self, z, scale_fac=1.0, force_update=False, pfield=None):
        """Returns the nonel rate vector and coupling matrix."""
        self._update_coupling_mat(z, scale_fac, force_update, pfield=pfield)
        if _cupy_backend_active(self.prince_run):
            return self._coupling_mat_gpu
        return self.coupling_mat

    def single_interaction_length(self, pid, z, pfield=None):
        """Returns energy loss length in cm
        (convenience function for plotting)
        """
        species = self.spec_man.pdgid2sref[pid]
        egrid = self.e_cosmicray.grid * species.A
        rate = (
            -1
            * self.get_hadr_jacobian(force_update=True, z=z, pfield=pfield)
            .toarray()[species.sl, species.sl]
            .diagonal()
        )

        with np.errstate(divide="ignore"):
            length = 1 / rate

        return egrid, length


class _ContinuousLossRateBase(object):
    """Shared boilerplate for continuous loss-rate classes.

    Both subclasses build a per-species vector keyed either to the
    cosmic-ray grid (``energy='grid'``, length ``dim_states``) or to its
    bin edges (``energy='bins'``, length ``dim_bins``).
    """

    def __init__(self, prince_run):
        info(3, "creating instance")
        self.spec_man = prince_run.spec_man
        self.e_cosmicray = prince_run.cr_grid
        self.dim_states = prince_run.dim_states
        self.dim_bins = prince_run.dim_bins
        # Per-home-grid registry (Tier 3): each species' block uses its own
        # energy array. Flag-off exposes only {"default": cr_grid}, so every
        # species resolves to the cr grid — bit-identical to the old path.
        self._grids = getattr(prince_run, "grids", {"default": prince_run.cr_grid})

    def _species_earr(self, spec, energy):
        """Return the home-grid energy array for ``spec`` (``grid`` → cell
        centres, ``bins`` → bin edges), matching that species' transport block
        size. Nuclei resolve to the cr grid; EM species to the EM grid."""
        grid = self._grids.get(spec.grid_tag, self.e_cosmicray)
        return grid.grid if energy == "grid" else grid.bins

    def _energy_axis(self, energy):
        """Resolve ``energy`` to ``(dim, e_array, lo_idx, hi_idx)``.

        ``lo_idx``/``hi_idx`` are callables taking a species and returning
        the per-species slice bounds in the resulting vector.
        """
        if energy == "grid":
            return (
                self.dim_states,
                self.e_cosmicray.grid,
                lambda s: s.lidx(),
                lambda s: s.uidx(),
            )
        if energy == "bins":
            return (
                self.dim_bins,
                self.e_cosmicray.bins,
                lambda s: s.lbin(),
                lambda s: s.ubin(),
            )
        raise ValueError(
            "Unexpected energy keyword ({}), use either 'grid' or 'bins'".format(
                energy
            )
        )


class ContinuousAdiabaticLossRate(_ContinuousLossRateBase):
    """Implementation of continuous pair production loss rates."""

    def __init__(self, prince_run, energy="grid", *args, **kwargs):
        super().__init__(prince_run)
        # Init adiabatic loss vector
        self.energy_vector = self._init_energy_vec(energy)

    def loss_vector(self, z, energy=None):
        """Returns all continuous losses on dim_states grid"""
        # return self.adiabatic_losses(z)
        from prince_cr.cosmology import H

        if energy is None:
            return H(z) * PRINCE_UNITS.cm2sec * self.energy_vector
        else:
            return H(z) * PRINCE_UNITS.cm2sec * energy

    def _init_energy_vec(self, energy):
        """Prepare vector for scaling with units, charge and mass.

        Adiabatic (redshift) loss applies to every species, so each block is
        filled with that species' own home-grid energy array (Tier 3). With a
        single grid this is bit-identical to filling every block with the cr
        grid.
        """
        dim, _, lo, hi = self._energy_axis(energy)
        energy_vector = np.zeros(dim)
        for spec in self.spec_man.species_refs:
            energy_vector[lo(spec) : hi(spec)] = self._species_earr(spec, energy)
        return energy_vector

    def single_loss_length(self, pid, z):
        """Returns energy loss length in cm
        (convenience function for plotting)
        """
        species = self.spec_man.pdgid2sref[pid]

        egrid = self.energy_vector[species.sl] * species.A
        rate = self.loss_vector(z)[species.sl] * species.A
        length = egrid / rate
        return egrid, length


class ContinuousPairProductionLossRate(_ContinuousLossRateBase):
    """Implementation of continuous pair production loss rates."""

    def __init__(self, prince_run, energy="grid", *args, **kwargs):
        super().__init__(prince_run)

        #: Owning PriNCeRun (source of `photon_field`, see property below)
        self.prince_run = prince_run

        self.e_photon = prince_run.ph_grid

        # xi is dimensionless (natural units) variable
        xi_steps = 400 if "xi_steps" not in kwargs else kwargs["xi_steps"]
        info(2, "using", xi_steps, "steps in xi")
        self.xi = np.logspace(np.log10(2 + 1e-8), 16.0, xi_steps)

        # weights for integration
        self.phi_xi2 = self._phi(self.xi) / (self.xi**2)

        # Scale vector containing the units and factors of Z**2 for nuclei
        self.scale_vec = self._init_scale_vec(energy)

        # Capture this instance's energy mode so loss_vector() places the
        # per-nucleus rate into the correct slice bounds and total length
        # (grid → dim_states/lidx; bins → dim_bins/lbin).
        self._loss_dim, _, self._loss_lo, self._loss_hi = self._energy_axis(energy)

        # Grid of photon energies for interpolation. `_energy_axis` already
        # validated `energy` above; reuse the same axis here.
        _, earr, _, _ = self._energy_axis(energy)
        gamma = earr / PRINCE_UNITS.m_proton
        self.photon_grid = np.outer(1 / gamma, self.xi) * PRINCE_UNITS.m_electron / 2.0
        self.pg_desort = self.photon_grid.reshape(-1).argsort()
        self.pg_sorted = self.photon_grid.reshape(-1)[self.pg_desort]

    @property
    def photon_field(self):
        return self.prince_run.photon_field

    def loss_vector(self, z, pfield=None):
        """Returns all continuous losses on dim_states grid"""

        rate_single = trapz(
            self.photon_vector(z, pfield=pfield) * self.phi_xi2, self.xi, axis=1
        )
        # Pair production is a nuclear process: scale_vec is non-zero only on
        # nucleus blocks (and rate_single lives on the cr grid). Place it into
        # each nucleus' slice; EM/other blocks stay zero. With a single grid
        # this reproduces the old ``np.tile(rate_single, nspec)`` exactly once
        # multiplied by ``scale_vec`` (which zeroed the non-nucleus blocks).
        rate_full = np.zeros(self._loss_dim)
        for spec in self.spec_man.species_refs:
            if spec.is_nucleus:
                rate_full[self._loss_lo(spec) : self._loss_hi(spec)] = rate_single
        pprod_loss_vector = self.scale_vec * rate_full

        return pprod_loss_vector

    def photon_vector(self, z, pfield=None):
        """Returns photon vector at redshift `z` on photon grid.

        This vector is in fact a matrix of vectors of the interpolated
        photon field with dimensions (dim_cr, xi_steps).

        Args:
            z (float): redshift
            pfield: optional photon field override; defaults to ``self.photon_field``.
        """
        pf = pfield if pfield is not None else self.photon_field
        photon_vector = np.zeros_like(self.photon_grid)
        photon_vector.reshape(-1)[self.pg_desort] = pf.get_photon_density(
            self.pg_sorted, z
        )

        return photon_vector

    def _init_scale_vec(self, energy):
        """Prepare vector for scaling with units, charge and mass."""
        dim, earr, lo, hi = self._energy_axis(energy)
        scale_vec = np.zeros(dim)
        units = (
            PRINCE_UNITS.fine_structure
            * PRINCE_UNITS.r_electron**2
            * PRINCE_UNITS.m_electron**2
        )
        for spec in self.spec_man.species_refs:
            if not spec.is_nucleus:
                continue
            scale_vec[lo(spec) : hi(spec)] = (
                units
                * abs(spec.charge) ** 2
                / float(spec.A)
                * np.ones_like(earr, dtype="double")
            )
        return scale_vec

    def single_loss_length(self, pid, z, pfield=None):
        """Returns energy loss length in cm
        (convenience function for plotting)
        """
        species = self.spec_man.pdgid2sref[pid]

        egrid = self.e_cosmicray.grid * species.A
        rate = self.loss_vector(z, pfield=pfield)[species.sl] * species.A
        length = egrid / rate

        return egrid, length

    def _phi(self, xi):
        """Phi function as in Blumental 1970"""

        # Simple ultrarelativistic approximation by Blumental 1970
        bltal_ultrarel = np.poly1d([2.667, -14.45, 50.95, -86.07])

        def phi_simple(xi):
            return xi * bltal_ultrarel(np.log(xi))

        # random fit parameters, see Chorodowski et al
        c1 = 0.8048
        c2 = 0.1459
        c3 = 1.137e-3
        c4 = -3.879e-6

        f1 = 2.91
        f2 = 78.35
        f3 = 1837

        res = np.zeros(xi.shape)

        le = np.where(xi < 25.0)
        he = np.where(xi >= 25.0)

        res[le] = (
            np.pi
            / 12.0
            * (xi[le] - 2) ** 4
            / (
                c1 * (xi[le] - 2) ** 1
                + c2 * (xi[le] - 2) ** 2
                + c3 * (xi[le] - 2) ** 3
                + c4 * (xi[le] - 2) ** 4
            )
        )

        res[he] = phi_simple(xi[he]) / (
            1 - f1 * xi[he] ** -1 - f2 * xi[he] ** -2 - f3 * xi[he] ** -3
        )

        return res
