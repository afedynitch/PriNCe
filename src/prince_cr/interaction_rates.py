"""The module contains classes for computations of interaction rates"""

import numpy as np
from scipy.integrate import trapezoid as trapz

from prince_cr.data import PRINCE_UNITS
from prince_cr.util import info
import prince_cr.config as config

using_cupy = False
# Use GPU support
if config.has_cupy and config.linear_algebra_backend.lower() == "cupy":
    import cupy

    using_cupy = True


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

    def _estimate_batch_matrix(self):
        """estimate dimension of the batch matrix"""
        dcr = self.e_cosmicray.d
        dph = self.e_photon.d

        batch_dim = 0
        for specid in self.spec_man.known_species:
            if specid < 100:
                continue
            # Add the main diagonal self-couplings (absorption)
            batch_dim += dcr
            # Daughter-only species have no outgoing reactions; skip them here.
            for rtup in self.cross_sections.reactions.get(specid, []):
                # Off main diagonal couplings (reinjection)
                if rtup in self.cross_sections.known_bc_channels:
                    batch_dim += dcr
                elif rtup in self.cross_sections.known_diff_channels:
                    # Only half of the elements can be non-zero (energy conservation)
                    batch_dim += int(dcr**2 / 2) + 1

        info(2, "Batch matrix dimensions are {0}x{1}".format(batch_dim, dph))
        self._batch_matrix = np.zeros((batch_dim, dph))
        self._batch_rows = []
        self._batch_cols = []
        info(3, "Memory usage: {0} MB".format(self._batch_matrix.nbytes / 1024**2))

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
        sp_id_ref = spec_man.ncoid2sref
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
        # nonel: dict[mo] → ΔR̂_nonel of length (dcr+dph-1)
        nonel_dR = {}
        for mo, intp in resp.nonel_intp.items():
            R = intp.antiderivative()(y_grid)
            nonel_dR[mo] = np.diff(R)
        # incl (boost-conserving): dict[(mo,da)] → ΔR̂_incl of length (dcr+dph-1)
        incl_dR = {}
        for key, intp in resp.incl_intp.items():
            R = intp.antiderivative()(y_grid)
            incl_dR[key] = np.diff(R)

        # ----- pre-sample 2D antiderivatives once -----
        # diff: dict[(mo,da)] → ΔΔR̂ of shape (dcr+dph-1, 2*dcr-1)
        diff_ddR = {}
        if resp.incl_diff_intp_integral:
            Y2d, X2d = np.meshgrid(y_grid, x_grid, indexing="ij")
            Yflat, Xflat = Y2d.ravel(), X2d.ravel()
            shape2d = Y2d.shape
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

        ibatch = 0
        emo_idcs = np.arange(dcr)
        eda_idcs = np.arange(dcr)
        # i_mo grid for diagonal-nonel handling in diff channels
        diag = np.arange(dcr)

        # Iteration order is irrelevant — `_init_coupling_mat` lex-sorts.
        known_species_rev = spec_man.known_species[::-1]
        import itertools

        for moid, daid in itertools.product(known_species_rev, known_species_rev):
            if moid < 100:
                continue
            # Daughter-only species have no cross sections; skip them as mothers.
            if moid not in self.cross_sections.reactions:
                continue

            has_nonel = moid == daid
            in_bc = (moid, daid) in self.cross_sections.known_bc_channels
            in_diff = (moid, daid) in self.cross_sections.known_diff_channels

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

                self._batch_matrix[ibatch:ibatch + dcr, :] = tile
                ibatch += dcr
                self._batch_rows.append(sp_id_ref[daid].lidx() + eda_idcs)
                self._batch_cols.append(sp_id_ref[moid].lidx() + emo_idcs)

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

                # x-cut filter (depends only on (i_mo, i_da))
                cut_low = x_cut_proton if daid == 101 else x_cut
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

                self._batch_matrix[ibatch:ibatch + n_kept, :] = kept
                ibatch += n_kept
                self._batch_rows.append(rows)
                self._batch_cols.append(cols)

            else:
                info(20, "Species combination not included in model", moid, daid)

        self._batch_matrix = self._batch_matrix[:ibatch, :]
        self._batch_rows = np.concatenate(self._batch_rows, axis=None)
        self._batch_cols = np.concatenate(self._batch_cols, axis=None)
        self._batch_vec = np.zeros(ibatch)

        info(2, f"Batch matrix shape: {self._batch_matrix.shape}")
        info(2, f"Batch rows shape: {self._batch_rows.shape}")
        info(2, f"Batch cols shape: {self._batch_cols.shape}")
        info(2, f"Batch vector shape: {self._batch_vec.shape}")

        memory = (
            self._batch_matrix.nbytes
            + self._batch_rows.nbytes
            + self._batch_cols.nbytes
            + self._batch_vec.nbytes
        ) / 1024**2
        info(3, "Memory usage after initialization: {:} MB".format(memory))

    def _init_coupling_mat(self):
        """Initialises the coupling matrix directly in sparse (csr) format."""
        info(0, "Initiating coupling matrix in ({:}) format".format("CSR"))

        from scipy.sparse import csr_matrix

        if using_cupy:
            # For GPU we initialize the csr matrix on the host and then cast to GPU
            from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix

            self.coupling_mat_np = csr_matrix(
                (
                    self._batch_vec.astype(np.float32),
                    (self._batch_rows, self._batch_cols),
                ),
                copy=True,
            )
            self.coupling_mat = cp_csr_matrix(self.coupling_mat_np, copy=True)
            self._batch_vec = self.coupling_mat.data
            del self.coupling_mat_np
        else:
            self.coupling_mat = csr_matrix(
                (self._batch_vec, (self._batch_rows, self._batch_cols)), copy=True
            )

        # create an index to sort by rows and then columns,
        # which is the same ordering CSR has internally
        # lexsort sorts by last argument first!!!
        self.sortidx = np.lexsort((self._batch_cols, self._batch_rows))

        self._batch_rows = self._batch_rows[self.sortidx]
        self._batch_cols = self._batch_cols[self.sortidx]

        # Reorder batch matrix according to order in coupling_mat
        if using_cupy:
            self._batch_matrix = cupy.array(
                self._batch_matrix[self.sortidx, :], dtype=np.float32
            )
        else:
            self._batch_matrix = self._batch_matrix[self.sortidx, :]

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
        if pfield is not None or self._ratemat_zcache != z or force_update:
            info(5, "Updating batch rate vectors.")

            if using_cupy:
                if isinstance(self._batch_matrix, np.ndarray):
                    self._init_coupling_mat()
                cupy.dot(
                    self._batch_matrix,
                    cupy.array(self.photon_vector(z, pfield=pfield), dtype=np.float32),
                    out=self._batch_vec,
                )
            else:
                np.dot(
                    self._batch_matrix,
                    self.photon_vector(z, pfield=pfield),
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
            # Sparsity-pattern contract (see CLAUDE.md): only `data` is mutated;
            # `indices`/`indptr` were fixed by `_init_coupling_mat`. SpMV backends
            # (scipy/MKL/cupy) rely on the pattern being intact.
            assert self.coupling_mat.data.size == self._batch_vec.size
            self.coupling_mat.data = scale_fac * self._batch_vec

    def get_hadr_jacobian(self, z, scale_fac=1.0, force_update=False, pfield=None):
        """Returns the nonel rate vector and coupling matrix."""
        self._update_coupling_mat(z, scale_fac, force_update, pfield=pfield)
        return self.coupling_mat

    def single_interaction_length(self, pid, z, pfield=None):
        """Returns energy loss length in cm
        (convenience function for plotting)
        """
        species = self.spec_man.ncoid2sref[pid]
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
        """Prepare vector for scaling with units, charge and mass."""
        dim, earr, lo, hi = self._energy_axis(energy)
        energy_vector = np.zeros(dim)
        for spec in self.spec_man.species_refs:
            energy_vector[lo(spec) : hi(spec)] = earr
        return energy_vector

    def single_loss_length(self, pid, z):
        """Returns energy loss length in cm
        (convenience function for plotting)
        """
        species = self.spec_man.ncoid2sref[pid]

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
        pprod_loss_vector = self.scale_vec * np.tile(rate_single, self.spec_man.nspec)

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
        species = self.spec_man.ncoid2sref[pid]

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
