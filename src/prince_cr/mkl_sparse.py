"""Intel MKL Sparse BLAS wrapper for ETD2 SpMVs.

Pattern vendored from MCEq's ``MCEq.solvers.MklSparseMatrix`` — see
``MCEq/src/MCEq/solvers.py`` for the original. Adapted to PriNCe's
single-matrix-per-handle layout (no companion ``int_m``/``dec_m`` split).

Design notes (kept from MCEq, abridged):

* RAII wrapper around an MKL sparse-matrix handle. CSR or BSR storage,
  same API. ``mkl_sparse_set_mv_hint`` + ``mkl_sparse_optimize`` run once
  at construction so the per-step SpMV reuses the optimised layout.
* MKL keeps raw pointers into the backing arrays — Python objects must
  outlive the handle, so we pin ``_data`` / ``_indices`` / ``_indptr``
  refs on ``self``.
* BSR requires the matrix dimension to be a multiple of ``blocksize``;
  the wrapper pads with zero rows/cols, so callers operating on padded
  buffers can slice ``[:n_orig]`` for elementwise math. ``blocksize=6``
  was MCEq's empirical sweet spot on SIBYLL21 matrices with MKL ≥ 2024;
  PriNCe's matrices are different so the optimum is an open question.

Precision + shape:

* CSR handles support **float32 or float64** (``mkl_sparse_{s,d}_*``) and
  **rectangular** matrices (rows != cols) — needed by the photo-nuclear
  response fold ``_batch_matrix @ photon`` (``batch_dim × dph``, fp32,
  constant for the whole solve → ``optimize=True``).
* The BSR path stays **float64 + square** (the ``M_off`` in-place-update
  use case); fp32/rectangular BSR raises.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from . import config


class MklSparseMatrix:
    """Thin RAII wrapper around an Intel MKL sparse-matrix handle.

    Args:
      csr (scipy.sparse.csr_matrix): float32 or float64 CSR matrix with
        int32-castable indices. May be rectangular (CSR path only).
      expected_calls (int): SpMV count hint for MKL planning.
      blocksize (int | None): If ``None``, store as CSR. If int >= 2,
        store as BSR with that block size (float64 + square only;
        auto-padding the matrix).
      optimize (bool): If True (default), call
        ``mkl_sparse_set_mv_hint`` + ``mkl_sparse_optimize`` once at
        construction. The optimised path is faster per gemv but
        **caches the data values internally** — subsequent in-place
        mutations of ``self._data`` are NOT picked up by gemv. Use this
        for matrices that are constant for the whole solve (e.g.
        PriNCe's FD operator ``D_off`` or the response fold kernel).
        If False, skip optimize. gemv is slower per call but reads
        ``self._data`` on every invocation, so :meth:`update_data`
        actually takes effect. Use for matrices whose values are
        refreshed across cache windows (e.g. PriNCe's photo-hadronic
        ``M_off``).
    """

    def __init__(self, csr, expected_calls=4000, blocksize=None, optimize=True):
        from ctypes import POINTER, Structure, byref, c_int, c_void_p
        from ctypes import c_double, c_float

        if config.mkl is None:
            raise RuntimeError(
                "MklSparseMatrix: MKL library is not loaded. "
                "Call config.set_mkl_threads(...) first."
            )
        if not sp.isspmatrix_csr(csr):
            raise TypeError(
                f"MklSparseMatrix expects a CSR matrix, got {type(csr).__name__}"
            )
        if csr.dtype == np.float64:
            self._dtype = np.dtype(np.float64)
            self._fl_ct = c_double
            suffix = "d"
        elif csr.dtype == np.float32:
            self._dtype = np.dtype(np.float32)
            self._fl_ct = c_float
            suffix = "s"
        else:
            raise TypeError(
                f"MklSparseMatrix expects float32/float64 data, got {csr.dtype}"
            )

        mkl = config.mkl
        self._mkl = mkl
        self._mv_fn = getattr(mkl, f"mkl_sparse_{suffix}_mv")
        self._mm_fn = getattr(mkl, f"mkl_sparse_{suffix}_mm")   # sparse × dense-matrix
        fl_pr = self._fl_ct

        n_rows, n_cols = csr.shape
        self.n_orig = n_rows          # rows (== cols for the square/BSR case)
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.blocksize = blocksize

        if blocksize is None:
            # ----- CSR path (fp32/fp64, rectangular OK) -----
            indices = csr.indices.astype(np.int32, copy=False)
            indptr = csr.indptr.astype(np.int32, copy=False)
            data = np.ascontiguousarray(csr.data, dtype=self._dtype)
            self._data = data
            self._indices = indices
            self._indptr = indptr
            self.nnz = csr.nnz
            self.n_padded = n_rows

            handle = c_void_p()
            data_p = data.ctypes.data_as(POINTER(fl_pr))
            ci_p = indices.ctypes.data_as(POINTER(c_int))
            pb_p = indptr[:-1].ctypes.data_as(POINTER(c_int))
            pe_p = indptr[1:].ctypes.data_as(POINTER(c_int))

            st = getattr(mkl, f"mkl_sparse_{suffix}_create_csr")(
                byref(handle), c_int(0), c_int(n_rows), c_int(n_cols),
                pb_p, pe_p, ci_p, data_p,
            )
            if st != 0:
                raise RuntimeError(
                    f"mkl_sparse_{suffix}_create_csr failed with status {st}"
                )
        else:
            # ----- BSR path (float64 + square only) -----
            if self._dtype != np.float64:
                raise NotImplementedError(
                    "MklSparseMatrix BSR path is float64-only "
                    f"(got {self._dtype})"
                )
            if n_rows != n_cols:
                raise ValueError(
                    "MklSparseMatrix BSR path requires a square matrix "
                    f"(got {n_rows}x{n_cols})"
                )
            n_orig = n_rows
            if not isinstance(blocksize, int) or blocksize < 2:
                raise ValueError(f"blocksize must be int >= 2, got {blocksize!r}")
            pad = (-n_orig) % blocksize
            if pad > 0:
                indptr_padded = np.concatenate(
                    [csr.indptr,
                     np.full(pad, csr.indptr[-1], dtype=csr.indptr.dtype)]
                )
                csr = sp.csr_matrix(
                    (csr.data, csr.indices, indptr_padded),
                    shape=(n_orig + pad, n_orig + pad),
                ).tocsr()
            n_padded = csr.shape[0]
            self.n_padded = n_padded

            # Save the (padded) CSR layout for the CSR→BSR-flat index
            # map below, so :meth:`update_data` can scatter a 1D CSR
            # data buffer into the BSR's pinned 3D data array.
            self._csr_nnz = csr.nnz

            B = csr.tobsr(blocksize=(blocksize, blocksize))
            # ``tobsr`` does NOT guarantee sorted block-column indices;
            # sort once so the per-block-row searchsorted in the
            # CSR→BSR index map below sees a sorted slice. ``sort_indices``
            # also reorders ``B.data`` to match.
            B.sort_indices()
            data = np.ascontiguousarray(B.data, dtype=np.float64)
            indices = B.indices.astype(np.int32, copy=False)
            indptr = B.indptr.astype(np.int32, copy=False)
            self._data = data
            self._indices = indices
            self._indptr = indptr
            self.nnz = data.size

            # Map each (padded-)CSR entry k to the flat position in
            # ``data`` (= n_blocks_nonzero * bs * bs). Within block-row
            # br, B.indices is sorted, so per-row searchsorted finds
            # the block index. flat = block_idx * bs² + intra_r·bs + intra_c.
            row_idx = np.repeat(
                np.arange(n_padded, dtype=np.int64), np.diff(csr.indptr)
            )
            col_idx = csr.indices.astype(np.int64, copy=False)
            block_row = row_idx // blocksize
            block_col = col_idx // blocksize
            intra = (row_idx % blocksize) * blocksize + (col_idx % blocksize)
            n_block_rows = n_padded // blocksize
            block_pos = np.empty(csr.nnz, dtype=np.int64)
            for br in range(n_block_rows):
                lo, hi = int(B.indptr[br]), int(B.indptr[br + 1])
                if lo == hi:
                    continue
                mask = block_row == br
                if not mask.any():
                    continue
                block_pos[mask] = lo + np.searchsorted(
                    B.indices[lo:hi], block_col[mask]
                )
            self._csr_to_bsr_flat = block_pos * (blocksize * blocksize) + intra

            handle = c_void_p()
            n_blocks = c_int(n_padded // blocksize)
            data_p = data.ctypes.data_as(POINTER(fl_pr))
            ci_p = indices.ctypes.data_as(POINTER(c_int))
            pb_p = indptr[:-1].ctypes.data_as(POINTER(c_int))
            pe_p = indptr[1:].ctypes.data_as(POINTER(c_int))
            # SPARSE_LAYOUT_ROW_MAJOR = 101 — scipy stores BSR blocks row-major.
            st = mkl.mkl_sparse_d_create_bsr(
                byref(handle), c_int(0), c_int(101),
                n_blocks, n_blocks, c_int(blocksize),
                pb_p, pe_p, ci_p, data_p,
            )
            if st != 0:
                raise RuntimeError(f"mkl_sparse_d_create_bsr failed with status {st}")
        self._handle = handle

        class _MatrixDescr(Structure):
            _fields_ = [
                ("type", c_int),
                ("mode", c_int),
                ("diag", c_int),
            ]

        descr = _MatrixDescr()
        descr.type = c_int(20)   # SPARSE_MATRIX_TYPE_GENERAL
        descr.mode = c_int(121)
        descr.diag = c_int(131)
        self._descr = descr
        self._operation = c_int(10)  # SPARSE_OPERATION_NON_TRANSPOSE

        if optimize:
            st = mkl.mkl_sparse_set_mv_hint(
                handle, self._operation, descr, c_int(int(expected_calls))
            )
            # set_mv_hint may return SPARSE_STATUS_NOT_SUPPORTED for some
            # (matrix, op) combos; that is non-fatal — optimize still helps.
            if st not in (0, 6):
                raise RuntimeError(f"mkl_sparse_set_mv_hint failed with status {st}")
            st = mkl.mkl_sparse_optimize(handle)
            if st != 0:
                raise RuntimeError(f"mkl_sparse_optimize failed with status {st}")
        self._optimized = optimize

    def update_data(self, new_data):
        """Overwrite the pinned data buffer in place.

        Empirically, ``mkl_sparse_optimize`` **caches the data values**
        internally on top of laying out the sparsity. After optimize,
        ``mkl_sparse_?_mv`` reads from MKL's internal copy, not from
        the source pointer — so in-place mutation has no effect on
        subsequent gemv calls. This method therefore requires
        ``optimize=False`` at construction; in that mode MKL re-reads
        the source pointer on every call and ``update_data`` works as
        expected.

        Args:
          new_data (np.ndarray): values (cast to the handle dtype). Two
            accepted shapes:

            * For CSR handles or for BSR with the full padded layout:
              length ``self._data.size``.
            * For BSR handles fed with 1D CSR data: length ``self._csr_nnz``
              (the original padded CSR.nnz, set at construction). The
              method then scatters the entries into the BSR's flat
              data buffer via the pre-computed CSR→BSR index map; padded
              positions inside blocks stay zero.
        """
        if self._handle is None:
            raise RuntimeError(
                "MklSparseMatrix.update_data: handle is closed."
            )
        if self._optimized:
            raise RuntimeError(
                "MklSparseMatrix.update_data requires optimize=False at "
                "construction; mkl_sparse_optimize caches data values "
                "internally and ignores in-place mutations."
            )
        if new_data is self._data:
            # Aliased buffer: ``np.ascontiguousarray`` returned the caller's
            # array unchanged at construction, so an in-place write to
            # ``new_data`` already updated MKL's (un-optimized) view. The
            # copyto would be a self-copy over the full nnz — at PriNCe's
            # ~45 M-nnz M_off that is ~20 ms/window wasted. Skip it.
            return
        if new_data.dtype != self._dtype:
            new_data = new_data.astype(self._dtype)
        if (
            self.blocksize is not None
            and new_data.size == getattr(self, "_csr_nnz", -1)
            and new_data.size != self._data.size
        ):
            # 1D CSR data → BSR flat data scatter. MKL's pinned buffer
            # is reused (no reallocation), so the address stays valid.
            self._data.flat[self._csr_to_bsr_flat] = new_data
            return
        if new_data.size != self._data.size:
            raise ValueError(
                f"update_data size mismatch: got {new_data.size}, "
                f"expected {self._data.size}"
                + (
                    f" or {self._csr_nnz} (1D CSR data)"
                    if self.blocksize is not None
                    else ""
                )
            )
        # In-place: preserves the buffer address that MKL has pinned.
        np.copyto(self._data, new_data)

    def gemv_ctargs(self, alpha, x_p, beta, y_p):
        """``y = alpha * A * x + beta * y`` via raw pointers (dtype-aware)."""
        fl_pr = self._fl_ct
        st = self._mv_fn(
            self._operation,
            fl_pr(alpha),
            self._handle,
            self._descr,
            x_p,
            fl_pr(beta),
            y_p,
        )
        if st != 0:
            raise RuntimeError(f"mkl_sparse_?_mv failed with status {st}")

    def gemv_preboxed(self, alpha_box, x_p, beta_box, y_p):
        """Like :meth:`gemv_ctargs` but takes pre-constructed float boxes
        for ``alpha`` and ``beta`` (must match the handle dtype).

        Constructing a ctypes float per call costs ~3–5 µs in CPython —
        comparable to the SpMV itself for ~2 k-DoF matrices, so we pre-
        box constants in the caller and pass them through. Skips the
        status check on the hot path; if MKL is going to fail it'll fail
        early on the first call (or via ``gemv_ctargs`` during warmup).
        """
        self._mv_fn(
            self._operation, alpha_box, self._handle, self._descr,
            x_p, beta_box, y_p,
        )

    def spmm(self, alpha, b_p, columns, ldb, beta, c_p, ldc, layout=101):
        """``C = alpha * A * B + beta * C`` for dense B, C via raw pointers.

        ``layout`` is the MKL ``sparse_layout_t``: 101 = ROW_MAJOR (a
        C-contiguous ``(k, columns)`` B / ``(m, columns)`` C, with
        ``ldb = ldc = columns``), 102 = COLUMN_MAJOR (Fortran, ``ldb = k``).
        Used by the multi-RHS ETD2 step (SpMV → SpMM over the K columns).
        """
        from ctypes import c_int
        fl = self._fl_ct
        st = self._mm_fn(
            self._operation, fl(alpha), self._handle, self._descr,
            c_int(layout), b_p, c_int(columns), c_int(ldb),
            fl(beta), c_p, c_int(ldc),
        )
        if st != 0:
            raise RuntimeError(f"mkl_sparse_?_mm failed with status {st}")

    def spmm_preboxed(self, alpha_box, b_p, columns_box, ldb_box,
                      beta_box, c_p, ldc_box, layout_box):
        """Like :meth:`spmm` but takes pre-constructed ctypes boxes for the
        scalar args (alpha/beta floats, layout/columns/ldb/ldc ints) — the
        multi-RHS hot loop boxes the K-fixed constants once and passes only
        the two data pointers per call. No status check on the hot path."""
        self._mm_fn(
            self._operation, alpha_box, self._handle, self._descr,
            layout_box, b_p, columns_box, ldb_box, beta_box, c_p, ldc_box,
        )

    def close(self):
        """Free the underlying MKL sparse handle. Idempotent."""
        handle = getattr(self, "_handle", None)
        mkl = getattr(self, "_mkl", None)
        if handle is None or mkl is None:
            return
        try:
            mkl.mkl_sparse_destroy(handle)
        except Exception:
            pass
        self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
