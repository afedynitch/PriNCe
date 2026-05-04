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
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from . import config


class MklSparseMatrix:
    """Thin RAII wrapper around an Intel MKL sparse-matrix handle.

    Args:
      csr (scipy.sparse.csr_matrix): float64 CSR matrix with int32 indices.
      expected_calls (int): SpMV count hint for MKL planning.
      blocksize (int | None): If ``None``, store as CSR. If int >= 2,
        store as BSR with that block size (auto-padding the matrix).
      optimize (bool): If True (default), call
        ``mkl_sparse_set_mv_hint`` + ``mkl_sparse_optimize`` once at
        construction. The optimised path is ~2× faster per gemv but
        **caches the data values internally** — subsequent in-place
        mutations of ``self._data`` are NOT picked up by gemv. Use this
        for matrices that are constant for the whole solve (e.g.
        PriNCe's FD operator ``D_off``).
        If False, skip optimize. gemv is slower per call but reads
        ``self._data`` on every invocation, so :meth:`update_data`
        actually takes effect. Use for matrices whose values are
        refreshed across cache windows (e.g. PriNCe's photo-hadronic
        ``M_off``).
    """

    def __init__(self, csr, expected_calls=4000, blocksize=None, optimize=True):
        from ctypes import POINTER, Structure, byref, c_int, c_void_p
        from ctypes import c_double as fl_pr

        if config.mkl is None:
            raise RuntimeError(
                "MklSparseMatrix: MKL library is not loaded. "
                "Call config.set_mkl_threads(...) first."
            )
        if not sp.isspmatrix_csr(csr):
            raise TypeError(
                f"MklSparseMatrix expects a CSR matrix, got {type(csr).__name__}"
            )
        if csr.dtype != np.float64:
            raise TypeError(
                f"MklSparseMatrix expects float64 data, got {csr.dtype}"
            )

        n_orig = csr.shape[0]
        self.n_orig = n_orig
        self.blocksize = blocksize

        mkl = config.mkl
        self._mkl = mkl

        if blocksize is None:
            # ----- CSR path -----
            indices = csr.indices.astype(np.int32, copy=False)
            indptr = csr.indptr.astype(np.int32, copy=False)
            data = csr.data
            self._data = data
            self._indices = indices
            self._indptr = indptr
            self.nnz = csr.nnz
            self.n_padded = n_orig

            handle = c_void_p()
            data_p = data.ctypes.data_as(POINTER(fl_pr))
            ci_p = indices.ctypes.data_as(POINTER(c_int))
            pb_p = indptr[:-1].ctypes.data_as(POINTER(c_int))
            pe_p = indptr[1:].ctypes.data_as(POINTER(c_int))

            st = mkl.mkl_sparse_d_create_csr(
                byref(handle), c_int(0), c_int(n_orig), c_int(n_orig),
                pb_p, pe_p, ci_p, data_p,
            )
            if st != 0:
                raise RuntimeError(f"mkl_sparse_d_create_csr failed with status {st}")
        else:
            # ----- BSR path -----
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

            B = csr.tobsr(blocksize=(blocksize, blocksize))
            data = np.ascontiguousarray(B.data, dtype=np.float64)
            indices = B.indices.astype(np.int32, copy=False)
            indptr = B.indptr.astype(np.int32, copy=False)
            self._data = data
            self._indices = indices
            self._indptr = indptr
            self.nnz = data.size

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
            if st != 0:
                raise RuntimeError(f"mkl_sparse_set_mv_hint failed with status {st}")
            st = mkl.mkl_sparse_optimize(handle)
            if st != 0:
                raise RuntimeError(f"mkl_sparse_optimize failed with status {st}")
        self._optimized = optimize

    def update_data(self, new_data):
        """Overwrite the pinned data buffer in place.

        Empirically, ``mkl_sparse_optimize`` **caches the data values**
        internally on top of laying out the sparsity. After optimize,
        ``mkl_sparse_d_mv`` reads from MKL's internal copy, not from
        the source pointer — so in-place mutation has no effect on
        subsequent gemv calls. This method therefore requires
        ``optimize=False`` at construction; in that mode MKL re-reads
        the source pointer on every call and ``update_data`` works as
        expected.

        Args:
          new_data (np.ndarray): float64 values, length ``self._data.size``.
            For BSR mode this includes the block-internal padding zeros.
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
        if new_data.size != self._data.size:
            raise ValueError(
                f"update_data size mismatch: got {new_data.size}, "
                f"expected {self._data.size}"
            )
        if new_data.dtype != np.float64:
            new_data = new_data.astype(np.float64)
        # In-place: preserves the buffer address that MKL has pinned.
        np.copyto(self._data, new_data)

    def gemv_ctargs(self, alpha, x_p, beta, y_p):
        """``y = alpha * A * x + beta * y`` via raw c_double pointers."""
        from ctypes import c_double as fl_pr

        st = self._mkl.mkl_sparse_d_mv(
            self._operation,
            fl_pr(alpha),
            self._handle,
            self._descr,
            x_p,
            fl_pr(beta),
            y_p,
        )
        if st != 0:
            raise RuntimeError(f"mkl_sparse_d_mv failed with status {st}")

    def gemv_preboxed(self, alpha_box, x_p, beta_box, y_p):
        """Like :meth:`gemv_ctargs` but takes pre-constructed ``c_double``
        boxes for ``alpha`` and ``beta``.

        Constructing a ``c_double`` per call costs ~3–5 µs in CPython —
        comparable to the SpMV itself for ~2 k-DoF matrices, so we pre-
        box constants in the caller and pass them through. Skips the
        status check on the hot path; if MKL is going to fail it'll fail
        early on the first call (or via ``gemv_ctargs`` during warmup).
        """
        self._mkl.mkl_sparse_d_mv(
            self._operation, alpha_box, self._handle, self._descr,
            x_p, beta_box, y_p,
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
