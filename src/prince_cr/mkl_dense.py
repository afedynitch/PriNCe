"""Intel MKL CBLAS DGEMV wrapper for the cache-rebuild dense matvec.

PriNCe builds its photo-nuclear rate vector via a single dense matvec at
each cache-window boundary:

    _batch_vec = _batch_matrix @ photon_vector   # ~60 % of solve time

When numpy is bundled-OpenBLAS (the default on this venv), that call
dispatches to OpenBLAS DGEMV with OpenBLAS's own threadpool (48 threads
by default on Zen 2). Combined with the MKL Sparse BLAS we run for the
SpMVs, we end up with two independent threadpools competing for the
same physical cores — bad on Zen 2 where the per-NUMA core count is 24.

Routing the dense matvec through MKL via this wrapper consolidates the
work onto a single threadpool — the one controlled by
:func:`prince_cr.config.set_mkl_threads`. The dense matvec ~5 % MKL-vs-
OpenBLAS gap on AVX2 hardware is essentially noise; the value is the
unified thread management.

This module is a thin ctypes layer with no Python-level state — the
caller passes A, x, y arrays and ``cblas_dgemv`` operates on them in
place via the buffer pointers numpy already exposes. No long-lived
handles, no caching surprises like the Sparse BLAS optimize step.
"""

from __future__ import annotations

from ctypes import POINTER, c_double, c_int

import numpy as np

from . import config

# CBLAS layout / op constants (mkl_cblas.h).
_CBLAS_ROW_MAJOR = 101
_CBLAS_COL_MAJOR = 102
_CBLAS_NO_TRANS = 111
_CBLAS_TRANS = 112


def _bind_argtypes(mkl):
    """Set ctypes argtypes on cblas_dgemv exactly once.

    Idempotent — guarded by an attribute on the cdll handle. Without
    explicit argtypes ctypes auto-converts python ints/floats but
    silently mismangles 64-bit pointers on some platforms; safer to be
    explicit.
    """
    if getattr(mkl, "_prince_cr_dgemv_bound", False):
        return
    mkl.cblas_dgemv.restype = None
    mkl.cblas_dgemv.argtypes = [
        c_int, c_int,           # layout, trans
        c_int, c_int,           # m, n
        c_double,               # alpha
        POINTER(c_double),      # A
        c_int,                  # lda
        POINTER(c_double),      # x
        c_int,                  # incx
        c_double,               # beta
        POINTER(c_double),      # y
        c_int,                  # incy
    ]
    mkl._prince_cr_dgemv_bound = True


def dgemv_y_eq_Ax(A: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``y = A @ x`` via ``cblas_dgemv`` — float64, row-major, contiguous.

    Args:
      A: 2D float64 row-major contiguous.
      x: 1D float64 contiguous, length A.shape[1].
      y: 1D float64 contiguous, length A.shape[0]. Overwritten in place.

    Returns ``y`` (the same object passed in) for caller convenience.

    Raises:
      RuntimeError: MKL not loaded — call ``config.set_mkl_threads`` first.
      TypeError: dtype isn't float64 — matches numpy DGEMV's contract.
      ValueError: shapes don't line up.
    """
    mkl = config.mkl
    if mkl is None:
        raise RuntimeError(
            "mkl_dense.dgemv: MKL not loaded. "
            "Call prince_cr.config.set_mkl_threads(N) first."
        )
    if A.dtype != np.float64 or x.dtype != np.float64 or y.dtype != np.float64:
        raise TypeError(
            "mkl_dense.dgemv expects float64 arrays; got "
            f"A.dtype={A.dtype}, x.dtype={x.dtype}, y.dtype={y.dtype}."
        )
    if A.ndim != 2 or x.ndim != 1 or y.ndim != 1:
        raise ValueError(
            f"shape mismatch: A.shape={A.shape}, x.shape={x.shape}, y.shape={y.shape}"
        )
    m, n = A.shape
    if x.shape[0] != n or y.shape[0] != m:
        raise ValueError(
            f"shape mismatch: A=({m},{n}), x=({x.shape[0]},), y=({y.shape[0]},)"
        )
    # cblas_dgemv expects unit-stride contiguous buffers (incx=incy=1) and
    # ``lda`` = A.shape[1] for row-major. Non-contiguous A would need a
    # contiguity copy here; the caller in interaction_rates.py builds A
    # from np.zeros + fancy indexing, both of which leave A C-contiguous.
    if not A.flags.c_contiguous:
        A = np.ascontiguousarray(A)
    if not x.flags.c_contiguous:
        x = np.ascontiguousarray(x)
    # y is written, so a non-contiguous y would silently break — better
    # to be loud than to copy.
    if not y.flags.c_contiguous:
        raise ValueError("mkl_dense.dgemv requires y to be C-contiguous.")

    _bind_argtypes(mkl)
    mkl.cblas_dgemv(
        c_int(_CBLAS_ROW_MAJOR), c_int(_CBLAS_NO_TRANS),
        c_int(m), c_int(n),
        c_double(1.0),
        A.ctypes.data_as(POINTER(c_double)),
        c_int(n),  # lda = n for row-major contiguous
        x.ctypes.data_as(POINTER(c_double)),
        c_int(1),
        c_double(0.0),
        y.ctypes.data_as(POINTER(c_double)),
        c_int(1),
    )
    return y
