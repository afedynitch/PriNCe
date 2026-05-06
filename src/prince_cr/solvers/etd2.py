"""ETD2 (exponential time-differencing RK2) kernel for PriNCe.

Vendored from MCEq's ``MCEq/solvers.py`` and adapted to the PriNCe RHS
layout::

    dn/dz = L(z) n + b(z)

where ``L(z) = J(z) + dl/dz · D · diag(κ(z))`` is the linear operator
(photo-hadronic + continuous-loss block) and ``b(z)`` is the inhomogeneous
injection term.

The kernel treats the diagonal of ``L`` exactly via ``exp(h·D)`` and the
off-diagonal explicitly with two SpMVs per stage (4 SpMVs / step). The
matrix is held constant across each step, so the rate-cache invalidation
threshold (``config.update_rates_z_threshold``) directly controls how often
the diagonal/off-diagonal split has to be recomputed.

The kernel is array-module-agnostic. ``state`` may be a numpy ndarray or
a cupy ndarray; the integrator dispatches all elementwise math through
``cupy.get_array_module(state)`` so the same code runs CPU-side
(numpy/scipy) or GPU-side (cupy/cupyx.scipy.sparse). Per-step buffers
inherit the array module of the state vector. The ``apply_F`` callback
provided by the solver decides which sparse backend (scipy / MKL /
cuSPARSE) drives the SpMVs — see ``UHECRPropagationSolverETD2``.

Reference: Cox & Matthews 2002 ("Exponential time differencing for stiff
systems"); Hochbruck & Ostermann 2010 §2.2 for inhomogeneous source.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp


# phi1(z) = (e^z - 1) / z              (limit 1   as z -> 0)
# phi2(z) = (e^z - 1 - z) / z**2       (limit 1/2 as z -> 0)
# Below the analytic-formula cutoffs we patch with the order-2 Taylor
# expansion via Horner. phi2 cancels at a wider radius than phi1 because
# its numerator has a leading z² term.
_PHI1_SMALL = 1e-6
_PHI2_SMALL = 1e-3
_INV_6 = 1.0 / 6.0
_INV_24 = 1.0 / 24.0


# Fused cupy ElementwiseKernels (Stage 3). Lazily compiled per dtype on
# first call. The non-graph cupy path uses these directly; the graph path
# captures their launches and reads scalar params (``h``, ``dldz``) from
# 1-element ``raw T`` device buffers so the captured graph picks up new
# values across replays without re-capture.
_CUPY_KERNELS = None


def _build_cupy_kernels(cp):
    phi_compute = cp.ElementwiseKernel(
        "T d, raw T hbuf",
        "T eD, T phi1, T phi2",
        f"""
        T h = hbuf[0];
        T hd = h * d;
        T e = exp(hd);
        eD = e;
        T abs_hd = (hd >= T(0)) ? hd : -hd;
        if (abs_hd > T({_PHI1_SMALL!r})) {{
            phi1 = (e - T(1)) / hd;
        }} else {{
            phi1 = T(1) + hd * (T(0.5) + hd * T({_INV_6!r}));
        }}
        if (abs_hd > T({_PHI2_SMALL!r})) {{
            phi2 = (e - T(1) - hd) / (hd * hd);
        }} else {{
            phi2 = T(0.5) + hd * (T({_INV_6!r}) + hd * T({_INV_24!r}));
        }}
        """,
        "prince_etd2_phi_compute",
    )
    post_apply1 = cp.ElementwiseKernel(
        "T eD, T state, T phi1, T F_phi, raw T hbuf",
        "T a",
        "a = eD * state + hbuf[0] * phi1 * F_phi;",
        "prince_etd2_post_apply1",
    )
    post_apply2 = cp.ElementwiseKernel(
        "T a, T F_a, T F_phi, T phi2, raw T hbuf",
        "T state",
        "state = a + hbuf[0] * phi2 * (F_a - F_phi);",
        "prince_etd2_post_apply2",
    )
    # Fused ``out = out * dldz + b`` (apply_F tail, source-on case) and
    # ``out = out * dldz`` (source-off case). Output writes back into the
    # same buffer as the input so the kernel is in-place from the caller's
    # perspective.
    scale_b = cp.ElementwiseKernel(
        "T x, T b, raw T dldz_buf",
        "T y",
        "y = x * dldz_buf[0] + b;",
        "prince_etd2_scale_b",
    )
    scale_no_b = cp.ElementwiseKernel(
        "T x, raw T dldz_buf",
        "T y",
        "y = x * dldz_buf[0];",
        "prince_etd2_scale_no_b",
    )
    # Fused ``d = dldz · (M_diag + κ ⊙ D_diag)`` for the diagonal of
    # ``L(z)``. Used by the cupy graph path so the entire per-step body
    # (including the diagonal rebuild) lives inside the captured graph.
    compute_d = cp.ElementwiseKernel(
        "T M_diag, T kappa, T D_diag, raw T dldz_buf",
        "T d_out",
        "d_out = dldz_buf[0] * (M_diag + kappa * D_diag);",
        "prince_etd2_compute_d",
    )
    compute_d_no_kappa = cp.ElementwiseKernel(
        "T M_diag, raw T dldz_buf",
        "T d_out",
        "d_out = dldz_buf[0] * M_diag;",
        "prince_etd2_compute_d_no_kappa",
    )
    return SimpleNamespace(
        phi_compute=phi_compute,
        post_apply1=post_apply1,
        post_apply2=post_apply2,
        scale_b=scale_b,
        scale_no_b=scale_no_b,
        compute_d=compute_d,
        compute_d_no_kappa=compute_d_no_kappa,
    )


def cupy_kernels():
    """Return the (lazily compiled) Stage 3 cupy ElementwiseKernel set.

    Module-level singleton — first call imports cupy and builds the
    kernels; subsequent calls return the cached namespace. ElementwiseKernel
    objects compile their CUDA source on first launch per dtype, not on
    construction, so this call is cheap.
    """
    global _CUPY_KERNELS
    if _CUPY_KERNELS is None:
        import cupy as cp

        _CUPY_KERNELS = _build_cupy_kernels(cp)
    return _CUPY_KERNELS


# Custom warp-per-row CSR SpMV. Required for the CUDA Graph path because
# cupy 14 explicitly blocks cuSPARSE calls during stream capture
# ("NotImplementedError: calling cuSPARSE API during stream capture is
# currently unsupported", `_setStream` in cusparse.pyx). The kernel is
# also used for the eager fused path so both code paths exercise the
# same SpMV implementation — see :class:`UHECRPropagationSolverETD2`.
_CSR_SPMV_CACHE = {}


def _csr_spmv_kernel(dtype, accumulate):
    """Compile (cached) a warp-per-row CSR SpMV ``RawKernel``.

    ``accumulate=False`` writes ``y = A·x``; ``accumulate=True`` writes
    ``y = y + A·x`` (no scalar multipliers — α is fused into the
    downstream ``scale_b`` / ``post_apply`` ElementwiseKernels). One
    warp per row; threads in the warp stride across the row's nnz and
    do a butterfly reduction. At the photo-hadronic ``M_off`` density
    (~150 nnz/row) the warp loop is fed; at the FD ``D_off`` density
    (~4 nnz/row) most lanes idle but the kernel is still cheap.
    """
    import cupy as cp

    key = (np.dtype(dtype).name, bool(accumulate))
    cached = _CSR_SPMV_CACHE.get(key)
    if cached is not None:
        return cached
    T = "float" if np.dtype(dtype) == np.float32 else "double"
    op = "y[row] += sum;" if accumulate else "y[row] = sum;"
    name = f"prince_csr_spmv_{T}_{'add' if accumulate else 'set'}"
    code = f"""
    extern "C" __global__ void {name}(
        const int n,
        const int* __restrict__ indptr,
        const int* __restrict__ indices,
        const {T}* __restrict__ data,
        const {T}* __restrict__ x,
        {T}* __restrict__ y
    ) {{
        const int warps_per_block = blockDim.x / 32;
        const int row = blockIdx.x * warps_per_block + (threadIdx.x / 32);
        const int lane = threadIdx.x & 31;
        if (row >= n) return;
        const int row_start = indptr[row];
        const int row_end = indptr[row + 1];
        {T} sum = ({T})0;
        for (int j = row_start + lane; j < row_end; j += 32) {{
            sum += data[j] * x[indices[j]];
        }}
        // Warp shuffle reduction.
        for (int offset = 16; offset > 0; offset >>= 1) {{
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }}
        if (lane == 0) {op}
    }}
    """
    k = cp.RawKernel(code, name)
    _CSR_SPMV_CACHE[key] = k
    return k


_CSR_SPMV_THREADS = 128  # 4 warps/block — light occupancy works for our row count


def csr_spmv(M, x, y, accumulate=False):
    """Run the custom CSR SpMV on the active stream.

    ``M`` is a ``cupyx.scipy.sparse.csr_matrix`` (or anything exposing
    ``indptr`` / ``indices`` / ``data`` cupy arrays); ``x`` and ``y``
    are dense cupy ndarrays of compatible dtype and length ``M.shape[0]``.
    Capture-friendly: no host syncs, no allocations, no cuSPARSE calls.
    """
    n = M.shape[0]
    if n == 0:
        return
    k = _csr_spmv_kernel(M.dtype, accumulate)
    warps = n
    blocks = (warps * 32 + _CSR_SPMV_THREADS - 1) // _CSR_SPMV_THREADS
    k(
        (blocks,),
        (_CSR_SPMV_THREADS,),
        (n, M.indptr, M.indices, M.data, x, y),
    )


def split_operator(L):
    """Return (d, L_off) where d = diag(L) and L_off has its diagonal zeroed.

    Dispatches on L's array module: scipy sparse on host, cupyx.scipy.sparse
    on GPU. L is left untouched. L_off is returned in CSR format with
    explicit zeros eliminated.

    Dtype convention: ``d`` keeps L's dtype on the cupy path (float32 for
    the Stage 2 GPU backend), so elementwise ``exp(h*d)`` lands on fp32
    without an extra cast. On the host path, ``d`` is upcast to float64
    regardless of L's dtype so the downstream phi-function evaluation is
    well-conditioned.
    """
    if sp.issparse(L):
        d = np.asarray(L.diagonal(), dtype=np.float64)
        L_off = L - sp.diags(d, format=L.format)
        if not sp.isspmatrix_csr(L_off):
            L_off = L_off.tocsr()
        L_off.eliminate_zeros()
        return d, L_off

    # cupy/cupyx fallback. Imported lazily so the host-only path doesn't
    # require cupy to be installed.
    import cupy as cp
    import cupyx.scipy.sparse as csp

    if not csp.issparse(L):
        raise TypeError(
            f"split_operator: expected scipy/cupy sparse matrix, got {type(L)!r}."
        )
    d = L.diagonal()  # cupy ndarray, same dtype as L
    L_off = L - csp.diags(d, format=L.format)
    if not csp.isspmatrix_csr(L_off):
        L_off = L_off.tocsr()
    L_off.eliminate_zeros()
    return d, L_off


def _array_module(arr):
    """Return numpy or cupy depending on ``arr``'s type.

    Falls back to numpy if cupy isn't importable; uses
    ``cupy.get_array_module`` otherwise. Inlined here so the integrator
    doesn't need to thread an ``xp`` argument through every call.
    """
    try:
        import cupy

        return cupy.get_array_module(arr)
    except ImportError:
        return np


def _step_buffers(dim, xp=np, dtype=None):
    """Allocate per-step scratch arrays in ``xp``'s memory space.

    Mirrors MCEq's layout. ``xp`` ∈ {numpy, cupy}; cupy arrays land on
    the current default device. ``dtype`` defaults to ``xp.float64`` on
    numpy and ``xp.float32`` on cupy — matching the project convention
    that GPU work runs in fp32 on RTX-class hardware (no FP64 Tensor
    Cores; the fp32→fp64 cost on Ampere is ~2× even without Tensor
    Cores). The mask buffers are bool regardless of dtype.
    """
    if dtype is None:
        dtype = xp.float32 if xp is not np else xp.float64
    bufs = {
        "hD": xp.empty(dim, dtype=dtype),
        "eD": xp.empty(dim, dtype=dtype),
        "phi1": xp.empty(dim, dtype=dtype),
        "phi2": xp.empty(dim, dtype=dtype),
        "scratch": xp.empty(dim, dtype=dtype),
        "scratch2": xp.empty(dim, dtype=dtype),
        "abs_hD": xp.empty(dim, dtype=dtype),
        "mask1": xp.empty(dim, dtype=xp.bool_),
        "mask2": xp.empty(dim, dtype=xp.bool_),
        # Inverted mask scratch — reused for the two ``copyto(where=)``
        # patches per ``_compute_diag_factors`` call. Without these
        # buffers ``xp.logical_not(mask)`` allocates a fresh bool array
        # each call (4×/step → ~6 µs each at dim_states ≈ 6 k).
        "not_mask1": xp.empty(dim, dtype=xp.bool_),
        "not_mask2": xp.empty(dim, dtype=xp.bool_),
        "F_phi": xp.empty(dim, dtype=dtype),
        "F_a": xp.empty(dim, dtype=dtype),
        "a": xp.empty(dim, dtype=dtype),
    }
    if xp is not np:
        # Stage 3 fused kernels read ``h`` from a 1-element ``raw T``
        # device buffer so a captured CUDA Graph can pick up updated
        # ``h`` between replays without re-capture (uniform z-grid means
        # the value never actually changes within a solve, but we keep
        # the buffer for the truncated-final-step edge case and for
        # symmetry with the per-step ``dldz_buf``).
        bufs["h_buf"] = xp.empty(1, dtype=dtype)
    return bufs


def _compute_diag_factors(h, d, bufs, xp):
    """Fill ``bufs['eD'] / bufs['phi1'] / bufs['phi2']`` in place.

    Computes ``hD = h * d``, ``eD = exp(hD)``, and the two phi-functions of
    ``hD`` elementwise. Near zero we patch with order-2 Taylor expansions.

    cupy's ufuncs accept ``out=`` but reject ``where=`` on most operations
    (verified on cupy 14.0.1: ``cupy.divide(out=, where=)`` raises
    ``TypeError: Wrong arguments``). We therefore avoid ``where=`` on the
    arithmetic ufuncs and use ``copyto(where=)`` (which IS supported in
    cupy 14) for the conditional patches. The full-array Taylor evaluation
    is cheap at dim_states ≈ 6 k.
    """
    hD = bufs["hD"]
    eD = bufs["eD"]
    phi1 = bufs["phi1"]
    phi2 = bufs["phi2"]
    scratch = bufs["scratch"]
    scratch2 = bufs["scratch2"]
    abs_hD = bufs["abs_hD"]
    mask1 = bufs["mask1"]
    mask2 = bufs["mask2"]
    not_mask1 = bufs["not_mask1"]
    not_mask2 = bufs["not_mask2"]

    xp.multiply(d, h, out=hD)
    xp.exp(hD, out=eD)
    xp.abs(hD, out=abs_hD)

    # phi1: analytic (eD - 1) / hD where |hD| > _PHI1_SMALL,
    #       Taylor 1 + hD/2 + hD²/6 elsewhere.
    # Step 1: scratch = hD where mask1, 1.0 elsewhere (denominator with
    # 1.0 fallback so we never divide by zero, even where the result
    # will be discarded).
    xp.greater(abs_hD, _PHI1_SMALL, out=mask1)
    xp.invert(mask1, out=not_mask1)
    xp.copyto(scratch, hD)
    xp.copyto(scratch, 1.0, where=not_mask1)
    # phi1 = (eD - 1) / scratch (analytic value where mask1 is True)
    xp.subtract(eD, 1.0, out=phi1)
    xp.divide(phi1, scratch, out=phi1)
    # Taylor branch in scratch2; copy into phi1 where mask1 is False.
    xp.multiply(hD, _INV_6, out=scratch2)
    xp.add(scratch2, 0.5, out=scratch2)
    xp.multiply(scratch2, hD, out=scratch2)
    xp.add(scratch2, 1.0, out=scratch2)
    xp.copyto(phi1, scratch2, where=not_mask1)

    # phi2: analytic (eD - 1 - hD) / hD² where |hD| > _PHI2_SMALL,
    #       Taylor 1/2 + hD/6 + hD²/24 elsewhere.
    xp.greater(abs_hD, _PHI2_SMALL, out=mask2)
    xp.invert(mask2, out=not_mask2)
    xp.multiply(hD, hD, out=scratch)
    xp.copyto(scratch, 1.0, where=not_mask2)
    xp.subtract(eD, 1.0, out=phi2)
    xp.subtract(phi2, hD, out=phi2)
    xp.divide(phi2, scratch, out=phi2)
    xp.multiply(hD, _INV_24, out=scratch2)
    xp.add(scratch2, _INV_6, out=scratch2)
    xp.multiply(scratch2, hD, out=scratch2)
    xp.add(scratch2, 0.5, out=scratch2)
    xp.copyto(phi2, scratch2, where=not_mask2)


def etd2_step(state, h, d, apply_F, bufs, xp):
    """One ETD2 step, in place on ``state``.

    Solves ``dn/dz = L n + b`` over a step of size ``h`` with ``L`` and
    ``b`` frozen at the start of the step. ``d`` is the diagonal of ``L``;
    ``apply_F(x, out)`` writes ``F(x) = (L - diag(d)) x + b`` into ``out``.

    Update (operator frozen at start of step):
        F(x) = L_off · x + b
        a   = exp(h*D) * state + h * phi1(h*D) * F(state)
        state <- a + h * phi2(h*D) * (F(a) - F(state))

    Dispatches to the fused-kernel cupy variant when ``xp`` isn't numpy.
    """
    if xp is np:
        return _etd2_step_numpy(state, h, d, apply_F, bufs)
    return _etd2_step_cupy(state, h, d, apply_F, bufs)


def _etd2_step_numpy(state, h, d, apply_F, bufs):
    """Host-side ETD2 step. The original ufunc-chain implementation."""
    eD = bufs["eD"]
    phi1 = bufs["phi1"]
    phi2 = bufs["phi2"]
    scratch = bufs["scratch"]
    F_phi = bufs["F_phi"]
    F_a = bufs["F_a"]
    a = bufs["a"]

    _compute_diag_factors(h, d, bufs, np)

    apply_F(state, F_phi)

    # a = eD * state + h * phi1 * F_phi
    np.multiply(eD, state, out=a)
    np.multiply(phi1, F_phi, out=scratch)
    scratch *= h
    np.add(a, scratch, out=a)

    apply_F(a, F_a)

    # state <- a + h * phi2 * (F_a - F_phi)
    np.subtract(F_a, F_phi, out=scratch)
    scratch *= h
    np.multiply(scratch, phi2, out=scratch)
    np.add(a, scratch, out=state)
    return state


def _etd2_step_cupy(state, h, d, apply_F, bufs):
    """Stage 3 fused-kernel cupy step.

    Replaces the per-step ufunc chain (~17 launches in
    ``_compute_diag_factors`` + 4 launches each in the two post-apply
    blocks) with three ElementwiseKernels — ``phi_compute``,
    ``post_apply1``, ``post_apply2`` — for a measured ~30 launches/step
    saved at production grid (see wiki/results/prof-etd2-launch-bound.md).

    ``apply_F`` itself remains the caller's responsibility (cupy variant
    in :class:`UHECRPropagationSolverETD2._make_apply_F_cupy`); its
    ``out *= dldz; out += b`` tail is fused into a single ElementwiseKernel
    on the propagation side using :func:`cupy_kernels`.
    """
    K = cupy_kernels()
    eD = bufs["eD"]
    phi1 = bufs["phi1"]
    phi2 = bufs["phi2"]
    F_phi = bufs["F_phi"]
    F_a = bufs["F_a"]
    a = bufs["a"]
    h_buf = bufs["h_buf"]

    # Refresh the 1-element ``h`` buffer the kernels read. Cheap (~µs)
    # and the value is constant for almost every step in a uniform
    # z-grid; kept per-step for the truncated-final-step edge case.
    h_buf[0] = h

    K.phi_compute(d, h_buf, eD, phi1, phi2)
    apply_F(state, F_phi)
    K.post_apply1(eD, state, phi1, F_phi, h_buf, a)
    apply_F(a, F_a)
    K.post_apply2(a, F_a, F_phi, phi2, h_buf, state)
    return state


def integrate(state, z_grid, operator_at):
    """Integrate ``dn/dz = L(z) n + b(z)`` along ``z_grid`` with ETD2.

    ``state`` may be a numpy or a cupy ndarray; the integrator picks the
    array module from it and allocates per-step buffers in the same
    memory space. ``operator_at(z) -> (d, apply_F)`` is responsible for
    returning ``d`` in the same array module as ``state`` and an
    ``apply_F(x, out)`` that writes into ``out`` in-place.

    Parameters
    ----------
    state : np.ndarray or cupy.ndarray
        Initial state vector. Modified in place. Returned.
    z_grid : np.ndarray
        Monotonic sequence of redshift values to step through.
    operator_at : callable
        ``(z) -> (d, apply_F)``. Invoked once per step.

    Returns
    -------
    state : ndarray (same module as input)
        The propagated state.
    """
    xp = _array_module(state)
    bufs = _step_buffers(state.shape[0], xp=xp, dtype=state.dtype)

    nsteps = len(z_grid) - 1
    for k in range(nsteps):
        z0 = z_grid[k]
        z1 = z_grid[k + 1]
        h = z1 - z0
        d, apply_F = operator_at(z0)
        etd2_step(state, h, d, apply_F, bufs, xp)

    return state
