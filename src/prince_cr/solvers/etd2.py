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
    return {
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
    """
    eD = bufs["eD"]
    phi1 = bufs["phi1"]
    phi2 = bufs["phi2"]
    scratch = bufs["scratch"]
    F_phi = bufs["F_phi"]
    F_a = bufs["F_a"]
    a = bufs["a"]

    _compute_diag_factors(h, d, bufs, xp)

    apply_F(state, F_phi)

    # a = eD * state + h * phi1 * F_phi
    xp.multiply(eD, state, out=a)
    xp.multiply(phi1, F_phi, out=scratch)
    scratch *= h
    xp.add(a, scratch, out=a)

    apply_F(a, F_a)

    # state <- a + h * phi2 * (F_a - F_phi)
    xp.subtract(F_a, F_phi, out=scratch)
    scratch *= h
    xp.multiply(scratch, phi2, out=scratch)
    xp.add(a, scratch, out=state)
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
