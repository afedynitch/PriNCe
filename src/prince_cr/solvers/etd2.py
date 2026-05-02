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

    L is left untouched. L_off is returned in CSR format with explicit zeros
    eliminated. The diagonal vector d is float64 regardless of L.dtype, so
    elementwise ``exp(h*d)`` etc. are well-defined.
    """
    d = np.asarray(L.diagonal(), dtype=np.float64)
    L_off = L - sp.diags(d, format=L.format)
    if not sp.isspmatrix_csr(L_off):
        L_off = L_off.tocsr()
    L_off.eliminate_zeros()
    return d, L_off


def _step_buffers(dim):
    """Allocate per-step scratch arrays. Mirrors MCEq layout."""
    return {
        "hD": np.empty(dim, dtype=np.float64),
        "eD": np.empty(dim, dtype=np.float64),
        "phi1": np.empty(dim, dtype=np.float64),
        "phi2": np.empty(dim, dtype=np.float64),
        "scratch": np.empty(dim, dtype=np.float64),
        "abs_hD": np.empty(dim, dtype=np.float64),
        "mask1": np.empty(dim, dtype=bool),
        "mask2": np.empty(dim, dtype=bool),
        "F_phi": np.empty(dim, dtype=np.float64),
        "F_a": np.empty(dim, dtype=np.float64),
        "a": np.empty(dim, dtype=np.float64),
    }


def _compute_diag_factors(h, d, bufs):
    """Fill ``bufs['eD'] / bufs['phi1'] / bufs['phi2']`` in place.

    Computes ``hD = h * d``, ``eD = exp(hD)``, and the two phi-functions of
    ``hD`` elementwise. The Taylor patches near zero are applied only on the
    rows that need them via boolean masks.
    """
    hD = bufs["hD"]
    eD = bufs["eD"]
    phi1 = bufs["phi1"]
    phi2 = bufs["phi2"]
    scratch = bufs["scratch"]
    abs_hD = bufs["abs_hD"]
    mask1 = bufs["mask1"]
    mask2 = bufs["mask2"]

    np.multiply(d, h, out=hD)
    np.exp(hD, out=eD)

    np.abs(hD, out=abs_hD)
    np.greater(abs_hD, _PHI1_SMALL, out=mask1)
    np.greater(abs_hD, _PHI2_SMALL, out=mask2)

    # phi1: analytic (eD - 1) / hD where mask1, Taylor 1 + hD/2 + hD²/6 elsewhere.
    np.subtract(eD, 1.0, out=phi1)
    np.divide(phi1, hD, out=phi1, where=mask1)
    np.multiply(hD, _INV_6, out=scratch)
    np.add(scratch, 0.5, out=scratch)
    np.multiply(scratch, hD, out=scratch)
    np.add(scratch, 1.0, out=scratch)
    np.invert(mask1, out=mask1)
    np.copyto(phi1, scratch, where=mask1)

    # phi2: analytic (eD - 1 - hD) / hD² where mask2, Taylor 1/2 + hD/6 + hD²/24 else.
    np.subtract(eD, 1.0, out=phi2)
    np.subtract(phi2, hD, out=phi2)
    np.multiply(hD, hD, out=scratch)  # hD² in scratch
    np.divide(phi2, scratch, out=phi2, where=mask2)
    np.multiply(hD, _INV_24, out=scratch)
    np.add(scratch, _INV_6, out=scratch)
    np.multiply(scratch, hD, out=scratch)
    np.add(scratch, 0.5, out=scratch)
    np.invert(mask2, out=mask2)
    np.copyto(phi2, scratch, where=mask2)


def etd2_step(state, h, d, apply_F, bufs):
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

    _compute_diag_factors(h, d, bufs)

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


def integrate(
    state,
    z_grid,
    operator_at,
):
    """Integrate ``dn/dz = L(z) n + b(z)`` along ``z_grid`` with ETD2.

    Parameters
    ----------
    state : np.ndarray
        Initial state vector. Modified in place. Returned.
    z_grid : np.ndarray
        Monotonic sequence of redshift values to step through. ``z_grid[0]``
        is the initial z; ``z_grid[-1]`` is the final z. Steps are
        ``h_k = z_grid[k+1] - z_grid[k]``; sign is determined automatically.
    operator_at : callable
        ``(z) -> (d, apply_F)``: returns the diagonal vector and a callable
        ``apply_F(x, out)`` that writes ``F(x) = (L - diag(d)) x + b`` into
        ``out``. Invoked once per step, so any internal caching must be
        handled by the callable itself.

    Returns
    -------
    state : np.ndarray
        The propagated state.
    """
    bufs = _step_buffers(state.shape[0])

    nsteps = len(z_grid) - 1
    for k in range(nsteps):
        z0 = z_grid[k]
        z1 = z_grid[k + 1]
        h = z1 - z0
        d, apply_F = operator_at(z0)
        etd2_step(state, h, d, apply_F, bufs)

    return state
