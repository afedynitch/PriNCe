"""Analytical validation of the explicit-decay machinery in the ETD2 kernel.

Step A (this file): kernel-level unit tests that drive ``etd2.integrate``
directly with hand-built diagonal + sparse off-diagonal operators. No
PriNCe init machinery; the tests measure only the ETD2 integrator's
ability to reproduce textbook decay solutions.

  * ``test_single_mother_exponential_decay`` — diagonal-only operator
    n' = -n/τ; expect n(T) = n(0)·exp(-T/τ).

  * ``test_two_step_bateman_chain`` — 2×2 operator with off-diagonal
    feed; expect Bateman closed form for daughter buildup.

  * ``test_three_species_chain_terminal_stable`` — 3×3 chain a→b→c with
    c stable (zero diagonal); the daughter-of-daughter case.

  * ``test_solver_enable_decay_no_regression`` — Step B no-regression
    check: with the chain reducer on (default), no unstable species reach
    the state vector, so ``_etd2_Lambda_diag`` stays ``None`` and the
    solver's output with ``enable_decay=True`` is bit-identical to
    ``enable_decay=False``. Validates the wiring is plumbed but inert
    when there's nothing for it to act on.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_allclose

from prince_cr.cr_sources import AugerFitSource
from prince_cr.solvers import UHECRPropagationSolverETD2
from prince_cr.solvers.etd2 import integrate


_AUGER_PARAMS = {
    2212: (0.96, 10**9.68, 20.0),
    1000020040: (0.96, 10**9.68, 50.0),
    1000070140: (0.96, 10**9.68, 30.0),
}


def _const_operator(d_const, L_off_csr=None):
    """Return an ``operator_at(z) -> (d, apply_F)`` for a z-independent L.

    ``d_const`` is the diagonal of L (1-D array, length dim).
    ``L_off_csr`` is the off-diagonal (scipy CSR, dim×dim) — pass ``None``
    for a purely diagonal operator.
    """
    d_arr = np.asarray(d_const, dtype=np.float64)

    if L_off_csr is None:
        def apply_F(x, out):
            out.fill(0.0)
    else:
        def apply_F(x, out):
            np.copyto(out, L_off_csr @ x)

    def operator_at(z):
        return d_arr, apply_F

    return operator_at


def test_single_mother_exponential_decay():
    """Diagonal-only L = diag(-1/τ); n(T) = n(0)·exp(-T/τ).

    Drives ``etd2.integrate`` against a 1-D state with a single decay
    constant. The "z" axis here is just time; the test exercises only
    the ETD2 stepper's exponential-of-diagonal treatment.
    """
    tau = 0.7  # arbitrary lifetime in the integration variable
    T = 5.0    # five lifetimes — final n ≈ 6.7e-3 of initial
    n_steps = 1000

    z_grid = np.linspace(0.0, T, n_steps + 1)
    state = np.array([1.0], dtype=np.float64)

    op = _const_operator([-1.0 / tau])
    integrate(state, z_grid, operator_at=op)

    expected = np.exp(-T / tau)
    assert_allclose(state[0], expected, rtol=1e-10)


def test_two_step_bateman_chain():
    """L = [[-1/τa, 0], [+1/τa, -1/τb]] driven from n_a(0)=1, n_b(0)=0.

    Bateman closed form (τa ≠ τb):
        n_a(t) = exp(-t/τa)
        n_b(t) = (τb/(τa - τb)) · (exp(-t/τa) - exp(-t/τb))

    Validates that the off-diagonal feed (mother → daughter) integrates
    correctly through the ETD2 phi₁/phi₂ machinery.
    """
    tau_a, tau_b = 1.0, 2.5  # well-separated to avoid the τa=τb edge
    T = 6.0
    n_steps = 2000

    z_grid = np.linspace(0.0, T, n_steps + 1)
    state = np.array([1.0, 0.0], dtype=np.float64)

    d = np.array([-1.0 / tau_a, -1.0 / tau_b])
    L_off = sp.csr_matrix(np.array([[0.0, 0.0], [1.0 / tau_a, 0.0]]))

    op = _const_operator(d, L_off)
    integrate(state, z_grid, operator_at=op)

    n_a_expected = np.exp(-T / tau_a)
    n_b_expected = (tau_b / (tau_a - tau_b)) * (
        np.exp(-T / tau_a) - np.exp(-T / tau_b)
    )

    assert_allclose(state[0], n_a_expected, rtol=1e-8)
    assert_allclose(state[1], n_b_expected, rtol=1e-6)


def test_three_species_chain_terminal_stable():
    """a → b → c with c stable (d_c = 0): conservation + Bateman cascade.

    Total population is conserved (no loss to outside), so
    ``n_a + n_b + n_c == 1`` at all times. Closed form for ``n_c`` follows
    by mass conservation:
        n_c(t) = 1 - n_a(t) - n_b(t)

    Catches sign errors in the off-diagonal feed and ensures the ETD2
    treatment of a zero-diagonal entry doesn't blow up via phi-function
    catastrophic cancellation.
    """
    tau_a, tau_b = 0.8, 2.0
    T = 8.0
    n_steps = 4000

    z_grid = np.linspace(0.0, T, n_steps + 1)
    state = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    # L_diag handles loss; L_off handles gain. d_c = 0 (c is stable).
    d = np.array([-1.0 / tau_a, -1.0 / tau_b, 0.0])
    L_off = sp.csr_matrix(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0 / tau_a, 0.0, 0.0],
                [0.0, 1.0 / tau_b, 0.0],
            ]
        )
    )

    op = _const_operator(d, L_off)
    integrate(state, z_grid, operator_at=op)

    n_a_expected = np.exp(-T / tau_a)
    n_b_expected = (tau_b / (tau_a - tau_b)) * (
        np.exp(-T / tau_a) - np.exp(-T / tau_b)
    )
    n_c_expected = 1.0 - n_a_expected - n_b_expected

    assert_allclose(state[0], n_a_expected, rtol=1e-8)
    assert_allclose(state[1], n_b_expected, rtol=1e-6)
    assert_allclose(state[2], n_c_expected, rtol=1e-6)
    # Mass conservation is bounded by ETD2's accumulated finite-step error
    # (~1e-6 at 4 000 steps); a tighter sum tolerance would be inconsistent
    # with the per-component tolerance above.
    assert_allclose(state.sum(), 1.0, rtol=1e-5)


def test_no_off_diagonal_no_branching():
    """Pure-diagonal multi-species: each species decays independently.

    Three species with distinct lifetimes and no coupling. Validates
    that ``apply_F`` returning zero is handled correctly (dim>1, no
    SpMV) and that each diagonal entry is exponentiated independently.
    """
    taus = np.array([0.5, 1.5, 3.0])
    T = 4.0
    n_steps = 1000

    z_grid = np.linspace(0.0, T, n_steps + 1)
    state = np.ones(3, dtype=np.float64)

    op = _const_operator(-1.0 / taus)
    integrate(state, z_grid, operator_at=op)

    expected = np.exp(-T / taus)
    assert_allclose(state, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Step B: solver wiring no-regression check.
# ---------------------------------------------------------------------------


def _make_etd2(prince_run, **kw):
    solver = UHECRPropagationSolverETD2(
        initial_z=1.0,
        final_z=0.0,
        prince_run=prince_run,
        enable_pairprod_losses=True,
        enable_adiabatic_losses=True,
        enable_injection_jacobian=True,
        enable_partial_diff_jacobian=True,
        **kw,
    )
    solver.add_source_class(
        AugerFitSource(prince_run, norm=1e-50, params=_AUGER_PARAMS)
    )
    return solver


class _BackendOverride:
    """Pin ``prince_run.backend.linear_algebra_backend`` for a test scope.

    The session-scoped ``prince_run_m14`` fixture is shared across tests;
    we save the prior backend setting and restore it on exit so other
    tests still see the default dispatch.
    """

    def __init__(self, prince_run, backend_name):
        self.prince_run = prince_run
        self.target = backend_name
        self._saved = None

    def __enter__(self):
        self._saved = self.prince_run.backend.linear_algebra_backend
        self.prince_run.backend.linear_algebra_backend = self.target
        return self

    def __exit__(self, *exc):
        self.prince_run.backend.linear_algebra_backend = self._saved
        return False


def test_solver_enable_decay_no_regression(prince_run_m14):
    """``enable_decay=True`` with default chain reducer is a no-op.

    Step B wiring lands the diagonal-Λ infrastructure on
    ``ETD2SolverCPU`` but does not gate the cross-section chain reducer
    (that's Step C). With the chain reducer on, every unstable mother
    has been folded out at cross-section build time, so no species in
    the state vector has a finite ``lifetime``. ``_ensure_Lambda_split``
    detects that and leaves ``_etd2_Lambda_diag`` as ``None``, which the
    per-step operator then skips entirely.

    Validates two things:
      1. The wiring compiles and runs without error when
         ``enable_decay=True``.
      2. The output is bit-identical to ``enable_decay=False`` (no
         silent perturbation from the new code path).

    Forces the scipy/CPU backend so the assertion targets the code path
    that actually got the Step B treatment. The cupy backend's matching
    no-regression test arrives in Step E.
    """
    with _BackendOverride(prince_run_m14, "scipy"):
        solver_off = _make_etd2(prince_run_m14, enable_decay=False)
        solver_off.solve(dz=1e-3, verbose=False, progressbar=False)
        state_off = solver_off.state.copy()

        solver_on = _make_etd2(prince_run_m14, enable_decay=True)
        solver_on.solve(dz=1e-3, verbose=False, progressbar=False)
        state_on = solver_on.state.copy()

    # Step B: the new code path should be inert when no unstable species
    # reach the state vector. Bit-identical, not just "close".
    assert solver_on._etd2_Lambda_diag is None, (
        "Λ_diag should stay None when chain reducer is on; "
        "found a populated buffer — chain-reducer gate may have leaked"
    )
    assert_allclose(state_on, state_off, atol=0.0, rtol=0.0)
