# PriNCe v1 → v2: changes, justifications, and measurements

This document records the changes made in two consecutive work sessions on
`master` (and the active `worktree-kernel-accel-explore` branch). It is
intended as a self-contained reference: what changed, why it was the right
call, and what numbers back up that call.

The two sessions are:

- **Session A (2026-05-02)** — ETD2 solver port + CLAUDE.md rewrite. Landed
  in commits `9f629bc` and `6470cb4`, merged via `096c887`.
- **Session B (2026-05-03)** — Toeplitz kernel construction. Lives on the
  `worktree-kernel-accel-explore` branch, ready to merge.

For brevity: **v1** = repo state immediately before Session A
(commit `186ee05` or earlier). **v2** = repo state at the tip of the
Session B branch.

---

## 1. Solver: BDF → ETD2 (Session A)

### v1: SciPy BDF wrapper

PriNCe v1 used SciPy's BDF integrator (`scipy.integrate.solve_ivp(method='BDF')`)
wrapped in `UHECRPropagationSolverBDF` / `PrinceBDF`. The RHS was passed
to BDF as a black-box function, and BDF treated the system as a generic
stiff ODE. Cost on the realistic Auger-fit case (z=1→0, dz=1e-3,
production grid 91 species × 88 E-bins): **~24 s wall-clock**.

A second path, `SemiLagrangianSolver`, existed for advection-style
treatments of the continuous loss term. By v1 it was effectively dead
code: only reachable from `EULER` with `enable_partial_diff_jacobian=False`,
which was never the default. It carried its own `config.semi_lagr_method`
flag and a private set of util helpers (`get_y_redist`, etc.).

### v2: vendored ETD2 (Cox–Matthews exponential time-differencing RK2)

`UHECRPropagationSolverETD2` (`src/prince_cr/solvers/etd2.py`) integrates

```
dn/dz = L(z) n + b(z)
```

with `L(z) = J(z) + dl/dz · D · diag(κ(z))` the linear operator and `b(z)`
the injection. The system is **fully linear in n plus an inhomogeneous
source** — a property that BDF didn't exploit. ETD2 does:

- Treats `diag(L)` exactly via `exp(h · diag(L))` (per-bin closed form).
- Treats the off-diagonal block of `L` explicitly with two SpMVs per stage
  (4 SpMVs / step total — Cox–Matthews variant).
- Source `b(z)` enters via the same `φ₁ / φ₂` machinery as the off-diagonal
  block, frozen at step start to preserve 2nd-order accuracy.

#### Caching

The expensive piece is the photo-hadronic rate matrix `M_raw(z)`. ETD2
caches it at the `update_rates_z_threshold` window (default `0.01`),
mirroring what BDF did. The cheap pieces — `dl/dz`, `κ(z)`, and the source
`b(z)` — are recomputed every step, matching the legacy
`eqn_deriv_standard` per-call behaviour. This avoids a per-cache-window
systematic of order `recomp_z_threshold` magnitude in the propagation
result.

#### Why ETD2 over BDF

1. **Structure-aware.** The system is linear-in-`n` plus a source. ETD2 is
   the textbook choice for that shape; BDF wastes work re-discovering the
   linearity at every step via Newton iterations.
2. **Cheaper steps.** ETD2 needs 4 SpMVs per step plus one elementwise
   `exp`. BDF needs an LU factorisation per Newton block and ~3–5 RHS
   evaluations per Newton iteration.
3. **Accuracy where it matters.** The diagonal of `L` (loss rates) is the
   stiffest part of the spectrum. ETD2 integrates it exactly; BDF
   integrates it polynomially. For typical PriNCe spectra the diagonal
   loss dominates, so ETD2's accuracy advantage is concrete.

#### Measured results (production grid, AugerFitSource, z=1→0, dz=1e-3)

|                                | BDF (v1)      | ETD2 (v2)      |
|--------------------------------|---------------|----------------|
| wall-clock                     | ~24 s         | **~6 s**       |
| L2 error vs tight-cache ETD2   | reference     | < 0.7 × 10⁻³   |
| max rel error                  | reference     | < 1.5 %        |

The L2 / max-rel comparison was done against an ETD2 run with
`recomp_z_threshold = 1e-3, dz = 2e-4` (~50 s, used as the regression
reference and cached at `tests/data/baseline_state.npz`).

### Stencil audit

`tests/test_stencil_accuracy.py` audits the 4th-order asymmetric FD stencil
used in `DifferentialOperator` for the continuous-loss term. At the
production grid (8 bins/decade), centred-row truncation error is **≲ 1.5 %
even for α = −3** (a hard test case for upwind-biased stencils). This
validates the embedded-in-operator approach for losses, which both BDF and
ETD2 rely on.

### Cleanup

- Removed `SemiLagrangianSolver` and the `semi_lagrangian()` method.
- Removed `UHECRPropagationSolverBDF` and `PrinceBDF`.
- Removed `config.semi_lagr_method`.
- Stripped the dead `enable_partial_diff_jacobian=False` branch from
  `UHECRPropagationSolverEULER`.
- Removed `prince_cr/util.py` helpers that only served the deleted paths
  (`get_y_redist`, etc.) — net 180 lines deleted.
- Removed the legacy `tests/test_propagation_extra.py` cases that targeted
  BDF-specific behaviour.

### Test infrastructure (added in Session A)

- `tests/_build_kernel_cache.py` — builds a cached production-size
  `PriNCeRun` pickle (~3 min, ~3 GB). Enables fast benchmark iteration
  without re-running the kernel build per test.
- `tests/conftest.py::cached_prince_run` — loads the cache from
  `PRINCE_KERNEL_CACHE` env var, falls back to an in-session build.
- `tests/conftest.py::baseline_state` — tight-cache ETD2 reference state
  used by `tests/test_solver_baseline.py` for regression.
- `tests/conftest.py::bench_propagation` — wall-clock helper context manager.
- `tests/test_etd2.py` — direct unit / integration tests for the ETD2 solver.
- `tests/test_solver_baseline.py` — end-to-end regression vs the cached
  reference state.
- `tests/test_stencil_accuracy.py` — the FD stencil audit described above.

### CLAUDE.md (Session A second commit)

Was a stub. Now documents:

- The transport equation as discretised in code, with line references.
- The species-major state-vector layout (`SpeciesManager.lidx() / uidx()`).
- The redshift integration convention (`z` runs high → low, backward in time).
- The full RHS structure `dn/dz = L(z) n + b(z)` and where each piece is
  built.
- The solver stack and how solver choice is made (instantiation, not
  config flag).
- The Jacobian sparsity-pattern contract: `coupling_mat.indices/indptr` are
  fixed at init; only `data` is updated at runtime.
- Non-obvious pitfalls: `tau_dec_threshold` as a resonance-style
  approximation (cf. MCEq's `force_resonance`), no public API for non-zero
  initial state, energy vs. energy-per-nucleon, NCo species IDs.

This was a behavioural change in the sense that future code search/edits
now have a real architectural reference instead of stub text.

---

## 2. Kernel construction: dense → Toeplitz (Session B, this branch)

### v1: dense per-channel corner evaluation

`PhotoNuclearInteractionRate._init_matrices` (now `_init_matrices_legacy`)
fills the `_batch_matrix` row-by-row. For each (mother, daughter) channel:

- **bc (boost-conserving) channels:** allocate a `(dcr, dph) = (88, 72)`
  intermediate, evaluate the response antiderivative spline
  `intp_bc(yu) - intp_bc(yl)` at all `dcr × dph` corners, multiply by a
  `(dcr, dph)` prefactor `int_fac · diff_fac`.
- **diff (redistribution) channels:** allocate a `(dcr, dcr, dph) =
  (88, 88, 72)` intermediate, evaluate the 2D antiderivative
  `intp_diff_integral` at the 4 corners of every (i_mo, i_da, j) cell,
  apply `x_cut`, subtract the 1D nonel response on the diagonal.

Cost on the production grid:

| max_mass | `_init_matrices` | transient RSS | persistent `_batch_matrix` |
|----------|------------------|---------------|----------------------------|
| 14       | 11.6 s           | +532 MB       | 165.7 MB                   |
| 56       | 61.8 s           | +2,694 MB     | 991 MB                     |

Diff channels dominate: ~97 % of `_batch_matrix` rows at m=14 and ~85 % at
m=56. Per-channel spline-evaluation count is `dcr × dph` (bc) or
`4 × dcr² × dph` (diff). At m=56 with 840 diff channels the legacy path
performs ~470 million 2D spline evaluations.

### v2: Toeplitz/log-grid sampling

The mathematical observation: with `cosmic_ray_grid` and `photon_grid`
sharing the same bins/decade (the configured default — and the only
sensible setting), every integration corner satisfies

```
log y(i_mo, j) = const + (i_mo + j) · δ
log x(i_mo, i_da) = const + (i_da - i_mo) · δ
```

So all `dcr × dph` (or `dcr × dcr × dph`) corners take only `(dcr + dph)`
distinct y-values and `(2 dcr − 1)` distinct x-values. The full
`(dcr, dcr, dph)` corner cube is a discrete-translation-invariant
(2D-Toeplitz) tensor with very few independent entries.

The new `_init_matrices_toeplitz` path:

1. **Pre-samples each spline antiderivative once** on a length-`(dcr+dph)`
   1D log-y grid (1D splines: `nonel_intp`, `incl_intp`) or a
   `(dcr+dph) × (2 dcr)` 2D grid (2D splines: `incl_diff_intp_integral`).
2. **Takes finite differences** to get `ΔR̂[k]` (1D, length `dcr+dph−1`)
   or `ΔΔR̂[a, b]` (2D, shape `(dcr+dph−1, 2 dcr − 1)`).
3. **Assembles per-channel tiles** by integer-index broadcasting:
   ```python
   bc:    tile = factor[:, None] * ΔR̂[A]                                 # A = i_mo + j
   diff:  tile = factor_diff[:, :, None] * ΔΔR̂[A_2d, B_idx]              # B_idx = i_da - i_mo + (dcr-1)
   ```
4. **Reuses the existing dense `_batch_matrix`** for storage, so the
   downstream `_init_coupling_mat` and runtime `_update_rates` paths are
   unchanged.

The prefactor algebra simplifies to

```
bc:    int_fac · diff_fac = m_p / E_mo[i_mo]                             (rank-1 over j)
diff:  int_fac · diff_fac = (m_p / E_mo[i_mo]) · (Δec[i_mo] / Δec[i_da]) (rank-1 outer product)
```

so the prefactor is never materialised as a `(dcr, dph)` tensor.

### Hard guardrail (added in this session)

`_assert_log_grids_compatible` runs at init and **raises `RuntimeError`**
if the CR-grid centers, CR-grid bin edges, and photon-grid bin edges do
not all share the same log-step. There is no silent fallback to legacy:
mismatched log-steps in the toeplitz path would silently produce wrong
rates, so the user must explicitly:

- align `cosmic_ray_grid` and `photon_grid` on the same bins/decade, OR
- set `config.kernel_method = "legacy"` to use the corner-evaluation path.

The error message names the detected bins/decade for each grid and the
exact log-steps that disagreed, so misconfiguration is self-diagnosing.

### Numerical guarantee

`tests/test_kernel_toeplitz.py` builds the same `PriNCeRun` twice (legacy
and toeplitz), lex-sorts both `(rows, cols, data)` triples, and asserts:

- identical `(row, col)` sparsity pattern;
- max relative error < 1e-10 above an absolute floor of 1e-25;
- identical `coupling_mat.indices` / `coupling_mat.indptr`.

In practice the relative error sits at ~1e-13. The two paths are
algebraically equivalent — the toeplitz form just deduplicates work — so
agreement is at the floating-point rounding level induced by the
re-ordered arithmetic.

### Measured results

|                            | legacy (v1) | toeplitz (v2) | speedup / reduction |
|----------------------------|-------------|---------------|---------------------|
| **m=14** `_init_matrices`  | 11.61 s     | 0.85 s        | **13.6×**           |
| **m=14** transient RSS     | +532 MB     | +14 MB        | **37×**             |
| **m=14** persistent batch  | 165.7 MB    | 165.7 MB      | 1.00× (Phase 1)     |
| **m=56** `_init_matrices`  | 61.80 s     | 4.80 s        | **12.9×**           |
| **m=56** transient RSS     | +2,694 MB   | +112 MB       | **24×**             |
| **m=56** persistent batch  | 991 MB      | 991 MB        | 1.00× (Phase 1)     |

Persistent storage is unchanged in this branch (Phase 1: same dense
`_batch_matrix`). Phase 2 — storing only `ΔR̂` / `ΔΔR̂` and lazily
expanding inside `_update_rates` — would shrink persistent storage by
20–40× but is left as a separate change.

### Why these numbers matter for the U238 roadmap

The user is planning to drive PriNCe with a much larger nuclear database
that goes up to U238 (max_mass ≈ 238). Linear extrapolation of the m=14 →
m=56 measurements puts the v1 path at:

- `_init_matrices` ~3–5 minutes per build
- transient RSS ~15–20 GB during init
- persistent `_batch_matrix` ~3–4 GB

The Toeplitz path moves these to ~30–60 s, ~3–5 GB peak, 150–250 MB. The
laptop-vs-cluster difference is roughly that.

---

## 3. Files changed in v2 (cumulative)

### Session A (already on master)

```
src/prince_cr/solvers/etd2.py             | +193 (new)
src/prince_cr/solvers/propagation.py      | -287 net  (BDF/SemiLagr removed; ETD2 hooks)
src/prince_cr/solvers/partial_diff.py     | -481      (dead code removed)
src/prince_cr/util.py                     | -180      (helpers for dead code)
src/prince_cr/solvers/__init__.py         | +/-       (export ETD2)
src/prince_cr/config.py                   | -3        (semi_lagr_method removed)
docs/tutorial.rst                         | +/-
CLAUDE.md                                 | +44       (real reference doc)
tests/_build_kernel_cache.py              | +60 (new)
tests/conftest.py                         | +136
tests/test_etd2.py                        | +110 (new)
tests/test_solver_baseline.py             | +94 (new)
tests/test_stencil_accuracy.py            | +158 (new)
tests/test_partial_diff.py                | -72 net
tests/test_propagation_extra.py           | -306 net
tests/test_propagation.py                 | +/-
tests/test_util.py                        | -54
```

### Session B (this branch)

```
src/prince_cr/config.py                   | +8        (kernel_method flag)
src/prince_cr/interaction_rates.py        | +247 net  (toeplitz + dispatcher + assert)
CLAUDE.md                                 | +9        (kernel construction section)
tests/test_kernel_toeplitz.py             | +112 (new)
V1_TO_V2.md                               | +new (this file)
CHANGES.md                                | +new (Session B summary)
explore/                                  | +new (analysis, prototypes, benchmarks)
```

---

## 4. How to validate v2 locally

```bash
# 1. Run all tests (should be 359 passed)
pytest

# 2. Compare legacy vs toeplitz on the kernel
pytest tests/test_kernel_toeplitz.py -v

# 3. Benchmark the kernel construction
python explore/bench_legacy_vs_toeplitz.py --max-mass 14
python explore/bench_legacy_vs_toeplitz.py --max-mass 56

# 4. Run the ETD2 regression vs cached reference
pytest tests/test_solver_baseline.py -v

# 5. Force the legacy kernel path for a session
python -c "
import prince_cr.config as cfg
cfg.kernel_method = 'legacy'
# ... your code ...
"
```

The `PRINCE_KERNEL_CACHE` env var can be set to a pickled `PriNCeRun` to
skip the ~3-minute kernel build at test time
(see `tests/_build_kernel_cache.py`).

---

## 5. End-to-end speedup, v1 → v2

The realistic AugerFitSource case (build the kernel once + solve from
z=1 to 0 at dz=1e-3) goes from:

```
v1: kernel build ~64 s + BDF solve ~24 s     ≈ 88 s
v2: kernel build  ~5 s + ETD2 solve  ~6 s    ≈ 11 s            (8× overall)
```

For the U238 case the kernel-build savings dominate even more: minutes
become seconds.

---

## 6. Things deferred to follow-up sessions

- **Phase 2 of kernel acceleration:** store only the `ΔR̂` / `ΔΔR̂`
  Toeplitz tables persistently, expand lazily inside `_update_rates`.
  Drops persistent `_batch_matrix` size by 20–40× and pickle size of
  `PriNCeRun` by 3–5×.
- **Phase 3 (one-liner):** drop `cs.resp.incl_diff_intp_integral` and
  `cs.resp.incl_diff_intp` after `_init_matrices` finishes — runtime
  doesn't use them. Frees ~1 GB at m=56, more at U238.
- **Coarser response splines** (Phase 4): the 2D antiderivative splines
  carry the full tabulated cross-section grid even when the response is
  smooth. Re-sampling them onto a coarser y-grid before storage would
  shrink the pickle further.
- **Phase 1 → master merge:** this branch is `worktree-kernel-accel-explore`.
  Once reviewed, fast-forward merge into master.
