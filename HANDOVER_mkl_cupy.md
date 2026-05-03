# Hand-over: MKL / CuPy backend decision

Deferred from the deep-review pass on 2026-05-03. Pick this up on a machine
that has both `libmkl_rt` and `cupy` installed; the choice between options
A/B/C below requires actually exercising the GPU and MKL paths, which the
default macOS/conda env can't do.

## What is and isn't broken right now

Already cleaned up in this branch (see git log around the deep-review commit):

- `mkl_interface.py` — **deleted** (BLAS L2/L3 wrappers used only by the
  now-deleted `eqn_deriv_mkl` RHS).
- `UHECRPropagationSolverEULER` and its `eqn_deriv_standard / _cupy / _mkl`
  variants — **deleted**. ETD2 is the only solver.
- `_update_jacobian` (base class) — **deleted**.

What's left and in what state:

| Where | What it does | Tested? | Likely state |
|---|---|---|---|
| `interaction_rates.py:10-15` | Sets `using_cupy = True` if cupy + backend selects it | no | should work |
| `interaction_rates.py:542-555` | `_init_coupling_mat`: uploads `_batch_matrix` and `coupling_mat` to GPU as float32 | no | should work |
| `interaction_rates.py:570-573` | Reorders `_batch_matrix` lex-sort on GPU | no | should work |
| `interaction_rates.py:592-600` | `_update_rates`: `cupy.dot(_batch_matrix, photon_vector)` | no | should work, this is the actual GPU win |
| `partial_diff.py:16-19` | Converts `diff_operator` to `cupyx.scipy.sparse.csr_matrix` | no | **likely broken** under ETD2: `solvers/etd2.py:split_operator` calls `sp.diags(...)` and expects a scipy CSR, not cupy |
| `config.py:139` | `linear_algebra_backend = "MKL"` default | n/a | **dead config knob** — no code path consumes it anymore |
| `config.py:142-153, 183, 191-194, 211-216` | MKL/cupy autodetect | partly | `has_mkl` and `mkl_path` no longer drive any decision; `kernel_config` is just printed |
| `config.py:199-212` | `set_mkl_threads(...)` | no | useful even without our own MKL — controls thread count for **scipy's** sparse BLAS when scipy is linked against MKL (typical conda numpy) |

## The actual question

After the pair-prod cache fix, the ETD2 hot path on production grid is:

```
_refresh_z_caches (~100x/solve):  60% of time
  └─ np.dot(_batch_matrix, photon_vector)   <-- dense × vec, 165 MB matrix
4 csr_matvec / step (~4000x/solve): 33% of time
  └─ scipy sparse SpMV on coupling_mat       <-- sparse × vec, dim_states-sized
```

The 60% piece is dense `(n_batch, dph) @ (dph,)` — would benefit a lot
from GPU if the matrix lives there permanently (which the existing cupy
path already does).

The 33% piece is sparse — harder to GPU-accelerate well.

So the upside of the existing cupy path is real **for the cache rebuild
only**. The cupy path does NOT touch the per-step SpMV.

## Three options

### A. Rip everything out (my recommendation as of the review)

Delete:
- `interaction_rates.py` lines 10-15, 542-555, 570-573, 592-600 (the cupy
  branches in `_init_coupling_mat` and `_update_rates`)
- `partial_diff.py:16-19` (cupy conversion of diff operator)
- `config.py`: `linear_algebra_backend`, `has_mkl`, `mkl_path`, `mkl`,
  `kernel_config`, `has_cupy`, `set_mkl_threads`, the autodetect block
- All `import cupy` / `using_cupy` references

**Keep**: nothing.

**Pros**: simplest, no untested code, no broken cupy-in-partial_diff bug
waiting to happen, ~80 lines smaller.

**Cons**: gives up the rate-rebuild GPU speedup. If you actually want GPU
later, do it properly with a known-good cupy install.

### B. Keep cupy in `interaction_rates.py`, drop the rest

Delete:
- `partial_diff.py:16-19` (the broken cupy CSR conversion)
- `config.py`: `linear_algebra_backend`, `has_mkl`, `mkl_path`, `mkl`,
  `kernel_config`, the MKL-specific autodetect
- Rename `set_mkl_threads` → `set_blas_threads` (it controls scipy's
  threading too via the underlying MKL runtime when scipy is linked
  against it)

**Keep**: the cupy rate-rebuild path in `interaction_rates.py`, gated on
a simpler `use_gpu` flag instead of `linear_algebra_backend == "cupy"`.

**Pros**: preserves the only piece that could give a real speedup. Drops
the dead MKL config without losing the BLAS-thread knob.

**Cons**: keeps untested code. Need to actually verify GPU path against
scipy reference at production grid before trusting it.

### C. Keep status quo

Just merge the deep-review work as-is. No further changes.

**Pros**: zero risk to anything that currently works.

**Cons**: leaves dead config (`linear_algebra_backend = "MKL"` doesn't
do anything anymore), leaves the broken `partial_diff.py:16-19`
cupy-CSR conversion path latent, leaves untested cupy paths in the
hot rate-rebuild.

## Test plan when you do this

Whichever option, on the cupy/mkl machine:

1. **Establish parity baseline**: run the full ETD2 frozen-baseline test
   with scipy backend; record `state.copy()`. This is the reference.

2. **For option B (the only one with non-trivial behaviour)**:
   - run the same solve with cupy enabled, check `state` agrees with
     the scipy reference to ~1e-6 relative
   - profile the cache rebuild on production grid (`max_mass=56`) to
     measure the actual GPU speedup vs scipy
   - benchmark must include first-step warm-up cost (cupy JIT)
   - check that `partial_diff.py:16-19` actually breaks under ETD2 as
     I suspect, or works (in which case keep it)

3. **For options A and B**: confirm `set_mkl_threads(1)` vs
   `set_mkl_threads(N)` actually changes scipy SpMV wall time on the
   target machine. If it does, keep the knob (renamed) regardless of
   choice. If it doesn't, the entire MKL detection is just clutter.

4. **In all cases**: rerun `pytest -q` after the change. Should still
   be 355 tests passing.

## Files to touch — quick reference

```
src/prince_cr/interaction_rates.py    (cupy in _init_coupling_mat + _update_rates)
src/prince_cr/solvers/partial_diff.py (cupy CSR conversion at line 16-19)
src/prince_cr/config.py               (autodetect block, set_mkl_threads, defaults)
```

No changes to `solvers/etd2.py`, `solvers/propagation.py`,
`cross_sections/`, `data.py`, `photonfields.py`, `cosmology.py`.

## Bench reference numbers (current, scipy backend, M-series Mac)

production grid (8 bins/decade), max_mass=14, dim_states=2024,
z=1→0, dz=1e-3, recomp=0.01:

```
ETD2 default config:                  0.99 s
ETD2 tight cache (recomp=1e-3, dz=2e-4): 7.83 s
```

Full breakdown in `tests/test_etd2.py` and `tests/test_solver_baseline.py`.
