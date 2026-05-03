# Hand-over: pending items from the deep-review pass

This is the work that was not done in the deep-review session of 2026-05-03.
Each section is independent unless noted. See git log around the same date
for what _was_ done (dead-code cleanup, ETD2 pair-prod cache, EULER and
RHS-variant deletion).

## Quick wins (each is a one-shot edit, < 30 minutes)

1. **`partial_diff.py:79`** — replace `coo_matrix(np.diag(1/self.egrid).dot(op_matrix))`
   with `op_matrix * (1/self.egrid)[:, None]` then convert. Avoids the
   dense `dim_e × dim_e` materialisation. Cosmetic at production size,
   but it's wrong-on-the-face-of-it.
2. **`interaction_rates.py:56-69`** — the `photon_vector` docstring claims
   "Return value from cache if redshift value didn't change since last
   call" but the function does no caching. Either delete that line or
   add the cache it claims to have.
3. **`interaction_rates.py:613`** — `coupling_mat.data = scale_fac * self._batch_vec`
   has no assertion that the sparsity pattern is intact, even though
   CLAUDE.md documents the contract. Add `assert self.coupling_mat.data.size == self._batch_vec.size`
   or a one-line comment pointing to the contract.
4. **`PrinceSpecies` indexing API (`data.py:352-404`)** — 5 overlapping
   accessors: `sl`, `lidx`, `uidx`, `lbin`, `ubin`, `indices`. The rest
   of the codebase mixes three of them. Pick canonical ones (`sl` for
   slices, `lidx/uidx` for ints), keep `lbin/ubin` as the bin-edges
   variant, delete `indices`. Update call sites.
5. **`response.py:34`** — `is_differential` forwards via
   `CrossSectionBase.is_differential(self, ...)` with a "might break"
   comment. Either inline the check or define it as a real abstract
   method on the base.
6. **`core.py:103-109`** — `set_photon_field` fan-out across 5
   sub-objects. Either replace `self.photon_field` storage in
   `interaction_rates.py:28, 708` with a property reading through
   `prince_run.photon_field`, or keep the fan-out and document why.
7. **`interaction_rates.py:625-640, 814-823`** — `single_interaction_length`
   / `single_loss_length` save-and-restore `self.photon_field` via
   instance variable. Pass `pfield` through as a parameter rather than
   mutating `self`.

Items 1, 2, 3 are completely safe (no behaviour change). Items 4-7 touch
public-ish APIs and need test runs.

## Bigger refactors (each needs a planning pass)

8. **Unify the two `_init_matrices` paths** (`interaction_rates.py:165-346`
   toeplitz, `:348-534` legacy). The legacy path is justifiably kept for
   benchmarking and heterogeneous grids, but the channel-iteration
   scaffolding (~50 lines) is shared and could be lifted. Needs a
   bin-by-bin parity test first; some of `test_kernel_toeplitz.py` is
   already there.
9. **`cross_sections/base.py:_reduce_channels`** (~250 lines, 4-level
   recursion, closure with 8 outer captures, LRU cache built inside the
   function, scattered debug prints). Lift the closure to a class with
   explicit state; split the recursion's two arms
   (differential / non-differential daughters) into separate methods.
   Needs a regression test on the resulting `_incl_tab` first — there
   isn't one today.
10. **`cross_sections/photo_meson.py`** mixes SOPHIA superposition,
    decay scaling, and the empirical model in one file. The
    `EmpiricalModel._fill_multiplicity` placeholder-arrays trick
    (`:332-368`) signals a leaky abstraction with the base class.
    Move `EmpiricalModel` to its own module; figure out what the
    placeholder is really for.
11. **`ContinuousAdiabaticLossRate` and `ContinuousPairProductionLossRate`**
    (`interaction_rates.py:644-824`) share their `_init_energy_vec` /
    `_init_scale_vec` skeletons. Extract a base class.

## Deferred to its own session

12. **MKL/CuPy infrastructure** — see `HANDOVER_mkl_cupy.md` in this
    repo. Specific options A/B/C, file-and-line list, test plan. Needs
    a machine with both libs.

## Long-term / architectural

13. **Module-level globals for physics + grids in `config.py`** —
    impossible to instantiate two `PriNCeRun`s with different
    cosmologies in the same process. Tests work around with
    `importlib.reload`. Right fix: make config a per-`PriNCeRun`
    object, but it's a backwards-incompatible API change.
14. **`config.py` does HDF5 download at import time** — multi-second
    I/O latency on first import, error handling is awkward. Move to
    lazy-on-first-use.
15. **Tests rely on multi-GB cached pickle** (`tests/_build_kernel_cache.py`).
    CI story isn't great. Either check in a smaller fixture, or
    rebuild-on-CI with a smaller production grid.

## Intentionally kept (review flagged but left alone)

- `UHECRPropagationResult.to_dict / from_dict` — has tests in
  `test_propagation_result.py:85-93`.
- `partial_diff.py:solve_coefficients` — only the error path is tested,
  but the function is a public utility; `np.math.factorial` already
  fixed.
- The legacy `_init_matrices` path — preserved per CLAUDE.md rationale.

## Snapshot at hand-over

```
ETD2 default solve (production grid 8 bins/decade, max_mass=14,
                    dim_states=2024, z=1→0, dz=1e-3, recomp=0.01,
                    M-series Mac, scipy backend):
    0.99 s
ETD2 tight cache (recomp=1e-3, dz=2e-4):
    7.83 s
Per-step breakdown (% of solve):
    cache rebuild (100 calls):  60%
    4 csr_matvec / step:        33%
    adia loss_vector (per-step): 7%
```

355 tests pass.

When picking this up, first run `pytest -q` to confirm the baseline is
still 355 green before starting.
