# Changes: Toeplitz kernel construction (Phase 1)

## What changed

Added a structurally-aware path for building the photo-hadronic batch matrix
(`PhotoNuclearInteractionRate._init_matrices`). The legacy path is preserved
for benchmarking and as a fallback for incompatible grids.

### Files

- `src/prince_cr/config.py`
  Added `kernel_method = "toeplitz"` (default). Set to `"legacy"` to use the
  original implementation.

- `src/prince_cr/interaction_rates.py`
  - `_init_matrices` is now a thin dispatcher.
  - `_init_matrices_toeplitz` (new): structured construction.
  - `_init_matrices_legacy` (renamed from old `_init_matrices`): unchanged.
  - `_assert_log_grids_compatible` (new): hard guardrail. Raises
    `RuntimeError` if the CR and photon grids do not share the same
    bins/decade when toeplitz is requested. There is no silent fallback —
    mismatched grids would silently produce wrong rates, so the caller must
    either align the grids or explicitly set `kernel_method = "legacy"`.
  - The dispatcher also raises `ValueError` for unrecognised `kernel_method`
    values.

- `tests/test_kernel_toeplitz.py` (new)
  Asserts `_batch_matrix` and `coupling_mat` produced by the two paths agree
  to machine precision for `max_mass ∈ {1, 4, 14}`.

- `CLAUDE.md`
  Added a "Kernel Construction" section under Key Design Patterns.

## Mathematical content

For each channel `(mo, da)`:

```
bc:    B[i_mo, j_ph]            = (m_p / E_mo[i_mo]) · ΔR̂[i_mo + j_ph]
diff:  B[i_mo, i_da, j_ph]      = (m_p / E_mo[i_mo]) · (Δec[i_mo] / Δec[i_da])
                                  · ΔΔR̂[i_mo + j_ph,  i_da - i_mo + (dcr - 1)]
```

with the `_init_matrices` prefactor identity:
```
int_fac · diff_fac  =  m_p · Δec[i_mo] / (E_mo[i_mo] · Δec[i_da])
```

Both `ΔR̂` and `ΔΔR̂` are precomputed once per channel by sampling the response
antiderivative on a structured log-grid:
- 1D grid:  `y[k] = E_mo[0] · b_ph[0] / m_p · 10^(k δ)`, `k ∈ [0, dcr+dph)`
- 2D grid:  `y[k]` × `x[k] = b_cr[0] / E_mo[0] · 10^((k - (dcr-1)) δ)`, `k ∈ [0, 2 dcr)`

Per-channel tile assembly is then integer-index broadcasting:
```python
tile = factor[..., None] * dR[ A[i_mo, j_ph] ]                          # bc
tile = factor[..., None] * ddR[ A[i_mo, j_ph], B[i_mo, i_da] + dcr-1 ]  # diff
```

The `res[res<0] = 0` clip and the diagonal nonel subtraction in diff channels
are applied in the same order as the legacy path so per-element output matches
bit-by-bit (modulo floating-point rounding from the reorganised arithmetic;
measured max relative error ~1e-13 above the noise floor).

## Numerical guarantee

`tests/test_kernel_toeplitz.py` builds the same `PriNCeRun` twice (legacy and
toeplitz) and verifies:

  - identical `(row, col)` sparsity pattern after lex-sort
  - max relative error < 1e-10 for every entry above the 1e-25 absolute floor
  - identical `coupling_mat.indices` / `coupling_mat.indptr`

All 356 existing tests pass with `kernel_method = "toeplitz"` as the default.

## Performance (measured)

### max_mass=14, production grid (cr=8 bins/dec, ph=8 bins/dec)

|                                       | legacy   | toeplitz | ratio  |
|---------------------------------------|----------|----------|--------|
| `_init_matrices` wall-clock           | 11.61 s  | 0.85 s   | **13.6×** |
| transient RSS during init             | +532 MB  | +14 MB   | **37×**   |
| persistent `_batch_matrix`            | 165.7 MB | 165.7 MB | 1.00×  |

(persistent storage is unchanged in Phase 1 — same dense `_batch_matrix`)

### max_mass=56, production grid (cr=8 bins/dec, ph=8 bins/dec)

|                                       | legacy    | toeplitz | ratio  |
|---------------------------------------|-----------|----------|--------|
| `_init_matrices` wall-clock           | 61.80 s   | 4.80 s   | **12.9×** |
| transient RSS during init             | +2,694 MB | +112 MB  | **24×**   |
| persistent `_batch_matrix`            | 991 MB    | 991 MB   | 1.00×  |

Channel mix at m=56: 93 species, 2,892 bc channels, 840 diff channels.

The 12–14× speedup is consistent across both grids and is set by the ratio of
spline evaluations: per-channel `dcr × dph` (legacy) vs `dcr + dph` (toeplitz)
for bc, and `4 · dcr² · dph` vs `(dcr+dph) · 2 dcr` for diff. The transient
RSS savings come from never materialising the full `(dcr, dcr, dph)` tile.

## Risks and assumptions

The toeplitz path requires CR-grid centers, CR-grid bin edges, and photon-grid
bin edges to all be log-spaced with the same log-step. This is the case for
every `cosmic_ray_grid` / `photon_grid` pair where the third tuple element
(bins/decade) matches. The default config has both at 8 bins/decade.

`_assert_log_grids_compatible()` enforces this at init. If it fails — e.g.
someone configures `cosmic_ray_grid = (3, 14, 8)` and `photon_grid = (-15, -6, 4)` —
**`_init_matrices` raises `RuntimeError`** with a message that names the
detected bins/decade for each grid and tells the user to either align them or
set `config.kernel_method = "legacy"` explicitly. There is no silent fallback,
because mismatched log-steps in the toeplitz path would silently produce
wrong rates.

## What's NOT changed (Phase 1 scope)

- `_batch_matrix` storage layout. The dense form is still materialised and
  used as-is by `_update_rates`. Phase 2 (per-channel `dR` / `ddR` storage
  with on-demand expansion) would shrink persistent storage by 5–40×.
- The runtime SpMV in `_update_rates`. Same single `np.dot(...)` against
  `photon_vector(z)`.
- The pickle of `PriNCeRun`. The 2D `incl_diff_intp_integral` splines are
  still kept after init, even though `_update_rates` doesn't use them.
  Dropping them post-init (Phase 3, one line) would free ~1 GB at m=56 alone
  but is left as a separate change.

## Reproducing the benchmark

```bash
# from the worktree root
python explore/bench_legacy_vs_toeplitz.py --max-mass 14
python explore/bench_legacy_vs_toeplitz.py --max-mass 56
python explore/profile_kernel_init.py --max-mass 14   # detailed breakdown
```

## Validating against legacy

```bash
pytest tests/test_kernel_toeplitz.py -v
```

To force the legacy path globally for a session:
```python
import prince_cr.config as cfg
cfg.kernel_method = "legacy"
```
