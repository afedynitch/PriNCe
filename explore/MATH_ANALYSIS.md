# Cross-section integration: math model and acceleration opportunities

## What "kernel construction" means here

`PhotoNuclearInteractionRate._init_matrices` (`src/prince_cr/interaction_rates.py:97`) builds
the **batch matrix** `B[k, j]`. At runtime, the photo-hadronic part of the rate matrix is just

```
coupling_mat.data = scale_fac * (B @ ph_vec(z))                  (one SpMV per z-update)
```

Each row `k = (channel, i_mo)` (or `(channel, i_mo, i_da)` for redistributions) holds the
**bin-averaged response of one channel** evaluated against the photon energy bins:

```
B[k, j] = (1 / Δx · Δy) · ∫_{x_l(i)}^{x_u(i)} dx ∫_{y_l(i,j)}^{y_u(i,j)} dy   R_ch(x, y)
       · int_fac(i, j)                                                       (1)
```

with `y = E_ph · E_mo / m_p` and `x = E_da / E_mo`. The integral is evaluated as a finite
difference of an **antiderivative spline** `R̂_ch`. So the cost is dominated by 1D / 2D
spline evaluations on (dcr × dph) or (dcr × dcr × dph) point grids per channel.

## Empirical baseline (full production grid)

| Stage / quantity                | max_mass=14   | max_mass=56   | scaling 14→56  |
|---------------------------------|---------------|---------------|----------------|
| known_species                   | 25            | 93            | 3.7×           |
| known_bc_channels               | 86            | 2,892         | **34×**        |
| known_diff_channels             | 160           | 840           | 5.3×           |
| _init_matrices wall-clock       | 12.0 s        | **63.6 s**    | 5.3×           |
| batch_matrix rows               | 287,664       | 1,720,512     | 6.0×           |
| ↳ from bc                       | 7,568 (3 %)   | 254,496 (15 %)|                |
| ↳ from diff                     | 280,096 (97 %)| 1,466,016 (85 %)|              |
| batch_matrix size               | 165.7 MB      | **991 MB**    | 6.0×           |
| incl_diff_intp pickle           | 105 MB        | 551 MB        | 5.3×           |
| incl_diff_intp_integral pickle  | 105 MB        | 551 MB        | 5.3×           |
| peak RSS during init            | 1.5 GB        | **5.35 GB**   | 3.6×           |

For an extrapolated **max_mass=238 (U238) build** with ~250 species, bc channel count
grows roughly with the catalog of (mother, daughter) disintegration pairs (~8 k–10 k
estimated). Diff channels (mostly photo-meson + lepton secondaries) scale more gently
with the species list — call it ~2 k–3 k. Linear extrapolation of the table above:

  - batch_matrix ≈ 3–4 GB (current code, before any optimisation)
  - incl_diff_intp + integral ≈ 3 GB on disk
  - _init_matrices wall-clock ≈ 3–5 minutes
  - peak RSS ≈ 15–20 GB during init

Diff (redistribution) channels still dominate batch storage and per-channel build time;
**bc channels dominate row-count growth** (34× from m=14 to m=56) and so will dominate
sparsity-pattern construction at U238 scale.

## The structural identity that makes this cheap

### Step 1: prefactors collapse to a 1D function of i_mo

In `_init_matrices`, the factor `int_fac · diff_fac` simplifies to

```
int_fac = Δec[i_mo] · Δph[j] / E_mo[i_mo]
diff_fac = 1 / (Δec[i_da]/E_mo[i_mo]) / (Δph[j]·E_mo[i_mo]/m_p)   # diff
         = m_p / (Δec[i_da] · Δph[j])                              # diff
         = m_p / (Δec[i_mo] · Δph[j])                              # bc (i_da = i_mo)

int_fac · diff_fac = m_p / E_mo[i_mo]                              (2)
```

This is **independent of j**, dependent only on i_mo. The (dcr, dph) prefactor tensor that
the current code builds is rank-1.

### Step 2: log-grid structure of x and y

Both grids are log-spaced with the **same log-step** Δ (8 bins/decade for both, see
`config.py:65,67`). Let `δ ≡ log10(Δ_step) = 1/8`.

```
log y_l(i_mo, j) = log E_mo,0 + i_mo·δ + log b_ph,0 + j·δ - log m_p
                ≡ const_y + (i_mo + j)·δ                          (3a)

log y_u(i_mo, j) = const_y + (i_mo + j + 1)·δ                     (3b)

log x_l(i_mo, i_da) = log b_cr,0 - log E_mo,0 + (i_da - i_mo)·δ
                   ≡ const_x + (i_da - i_mo)·δ                    (4a)

log x_u(i_mo, i_da) = const_x + (i_da - i_mo + 1)·δ               (4b)
```

So **all** the (i_mo, i_da, j) integration corners live on a 1D log-grid in y and a 1D
log-grid in x — both indexed by simple integer offsets.

### Step 3: rewrite the kernel

For boost-conserving (bc) channels, after substituting (2), (3):

```
B_bc[ch, i_mo, j] = (m_p / E_mo[i_mo]) · ΔR̂_ch[i_mo + j]            (5)

where  ΔR̂_ch[k] = R̂_ch(y_grid[k+1]) - R̂_ch(y_grid[k])
       y_grid[k] = exp(const_y + k·δ ln 10),  k = 0 … dcr+dph-1
```

`ΔR̂_ch` is a **1D vector of length (dcr + dph)** per channel. The current code stores
the full (dcr, dph) tensor — i.e. it materialises a Toeplitz matrix that only has
(dcr + dph) distinct values.

For differential (diff) channels:

```
B_diff[ch, i_mo, i_da, j] = (m_p / E_mo[i_mo]) · ΔΔR̂_ch[i_mo + j, i_da - i_mo]    (6)

where ΔΔR̂_ch[a, b] = R̂_ch(y[a+1], x[b+1]) - R̂_ch(y[a], x[b+1])
                    - R̂_ch(y[a+1], x[b])   + R̂_ch(y[a], x[b])
```

Indices: `a ∈ [0, dcr+dph-1]`, `b ∈ [-(dcr-1), dcr-1]`. The full (dcr, dcr, dph) tensor
the current code materialises has **at most (dcr+dph) · (2 dcr - 1)** distinct values
— a ~20× factor for the production grid (88, 88, 72).

## What this buys us

### Compute (init time)

| Step                                  | Current                | Toeplitz                  | Ratio            |
|---------------------------------------|------------------------|---------------------------|------------------|
| 1D antiderivative evals per bc chan   | 2 · dcr · dph = 12 672 | dcr + dph + 1 = 161       | **~80×**         |
| 2D antideriv. evals per diff chan     | 4 · dcr² · dph ≈ 2.2 M | (dcr+dph)·(2 dcr-1) ≈ 28k | **~80×**         |
| Per-channel storage                   | dcr · dph (bc) /       | dcr+dph (bc) /            | ~40× / ~20×      |
|                                       | dcr²·dph/2 (diff)      | (dcr+dph)(2dcr-1) (diff)  |                  |

Actual init speedup will be smaller than the ideal because Python loop overhead and
prefactor work don't shrink as much. Expect **~5–10× wall-clock improvement** on the
inner loops, dominated by:
  - SciPy spline evaluation overhead per call (cuts down with vectorisation)
  - The cuts (`x_cut`, `x_cut_proton`) still have to be applied per (i_mo, i_da)

### Memory: kernel storage

For `max_mass=14` production grid:
  - bc rows: 7568 × 72 × 8 = 4.4 MB → with (dcr+dph) packing: 86 · 161 · 8 ≈ 110 KB (40×)
  - diff rows: 280k × 72 × 8 = 161 MB → with Toeplitz packing: 160 · 28k · 8 ≈ 36 MB (~5×)

For an extrapolated `max_mass=238` (U238) build:
  - assume ~10× channel counts → batch_matrix ≈ 1.6 GB current vs ~360 MB Toeplitz
  - peak transient memory in _init_matrices drops by the same factor

### Memory: response interpolator footprint

`incl_diff_intp_integral` is 105 MB at max_mass=14, and is just the bivariate antiderivative
tabulated on the response grid (typically (xcenters, ygrid) where ygrid has ~hundreds of
points and xcenters ~31). At max_mass=238 → ~1 GB on disk for these splines alone.

We don't strictly need to keep the full splines after kernel construction — once we've
sampled `ΔΔR̂_ch[a, b]` for the actual grid corners, the splines can be discarded for
runtime evaluations. **The only reason to keep them is `_update_rates` … which doesn't
use them.** So we can free them post-init.

## What changes about runtime SpMV?

`_update_rates` does:
```python
np.dot(self._batch_matrix, self.photon_vector(z), out=self._batch_vec)
```

This is the **dense GEMV** of B against the photon vector. With Toeplitz structure, we
could go further:

```
out_bc[ch, i_mo] = (m_p / E_mo[i_mo]) · sum_j ΔR̂_ch[i_mo + j] · ph[j]
                = (m_p / E_mo[i_mo]) · cross_corr(ΔR̂_ch, ph)[i_mo]
```

This is a 1D cross-correlation; `np.convolve` or a single `scipy.signal.fftconvolve` per
channel batched together is O(N log N). For diff channels the structure is 2D-Toeplitz
in (a, b), reducible to two 1D correlations or a single 2D one.

**But** — we should be careful: the runtime SpMV is only ~1 BLAS call already, and the
cache window (`update_rates_z_threshold`) means it runs maybe ~100 times per solve. The
init-time savings are the bigger lever; runtime SpMV optimization is bonus.

## Risks and edge cases

1. **Different log-steps** between cr_grid and ph_grid would break the i+j index trick.
   Currently both are 8 bins/decade — but the code allows them to differ. We need to
   either assert equal steps OR use a 2D rescale lookup. The latter still gives most
   of the savings (one antideriv evaluation per (i_mo+j)·δ_y rather than per (i_mo, j)).

2. **`grid_scale = "logE"`**: the alternative grid scale (`config.py:72`) needs separate
   handling; currently default and only used path is `'E'` (log-spaced energy).

3. **`x_cut` / `x_cut_proton` cuts** drop entries based on x ≥ cut. With Toeplitz, this
   becomes a cut on `(i_da - i_mo)` index — easy. The current code already uses
   `cuts[:, :, 0]` which is independent of j; this carries over.

4. **The nonel subtraction on the diagonal** (lines 250–256 of `interaction_rates.py`):
   when `mo == da` and the channel is differential, the nonel cross-section is
   subtracted on `i_mo == i_da`. With Toeplitz indexing this is `b = 0` only.

5. **`_init_coupling_mat` lexsort** still works — we'd produce the same `(rows, cols)`
   index arrays, just with a different (smaller) data array. The CSR builder downstream
   doesn't care.

## Proposed plan

1. **Phase 1 (low-risk)**: keep the dense `_batch_matrix` interface but rewrite
   `_init_matrices` to use the Toeplitz identity internally. We still produce the same
   `(rows, cols, data)` triple at the end, so nothing else has to change. Expected wins:
   3–10× faster init, ~50% transient memory drop.

2. **Phase 2 (storage win)**: replace the dense `_batch_matrix` with per-channel ΔR̂ /
   ΔΔR̂ vectors plus an index map. Keep `coupling_mat.data` API so all solver code
   still works. Expected: 5–10× smaller persistent memory; same SpMV semantics (we
   "expand" the Toeplitz matrix to dense before SpMV — or do the cross-correlation
   directly).

3. **Phase 3 (optional, runtime)**: replace the per-z dense GEMV with batched
   correlations. Only worth it if profiling shows GEMV is a hotspot at U238 scale.

4. **Phase 4 (interpolator memory)**: drop `incl_diff_intp` and `incl_diff_intp_integral`
   from the run object after `_init_matrices` finishes. The kernel doesn't need them
   for SpMV updates.

5. **Phase 5 (orthogonal)**: investigate whether the response splines can be sampled on
   a coarser y-grid. The 2D RectBivariateSpline holds knot/coeff arrays whose size is
   set by the underlying tabulated data; this is independent of (1)–(4) and could
   shrink the pickle further at U238.

## Concrete next prototype

A self-contained script that:
  - takes a real `PhotoNuclearInteractionRate` object (built once)
  - re-builds the bc-channel slice using identity (5)
  - asserts numerical agreement against the existing `_batch_matrix` rows
  - times both paths

If that lands clean, do the same for the diff channels using identity (6).
