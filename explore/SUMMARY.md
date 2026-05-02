# Summary: cross-section integration acceleration

## TL;DR

The dominant cost in `PhotoNuclearInteractionRate._init_matrices` is **redundant
spline evaluation**: the same response function is sampled at points whose values
all lie on a single 1D log-grid (or 2D log-grid for redistributions), but the
current code evaluates at every (i_mo, i_da, j_ph) corner independently.

Reformulating the integration to **exploit the shared log-step between the CR
and photon grids** gives — measured on the production grid (88, 88, 72) at
max_mass=14:

| Metric                       | Current   | Toeplitz reformulation | Ratio |
|------------------------------|-----------|------------------------|-------|
| **bc** channel build, 97 ch  | 21.6 ms   | 4.4 ms                 | 5.0×  |
| **bc** storage (per channel) | 50,688 B  | 1,280 B                | 39.6× |
| **diff** channel build (30)  | 2,774 ms  | 197 ms                 | 14.1× |
| **diff** storage (per chan)  | 4,460,544 B | 224,000 B            | 19.9× |
| Numerical agreement          | —         | max rel err 4e-14      | exact |

Both paths sample the same antiderivative spline, so disagreement is at
machine-epsilon level (3e-12 for bc, 4e-14 for diff).

Extrapolated to a U238 build (max_mass=238):

|                               | Current (extrapolated) | After Toeplitz |
|-------------------------------|------------------------|----------------|
| `_init_matrices` wall-clock   | ~3–5 min               | ~30–60 s       |
| `_batch_matrix` size          | ~3–4 GB                | ~150–250 MB    |
| Peak RSS during init          | ~15–20 GB              | ~3–5 GB        |
| Pickle of full PriNCeRun      | ~10 GB                 | ~3 GB          |

The 5–14× wall-clock improvement comes purely from doing fewer spline
evaluations; the 20–40× memory improvement comes from not materialising the
full Toeplitz matrix.

## Why this works (one paragraph)

The kernel block for one channel is

```
B[i_mo, i_da, j] = factor(i_mo, i_da) · ΔΔR̂_ch(xu, xl, yu, yl)
```

with `y = E_ph · E_mo / m_p`, `x = E_da / E_mo`. With logarithmic CR-energy and
photon grids that share the same `bins/decade` (the configured default), the
integration corners satisfy:

```
log y_l(i_mo, j) = const + (i_mo + j) · δ
log x_l(i_mo, i_da) = const + (i_da - i_mo) · δ
```

So the entire response sampling lives on a 1D log-y grid of size `dcr+dph` (bc)
or a 2D log-(y,x) grid of size `(dcr+dph) × (2 dcr - 1)` (diff). A single
batched spline evaluation fills the whole channel; the Toeplitz/translation-
invariant structure is exposed to memory and to any downstream FFT or
correlation if we want it later.

The prefactor `int_fac · diff_fac` simplifies to:
- bc:    `m_p / E_mo[i_mo]`           (rank-1, function of i_mo only)
- diff:  `m_p / E_mo[i_mo] · Δec[i_mo] / Δec[i_da]`  (rank-1 outer product)

both of which are constant arrays computed once per kernel.

## What's in this worktree

- `explore/MATH_ANALYSIS.md` — full derivation, baseline measurements, risks.
- `explore/profile_kernel_init.py` — reproducible profiler for any (max_mass, grid).
- `explore/profile_m56.log` — measurements for max_mass=56.
- `explore/prototype_bc_toeplitz.py` — bc-channel reformulation, validated on m=14.
- `explore/prototype_diff_toeplitz.py` — diff-channel reformulation, validated on m=14.

Run any of the profilers with `python explore/<name>.py --max-mass N` from the
worktree root.

## Observed empirical scaling

| Grid                                 | m=14  | m=56     | factor |
|--------------------------------------|-------|----------|--------|
| nspec                                | 25    | 93       | 3.7×   |
| n_bc_channels                        | 86    | 2,892    | 34×    |
| n_diff_channels                      | 160   | 840      | 5.3×   |
| `_init_matrices` time                | 12 s  | 64 s     | 5.3×   |
| batch_matrix bytes                   | 166 MB| 991 MB   | 6.0×   |
| incl_diff_intp_integral pickle       | 105 MB| 551 MB   | 5.3×   |

The number of bc channels grows roughly with the catalog of (mother, daughter)
disintegration channels — fastest. Diff channels grow more slowly because
they're mostly `(nucleus, secondary meson/lepton)`. _init_matrices time scales
with diff-channel count × dcr² × dph spline evals, which is why 5.3× = ratio
of diff-channel counts.

## Recommended implementation roadmap (in priority order)

1. **(Phase 1, 1–2 days)**  Replace the inner loops of `_init_matrices` with
   the Toeplitz path. Keep producing the same `(rows, cols, data)` triple
   (i.e. expand the Toeplitz form to dense before stuffing into
   `_batch_matrix`). Zero behavioural change downstream; expected:
   - 5–14× init speedup (production grid)
   - peak RSS during init drops by ~3× (no more (dcr,dcr,dph) intermediates)
   - storage of `_batch_matrix` unchanged
   - pickled `PriNCeRun` unchanged in size

2. **(Phase 2, 2–4 days)**  Replace `_batch_matrix` with a per-channel store
   (`ΔR̂` for bc, `ΔΔR̂` for diff) plus the index map. Keep `coupling_mat.data`
   semantics by computing the SpMV directly from the Toeplitz form — either
   expand to dense lazily inside `_update_rates`, or do per-channel 1D / 2D
   correlations against the photon vector (FFT-based for diff). Expected:
   - persistent `_batch_matrix` size: 20–40× smaller
   - pickled `PriNCeRun` size: 3–5× smaller (depends on what else is in there)
   - runtime SpMV: same speed if we expand lazily; possibly faster if we
     correlate (but this needs benchmarking — current SpMV is already a
     single BLAS call cached at `update_rates_z_threshold`)

3. **(Phase 3, 0.5 days)**  After `_init_matrices` finishes, drop
   `cs.resp.incl_diff_intp` and `cs.resp.incl_diff_intp_integral` from the
   PriNCeRun. Nothing in the runtime path uses them. Expected:
   - 1+ GB savings in pickled `PriNCeRun` at max_mass=56 alone
   - this is worth doing **even without Phases 1–2**, as a one-liner.

4. **(Phase 4, optional)**  If U238 still doesn't fit, look at the response
   splines themselves: `incl_diff_intp_integral` is built by
   `cumulative_trapezoid` on the response grid, which is set by the underlying
   tabulated cross-section data. For most channels the response is smooth and
   could be re-sampled on a coarser grid before the spline is built.

## Risks (already documented in MATH_ANALYSIS.md, repeated here)

- The Toeplitz identity requires CR and photon grids to share the same log-step.
  Currently both are 8 bins/decade — assert this at init and fall back to a
  general 2D-rescale path if not. (Almost-as-fast: just a few extra spline
  evaluations per channel.)
- `grid_scale = "logE"` (an alternative grid mode in `config.py`) is not used
  in production; the analysis here is for the default `'E'` mode.
- `x_cut` / `x_cut_proton` cuts apply to the (i_mo, i_da) plane only and are
  trivially preserved.
- The `res[res<0] = 0.0` clipping happens **before** the nonel subtraction in
  the existing code; the prototype matches this ordering exactly.

## Numerical-agreement caveat

The prototype validates that the Toeplitz formula reproduces the dense
`_batch_matrix` to 4e-14 relative accuracy. This is _exactly_ the same
computation, just reorganised, so no physics is approximated. The only thing a
Phase-1 patch would change is **performance** and **memory**.
