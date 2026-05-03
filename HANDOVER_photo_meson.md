# Hand-over: photo_meson.py module split + `_fill_multiplicity` placeholder

This is HANDOVER.md item 10, deferred. The other half of the task — the
new database — overlaps with the data-file gaps in `EmpiricalModel`, so
this is a hand-off to the database-rework agent.

## Quick summary

`src/prince_cr/cross_sections/photo_meson.py` (478 lines) bundles two
unrelated models behind one file:

- **`SophiaSuperposition`** (lines 12–228): production model. Loads SOPHIA
  proton/neutron tables from the HDF5 db; nuclei become Z·p + N·n
  superpositions. Used by every test and the `composite` fixture.
- **`EmpiricalModel(SophiaSuperposition)`** (lines 231–478): Morejon et
  al. JCAP 11 (2019) 007. Marked `# pragma: no cover`. Has no tests,
  no callers in production, and **its data files are not shipped** in
  `src/prince_cr/data/` — see "Missing data files" below.

The HANDOVER recommendation is: move `EmpiricalModel` to its own module
and figure out what the `_fill_multiplicity` placeholder trick is really
for. Both questions are entangled with the new database, which is why
they belong with you.

---

## The placeholder trick (`_fill_multiplicity`, lines 332–368)

This is the "leaky abstraction" call-out in the parent HANDOVER.md.

### What it does

`EmpiricalModel.__init__` (line 246) calls `_fill_multiplicity()`, which
populates the three tab dicts that every `CrossSectionBase` subclass
uses to declare channel membership:

```python
self._nonel_tab = {100: (), 101: ()}                  # tuple sentinel
for mom in self._nonel_tab:
    for dau in [2, 3, 4, 100, 101]:
        self._incl_diff_tab[mom, dau] = ()            # tuple sentinel

# ...later, per nucleus mother:
self._nonel_tab[mom] = ()
for dau in [2, 3, 4, 100, 101]:
    self._incl_diff_tab[mom, dau] = SophiaSuperposition.incl_diff(self, mom, dau)[1]
for dau, mult in mults.items():
    new_multiplicity[mom, dau] = mult
    self._incl_tab[mom, dau] = np.array([])           # empty-array sentinel
```

So most tab values are `()` or `np.array([])` — placeholders, not real
cross sections. The actual numbers come from `EmpiricalModel.nonel`,
`incl`, `incl_diff` overrides (lines 370–478), which compute on demand
from `multiplicity_table()`, the universal spline, and the pion spline.

### Why the placeholders exist

`CrossSectionBase` confounds two concepts in one data structure:

1. **Channel membership.** `_update_indices()` (`base.py:166-168`)
   derives `nonel_idcs / incl_idcs / incl_diff_idcs` from
   `dict.keys()` of the three tabs. All downstream code (the kernel
   construction in `interaction_rates.py`, `CompositeCrossSection._join_*`,
   `_reduce_channels`) reads channel membership from those `_idcs`
   lists.
2. **Cross-section data.** Tab values are expected to be ndarrays
   (or `(egrid, ndarray)` tuples).

`EmpiricalModel` wants to declare membership without storing data, since
it computes everything lazily. The placeholder trick fakes (2) to
satisfy (1). It works because:

- Default `config.tau_dec_threshold = np.inf` (config.py and
  conftest.py both set this), so `_reduce_channels` never actually
  recurses on a placeholder value — it only ever exits via the
  "stable terminal" arm, where the placeholder gets dropped into
  `new_incl_tab` / `new_dec_diff_tab` but is then immediately
  overwritten by `EmpiricalModel.{nonel,incl,incl_diff}` overrides
  that ignore the tab values.
- `_optimize_and_generate_index` is **commented out** in
  `EmpiricalModel.__init__` (line 248: `# self._optimize_and_generate_index() # without this works!!`).
  That comment is load-bearing: with placeholders feeding in, calling
  `_optimize_and_generate_index → _reduce_channels` produces nonsense
  values in the tabs. Skipping the call leaves the placeholders alone
  and lets the lazy overrides do their thing.

If anyone ever lowers `tau_dec_threshold` for an `EmpiricalModel` run
(or someone uncomments the `_optimize_and_generate_index()` call), the
placeholder values would be multiplied by branching ratios and convolved
with decay distributions, producing garbage. The model would silently
break. There is no test that would catch this.

### The two consequences

- **The base/subclass contract is muddied.** `CrossSectionBase` says
  "subclasses fill in the tabs." `EmpiricalModel` actually says "I
  fill in only the keys; the values are dummies; please call my
  override methods to get real numbers."

- **`_reduce_channels` (in base.py, just refactored into
  `_DecayChainReducer`) cannot tell the two cases apart.** It assumes
  every value is a real cross section it can multiply by a branching
  ratio. The `EmpiricalModel` workflow only survives because the
  default config never lets it execute against a placeholder.

### Recommended fix shape

When the new database lands, the cleanest split is:

1. **Separate channel-membership from value storage on the base.**
   Add an explicit channel-set attribute (e.g.,
   `_declared_channels: set[tuple]`) that `_update_indices` reads
   instead of `dict.keys()`. Subclasses populate `_declared_channels`
   directly; `_nonel_tab`/`_incl_tab`/`_incl_diff_tab` then only carry
   real values for subclasses that have them.

2. **`EmpiricalModel` fills `_declared_channels` and overrides the
   accessor methods.** No more empty-tuple/empty-array sentinels.
   Drop `_fill_multiplicity`'s placeholder loops; keep the
   `multiplicity` dict (it's a real piece of state used by `incl`).

3. **Re-enable `_optimize_and_generate_index()` for `EmpiricalModel`.**
   With placeholders gone, the chain-reduction path will only see
   real values from subclasses that have them, so the call becomes
   safe again.

If fix (1) is too invasive, the cheaper alternative is to keep the
two-step contract but make it explicit: a base-class method
`_declare_channel_keys_only(mom, dau)` that records membership without
inserting a placeholder; `_reduce_channels` then skips channels whose
values are unset (and `EmpiricalModel`'s overrides handle their
fetches). This is a smaller change but leaves the conceptual confusion.

---

## Other issues co-located in `photo_meson.py`

Lower-priority but worth folding into the same pass while the file is
open:

- **Two unrelated models in one file.** Move `EmpiricalModel` to its
  own module (`empirical_model.py` next to `empirical_relations.py`).
  `SophiaSuperposition` is the only class that the production
  `CompositeCrossSection` chain instantiates by default; keeping it in
  `photo_meson.py` is fine.

- **Hardcoded magic numbers in `EmpiricalModel.incl_diff`** (line 401
  onwards). `e_max = 1.2`, `e_scale = 0.3`, fade indices
  `range(32)` and `range(55, 95)` are flagged in source comments as
  "found manually" / "hardcoded index". If your new database
  re-derives the universal/pion splines, expect these to need
  re-tuning. Worth adding a docstring to `incl_diff` capturing what
  energy regime each magic number maps to (where on `egrid`).

- **Two near-identical loaders.** `_load_universal_function`
  (lines 310–319) and `_load_pion_function` (lines 321–330) only
  differ by filename and target attribute. Trivially DRY-able to
  `_load_spline(filename, attr_name)`.

- **Dead code in `incl_diff`.** Lines 455–471 are a commented-out
  `sigm` definition; lines 477+ have other commented-out sections
  earlier in the function. Delete on the way through.

- **`SophiaSuperposition.incl_diff` mutates `self.incl_diff_idcs`
  on every call** (lines 178, 184): "created incl. diff. index for
  all particle created in p-gamma". This is a side effect of a
  read accessor, which makes `incl_diff` order-dependent. Likely a
  vestige of the channel-discovery flow; check whether
  `CompositeCrossSection._join_models` or `generate_incl_channels`
  already covers this and the mutation can be deleted.

---

## Missing data files (blocks any work on `EmpiricalModel`)

`EmpiricalModel.__init__` needs three files that are NOT shipped:

| File | Loaded at | Source |
|---|---|---|
| `scaling_lines` (no extension; pickle) | `A_eff`, line 273 | `np.load(..., allow_pickle=True, encoding="latin1")` — Python-2-pickled |
| `universal-spline.pkl` | `_load_universal_function`, line 315 | scipy `UnivariateSpline` tck tuple, Python-2-pickled |
| `pion_spline.pkl` | `_load_pion_function`, line 326 | same |

`src/prince_cr/data/` currently contains only `particle_data.ppo` and
`prince_db_05.h5`. There is no test that builds an `EmpiricalModel`,
which is why CI doesn't notice the missing files.

When the new database lands, these three files should either:
- be folded into the HDF5 (preferred, single-source);
- or shipped alongside as numpy/json with a clean format (no Python-2
  pickle / `latin1` encoding hacks).

The `latin1` encoding bit is a Python-2-pickle compatibility shim and
will be unnecessary if the data is re-serialised cleanly.

---

## File-and-line index

For a one-screen orientation:

```
photo_meson.py
   12  class SophiaSuperposition(CrossSectionBase)
   22    .__init__    -> ._load() (HDF5 SOPHIA tables)
  120    .nonel       (Z·p + N·n superposition; production path)
  144    .incl        (boost-conserving + side-effect on incl_diff_idcs)
  193    .incl_diff   (xL distribution from p/n redistribution tables)

  231  class EmpiricalModel(SophiaSuperposition)              # pragma: no cover
  236    .__init__    -> ._load_universal_function/_load_pion_function/_fill_multiplicity
  248                  # commented-out _optimize_and_generate_index — load-bearing
  250    .A_eff       (sigmoid-blended A^y scaling, scaling_lines pickle)
  292    .fade        (sigmoid blend of two cs distributions)
  310    ._load_universal_function  (universal-spline.pkl)
  321    ._load_pion_function       (pion_spline.pkl)
  332    ._fill_multiplicity        (the placeholder trick — see above)
  370    .nonel       (A·univ_spl below 1.2 GeV; A_eff scaling above)
  384    .incl        (multiplicity[mo,da] · nonel)
  401    .incl_diff   (Sophia + multiplicity bump; pion A^2/3 rescaling above 200)
```

`empirical_relations.py` is the math backend
(`multiplicity_table`, `gxn_multiplicities`, `cs_g{p,n,pi,xn}`, etc.) — 578
lines, untouched by this hand-over.

Refactor of `_reduce_channels` into `_DecayChainReducer` already landed
in `14b464e` — `_reduce_channels` is now ~95 lines in `base.py`, and
`_DecayChainReducer` is ~125 lines at the bottom of the same file.
The placeholder-trick failure mode described above lives entirely
inside `_DecayChainReducer.follow / _convolve / _record_stable`, so
that's where to assert "value must be a real ndarray" if you take the
explicit-channel-set route.
