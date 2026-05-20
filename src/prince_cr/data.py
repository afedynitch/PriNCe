"""Module inteded to contain some prince-specific data structures."""

import pickle as pickle
import os.path as path
import numpy as np
import scipy.constants as spc
import h5py


from prince_cr.util import convert_to_namedtuple, info, make_nucleus_pdg, get_AZN
import prince_cr.config as config


# v2-sparse CSR layout (mirrors prince-fluka-utils' `_pack_csr_dataset`):
#   <name>           [2, total_nnz] float64   (data row 0; col indices row 1)
#   <name>_indptrs   [n_ch, n_rows+1] int64
#   <name>_len_data  [n_ch] int64             (per-channel nnz)
#   <name>.attrs:    mat_shape, format='v2-sparse'
# Reader rebuilds each channel's CSR -> dense ndarray on demand. With
# `channels=...`, only those rows of indptrs and the corresponding slices
# of the values dataset are read, so `max_mass` filtering pays I/O only
# for the channels it keeps.
def _read_csr_yields(grp, name, *, channels=None):
    """Return a dense ndarray for `grp/name` (always v2-sparse).

    `channels` selects which channel rows to materialise (in the given
    order). v1 dense db files are no longer supported — regenerate the
    db with prince-fluka-utils ≥ d5c8a4f.
    """
    ds = grp[name]
    fmt = ds.attrs.get("format")
    if fmt != b"v2-sparse" and fmt != "v2-sparse":
        raise RuntimeError(
            f"FLUKA db dataset {grp.name}/{name} is not v2-sparse "
            f"(format={fmt!r}). Regenerate with prince-fluka-utils ≥ "
            f"d5c8a4f; v1 dense db files are no longer read."
        )

    from scipy.sparse import csr_matrix

    indptrs_ds = grp[name + "_indptrs"]
    len_data = np.asarray(grp[name + "_len_data"][:], dtype=np.int64)
    mat_shape_attr = tuple(int(x) for x in ds.attrs["mat_shape"])
    is_vector = (len(mat_shape_attr) == 1)
    csr_shape = (1, mat_shape_attr[0]) if is_vector else mat_shape_attr

    n_ch_total = len(len_data)
    if channels is None:
        ch_indices = np.arange(n_ch_total, dtype=np.int64)
    else:
        ch_indices = np.asarray(list(channels), dtype=np.int64)

    if ch_indices.size == 0:
        return np.zeros((0, *mat_shape_attr), dtype=np.float64)

    n_rows = 1 if is_vector else mat_shape_attr[0]
    n_cols = mat_shape_attr[-1]

    # Build a contiguous packed buffer for the requested channels only:
    # concatenate each selected channel's `mat[:, off:off+n]` slice plus the
    # corresponding indptr row. The resulting big CSR has shape
    # (n_ch_sel * n_rows, n_cols) and decodes to dense in a single
    # vectorised call — far faster than a Python loop of 14k csr_matrix()
    # constructions when the consumer asks for many channels at once.
    offsets = np.concatenate(([0], np.cumsum(len_data)))

    # Fast path: ask for ALL channels in order. Read the full mat once;
    # h5py + lzf turns this into a single decompression pass with no
    # Python-level overhead per channel.
    full_scan = (
        ch_indices.size == n_ch_total
        and ch_indices[0] == 0
        and ch_indices[-1] == n_ch_total - 1
        and np.all(np.diff(ch_indices) == 1)
    )
    if full_scan:
        mat = ds[:]
        indptrs_all = indptrs_ds[:]
        # Shift each channel's indptr into the concatenated coordinate
        # system: row r of channel i maps to absolute row i*n_rows + r,
        # and its data range starts at offsets[i] + indptrs_all[i, r].
        # Drop the leading 0 of every channel because it's redundant once
        # we've concatenated; prepend a single 0 for the global indptr.
        shifts = offsets[:-1][:, None]                # (n_ch, 1)
        big_indptr = np.concatenate(
            ([0], (indptrs_all[:, 1:] + shifts).ravel())
        )
    else:
        # Selective path: gather only the slices we need.
        ch_lens = len_data[ch_indices].astype(np.int64)
        ch_offs = offsets[ch_indices].astype(np.int64)
        total_nnz = int(ch_lens.sum())
        mat = np.empty((2, total_nnz), dtype=np.float64)
        cur = 0
        # h5py supports cheap contiguous slices; one read per channel beats
        # a single `ds[:]` when most channels are skipped (max_mass cap).
        for i in range(ch_indices.size):
            n = int(ch_lens[i])
            if n == 0:
                continue
            off = int(ch_offs[i])
            mat[:, cur:cur + n] = ds[:, off:off + n]
            cur += n
        # Indptrs: read just the requested rows.
        indptrs_sel = indptrs_ds[ch_indices, :]
        # Per-channel cumulative shift in the new packed buffer.
        shifts_sel = np.concatenate(([0], np.cumsum(ch_lens[:-1])))[:, None]
        big_indptr = np.concatenate(
            ([0], (indptrs_sel[:, 1:] + shifts_sel).ravel())
        )

    data = mat[0]
    col = mat[1].astype(np.int64)
    n_rows_total = ch_indices.size * n_rows
    big_csr = csr_matrix((data, col, big_indptr), shape=(n_rows_total, n_cols))
    dense = big_csr.toarray().reshape(ch_indices.size, n_rows, n_cols)
    if is_vector:
        return dense[:, 0, :]
    return dense


# Boundary translation: the legacy ``particle_data.ppo`` is keyed by Neucosma
# (NCo) IDs (proton=101, neutron=100, A*100+Z for nuclei; small integers for
# light particles). PriNCe runs entirely in PDG IDs internally, so we translate
# once at module load and never look at NCo again.
#
# Helicity-resolved muon NCo slots (5/6/8/9) collapse onto PDG ±13 — chromo's
# Pythia post-decay produces helicity-mixed muons, and the production FLUKA
# pipeline never populates the helicity-resolved slots. Helicity-0 NCo slots
# (7 = μ+, 10 = μ-) carry the canonical entries; the duplicates are dropped.
_NCO_TO_PDG_LIGHT = {
    0:   22,    # gamma
    2:   211,   # pi+
    3:  -211,   # pi-
    4:   111,   # pi0
    7:  -13,    # mu+ (helicity 0)
    10:  13,    # mu-  (helicity 0)
    11:  12,    # nu_e
    12: -12,    # nu_ebar
    13:  14,    # nu_mu
    14: -14,    # nu_mubar
    15:  16,    # nu_tau
    16: -16,    # nu_taubar
    20:  11,    # e-
    21: -11,    # e+
    50:  321,   # K+
    51: -321,   # K-
    100: 2112,  # neutron
    101: 2212,  # proton
}
# Helicity-resolved slots map onto the helicity-0 PDG codes (collapsed).
_NCO_HELICITY_RESOLVED_TO_PDG = {5: -13, 6: -13, 8: 13, 9: 13}


def _nco_to_pdg(nco_id):
    """Translate a single NCo ID (int) to its PDG equivalent.

    Nuclei (NCo >= 200, encoded as 100*A + Z) become 10LZZZAAAI ground-state
    PDG codes via :func:`prince_cr.util.make_nucleus_pdg`. Free p/n use the
    canonical 2212/2112 codes. Light particles use the table above. The
    helicity-resolved muon slots collapse onto PDG ±13.
    """
    nco_id = int(nco_id)
    if nco_id in _NCO_TO_PDG_LIGHT:
        return _NCO_TO_PDG_LIGHT[nco_id]
    if nco_id in _NCO_HELICITY_RESOLVED_TO_PDG:
        return _NCO_HELICITY_RESOLVED_TO_PDG[nco_id]
    if nco_id >= 100:
        Z = nco_id % 100
        A = (nco_id - Z) // 100
        return make_nucleus_pdg(A, Z)
    raise KeyError("Unknown NCo ID {0} in spec_data translation".format(nco_id))


def _translate_spec_data(nco_keyed):
    """Return a PDG-keyed copy of the legacy NCo-keyed ``particle_data.ppo``.

    Recursively rewrites daughter IDs in each entry's ``branchings`` list and
    rewrites the ``non_nuclear_species`` index list. Helicity-resolved muon
    duplicates collapse onto the ±13 entry; later writes are no-ops because
    the data is identical (mass/lifetime/branchings agree by construction).
    """
    pdg_keyed = {}
    for key, value in nco_keyed.items():
        if isinstance(key, str):
            if key == "non_nuclear_species":
                pdg_keyed[key] = sorted({_nco_to_pdg(p) for p in value})
            else:
                pdg_keyed[key] = value
            continue
        new_key = _nco_to_pdg(key)
        new_entry = dict(value)
        new_branchings = []
        for branching, daughters in value.get("branchings", []):
            new_daughters = [_nco_to_pdg(d) for d in daughters]
            new_branchings.append((branching, new_daughters))
        new_entry["branchings"] = new_branchings
        pdg_keyed[new_key] = new_entry
    return pdg_keyed


#: Dictionary containing particle properties, keyed by PDG ID. Loaded from the
#: NCo-keyed legacy pickle and translated on import.
try:
    with open(path.join(config.data_dir, "particle_data.ppo"), "rb") as _f:
        _raw = pickle.load(_f)
except UnicodeDecodeError:
    with open(path.join(config.data_dir, "particle_data.ppo"), "rb") as _f:
        _raw = pickle.load(_f, encoding="latin1")
    spec_data = _translate_spec_data(_raw)
except FileNotFoundError:
    info(0, 'Warning, particle database "particle_data.ppo" file not found.')
    spec_data = {}
else:
    spec_data = _translate_spec_data(_raw)
finally:
    try:
        del _raw
    except NameError:
        pass


# ----------------------------------------------------------------------------
# Tabulated decay backend (FLUKA + Pythia).
#
# The v1 prince-fluka-utils db carries two decay groups:
#   /decays/FLUKA_DECAY_2025/   — 2813 nuclear mothers (β±/EC/α, …)
#   /decays/PYTHIA_HADRON_2025/ —  10 hadron mothers   (μ±, π±, π⁰, K±, K⁰_S/L, n)
# Both store per-decay light-secondary yields on the shared xbins
# (`x = E_total_rest / m_nucleon` for FLUKA, `x = E_total_rest / m_parent`
# for Pythia — both happen to coincide with PriNCe's per-nucleon kernel x
# convention, see `cross_sections/fluka.py` for the photo-nuclear analogue).
#
# At module load we merge the tabulated info into `spec_data` (overriding
# legacy `particle_data.ppo` entries where they exist, e.g. tritium is
# `stable=True` in the ppo) and stash dN/dx tables in `_TABULATED_DECAY_DX`
# for `decays.get_decay_matrix_bin_average` to dispatch into.
# ----------------------------------------------------------------------------

#: (mother_pdg, daughter_pdg) -> ndarray dN/dx per decay on `_TABULATED_DECAY_XBINS`.
_TABULATED_DECAY_DX: dict = {}
#: Shared xbins for the tabulated decay grid (FLUKA decay = Pythia = photo-nuclear).
_TABULATED_DECAY_XBINS = None


def fluka_decay_db():
    """Read `/decays/FLUKA_DECAY_2025/` from the FLUKA db. Returns a dict, or
    None if the db file isn't present at `config.fluka_db_path`."""
    fpath = path.join(config.fluka_db_path, config.fluka_db_fname)
    if not path.isfile(fpath):
        return None
    with h5py.File(fpath, "r") as f:
        if "decays/FLUKA_DECAY_2025" not in f:
            return None
        g = f["decays/FLUKA_DECAY_2025"]
        return {
            "decay_mothers":     g["decay_mothers"][:],
            "half_lives":        g["half_lives"][:],
            "rest_masses":       g["rest_masses"][:],
            "nuclear_daughters": g["nuclear_daughters"][:],
            "branching_ratios":  g["branching_ratios"][:],
            "light_daughters":   g["light_daughters"][:],
            "light_yields":      _read_csr_yields(g, "light_yields"),
            "xbins":             g["xbins"][:],
            "n_decays_per_mother":
                int(g.attrs.get("n_decays_per_mother", 1)),
        }


def pythia_hadron_db():
    """Read `/decays/PYTHIA_HADRON_2025/` from the FLUKA db. Returns a dict,
    or None if the db file isn't present at `config.fluka_db_path`."""
    fpath = path.join(config.fluka_db_path, config.fluka_db_fname)
    if not path.isfile(fpath):
        return None
    with h5py.File(fpath, "r") as f:
        if "decays/PYTHIA_HADRON_2025" not in f:
            return None
        g = f["decays/PYTHIA_HADRON_2025"]
        return {
            "decay_mothers":   g["decay_mothers"][:],
            "rest_masses":     g["rest_masses"][:],
            "light_daughters": g["light_daughters"][:],
            "light_yields":    _read_csr_yields(g, "light_yields"),
            "xbins":           g["xbins"][:],
            "n_decays_per_mother":
                int(g.attrs.get("n_decays_per_mother", 1)),
        }


def fluka_photo_nuclear_inel_mothers():
    """Return the `inel_mothers` array from `/photo_nuclear/FLUKA_2025/`,
    or None if the db file isn't present at `config.fluka_db_path`. Used
    to fill in `spec_data` entries for stable mothers that aren't carried
    by either the legacy ppo or the FLUKA decay group."""
    fpath = path.join(config.fluka_db_path, config.fluka_db_fname)
    if not path.isfile(fpath):
        return None
    with h5py.File(fpath, "r") as f:
        if "photo_nuclear/FLUKA_2025/inel_mothers" not in f:
            return None
        return f["photo_nuclear/FLUKA_2025/inel_mothers"][:]


_LN2 = float(np.log(2.0))


def _synthesize_from_particle_db(pdg):
    """Build a `spec_data`-shaped dict for `pdg` from the `particle`
    package, or return None if the PDG isn't recognised.

    Used as the canonical fallback for elementary species that don't
    appear in the legacy `particle_data.ppo` (anti-baryons, K⁰_S/K⁰_L,
    τ leptons, ...) but show up as photo-nuclear / decay daughters or as
    Pythia decay mothers.

    Unit conventions: `particle` reports mass in MeV, `three_charge` in
    thirds, lifetime in ns. PriNCe's `spec_data` uses GeV for mass,
    integer charge in units of e, seconds for lifetime; conversion is
    done here. Stable particles (lifetime None or +inf in `particle`)
    map to `lifetime=np.inf`, `stable=True`.
    """
    try:
        from particle import Particle, ParticleNotFound  # type: ignore
    except ImportError:
        return None
    try:
        p = Particle.from_pdgid(pdg)
    except ParticleNotFound:
        return None

    mass_MeV = p.mass
    if mass_MeV is None or not np.isfinite(mass_MeV):
        # Nominal-mass particles like neutrinos return None — fine for
        # spec_data which only requires the field to exist.
        mass_GeV = 0.0
    else:
        mass_GeV = float(mass_MeV) * 1e-3

    if p.three_charge is None:
        charge = 0
    else:
        charge = int(p.three_charge) // 3

    lt_ns = p.lifetime
    if lt_ns is None or lt_ns == float("inf") or not np.isfinite(lt_ns):
        lifetime = np.inf
        stable = True
    else:
        lifetime = float(lt_ns) * 1e-9
        stable = False

    return {
        "mass": mass_GeV,
        "charge": charge,
        "lifetime": lifetime,
        "stable": stable,
        "incomplete": False,
        "branchings": [],
    }


def _heuristic_beta_branching(mo_pdg):
    """β±-direction last-resort branching for unstable FLUKA decay
    mothers whose chromo / SPDCEV sampling failed (empty channels in
    the db). Direction picked from neutron excess: β⁻ if Z·2 < A
    (n-rich), else β⁺ / EC. Returns the same `[(mult, [da])]` shape
    PriNCe's chain reducer expects, with the nuclear daughter, an
    electron(/positron), and the matching neutrino as separate
    one-daughter entries.

    Use of this heuristic is FUDGE — every invocation should fire
    a `RuntimeWarning` so the gap is visible. The proper fix is at
    db-build time in prince-fluka-utils (rerun chromo's decay sampler
    until SPDCEV succeeds, or add the missing isotope's branching
    info from PDG / NNDC by hand into the schema). Cycle detection
    in `_DecayChainReducer.follow` (base.py) catches the case where
    successive heuristic steps land on other empty-channel mothers
    and chain back."""
    from prince_cr.util import get_AZN, make_nucleus_pdg

    A, Z, _ = get_AZN(mo_pdg)
    if A < 1 or Z is None or Z < 0:
        return None
    if Z * 2 < A:                                 # n-rich → β⁻
        return [
            (1.0, [make_nucleus_pdg(A, Z + 1)]),  # nucleus(Z+1)
            (1.0, [-12]),                         # ν̄_e
            (1.0, [11]),                          # e⁻
        ]
    else:                                         # p-rich → β⁺ / EC
        return [
            (1.0, [make_nucleus_pdg(A, Z - 1)]),  # nucleus(Z−1)
            (1.0, [12]),                          # ν_e
            (1.0, [-11]),                         # e⁺
        ]


def _merge_tabulated_decays(spec_data):
    """Override `spec_data[mo]` lifetimes / branchings from the tabulated
    decay groups, and populate `_TABULATED_DECAY_DX` for the chain reducer
    to convolve light-secondary distributions against.

    Silently no-ops if `config.fluka_db_path` doesn't resolve to a file
    or if the decay groups are missing. `spec_data` mass / charge fields
    are preserved from the legacy ppo where they exist.
    """
    global _TABULATED_DECAY_XBINS

    fluka = fluka_decay_db()
    pythia = pythia_hadron_db()
    if fluka is None and pythia is None:
        info(2, "Tabulated decay db not found; using legacy spec_data only")
        return

    # Coverage gap accumulators — surfaced as a single RuntimeWarning at
    # the end of the merge so the operator sees the full list at once
    # instead of one warning per channel. Two categories:
    #   - heuristic_fallback: unstable FLUKA mother with empty channels;
    #     β-direction heuristic guessed a daughter (chromo / SPDCEV
    #     failure — fix at db-build).
    #   - heuristic_unparseable: half-life finite but PDG can't be parsed
    #     into A,Z (shouldn't happen in practice; logged just in case).
    _heuristic_fallback_mothers = []
    _heuristic_unparseable = []

    if fluka is not None:
        _TABULATED_DECAY_XBINS = fluka["xbins"]
        xb = fluka["xbins"]
        xw = xb[1:] - xb[:-1]
        nd_mo = fluka["nuclear_daughters"][:, 0]
        nd_da = fluka["nuclear_daughters"][:, 1]
        ld_mo = fluka["light_daughters"][:, 0]
        ld_da = fluka["light_daughters"][:, 1]

        for i, mo_raw in enumerate(fluka["decay_mothers"]):
            mo_pdg = int(mo_raw)
            half_life = float(fluka["half_lives"][i])
            lifetime = (
                half_life / _LN2
                if np.isfinite(half_life) and half_life > 0
                else np.inf
            )

            branchings = []
            for ch in np.where(nd_mo == mo_raw)[0]:
                branchings.append(
                    (float(fluka["branching_ratios"][ch]),
                     [int(nd_da[ch])])
                )
            for ch in np.where(ld_mo == mo_raw)[0]:
                yld = fluka["light_yields"][ch]
                mult = float(yld.sum())   # per-bin yield is already per-decay
                if mult <= 0.0:
                    continue
                da_pdg = int(ld_da[ch])
                branchings.append((mult, [da_pdg]))
                _TABULATED_DECAY_DX[(mo_pdg, da_pdg)] = yld / xw

            if not branchings and lifetime != np.inf:
                heur = _heuristic_beta_branching(mo_pdg)
                if heur is not None:
                    branchings = heur
                    _heuristic_fallback_mothers.append(
                        (mo_pdg, half_life, branchings[0][1][0])
                    )
                else:
                    _heuristic_unparseable.append((mo_pdg, half_life))
                    lifetime = np.inf

            if mo_pdg in spec_data:
                spec_data[mo_pdg]["lifetime"] = lifetime
                spec_data[mo_pdg]["stable"] = (lifetime == np.inf)
                if branchings:
                    spec_data[mo_pdg]["branchings"] = branchings
            else:
                # Unknown to the legacy ppo (e.g. exotic isotopes). Synthesize
                # a minimal entry from the FLUKA rest-mass + PDG-derived charge.
                from prince_cr.util import get_AZN
                _, Z, _ = get_AZN(mo_pdg)
                spec_data[mo_pdg] = {
                    "mass": float(fluka["rest_masses"][i]),
                    "stable": (lifetime == np.inf),
                    "lifetime": lifetime,
                    "incomplete": False,
                    "charge": int(Z) if Z is not None else 0,
                    "branchings": branchings,
                }

    if pythia is not None:
        if _TABULATED_DECAY_XBINS is None:
            _TABULATED_DECAY_XBINS = pythia["xbins"]
        xb = pythia["xbins"]
        xw = xb[1:] - xb[:-1]
        ld_mo = pythia["light_daughters"][:, 0]
        ld_da = pythia["light_daughters"][:, 1]

        # Pythia carries no lifetime field; PDG values from particle_data.ppo
        # are kept where they exist. We overwrite the branching list with
        # Pythia's actual tabulated final-state composition + dN/dx tables,
        # and synthesize a spec_data entry from the `particle` package for
        # mothers the legacy ppo doesn't carry (e.g. K⁰_S/K⁰_L, anti-n).
        # Pythia's `rest_masses` is the mass fallback if `particle` is
        # somehow unavailable.
        pythia_synth_added = []
        for i_mo, mo_raw in enumerate(pythia["decay_mothers"]):
            mo_pdg = int(mo_raw)
            branchings = []
            for ch in np.where(ld_mo == mo_raw)[0]:
                yld = pythia["light_yields"][ch]
                mult = float(yld.sum())
                if mult <= 0.0:
                    continue
                da_pdg = int(ld_da[ch])
                branchings.append((mult, [da_pdg]))
                _TABULATED_DECAY_DX[(mo_pdg, da_pdg)] = yld / xw

            if mo_pdg not in spec_data:
                synth = _synthesize_from_particle_db(mo_pdg)
                if synth is None:
                    # particle package missing or PDG unrecognised — fall
                    # back to Pythia rest-mass + best-effort flags.
                    synth = {
                        "mass": float(pythia["rest_masses"][i_mo]),
                        "charge": 0,
                        "lifetime": np.inf,
                        "stable": True,
                        "incomplete": True,
                        "branchings": [],
                    }
                spec_data[mo_pdg] = synth
                pythia_synth_added.append(mo_pdg)

            if branchings:
                spec_data[mo_pdg]["branchings"] = branchings

        if pythia_synth_added:
            info(2, "Synthesized {0} Pythia mother spec_data entries: {1}"
                    .format(len(pythia_synth_added), pythia_synth_added))

    # Synthesize stable-mother spec_data entries for photo-nuclear mothers
    # that the legacy ppo doesn't carry (everything beyond A=56 stable, plus
    # the bound H-1 / n-1 codes) and that aren't in the FLUKA decay db
    # (which only tabulates unstable species). Without this, the chain
    # reducer's `_reduce_channels` drops these mothers via
    # `mother not in spec_data` and the kernel never sees them.
    inel_mothers = fluka_photo_nuclear_inel_mothers()
    if inel_mothers is not None:
        from prince_cr.util import get_AZN
        # mass / charge constants (kept local to avoid forward refs to UNITS)
        m_nucleon = (
            float(fluka["rest_masses"][0]) / max(int(get_AZN(int(fluka["decay_mothers"][0]))[0]), 1)
            if fluka is not None and len(fluka["decay_mothers"])
            else 0.93827208816
        )
        # bound H-1 (1000010010) gets remapped to free p (2212) in the
        # photo-nuclear loader (`_normalize_pdg`); ensure both forms have
        # entries. n-1 (1000000010) is filtered upstream — never appears.
        bound_to_free = {1000010010: 2212, 1000000010: 2112}
        n_added = 0
        for raw in inel_mothers:
            keys = {int(raw)}
            if int(raw) in bound_to_free:
                keys.add(bound_to_free[int(raw)])
            for mo_pdg in keys:
                if mo_pdg in spec_data:
                    continue
                A, Z, _ = get_AZN(mo_pdg)
                if A is None or Z is None:
                    continue
                spec_data[mo_pdg] = {
                    "mass": float(A) * m_nucleon,
                    "stable": True,
                    "lifetime": np.inf,
                    "incomplete": False,
                    "charge": int(Z),
                    "branchings": [],
                }
                n_added += 1
        if n_added:
            info(2, "Synthesized {0} stable spec_data entries from photo-nuclear "
                    "inel_mothers".format(n_added))

    # Scan all elementary daughter PDGs that show up in cross-section
    # channels and tabulated decays, and synthesize spec_data entries from
    # the `particle` package for any unknowns. Catches species that appear
    # only as daughters (never as mothers) — anti-p is the canonical case:
    # FLUKA emits it as a redistribution daughter from heavy mothers, but
    # it isn't carried by the legacy ppo, isn't an unstable nuclear mother
    # in the FLUKA decay db, and isn't a Pythia decay mother. Without this
    # scan, the chain reducer's `if da not in spec_data: return` silently
    # discards all flux through anti-p (cf. base.py:_DecayChainReducer.follow);
    # the explicit-decay path would lift the discarded entries into
    # `known_species` and trigger `KeyError(-2212)` at SpeciesManager init.
    daughter_pdgs = set()
    if fluka is not None:
        daughter_pdgs.update(int(p) for p in fluka["nuclear_daughters"][:, 1])
        daughter_pdgs.update(int(p) for p in fluka["light_daughters"][:, 1])
    if pythia is not None:
        daughter_pdgs.update(int(p) for p in pythia["light_daughters"][:, 1])

    fpath = path.join(config.fluka_db_path, config.fluka_db_fname)
    if path.isfile(fpath):
        try:
            with h5py.File(fpath, "r") as f:
                ed_path = "photo_nuclear/FLUKA_2025/elementary_daughters"
                if ed_path in f:
                    ed = f[ed_path][:]
                    daughter_pdgs.update(int(p) for p in ed.flatten())
        except OSError:
            pass

    # Skip nuclei (10-digit PDG codes 10LZZZAAAI starting at 1e9) — those
    # are handled by the inel_mothers synthesis above. Skip 0 (placeholder
    # / null daughter rows in v2-sparse pads).
    elementary_unknown = sorted(
        pdg for pdg in daughter_pdgs
        if pdg != 0 and abs(pdg) < 1_000_000_000 and pdg not in spec_data
    )
    synthesized_from_particle = []
    skipped_unknown_to_particle = []
    for da_pdg in elementary_unknown:
        synth = _synthesize_from_particle_db(da_pdg)
        if synth is None:
            skipped_unknown_to_particle.append(da_pdg)
            continue
        spec_data[da_pdg] = synth
        synthesized_from_particle.append(da_pdg)
    if synthesized_from_particle:
        info(2, "Synthesized {0} elementary spec_data entries from `particle` "
                "package: {1}".format(len(synthesized_from_particle),
                                       synthesized_from_particle))
    if skipped_unknown_to_particle:
        import warnings
        warnings.warn(
            "FLUKA db references {0} elementary PDG IDs that aren't in "
            "spec_data and that the `particle` package doesn't recognise: "
            "{1}. The chain reducer will silently drop cross-section flux "
            "through these daughters, and explicit-decay mode will raise "
            "KeyError on PriNCeRun init. Add hand-coded entries to "
            "particle_data.ppo or extend the FLUKA db filter to drop these "
            "channels at write time.".format(
                len(skipped_unknown_to_particle), skipped_unknown_to_particle,
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    # Surface coverage gaps as a single RuntimeWarning so the operator
    # sees the full list of fudges at module load (instead of silently
    # carrying them or one warning per channel). Each entry is an
    # unstable nucleus that the FLUKA decay db couldn't sample (mostly
    # chromo SPDCEV failures); the β-direction heuristic guesses a
    # daughter to keep the chain reducer from running into a sink. The
    # proper fix is at db-build time — rerun chromo's decay sampler
    # for these mothers, or hand-extend the schema with PDG/NNDC
    # branching info. See wiki/open-questions.md.
    if _heuristic_fallback_mothers:
        import warnings
        from prince_cr.util import get_AZN
        # Compact tabular summary, sorted by half-life ascending.
        _heuristic_fallback_mothers.sort(key=lambda x: x[1])
        lines = []
        for pdg, hl, da in _heuristic_fallback_mothers:
            A, Z, _ = get_AZN(pdg)
            lines.append(
                "  PDG={0}  A={1} Z={2}  half_life={3:.3e}s  "
                "→ heuristic daughter PDG={4}".format(pdg, A, Z, hl, da)
            )
        warnings.warn(
            "FLUKA decay db has empty channels for {0} unstable nuclei "
            "(SPDCEV / sampling failures). Falling back to β-direction "
            "heuristic per-mother — this is a FUDGE; verify the chain "
            "reducer doesn't hit cycle warnings downstream and fix at "
            "the next prince-fluka-utils rebuild:\n{1}".format(
                len(_heuristic_fallback_mothers), "\n".join(lines)
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    if _heuristic_unparseable:
        import warnings
        warnings.warn(
            "FLUKA decay db has empty channels for {0} mothers whose PDG "
            "couldn't be parsed into (A,Z); forcing stable terminal: "
            "{1}".format(len(_heuristic_unparseable), _heuristic_unparseable),
            RuntimeWarning,
            stacklevel=2,
        )

    info(
        2,
        "Tabulated FLUKA/Pythia decays merged: "
        "{0} (mo,da) channels{1}".format(
            len(_TABULATED_DECAY_DX),
            "; {0} heuristic fallbacks".format(len(_heuristic_fallback_mothers))
                if _heuristic_fallback_mothers else "",
        ),
    )


try:
    _merge_tabulated_decays(spec_data)
except Exception as _exc:                                        # noqa: BLE001
    info(0, "Could not merge tabulated decays: {0}".format(_exc))


# Default units in Prince are ***cm, s, GeV***
# Define here all constants and unit conversions and use
# throughout the code. Don't write c=2.99.. whatever.
# Write clearly which units a function returns.
# Convert them if not standard unit
# Accept only arguments in the units above

UNITS_AND_CONVERSIONS_DEF = dict(
    c=1e2 * spc.c,
    cm2Mpc=1.0 / (spc.parsec * spc.mega * 1e2),
    Mpc2cm=spc.mega * spc.parsec * 1e2,
    m_proton=spc.physical_constants["proton mass energy equivalent in MeV"][0] * 1e-3,
    m_electron=spc.physical_constants["electron mass energy equivalent in MeV"][0] * 1e-3,
    r_electron=spc.physical_constants["classical electron radius"][0] * 1e2,
    fine_structure=spc.fine_structure,
    GeV2erg=1.0 / 624.15,
    erg2GeV=624.15,
    km2cm=1e5,
    yr2sec=spc.year,
    Gyr2sec=spc.giga * spc.year,
    cm2sec=1e-2 / spc.c,
    sec2cm=spc.c * 1e2,
)

# This is the immutable unit object to be imported throughout the code
PRINCE_UNITS = convert_to_namedtuple(UNITS_AND_CONVERSIONS_DEF, "PriNCeUnits")


class InterpolatorWrapper:
    """Wrapper class to make RegularGridInterpolator behave like interp2d."""

    def __init__(self, rgi):
        self.rgi = rgi

    def __call__(self, x, y):
        import numpy as np

        x, y = np.broadcast_arrays(x, y)
        points = np.column_stack([x.ravel(), y.ravel()])
        result = self.rgi(points)
        return result.reshape(x.shape) if x.shape else result.item()


class PrinceDB(object):
    """Provides access to data stored in an HDF5 file.

    The file contains all tables for runnin PriNCe. Currently
    the only still required file is the particle database. The tools
    to generate this database are publicly available in
    `PriNCe-data-utils <https://github.com/joheinze/PriNCe-data-utils>`_.

    """

    def __init__(self):
        info(2, "Opening HDF5 file", config.db_fname)
        self.prince_db_fname = path.join(config.data_dir, config.db_fname)
        if not path.isfile(self.prince_db_fname):
            raise Exception(
                'Prince DB file {0} not found in "data" directory.'.format(
                    config.db_fname
                )
            )

        with h5py.File(self.prince_db_fname, "r") as prince_db:
            self.version = prince_db.attrs["version"]

    def _check_subgroup_exists(self, subgroup, mname):
        available_models = list(subgroup)
        if mname not in available_models:
            info(0, "Invalid choice/model", mname)
            info(0, "Choose from:\n", "\n".join(available_models))
            raise Exception("Unknown selections.")

    @staticmethod
    def _energy_slice(egrid, e_range):
        """Compute a contiguous slice for the energy axis.

        Args:
            egrid (numpy.array): Full energy grid from the database.
            e_range (tuple or None): ``(e_min, e_max)`` bounds. ``None``
                means use the full grid.

        Returns:
            slice: A contiguous slice object for indexing the energy axis.
        """
        if e_range is None:
            return slice(None)
        e_min, e_max = e_range
        mask = (egrid >= e_min) & (egrid <= e_max)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            raise ValueError(
                "e_range ({}, {}) selects no points from the energy grid [{}, {}]".format(
                    e_min, e_max, egrid[0], egrid[-1]
                )
            )
        return slice(int(idx[0]), int(idx[-1]) + 1)

    def photo_nuclear_db(self, model_tag, e_range=None):
        info(10, "Reading photo-nuclear db. tag={0}".format(model_tag))
        db_entry = {}
        with h5py.File(self.prince_db_fname, "r") as prince_db:
            self._check_subgroup_exists(prince_db["photo_nuclear"], model_tag)
            grp = prince_db["photo_nuclear"][model_tag]

            # Read energy grid first to compute the slice
            egrid_full = grp["energy_grid"][:]
            sl = self._energy_slice(egrid_full, e_range)
            db_entry["energy_grid"] = egrid_full[sl]

            # Index arrays — no energy dimension
            for entry in ["inel_mothers", "mothers_daughters"]:
                info(10, "Reading entry {0} from db.".format(entry))
                db_entry[entry] = grp[entry][:]

            # Cross section arrays — energy is the last axis
            for entry in ["inelastic_cross_sctions", "fragment_yields"]:
                info(10, "Reading entry {0} from db.".format(entry))
                db_entry[entry] = grp[entry][..., sl]

        return db_entry

    def photo_meson_db(self, model_tag, e_range=None):
        info(10, "Reading photo-nuclear db. tag={0}".format(model_tag))
        db_entry = {}
        with h5py.File(self.prince_db_fname, "r") as prince_db:
            self._check_subgroup_exists(prince_db["photo_nuclear"], model_tag)
            grp = prince_db["photo_nuclear"][model_tag]

            # Read energy grid first to compute the slice
            egrid_full = grp["energy_grid"][:]
            sl = self._energy_slice(egrid_full, e_range)
            db_entry["energy_grid"] = egrid_full[sl]

            # xbins and index arrays — no energy dimension
            db_entry["xbins"] = grp["xbins"][:]
            for entry in ["inel_mothers", "mothers_daughters"]:
                info(10, "Reading entry {0} from db.".format(entry))
                db_entry[entry] = grp[entry][:]

            # inelastic_cross_sctions: shape (n_mothers, n_energy)
            info(10, "Reading entry inelastic_cross_sctions from db.")
            db_entry["inelastic_cross_sctions"] = grp["inelastic_cross_sctions"][:, sl]

            # fragment_yields: shape (n_channels, n_energy, n_xbins)
            # Energy is axis 1 (middle), not the last axis
            info(10, "Reading entry fragment_yields from db.")
            db_entry["fragment_yields"] = grp["fragment_yields"][:, sl, :]

        return db_entry

    def fluka_photo_nuclear_db(self, model_tag, e_range=None, max_mass=None,
                               db_path=None, db_fname=None):
        """Read photo_nuclear/<tag>/ from the FLUKA db (a separate file from
        ``db_fname``). Lazy: opens the file on each call.

        Args:
            model_tag (str): subgroup tag, e.g. ``"FLUKA_2025"``.
            e_range (tuple or None): ``(e_min, e_max)`` GeV bounds.
            max_mass (int or None): if given, only mother rows with
                ``A_mother <= max_mass`` are read from disk. Skips both the
                small index arrays and the large ``elementary_yields`` /
                ``fragment_yields`` channel rows for heavier mothers, so
                lighter caps load proportionally faster from a v2-sparse
                db. ``None`` means no filter (read everything).
            db_path / db_fname (str or None): explicit DB location. When
                ``None``, the module-global ``config.fluka_db_path`` /
                ``config.fluka_db_fname`` are consumed (legacy default).
                Pass explicitly to bypass the global — useful in tests
                or when running multiple builds against different dbs
                from the same process.

        Returns:
            dict with keys: ``energy_grid``, ``xbins``, ``inel_mothers``,
            ``mothers_daughters``, ``elementary_daughters``,
            ``inelastic_cross_sctions`` (typo preserved), ``fragment_yields``
            (n_ch, n_E), ``elementary_yields`` (n_em, n_E, n_x).
        """
        info(10, "Reading FLUKA photo-nuclear db. tag={0}".format(model_tag))
        if db_path is None:
            db_path = config.fluka_db_path
        if db_fname is None:
            db_fname = config.fluka_db_fname
        fpath = path.join(db_path, db_fname)
        if not path.isfile(fpath):
            raise FileNotFoundError(
                f"FLUKA db not found at {fpath}. "
                "Pass db_path/db_fname explicitly to FlukaPhotoNuclear, "
                "or set prince_cr.config.fluka_db_path to its directory."
            )

        db_entry = {}
        with h5py.File(fpath, "r") as fdb:
            self._check_subgroup_exists(fdb["photo_nuclear"], model_tag)
            grp = fdb["photo_nuclear"][model_tag]

            # schema_version is a v4 addition; older dbs lack the attr.
            # Default to v3 (no per-nucleon rescale at write time) so existing
            # v3 dbs keep loading exactly as before.
            db_entry["schema_version"] = int(grp.attrs.get("schema_version", 3))

            egrid_full = grp["energy_grid"][:]
            sl = self._energy_slice(egrid_full, e_range)
            db_entry["energy_grid"] = egrid_full[sl]
            db_entry["xbins"] = grp["xbins"][:]

            inel_mothers_all = grp["inel_mothers"][:]
            md_all = grp["mothers_daughters"][:]
            ed_all = grp["elementary_daughters"][:]

            if max_mass is None:
                m_keep = np.ones(len(inel_mothers_all), dtype=bool)
                bc_keep = np.ones(len(md_all), dtype=bool)
                em_keep = np.ones(len(ed_all), dtype=bool)
            else:
                # Filter on mother A only. Daughter A is bounded by mother A
                # at write time (`_filter_boost_keys`), so a mother-A cap is
                # sufficient for the boost bucket, and elementary daughters
                # are species-id'd not mass-cut.
                m_a = np.array([get_AZN(int(p))[0] for p in inel_mothers_all])
                bc_a = np.array([get_AZN(int(p))[0] for p in md_all[:, 0]]) \
                    if md_all.size else np.empty(0, dtype=np.int64)
                em_a = np.array([get_AZN(int(p))[0] for p in ed_all[:, 0]]) \
                    if ed_all.size else np.empty(0, dtype=np.int64)
                m_keep = m_a <= max_mass
                bc_keep = bc_a <= max_mass
                em_keep = em_a <= max_mass

            db_entry["inel_mothers"] = inel_mothers_all[m_keep]
            db_entry["mothers_daughters"] = md_all[bc_keep]
            db_entry["elementary_daughters"] = ed_all[em_keep]

            m_idx = np.flatnonzero(m_keep)
            bc_idx = np.flatnonzero(bc_keep)
            em_idx = np.flatnonzero(em_keep)

            n_E_sl = len(egrid_full[sl])
            # σ_inel: (n_m, n_E). Skip the fancy index when no mass cut so we
            # don't pull the full block off disk just to drop nothing.
            if max_mass is None:
                db_entry["inelastic_cross_sctions"] = (
                    grp["inelastic_cross_sctions"][:, sl]
                )
            elif m_idx.size:
                db_entry["inelastic_cross_sctions"] = (
                    grp["inelastic_cross_sctions"][m_idx, :][:, sl]
                )
            else:
                db_entry["inelastic_cross_sctions"] = np.zeros(
                    (0, n_E_sl), dtype=np.float64,
                )
            # heavy yields: (n_ch, n_E) — stays dense (no x axis to gain on).
            if max_mass is None:
                db_entry["fragment_yields"] = grp["fragment_yields"][:, sl]
            elif bc_idx.size:
                db_entry["fragment_yields"] = grp["fragment_yields"][bc_idx, :][:, sl]
            else:
                db_entry["fragment_yields"] = np.zeros((0, n_E_sl), dtype=np.float64)
            # elementary yields: (n_em, n_E, n_x) — sparse-packed in v2 db.
            ey_full = _read_csr_yields(grp, "elementary_yields", channels=em_idx)
            db_entry["elementary_yields"] = ey_full[:, sl, :]

        return db_entry

    def ebl_spline(self, model_tag, subset="base"):
        from scipy.interpolate import RegularGridInterpolator

        info(10, "Reading EBL field splines. tag={0}".format(model_tag))
        with h5py.File(self.prince_db_fname, "r") as prince_db:
            self._check_subgroup_exists(prince_db["EBL_models"], model_tag)
            self._check_subgroup_exists(prince_db["EBL_models"][model_tag], subset)
            spl_gr = prince_db["EBL_models"][model_tag][subset]

            # Create RegularGridInterpolator which is the modern replacement for interp2d
            x_coords = spl_gr["x"][:]
            y_coords = spl_gr["y"][:]
            z_values = spl_gr["z"][:]

            # RegularGridInterpolator expects z_values to have shape (len(x_coords), len(y_coords))
            # If z_values needs transposing to match this requirement, do it
            if z_values.shape != (len(x_coords), len(y_coords)):
                z_values = z_values.T

            # Create the interpolator
            rgi = RegularGridInterpolator(
                (x_coords, y_coords),
                z_values,
                method="linear",
                bounds_error=False,
                fill_value=0.0,
            )

            # Return wrapper that maintains interp2d interface and is picklable
            return InterpolatorWrapper(rgi)


#: db_handler is the HDF file interface
db_handler = PrinceDB()


class EnergyGrid(object):
    """Class for constructing a grid for discrete distributions.

    Since we discretize everything in energy, the name seems appropriate.
    All grids are log spaced.

    Args:
        lower (float): log10 of low edge of the lowest bin
        upper (float): log10 of upper edge of the highest bin
        bins_dec (int): bins per decade of energy
    """

    def __init__(self, lower, upper, bins_dec):
        self.bins = np.logspace(lower, upper, int((upper - lower) * bins_dec + 1))
        self.grid = 0.5 * (self.bins[1:] + self.bins[:-1])
        self.widths = self.bins[1:] - self.bins[:-1]
        self.d = self.grid.size
        info(
            5,
            "Energy grid initialized {0:3.1e} - {1:3.1e}, {2} bins".format(
                self.bins[0], self.bins[-1], self.grid.size
            ),
        )


_PDG_MESONS = frozenset({211, -211, 111, 321, -321})
_PDG_CHARGED_LEPTONS = frozenset({11, -11, 13, -13, 15, -15})
_PDG_NEUTRINOS = frozenset({12, -12, 14, -14, 16, -16})
_PDG_GAMMA = 22


class PrinceSpecies(object):
    """Bundles different particle properties for simplified
    availability of particle properties in :class:`prince_cr.core.PriNCeRun`.

    Args:
      pdgid (int): PDG ID of the particle
      particle_db (object): a dictionary with particle properties
      d (int): dimension of the energy grid
    """

    @staticmethod
    def calc_AZN(pdg_id):
        """Returns ``(A, Z, N)`` for a nucleus PDG ID; ``(0, 0, 0)`` otherwise."""
        from prince_cr.util import get_AZN

        return get_AZN(pdg_id)

    def __init__(self, pdgid, princeidx, d):
        info(5, "Initializing new species", pdgid)

        #: PDG Monte Carlo ID of particle
        self.pdgid = int(pdgid)
        #: (bool) particle is a hadron (meson or baryon)
        self.is_hadron = False
        #: (bool) particle is a meson
        self.is_meson = False
        #: (bool) particle is a baryon
        self.is_baryon = False
        #: (bool) particle is a lepton
        self.is_lepton = False
        #: (bool) if it's an electromagnetic particle
        self.is_em = False
        #: (bool) particle is a lepton
        self.is_charged = False
        #: (bool) particle is a nucleus
        self.is_nucleus = False
        #: (bool) particle has an energy redistribution
        self.has_redist = False
        #: (bool) particle is stable
        self.is_stable = True
        #: (float) lifetime
        self.lifetime = np.inf
        #: (bool) particle is an alias (PDG ID encodes special scoring behavior)
        self.is_alias = False
        #: (bool) passive tracking copy (see :class:`PrinceTrackedSpecies`)
        self.is_tracking = False
        #: (str) species name in string representation
        self.sname = None
        #: decay channels if any
        self.decay_channels = {}
        #: Mass, charge, neutron number
        self.A, self.Z, self.N = 1, None, None
        #: Mass in atomic units or GeV
        self.mass = None

        #: (int) Prince index (in state vector)
        self.princeidx = princeidx

        # (dict) Dimension of energy grids (for idx calculations)
        self.grid_dims = {"default": d}

        # Obtain values for the attributes
        self._init_species()

    def _init_species(self):
        """Fill all class attributes with values from :var:`spec_data`,
        dispatching on the PDG ID's particle class.
        """
        from prince_cr.util import is_nucleus, get_AZN

        pdgid = self.pdgid
        dbentry = spec_data[pdgid]

        if is_nucleus(pdgid):
            self.is_nucleus = True
            self.A, self.Z, self.N = get_AZN(pdgid)
            if self.A == 1:  # free p / n
                self.is_hadron = True
                self.is_baryon = True
        elif pdgid == _PDG_GAMMA:
            self.is_em = True
        elif pdgid in _PDG_MESONS:
            self.is_hadron = True
            self.is_meson = True
        elif pdgid in _PDG_CHARGED_LEPTONS:
            self.is_lepton = True
            if abs(pdgid) == 11:  # e±
                self.is_em = True
            elif abs(pdgid) == 13:  # μ±
                self.is_alias = True
        elif pdgid in _PDG_NEUTRINOS:
            self.is_lepton = True

        self.AZN = self.A, self.Z, self.N

        # All non-nuclei (and free p/n) get redistributed; A>=2 nuclei are
        # boost-conserving. The free nucleon case is the same in PDG and NCo:
        # PriNCe stores p/n yields differentially (x = E_da/E_γ).
        if (not self.is_nucleus) or self.A == 1:
            self.has_redist = True

        if "name" not in dbentry:
            info(5, "Name for species", pdgid, "not defined")
            self.sname = "nucleus_{0}".format(pdgid)
        else:
            self.sname = dbentry["name"]

        self.charge = dbentry["charge"]
        self.is_charged = self.charge != 0
        self.is_stable = dbentry["stable"]
        self.lifetime = dbentry["lifetime"]
        self.mass = dbentry["mass"]
        self.decay_channels = dbentry["branchings"]

    @property
    def sl(self):
        """Return the slice for this species on the grid
           can be used as spec[s.sl]

        Returns:
          (slice): a slice object pointing to the species in the state vecgtor
        """
        idx = self.princeidx
        dim = self.grid_dims["default"]
        return slice(idx * dim, (idx + 1) * dim)

    def lidx(self, grid_tag="default"):
        """Returns lower index of particle range in state vector.

        Returns:
          (int): lower index in state vector :attr:`PrinceRun.phi`
        """
        return self.princeidx * self.grid_dims[grid_tag]

    def uidx(self, grid_tag="default"):
        """Returns upper index of particle range in state vector.

        Returns:
          (int): upper index in state vector :attr:`PrinceRun.phi`
        """
        return (self.princeidx + 1) * self.grid_dims[grid_tag]

    def lbin(self, grid_tag="default"):
        """Returns lower bin of particle range in state vector.

        Returns:
          (int): lower bin in state vector :attr:`PrinceRun.phi`
        """
        return self.princeidx * (self.grid_dims[grid_tag] + 1)

    def ubin(self, grid_tag="default"):
        """Returns upper bin of particle range in state vector.

        Returns:
          (int): upper bin in state vector :attr:`PrinceRun.phi`
        """
        return (self.princeidx + 1) * (self.grid_dims[grid_tag] + 1)


#: Synthetic PDG namespace for tracked species. Chosen disjoint from the
#: nuclear PDG range (max ~1.01e9 at ``10LZZZAAAI``) and the hadron/lepton
#: range (|PDG| < 10^6). ``util.is_nucleus`` / ``util.get_AZN`` return
#: False / (0,0,0) for IDs in this band, so kernel branches that dispatch
#: on PDG arithmetic naturally skip tracked species as mothers — passive-
#: observer semantics fall out without an explicit guard.
_TRACKING_PDG_BASE = 2_000_000_000


_TRACKING_PROCESS_CLASSES = ("photo-nuclear", "decay", "both")


class PrinceTrackedSpecies(PrinceSpecies):
    """Passive tracking copy of an existing :class:`PrinceSpecies`.

    The species occupies its own slot in the state vector but takes all
    physics (``is_nucleus``, ``A/Z/N``, ``lifetime``, ``mass``,
    ``has_redist``, decay channels) from a *real* daughter identified by
    ``real_pdgid``. The synthetic ``pdgid`` is only an indexing key.

    See ``wiki/methods/tracking-species-design.md`` for the design.
    """

    def __init__(
        self,
        real_pdgid,
        tracking_pdgid,
        princeidx,
        d,
        *,
        parent_pdgs,
        process_class,
        e_gamma_range=None,
        alias=None,
    ):
        if process_class not in _TRACKING_PROCESS_CLASSES:
            raise ValueError(
                "process_class must be one of {0}, got {1!r}".format(
                    _TRACKING_PROCESS_CLASSES, process_class
                )
            )
        if process_class == "decay" and e_gamma_range is not None:
            raise ValueError(
                "e_gamma_range applies to the photon-energy axis of the "
                "photo-nuclear kernel; decay-only trackers must pass None."
            )
        self.real_pdgid = int(real_pdgid)
        self._tracking_alias = alias
        self.parent_pdgs = frozenset(int(p) for p in parent_pdgs)
        self.process_class = process_class
        self.e_gamma_range = (
            (float(e_gamma_range[0]), float(e_gamma_range[1]))
            if e_gamma_range is not None
            else None
        )
        super().__init__(tracking_pdgid, princeidx, d)
        # ``PrinceSpecies.__init__`` set is_tracking=False; flip after super.
        self.is_tracking = True

    def _init_species(self):
        """Dispatch physics on ``self.real_pdgid`` while keeping
        ``self.pdgid`` set to the synthetic indexing key.
        """
        synthetic = self.pdgid
        self.pdgid = self.real_pdgid
        try:
            super()._init_species()
        finally:
            self.pdgid = synthetic
        if self._tracking_alias:
            self.sname = self._tracking_alias


class SpeciesManager(object):
    """Provides a database with particle and species."""

    def __init__(self, pdgid_list, ed):
        # (dict) Dimension of primary grid
        self.grid_dims = {"default": ed}
        # Particle index shortcuts
        #: (dict) Converts PDG ID to index in state vector
        self.pdgid2princeidx = {}
        #: (dict) Converts particle name to index in state vector
        self.sname2princeidx = {}
        #: (dict) Converts PDG ID to reference of :class:`data.PrinceSpecies`
        self.pdgid2sref = {}
        #: (dict) Converts particle name to reference of
        #:class:`data.PrinceSpecies`
        self.sname2sref = {}
        #: (dict) Converts prince index to reference of
        #:class:`data.PrinceSpecies`
        self.princeidx2sref = {}
        #: (dict) Converts index in state vector to PDG ID
        self.princeidx2pdgid = {}
        #: (dict) Converts index in state vector to reference
        # of :class:`data.PrinceSpecies`
        self.princeidx2pname = {}
        #: (int) Total number of species
        self.nspec = 0

        #: Counter for synthetic tracked-species PDGs. Each
        #: :meth:`add_tracking_species` call increments and offsets
        #: ``_TRACKING_PDG_BASE`` to allocate a fresh synthetic ID.
        self._tracking_counter = 0
        #: Dict[int, list[PrinceTrackedSpecies]]: maps a *real* daughter
        #: PDG to the list of tracked species that mirror it. Built by
        #: :meth:`add_tracking_species`; consumed by cross-section and
        #: solver tracking hooks.
        self._tracked_by_real_da = {}

        self._gen_species(pdgid_list)
        self._init_species_tables()

    def _gen_species(self, pdgid_list):
        info(4, "Generating list of species.")

        # Make sure list is unique and sorted
        pdgid_list = sorted(list(set(pdgid_list)))

        self.species_refs = []
        # Define position in state vector (princeidx) by simply
        # incrementing it with the (sorted) list of PDG IDs
        for princeidx, pdgid in enumerate(pdgid_list):
            info(4, "Appending species {0} at position {1}".format(pdgid, princeidx))
            self.species_refs.append(
                PrinceSpecies(pdgid, princeidx, self.grid_dims["default"])
            )

        self.known_species = [s.pdgid for s in self.species_refs]
        self.redist_species = [s.pdgid for s in self.species_refs if s.has_redist]
        self.boost_conserv_species = [
            s.pdgid for s in self.species_refs if not s.has_redist
        ]

    def _init_species_tables(self):
        for s in self.species_refs:
            self.pdgid2princeidx[s.pdgid] = s.princeidx
            self.sname2princeidx[s.sname] = s.princeidx
            self.princeidx2pdgid[s.princeidx] = s.pdgid
            self.princeidx2pname[s.princeidx] = s.sname
            self.pdgid2sref[s.pdgid] = s
            self.princeidx2sref[s.princeidx] = s
            self.sname2sref[s.sname] = s

        self.nspec = len(self.species_refs)

    def add_tracking_species(
        self,
        *,
        parent_pdgs,
        daughter_pdg,
        process_class="both",
        e_gamma_range=None,
        alias=None,
    ):
        """Register a passive tracked copy of ``daughter_pdg``.

        ``parent_pdgs`` may be an iterable of PDG IDs or a predicate
        ``callable(PrinceSpecies) -> bool``; predicates are resolved
        against the current ``species_refs`` and frozen at registration
        time.

        Returns the new :class:`PrinceTrackedSpecies`. The species is
        appended to ``species_refs`` with a fresh ``princeidx``; callers
        must register all tracked species *before* dimensions like
        ``PriNCeRun.dim_states`` are computed and *before* the
        interaction-rate / solver matrices are built.
        """
        daughter_pdg = int(daughter_pdg)
        if daughter_pdg not in self.pdgid2sref:
            raise KeyError(
                "Real daughter PDG {0} is not in the species set; add it to "
                "the species list before registering a tracked copy.".format(
                    daughter_pdg
                )
            )

        if callable(parent_pdgs):
            predicate = parent_pdgs
            resolved = [
                s.pdgid for s in self.species_refs
                if not getattr(s, "is_tracking", False) and predicate(s)
            ]
        else:
            resolved = [int(p) for p in parent_pdgs]
        if not resolved:
            raise ValueError(
                "No parent species matched for tracked daughter {0}; check "
                "the parent_pdgs argument.".format(daughter_pdg)
            )

        self._tracking_counter += 1
        synthetic = _TRACKING_PDG_BASE + self._tracking_counter

        if alias is None:
            real_sname = self.pdgid2sref[daughter_pdg].sname
            alias = "{0}_tracked_{1}".format(real_sname, self._tracking_counter)
        if alias in self.sname2sref:
            raise ValueError(
                "Tracked-species alias {0!r} clashes with an existing "
                "species name; pass a unique ``alias=``.".format(alias)
            )

        # Sanity-warn on overlapping e_gamma windows in the same
        # (parent_set, daughter, process_class) bucket.
        if e_gamma_range is not None:
            import warnings

            E_lo, E_hi = float(e_gamma_range[0]), float(e_gamma_range[1])
            for other in self._tracked_by_real_da.get(daughter_pdg, []):
                if other.process_class != process_class:
                    continue
                if other.parent_pdgs != frozenset(resolved):
                    continue
                if other.e_gamma_range is None:
                    continue
                o_lo, o_hi = other.e_gamma_range
                # Half-open [lo, hi) — touching edges don't overlap.
                if E_lo < o_hi and o_lo < E_hi:
                    warnings.warn(
                        "Tracked species {0!r} overlaps energy window of "
                        "existing {1!r} ([{2}, {3}) vs [{4}, {5})); the "
                        "overlap region will be double-counted across "
                        "trackers.".format(
                            alias, other.sname, E_lo, E_hi, o_lo, o_hi
                        ),
                        RuntimeWarning,
                        stacklevel=2,
                    )

        princeidx = len(self.species_refs)
        trk = PrinceTrackedSpecies(
            real_pdgid=daughter_pdg,
            tracking_pdgid=synthetic,
            princeidx=princeidx,
            d=self.grid_dims["default"],
            parent_pdgs=resolved,
            process_class=process_class,
            e_gamma_range=e_gamma_range,
            alias=alias,
        )
        # Apply any non-default grid_tags registered before this call so the
        # new species reports correct lidx / uidx for them.
        trk.grid_dims = self.grid_dims

        self.species_refs.append(trk)
        self.pdgid2princeidx[synthetic] = princeidx
        self.sname2princeidx[alias] = princeidx
        self.princeidx2pdgid[princeidx] = synthetic
        self.princeidx2pname[princeidx] = alias
        self.pdgid2sref[synthetic] = trk
        self.princeidx2sref[princeidx] = trk
        self.sname2sref[alias] = trk
        self.known_species.append(synthetic)
        if trk.has_redist:
            self.redist_species.append(synthetic)
        else:
            self.boost_conserv_species.append(synthetic)
        self.nspec = len(self.species_refs)
        self._tracked_by_real_da.setdefault(daughter_pdg, []).append(trk)
        return trk

    def tracked_species_for(self, da_real_pdg):
        """Return tracked species whose ``real_pdgid`` equals
        ``da_real_pdg``; empty list when none exist."""
        return list(self._tracked_by_real_da.get(int(da_real_pdg), ()))

    def has_tracked_species(self):
        return self._tracking_counter > 0

    def add_tracking_neutrinos_from_nuclei(self, min_A=2):
        """Register six tracked neutrinos (ν_e, ν̄_e, ν_μ, ν̄_μ, ν_τ, ν̄_τ)
        each capturing the flux produced from decays of nuclei with
        ``A >= min_A``. Returns the list of new tracked species. Skips
        flavours whose real species is absent from the state vector.
        """
        nu_pdgs = (12, -12, 14, -14, 16, -16)
        from prince_cr.util import is_nucleus, get_AZN
        registered = []
        for nu in nu_pdgs:
            if nu not in self.pdgid2sref:
                continue
            real_sname = self.pdgid2sref[nu].sname
            trk = self.add_tracking_species(
                parent_pdgs=lambda s, _min=min_A: (
                    is_nucleus(s.pdgid) and get_AZN(s.pdgid)[0] >= _min
                ),
                daughter_pdg=nu,
                process_class="decay",
                alias="{0}_from_nuclei".format(real_sname),
            )
            registered.append(trk)
        return registered

    def add_tracking_charge_exchange(self, parent_pdgid=2212, daughter_pdgid=2112):
        """Register a tracked daughter (default: neutron) capturing the
        flux produced from photo-hadronic conversion of ``parent_pdgid``
        (default: proton). Returns the new tracked species."""
        real = self.pdgid2sref[int(daughter_pdgid)].sname
        parent = self.pdgid2sref[int(parent_pdgid)].sname
        return self.add_tracking_species(
            parent_pdgs=[int(parent_pdgid)],
            daughter_pdg=int(daughter_pdgid),
            process_class="photo-nuclear",
            alias="{0}_from_{1}".format(real, parent),
        )

    def add_tracking_photo_nuclear_regimes(
        self,
        *,
        parent_pdgs,
        daughter_pdg,
        e_gamma_partition,
    ):
        """Register one tracked species per energy bucket in
        ``e_gamma_partition``. Each partition entry is
        ``(E_lo, E_hi, label)``; the resulting alias is
        ``"<daughter_sname>_from_<parent_alias>_<label>"``. The buckets
        are not checked for full coverage — caller can pass an extra
        ``e_gamma_range=None`` tracker as a sum-rule check.

        Returns the list of new tracked species in the order given.
        """
        parents = list(parent_pdgs)
        if len(parents) == 1:
            parent_label = self.pdgid2sref[int(parents[0])].sname
        else:
            parent_label = "_".join(
                self.pdgid2sref[int(p)].sname for p in parents
            )
        real = self.pdgid2sref[int(daughter_pdg)].sname
        registered = []
        for E_lo, E_hi, label in e_gamma_partition:
            trk = self.add_tracking_species(
                parent_pdgs=parents,
                daughter_pdg=int(daughter_pdg),
                process_class="photo-nuclear",
                e_gamma_range=(E_lo, E_hi),
                alias="{0}_from_{1}_{2}".format(real, parent_label, label),
            )
            registered.append(trk)
        return registered

    def add_grid(self, grid_tag, dimension):
        """Defines additional grid dimensions under a certain tag.

        Propagates changes to this variable to all known species.
        """
        info(2, "New grid_tag", grid_tag, "with dimension", dimension)
        self.grid_dims[grid_tag] = dimension

        for s in self.species_refs:
            s.grid_dims = self.grid_dims

    def __repr__(self):
        str_out = ""
        ident = 3 * " "
        for s in self.species_refs:
            str_out += s.sname + "\n" + ident
            str_out += "PDG id : " + str(s.pdgid) + "\n" + ident
            str_out += "PriNCe idx : " + str(s.princeidx) + "\n\n"

        return str_out
