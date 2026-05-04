"""Module inteded to contain some prince-specific data structures."""

import pickle as pickle
import os.path as path
import numpy as np
import scipy.constants as spc
import h5py


from prince_cr.util import convert_to_namedtuple, info, make_nucleus_pdg
import prince_cr.config as config


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
            "light_yields":      g["light_yields"][:],
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
            "light_yields":    g["light_yields"][:],
            "xbins":           g["xbins"][:],
            "n_decays_per_mother":
                int(g.attrs.get("n_decays_per_mother", 1)),
        }


_LN2 = float(np.log(2.0))


def _heuristic_beta_branching(mo_pdg):
    """Best-effort branchings for an unstable nucleus that the FLUKA decay
    db has *no* explicit channels for (SPDCEV failures — ~4% of mothers).

    Direction picked from neutron excess: β⁻ if Z < A/2, β⁺ otherwise.
    Returns a `[(mult, [da])]` list in the same shape PriNCe's chain
    reducer expects, with the nuclear daughter, an electron(/positron),
    and the matching neutrino as separate one-daughter entries (so each
    can be followed independently)."""
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
                heuristic = _heuristic_beta_branching(mo_pdg)
                if heuristic:
                    branchings = heuristic

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
        # are kept. We only overwrite the branching list with Pythia's actual
        # tabulated final-state composition + dN/dx tables.
        for mo_raw in pythia["decay_mothers"]:
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

            if mo_pdg in spec_data and branchings:
                spec_data[mo_pdg]["branchings"] = branchings

    info(
        2,
        "Tabulated FLUKA/Pythia decays merged: "
        "{0} (mo,da) channels".format(len(_TABULATED_DECAY_DX)),
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

    def fluka_photo_nuclear_db(self, model_tag, e_range=None):
        """Read photo_nuclear/<tag>/ from the FLUKA db (a separate file from
        ``db_fname``). Lazy: opens the file on each call.

        Args:
            model_tag (str): subgroup tag, e.g. ``"FLUKA_2025"``.
            e_range (tuple or None): ``(e_min, e_max)`` GeV bounds.

        Returns:
            dict with keys: ``energy_grid``, ``xbins``, ``inel_mothers``,
            ``mothers_daughters``, ``elementary_daughters``,
            ``inelastic_cross_sctions`` (typo preserved), ``fragment_yields``
            (n_ch, n_E), ``elementary_yields`` (n_em, n_E, n_x).
        """
        info(10, "Reading FLUKA photo-nuclear db. tag={0}".format(model_tag))
        fpath = path.join(config.fluka_db_path, config.fluka_db_fname)
        if not path.isfile(fpath):
            raise FileNotFoundError(
                f"FLUKA db not found at {fpath}. "
                "Set prince_cr.config.fluka_db_path to its directory "
                "(e.g. the prince-fluka-utils repo root)."
            )

        db_entry = {}
        with h5py.File(fpath, "r") as fdb:
            self._check_subgroup_exists(fdb["photo_nuclear"], model_tag)
            grp = fdb["photo_nuclear"][model_tag]

            egrid_full = grp["energy_grid"][:]
            sl = self._energy_slice(egrid_full, e_range)
            db_entry["energy_grid"] = egrid_full[sl]
            db_entry["xbins"] = grp["xbins"][:]

            for entry in ("inel_mothers", "mothers_daughters", "elementary_daughters"):
                info(10, "Reading entry {0} from FLUKA db.".format(entry))
                db_entry[entry] = grp[entry][:]

            # σ_inel: (n_m, n_E) — slice last axis
            db_entry["inelastic_cross_sctions"] = grp["inelastic_cross_sctions"][:, sl]
            # heavy yields: (n_ch, n_E) — slice last axis
            db_entry["fragment_yields"] = grp["fragment_yields"][:, sl]
            # elementary yields: (n_em, n_E, n_x) — slice middle axis
            db_entry["elementary_yields"] = grp["elementary_yields"][:, sl, :]

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
