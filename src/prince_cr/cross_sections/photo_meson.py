"""SOPHIA photo-meson cross sections (proton / neutron), PDG-native.

Reads the **repacked** SOPHIA database (``config.sophia_db_fname``, default
``prince_db_sophia_pdg.h5``) produced by ``scripts/repack_sophia_pdg.py`` from
the legacy ``photo_nuclear/SOPHIA`` tables. The repacked db already carries PDG
mother/secondary ids (p=2212, n=2112, gamma=22, e+-=+-11, pi+-0=211/-211/111,
mu=13/-13, K=321/-321, neutrinos), so this loader does no id translation.

The mesons are stored **pre-decay**; ``_optimize_and_generate_index`` runs the
modern decay-chain reducer, which folds pi0->gamma gamma and pi+- -> mu -> e nu
into the stable final-state channels (gamma / e / nu) using the PDG decay
database — the same machinery the FLUKA model relies on, so no decay tooling is
duplicated here.

This restored variant covers free nucleons (the model needed for proton-only
propagation and for SOPHIA-vs-FLUKA photopion-yield studies); nuclear
superposition is intentionally not reimplemented.
"""

from os import path

import numpy as np

from prince_cr.util import info, get_AZN
import prince_cr.config as config

from .base import CrossSectionBase


class SophiaSuperposition(CrossSectionBase):
    """SOPHIA proton/neutron photo-meson model (PDG-native loader)."""

    def __init__(self, *args, **kwargs):
        self.supports_redistributions = True
        CrossSectionBase.__init__(self)
        self._load()
        self._optimize_and_generate_index()

    def _load(self):
        from prince_cr.data import db_handler

        db_file = path.join(config.sophia_db_path, config.sophia_db_fname)
        info(2, "Load SOPHIA photo-meson cross sections from {0}".format(db_file))
        tab = db_handler.photo_meson_db(
            "SOPHIA", e_range=config.cross_section_e_range, db_fname=db_file
        )

        self._egrid_tab = tab["energy_grid"]
        self.xbins = tab["xbins"]
        pid_nonel = tab["inel_mothers"]
        pids_incl = tab["mothers_daughters"]
        nonel_raw = tab["inelastic_cross_sctions"]
        incl_raw = tab["fragment_yields"]

        # Non-elastic (total photo-meson) cross sections for p / n.
        self.cs_proton_grid = nonel_raw[pid_nonel == 2212].flatten()
        self.cs_neutron_grid = nonel_raw[pid_nonel == 2112].flatten()

        # Redistribution functions, keyed by PDG daughter.
        self.redist_proton = {}
        self.redist_neutron = {}
        for (mo, da), csgrid in zip(pids_incl, incl_raw):
            arr = np.asarray(csgrid, dtype=float)
            if mo == 2212:
                self.redist_proton[int(da)] = arr
            elif mo == 2112:
                self.redist_neutron[int(da)] = arr
            else:
                raise Exception(
                    "SOPHIA model only knows nucleons, but mother id is {0}".format(mo)
                )

        # Materialise PDG-keyed tabs. Raw redist arrays are (nE, nx); the
        # pipeline expects differential channels as (nx, nE).
        self._nonel_tab = {2212: self.cs_proton_grid, 2112: self.cs_neutron_grid}
        self._incl_tab = {}
        self._incl_diff_tab = {}
        for pdg, grid in self.redist_proton.items():
            self._incl_diff_tab[(2212, pdg)] = grid.T
        for pdg, grid in self.redist_neutron.items():
            self._incl_diff_tab[(2112, pdg)] = grid.T

        self.redist_shape = (self.xbins.shape[0], self._egrid_tab.shape[0])
        self.set_range()
        info(2, "SOPHIA photo-meson loading finished")

    def nonel(self, mother):
        r"""Non-elastic (total photo-meson) cross section for a free nucleon."""
        _, Z, N = get_AZN(mother)
        cgrid = Z * self.cs_proton_grid + N * self.cs_neutron_grid
        return self.egrid, cgrid[self._range]
