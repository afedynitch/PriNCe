"""The module contains everything to handle cross section interfaces."""

from abc import ABCMeta

import numpy as np

import prince_cr.decays as decs
from prince_cr.data import spec_data
from prince_cr.util import (
    info,
    bin_widths,
    get_AZN,
    is_nucleus,
)
import prince_cr.config as config


def _is_redistributed(pdg_id):
    """True iff secondaries of this PDG ID are stored differentially in
    ``x = E_secondary / E_γ`` (free p/n and all elementary species), False
    if they are boost-conserving (heavy nuclei A>=2).
    """
    A, _, _ = get_AZN(pdg_id)
    if A >= 2:
        return False
    return True


class CrossSectionBase(object, metaclass=ABCMeta):
    """Base class for cross section interfaces to tabulated models.

    The class is abstract and it is not inteded to be instantiated.

    Child Classes either define the tables:
        self._egrid_tab
        self._nonel_tab
        self._incl_tab
        self._incl_diff

    Or directly reimplememnt the functions
        nonel(self, mother, daughter)
        incl(self, mother, daughter)
        incl_diff(self, mother, daughter)

    The flag self.supports_redistributions = True/False should be set
    To tell the class to include/ignore incl_diff
    """

    def __init__(self):
        # Tuple, defining min and max energy cuts on the grid
        self._range = None
        # Energy grid, as defined in files
        self._egrid_tab = None
        # Dictionary of nonel. cross sections on egrid, indexed by (mother)
        self._nonel_tab = {}
        # Dictionary of incl. cross sections on egrid, indexed by (mother, daughter)
        self._incl_tab = {}
        # Dictionary of incl. diff. cross sections on egrid, indexed by (mother, daughter)
        self._incl_diff_tab = {}
        # List of available mothers for nonel cross sections
        self.nonel_idcs = []
        # List of available (mothers,daughter) reactions in incl. cross sections
        self.incl_idcs = []
        # List of available (mothers,daughter) reactions in incl. diff. cross sections
        self.incl_diff_idcs = []
        # O(1) membership companion to incl_diff_idcs. Kept in sync by
        # `_update_indices`; queried by the hot ``is_differential`` path
        # in the decay-chain reducer (606k calls at max_mass=245).
        self._incl_diff_idcs_set = set()
        # Common grid in x (the redistribution variable)
        self.xbins = None

        # Flag, which tells if the model supports secondary redistributions
        if not hasattr(self, "supports_redistributions"):
            self.supports_redistributions = (
                None  # JH: to differ from explicitly set False
            )
        # List of all known particles (after optimization)
        self.known_species = []
        # List of all boost conserving inclusive channels (after optimization)
        self.known_bc_channels = []
        # List of all differential inclusive channels (after optimization)
        self.known_diff_channels = []
        # O(1) companions to the lists above — `interaction_rates`
        # iterates `dcr × dcr` species pairs and queries channel
        # membership for each, which is hundreds of thousands of
        # `tuple in list` checks at heavy max_mass. Synced inside
        # `_optimize_and_generate_index`.
        self.known_bc_channels_set = set()
        self.known_diff_channels_set = set()
        # Dictionary of (mother, daughter) reactions for each mother
        self.reactions = {}

        # Class name of the model
        self.mname = self.__class__.__name__

    def set_range(self, e_min=None, e_max=None):
        """Set energy range within which to return tabulated data.

        Args:
            e_min (float): minimal energy in GeV
            e_max (float): maximal energy in GeV
        """
        if e_min is None:
            e_min = np.min(self._egrid_tab)
        if e_max is None:
            e_max = np.max(self._egrid_tab)

        info(5, "Setting range to {0:3.2e} - {1:3.2e}".format(e_min, e_max))
        self._range = np.where((self._egrid_tab >= e_min) & (self._egrid_tab <= e_max))[0]
        info(
            2,
            "Range set to {0:3.2e} - {1:3.2e}".format(
                np.min(self._egrid_tab[self._range]),
                np.max(self._egrid_tab[self._range]),
            ),
        )

    @property
    def egrid(self):
        """Returns energy grid of the tabulated data in selected range.

        Returns:
            (numpy.array): Energy grid in GeV
        """

        return self._egrid_tab[self._range]

    @property
    def xcenters(self):
        """Returns centers of the grid in x.

        Returns:
            (numpy.array): x grid
        """

        return 0.5 * (self.xbins[1:] + self.xbins[:-1])

    @property
    def xwidths(self):
        """Returns bin widths of the grid in x.

        Returns:
            (numpy.array): x widths
        """

        return self.xbins[1:] - self.xbins[:-1]

    @property
    def resp(self):
        """Return ResponseFunction corresponding to this cross section
        Will only create the Response function once.
        """
        if not hasattr(self, "_resp"):
            info(2, "First Call, creating instance of ResponseFunction now")
            from .response import ResponseFunction

            self._resp = ResponseFunction(self)
        return self._resp

    def is_differential(self, mother, daughter):
        """Returns true if the model supports redistributions and requested
        mother/daughter combination should return non-zero redistribution matrices.

        Args:
            mother (int): PDG ID of mother particle
            daughter (int): PDG ID of daughter particle

        Returns:
            (bool): ``True`` if the model has this particular redistribution function
        """
        if (
            _is_redistributed(daughter)
            or (mother, daughter) in self._incl_diff_idcs_set
        ):
            info(60, "Daughter requires redistribution.", mother, daughter)
            return True
        info(60, "Daughter conserves boost.", mother, daughter)
        return False

    def _update_indices(self):
        """Updates the list of indices according to entries in the
        _tab variables"""

        self.nonel_idcs = sorted(self._nonel_tab.keys())
        self.incl_idcs = sorted(self._incl_tab.keys())
        self.incl_diff_idcs = sorted(self._incl_diff_tab.keys())
        # Keep the O(1) membership companion in sync. `is_differential`
        # is called O(1e6) times during chain reduction at heavy
        # max_mass, so a list scan dominates init.
        self._incl_diff_idcs_set = set(self.incl_diff_idcs)

    def generate_incl_channels(self, mo_indices):
        """Generates indices for all allowed channels given mo_indices
            Note: By default this returns an empty list,
                  meant to be overwritten in cases where
                  the child class needs to dynamically generate indices

        Args:
            mo_indices (list of ints): list of indices for mother nuclei

        Returns:
           Returns:
            list of tuples: list of allowed channels given as (mo_idx, da_idx)
        """
        incl_channels = []

        return incl_channels

    def _optimize_and_generate_index(self):
        """Construct a list of mothers and (mother, daughter) indices.

        Args:
            just_reactions (bool): If True then fill just the reactions index.
        """

        # Integrate out short lived processes and leave only stable particles
        # in the databases
        self._reduce_channels()

        # Go through all three cross section categories
        # index contents in the ..known..variable
        self.reactions = {}

        self._update_indices()

        for mo, da in self.incl_idcs:
            if is_nucleus(da) and get_AZN(da)[0] > get_AZN(mo)[0]:
                raise Exception(
                    "Daughter {0} heavier than mother {1}. Physics??".format(da, mo)
                )

            if mo not in self.reactions:
                self.reactions[mo] = []
                self.known_species.append(mo)

            if (mo, da) not in self.reactions[mo]:
                # Make sure it's a unique list
                self.reactions[mo].append((mo, da))
            if self.is_differential(mo, da):
                # Move the distributions which are expected to be differential
                # to _incl_diff_tab
                self._incl_diff_tab[(mo, da)] = self._arange_on_xgrid(
                    self._incl_tab.pop((mo, da))
                )
                info(10, "Channel {0} -> {1} forced to be differential.")
            else:
                self.known_bc_channels.append((mo, da))
                self.known_species.append(da)

        for mo, da in list(self._incl_diff_tab.keys()):
            if is_nucleus(da) and get_AZN(da)[0] > get_AZN(mo)[0]:
                raise Exception(
                    "Daughter {0} heavier than mother {1}. Physics??".format(da, mo)
                )

            if mo not in self.reactions:
                self.reactions[mo] = []
                self.known_species.append(mo)

            if (mo, da) not in self.reactions[mo]:
                # Make sure it's a unique list to avoid unnecessary loops
                self.reactions[mo].append((mo, da))
                self.known_diff_channels.append((mo, da))
                self.known_species.append(da)

        # Remove duplicates
        self.known_species = sorted(list(set(self.known_species)))
        self.known_bc_channels = sorted(list(set(self.known_bc_channels)))
        self.known_diff_channels = sorted(list(set(self.known_diff_channels)))

        for sp in self.known_species:
            if is_nucleus(sp) and (sp, sp) not in self.known_diff_channels:
                self.known_bc_channels.append((mo, mo))
            if (mo, mo) not in self.reactions[mo]:
                self.reactions[mo].append((mo, mo))

        # Membership companions used by `interaction_rates` to bypass
        # O(N) `tuple in list` lookups in the dcr × dcr species product.
        self.known_bc_channels_set = set(self.known_bc_channels)
        self.known_diff_channels_set = set(self.known_diff_channels)

        # Make sure the indices are up to date
        self._update_indices()


    def _reduce_channels(self):
        """Follows decay chains until all inclusive reactions point to
        stable final state particles.

        The ``tau_dec_threshold`` parameter in the config controls the
        definition of stable. Unstable nuclei for which no decay channels
        are known are simply dropped (no forced beta decay).
        """
        threshold = config.tau_dec_threshold
        info(2, "Integrating out species with lifetime smaller than", threshold)
        info(
            3,
            (
                "Before optimization, the number of known primaries is {0} with "
                + "in total {1} inclusive channels"
            ).format(len(self._nonel_tab), len(self._incl_tab)),
        )

        # Drop unstable mothers from the nonel dictionary first; the chain
        # reduction below only walks daughters of mothers that survive.
        for mother in sorted(self._nonel_tab.keys()):
            if mother not in spec_data or spec_data[mother]["lifetime"] < threshold:
                info(
                    20,
                    "Primary species {0} does not fulfill stability criteria.".format(
                        mother
                    ),
                )
                _ = self._nonel_tab.pop(mother)
        self._update_indices()

        # Drop incl entries whose mother is no longer stable; for the rest,
        # move differential channels to _incl_diff_tab.
        for mother, daughter in self.incl_idcs:
            if mother not in self.nonel_idcs:
                info(
                    30,
                    "Removing {0}/{1} from incl, since mother not stable ".format(
                        mother, daughter
                    ),
                )
                _ = self._incl_tab.pop((mother, daughter))

            elif self.is_differential(mother, daughter):
                self._incl_diff_tab[(mother, daughter)] = self._arange_on_xgrid(
                    self._incl_tab.pop((mother, daughter))
                )

        self._update_indices()

        for mother, daughter in self.incl_diff_idcs:
            if mother not in self.nonel_idcs:
                info(
                    30,
                    "Removing {0}/{1} from diff incl, since mother not stable ".format(
                        mother, daughter
                    ),
                )
                _ = self._incl_diff_tab.pop((mother, daughter))

        self._update_indices()

        # Walk every channel's decay chain and accumulate stable-final-state
        # contributions into the reducer's two output dicts.
        reducer = _DecayChainReducer(self, threshold)
        for (mo, da), value in list(self._incl_tab.items()):
            reducer.follow(mo, da, value)
        for (mo, da), value in list(self._incl_diff_tab.items()):
            reducer.follow(mo, da, value)

        self._incl_tab = dict(reducer.new_incl_tab)
        self._incl_diff_tab = dict(reducer.new_dec_diff_tab)

        info(
            3,
            (
                "After optimization, the number of known primaries is {0} with "
                + "in total {1} inclusive channels"
            ).format(
                len(self._nonel_tab), len(self._incl_tab) + len(self._incl_diff_tab)
            ),
        )
        info(
            2,
            f"Cache used for decays: {reducer.decay_cache_size} entries.",
        )

    def nonel_scale(self, mother, scale="A"):
        """Returns the nonel cross section scaled by `scale`.

        Convenience funtion for plotting, where it is important to
        compare the cross section per nucleon.

        Args:
            mother (int): Mother nucleus(on)
            scale (float): If `A` then nonel/A is returned, otherwise
                           scale can be any float.

        Returns:
            (numpy.array, numpy.array): Tuple of Energy grid in GeV,
                                        scale * inclusive cross section
                                        in :math:`cm^{-2}`
        """

        egr, csection = self.nonel(mother)

        if scale == "A":
            scale = 1.0 / get_AZN(mother)[0]

        return egr, scale * csection

    def incl_scale(self, mother, daughter, scale="A"):
        """Same as :func:`~cross_sections.CrossSectionBase.nonel_scale`,
        just for inclusive cross sections.
        """

        egr, csection = self.incl(mother, daughter)

        if scale == "A":
            scale = 1.0 / get_AZN(mother)[0]

        return egr, scale * csection

    def nonel(self, mother):
        """Returns non-elastic cross section.

        Absorption cross section of `mother`, which is
        the total minus elastic, or in other words, the inelastic
        cross section.

        Args:
            mother (int): Mother nucleus(on)

        Returns:
            (numpy.array, numpy.array): Tuple of Energy grid in GeV, inclusive cross
                                        section in :math:`cm^{-2}`
        """

        if mother not in self._nonel_tab:
            raise Exception("Mother {0} unknown.".format(mother))

        if isinstance(self._nonel_tab[mother], tuple):
            return self._nonel_tab[mother]
        else:
            return self.egrid, self._nonel_tab[mother][self._range]

    def incl(self, mother, daughter):
        """Returns inclusive cross section.

        Inclusive cross section for daughter in photo-nuclear
        interactions of `mother`.

        Args:
            mother (int): Mother nucleus(on)
            daughter (int): Daughter nucleus(on)

        Returns:
            (numpy.array): Inclusive cross section in :math:`cm^{-2}`
                           on self._egrid_tab
        """

        from scipy.integrate import trapezoid as trapz

        if (mother, daughter) in self._incl_diff_tab:
            # Return the integral of the differential for the inclusive
            egr_incl, cs_diff = self.incl_diff(mother, daughter)
            # diff_mat = diff_mat.transpose()
            cs_incl = trapz(cs_diff, x=self.xcenters, dx=bin_widths(self.xbins), axis=0)

            if isinstance(self._incl_diff_tab[(mother, daughter)], tuple):
                return egr_incl, cs_incl

            return self.egrid, cs_incl[self._range]

        elif (mother, daughter) not in self._incl_tab:
            raise Exception(
                self.__class__.__name__
                + "::({0},{1}) combination not in inclusive cross sections".format(
                    mother, daughter
                )
            )

        # If _nonel_tab contains tuples of (egrid, cs) return tuple
        # otherwise return (egrid, cs) in range defined by self.range

        if isinstance(self._incl_tab[(mother, daughter)], tuple):
            return self._incl_tab[(mother, daughter)]
        return self.egrid, self._incl_tab[(mother, daughter)][self._range]

    def incl_diff(self, mother, daughter):
        """Returns inclusive cross section.

        Inclusive differential cross section for daughter in photo-nuclear
        interactions of `mother`. Only defined, if the daughter is distributed
        in :math:`x = E_{da} / E_{mo}`

        Args:
            mother (int): Mother nucleus(on)
            daughter (int): Daughter nucleus(on)

        Returns:
            (numpy.array): Inclusive cross section in :math:`cm^{-2}`
                           on self._egrid_tab
        """

        if (mother, daughter) not in self._incl_diff_tab:
            raise Exception(
                self.__class__.__name__
                + "({0},{1}) combination not in inclusive differential cross sections".format(
                    mother, daughter
                )
            )

        # If _nonel_tab contains tuples of (egrid, cs) return tuple
        # otherwise return (egrid, cs) in range defined by self.range

        if isinstance(self._incl_diff_tab[(mother, daughter)], tuple):
            return self._incl_diff_tab[(mother, daughter)]
        return self.egrid, self._incl_diff_tab[(mother, daughter)][:, self._range]

    def _arange_on_xgrid(self, incl_cs):
        """Returns the inclusive cross section on an xgrid at x=1."""

        egr, cs = None, None

        if isinstance(incl_cs, tuple):
            egr, cs = incl_cs
        else:
            cs = incl_cs

        nxbins = len(self.xbins) - 1
        if len(cs.shape) > 1 and cs.shape[0] != nxbins:
            raise Exception(
                "One dimensional cross section expected, instead got",
                cs.shape,
                "\n",
                cs,
            )
        elif len(cs.shape) == 2 and cs.shape[0] == nxbins:
            info(20, "Supplied 2D distribution seems to be distributed in x.")
            if isinstance(incl_cs, tuple):
                return egr, cs
            return cs

        csec = np.zeros((nxbins, cs.shape[0]))
        # NOTE: The factor 2 in the following line is a workarround to account for the latter linear interpolation
        #       This is needed because linear spline integral will result in a trapz,
        #       which has half the area of the actual first bin
        corr_factor = 2 * self.xwidths[-1] / (self.xcenters[-1] - self.xcenters[-2])
        csec[-1, :] = cs / self.xwidths[-1] * corr_factor
        info(
            4,
            "Warning! Workaround to account for linear interpolation in x, factor 2 added!",
        )
        if isinstance(incl_cs, tuple):
            return egr, csec
        return csec

    def multiplicities(self, mother, daughter):
        """Return the multiplicities from either the inclusive channels, or the
        differential ones integrated by x, as a function of Energy.
        """
        egrid_incl, cs_incl = self.incl(mother, daughter)
        egrid_nonel, cs_nonel = self.nonel(mother)

        if egrid_incl.shape != egrid_nonel.shape:
            raise Exception("Problem with different grid shapes")

        multiplicities = cs_incl / np.where(cs_nonel == 0, np.inf, cs_nonel)

        return egrid_nonel, multiplicities


class _DecayChainReducer(object):
    """Walk decay chains for `CrossSectionBase._reduce_channels`.

    For a given owning cross-section and lifetime threshold, `follow` recurses
    on a (mother, daughter) channel until the daughter is stable (lifetime
    >= threshold), accumulating the cross section -- multiplied by the
    branching ratios along the way -- into one of two output dicts on this
    instance:

      - `new_incl_tab`:    boost-conserving stable daughters
      - `new_dec_diff_tab`: stable daughters that carry an x redistribution

    The decay distribution between two species (`int_scale * decay_matrix`)
    is cached on the instance so we don't rebuild it for repeated chained
    daughters.
    """

    def __init__(self, owner, threshold):
        from prince_cr.util import AdditiveDictionary

        self.owner = owner
        self.threshold = threshold
        self.new_incl_tab = AdditiveDictionary()
        self.new_dec_diff_tab = AdditiveDictionary()

        bc = owner.xcenters
        bw = bin_widths(owner.xbins)
        dec_bins = np.outer(owner.xbins, 1 / bc)
        self._dec_bins_lower = dec_bins[:-1]
        self._dec_bins_upper = dec_bins[1:]
        self._int_scale = np.tile(bw / bc, (len(bc), 1))

        self._decay_cache = {}

    @property
    def decay_cache_size(self):
        return len(self._decay_cache)

    @staticmethod
    def _dbg_indent(reclev):
        return 4 * reclev * "-" + ">" if reclev else ""

    def _decay_dist(self, mother, daughter):
        key = (mother, daughter)
        cached = self._decay_cache.get(key)
        if cached is None:
            cached = self._int_scale * decs.get_decay_matrix_bin_average(
                mother, daughter, self._dec_bins_lower, self._dec_bins_upper
            )
            self._decay_cache[key] = cached
        return cached

    def _convolve(self, diff_dist, mother, daughter, branching_ratio):
        r"""Convolve `diff_dist` (already on the x-grid) with the
        mother→daughter decay distribution and scale by the branching ratio.

        :math:`\frac{{\rm d}N^{A\gamma \to \mu}}{{\rm d}x_j} =
        \sum_{i=0}^{N_x}~\Delta x_i
        \frac{{\rm d}N^{A\gamma \to \pi}}{{\rm d} x_i}~
        \frac{{\rm d}N^{\pi \to \mu}}{{\rm d} x_j}`
        """
        dec_dist = self._decay_dist(mother, daughter)
        info(20, "convolving with decay dist", mother, daughter)
        if not isinstance(diff_dist, tuple):
            return branching_ratio * dec_dist.dot(diff_dist)
        return diff_dist[0], branching_ratio * dec_dist.dot(diff_dist[1])

    #: Hard cap on chain depth. Even acyclic β-chains in nature don't exceed
    #: ~20 hops (U-238 → Pb-206 is 14); anything deeper is almost certainly
    #: a cycle that escaped the visited-set check, or a pathology in the
    #: tabulated decay db. Treated as a stable terminal at the cap.
    _MAX_CHAIN_DEPTH = 64

    def follow(self, first_mo, da, csection, reclev=0, visited=None):
        """Recurse into `da`'s decay chain, anchored on `first_mo`.

        Either records `csection` against a stable terminal daughter, or
        delegates to `_recurse_into_daughters` to walk one more level.

        ``visited`` carries the set of unstable daughters already walked
        on the current branch so we can detect A→…→A cycles introduced
        by the tabulated FLUKA decay db (adjacent isobar pairs sometimes
        carry both β⁻ and β⁺ entries that loop back). On a cycle hit we
        record the entry as stable terminal — flux-preserving and a
        better approximation than running the chain to numerical death.
        """
        info(10, self._dbg_indent(reclev), "Entering with", first_mo, da)

        if da not in spec_data:
            info(
                3,
                self._dbg_indent(reclev),
                "daughter {0} unknown. Force beta decay not implemented!!".format(da),
            )
            return

        if spec_data[da]["lifetime"] >= self.threshold:
            self._record_stable(first_mo, da, csection, reclev)
            return

        if reclev >= self._MAX_CHAIN_DEPTH:
            import warnings
            warnings.warn(
                "Decay chain depth {0} reached on PDG {1} (primary mother "
                "{2}); recording as terminal stable to break runaway. "
                "Natural β-chains shouldn't exceed ~20 hops — check the "
                "tabulated decay db for corrupted half_lives or missing "
                "stable terminals.".format(reclev, da, first_mo),
                RuntimeWarning,
                stacklevel=2,
            )
            self._record_stable(first_mo, da, csection, reclev)
            return

        if visited is None:
            visited = frozenset()
        if da in visited:
            import warnings
            warnings.warn(
                "Decay-chain CYCLE at PDG {0} (primary mother {1}). "
                "Visited so far: {2}. Treating as terminal stable. "
                "This is a pathology in the tabulated decay db — adjacent "
                "isobars carrying both β⁻ and β⁺ entries that loop back, "
                "or a hand-coded branching that points back into a "
                "FLUKA-tabulated chain. Investigate.".format(
                    da, first_mo, sorted(visited),
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            self._record_stable(first_mo, da, csection, reclev)
            return

        self._recurse_into_daughters(first_mo, da, csection, reclev, visited | {da})

    def _record_stable(self, first_mo, da, csection, reclev):
        """Terminal arm: daughter is stable. Route to the right output dict."""
        if self.owner.is_differential(None, da):
            info(
                20,
                self._dbg_indent(reclev),
                "daughter {0} stable and differential. Adding to ({1}, {2})".format(
                    da, first_mo, da
                ),
            )
            self.new_dec_diff_tab[(first_mo, da)] = csection
        else:
            info(
                20,
                self._dbg_indent(reclev),
                "daughter {0} stable. Adding to ({1}, {2})".format(da, first_mo, da),
            )
            self.new_incl_tab[(first_mo, da)] = csection

    def _recurse_into_daughters(self, first_mo, da, csection, reclev, visited):
        """Recursive arm: daughter is unstable. For each branching, recurse on
        each chained daughter, scaling the cross section by the branching
        ratio and convolving with the decay distribution if the chained
        daughter requires x-redistribution. ``visited`` is forwarded so the
        cycle check in ``follow`` sees the chain prefix this call extends.
        """
        for br, daughters in spec_data[da]["branchings"]:
            info(
                10,
                self._dbg_indent(reclev),
                ("{3} -> {0:4d} -> {2:4.2f}: {1}").format(
                    da, ", ".join(map(str, daughters)), br, first_mo
                ),
            )
            for chained_daughter in daughters:
                if self.owner.is_differential(None, chained_daughter):
                    info(10, "daughter", chained_daughter, "of", da, "is differential")
                    self.follow(
                        first_mo,
                        chained_daughter,
                        self._convolve(
                            self.owner._arange_on_xgrid(csection),
                            da,
                            chained_daughter,
                            br,
                        ),
                        reclev + 1,
                        visited,
                    )
                else:
                    self.follow(
                        first_mo, chained_daughter, br * csection,
                        reclev + 1, visited,
                    )


if __name__ == "__main__":
    pass
