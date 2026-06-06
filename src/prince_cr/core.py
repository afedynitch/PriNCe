"""Provides user interface and runtime management."""

import pickle as pickle


from prince_cr import cross_sections, data, interaction_rates
from prince_cr.data import EnergyGrid
from prince_cr.util import info, get_AZN, is_nucleus
import prince_cr.config as config


class PriNCeRun(object):
    """Top-level orchestrator: holds grids, species, cross sections,
    photon field, interaction rates, and the per-run
    :class:`~prince_cr.config.BackendConfig`. Solvers consume this as
    a single handle.
    """

    def __init__(self, *args, **kwargs):
        # max_mass: explicit kwarg wins; otherwise fall back to the
        # module-global default. The previous gated-pop form silently
        # NameError'd when neither kwargs nor species_list was given.
        max_mass = kwargs.pop("max_mass", config.max_mass)

        # Per-run backend dispatch handle. Solver and interaction-rate
        # code reads ``self.backend.<knob>`` rather than ``config.<knob>``.
        backend = kwargs.pop("backend", None)
        self.backend = backend if backend is not None else config.BackendConfig.from_globals()

        # Tracked-species spec: list of kwargs dicts to pass to
        # ``SpeciesManager.add_tracking_species``. Resolved after
        # spec_man is built but before interaction_rates / response,
        # so the matrix builders see the augmented species set.
        # See methods/tracking-species-design.md.
        tracked_species = kwargs.pop("tracked_species", None) or []

        # EM cascade: evolve γ/e± as active species in transport (opt-in).
        # Default off keeps the nuclear-only path bit-for-bit unchanged.
        # See methods/em-cascade-in-transport.md.
        self.enable_em_cascade = kwargs.pop(
            "enable_em_cascade", getattr(config, "enable_em_cascade", False)
        )

        # Initialize energy grid
        if config.grid_scale == "E":
            info(1, "initialising Energy grid")
            cr_cfg = config.cosmic_ray_grid
            # EM cascade products (γ, e±) reach down to ~0.1 GeV; extend the
            # shared grid's low end when the co-evolved EM cascade is on.
            if self.enable_em_cascade:
                lo, hi, ppd = cr_cfg
                cr_cfg = (min(lo, -1), hi, ppd)
            self.cr_grid = EnergyGrid(*cr_cfg)
            self.ph_grid = EnergyGrid(*config.photon_grid)
        else:
            raise Exception(
                "Unknown energy grid scale {:}, adjust config.grid_scale".format(
                    config.grid_scale
                )
            )

        # Cross section handler
        if "cross_sections" in kwargs:
            self.cross_sections = kwargs["cross_sections"]
        else:
            self.cross_sections = cross_sections.FlukaPhotoNuclear()

        # Photon field handler
        if "photon_field" in kwargs:
            self.photon_field = kwargs["photon_field"]
        else:
            import prince_cr.photonfields as pf

            self.photon_field = pf.CombinedPhotonField(
                [pf.CMBPhotonSpectrum, pf.CIBGilmore2D]
            )

        # Limit max nuclear mass of eqn system
        if "species_list" in kwargs:
            system_species = list(
                set(kwargs["species_list"]) & set(self.cross_sections.known_species)
            )
        else:
            system_species = [
                s for s in self.cross_sections.known_species if get_AZN(s)[0] <= max_mass
            ]
        # Disable photo-meson production
        if not config.secondaries:
            system_species = [s for s in system_species if is_nucleus(s)]
        # Remove particles that are explicitly excluded. Keep the EM species
        # (γ, e±) as active species when the co-evolved EM cascade is on.
        em_keep = {22, 11, -11} if self.enable_em_cascade else set()
        for pid in config.ignore_particles:
            if pid in system_species and pid not in em_keep:
                system_species.remove(pid)
        for pid in em_keep:
            if pid not in system_species:
                system_species.append(pid)

        # Initialize species manager for all species for which cross sections are known
        self.spec_man = data.SpeciesManager(system_species, self.cr_grid.d)

        # Register tracked species (passive observers) before sizing the
        # state vector. Each entry is a kwargs dict for
        # ``SpeciesManager.add_tracking_species``; tracked species are
        # appended to ``species_refs`` with synthetic PDG IDs.
        for spec in tracked_species:
            self.spec_man.add_tracking_species(**spec)
        if self.spec_man.has_tracked_species():
            # Mirror matching (mo, real_da) channels onto (mo, tracked_pdg)
            # so the response builder and rate kernel pick them up. Hook
            # is a no-op when no tracked species require photo-nuclear
            # tracking (e.g. all are decay-only).
            self.cross_sections.emit_tracking_channels(self.spec_man)

        # Total dimension of system
        self.dim_states = self.cr_grid.d * self.spec_man.nspec
        self.dim_bins = (self.cr_grid.d + 1) * self.spec_man.nspec

        # Initialize continuous energy losses
        self.adia_loss_rates_grid = interaction_rates.ContinuousAdiabaticLossRate(
            prince_run=self, energy="grid"
        )
        self.pair_loss_rates_grid = interaction_rates.ContinuousPairProductionLossRate(
            prince_run=self, energy="grid"
        )
        self.adia_loss_rates_bins = interaction_rates.ContinuousAdiabaticLossRate(
            prince_run=self, energy="bins"
        )
        self.pair_loss_rates_bins = interaction_rates.ContinuousPairProductionLossRate(
            prince_run=self, energy="bins"
        )

        # Initialize the interaction rates
        self.int_rates = interaction_rates.PhotoNuclearInteractionRate(prince_run=self)

        # Co-evolved EM cascade: γ/e± are active species; the per-z saturated
        # cascade transfer is applied by the solver (operator-split per step,
        # see solvers/propagation). The stiff γγ/IC reprocessing is handled
        # semi-analytically there, NOT as an M-sink — so em_int_rates is unused.
        self.em_int_rates = None

        # Let species manager know about the photon grid dimensions (for idx calculations)
        # it is accesible under index "ph" for lidx(), uidx() calls
        self.spec_man.add_grid("ph", self.ph_grid.d)

    def set_photon_field(self, pfield):
        # Sub-objects (`int_rates`, `pair_loss_rates_*`) read `photon_field`
        # via a property that delegates back to this PriNCeRun, so no fan-out
        # is needed. `ContinuousAdiabaticLossRate` does not depend on the
        # photon field at all.
        self.photon_field = pfield
