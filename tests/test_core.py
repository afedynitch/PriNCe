"""Tests for prince_cr.core module."""

import numpy as np
import pytest

import prince_cr.config as config

config.debug_level = 0
config.max_mass = 4

from prince_cr import core, cross_sections, photonfields


@pytest.fixture(scope="module")
def pf():
    return photonfields.CombinedPhotonField(
        [photonfields.CMBPhotonSpectrum, photonfields.CIBGilmore2D]
    )


@pytest.fixture(scope="module")
def cs():
    return cross_sections.CompositeCrossSection(
        [
            (0.0, cross_sections.TabulatedCrossSection, ("CRP2_TALYS",)),
            (0.14, cross_sections.SophiaSuperposition, ()),
        ]
    )


@pytest.fixture(scope="module")
def prince_run(pf, cs):
    return core.PriNCeRun(max_mass=4, photon_field=pf, cross_sections=cs)


class TestPriNCeRun:
    def test_init(self, prince_run):
        assert prince_run.cr_grid is not None
        assert prince_run.ph_grid is not None
        assert prince_run.spec_man is not None

    def test_dim_states(self, prince_run):
        expected = prince_run.cr_grid.d * prince_run.spec_man.nspec
        assert prince_run.dim_states == expected

    def test_dim_bins(self, prince_run):
        expected = (prince_run.cr_grid.d + 1) * prince_run.spec_man.nspec
        assert prince_run.dim_bins == expected

    def test_has_int_rates(self, prince_run):
        assert prince_run.int_rates is not None

    def test_has_loss_rates(self, prince_run):
        assert prince_run.adia_loss_rates_grid is not None
        assert prince_run.pair_loss_rates_grid is not None
        assert prince_run.adia_loss_rates_bins is not None
        assert prince_run.pair_loss_rates_bins is not None

    def test_set_photon_field(self, prince_run):
        new_pf = photonfields.CombinedPhotonField([photonfields.CMBPhotonSpectrum])
        old_pf = prince_run.photon_field
        prince_run.set_photon_field(new_pf)
        assert prince_run.photon_field is new_pf
        assert prince_run.adia_loss_rates_grid.photon_field is new_pf
        assert prince_run.pair_loss_rates_grid.photon_field is new_pf
        # Restore
        prince_run.set_photon_field(old_pf)

    def test_with_species_list(self, pf, cs):
        """Test PriNCeRun with explicit species_list."""
        run = core.PriNCeRun(
            max_mass=4,
            photon_field=pf,
            cross_sections=cs,
            species_list=[101, 402],
        )
        assert 101 in run.spec_man.known_species

    def test_invalid_grid_scale_raises(self, pf, cs):
        old_scale = config.grid_scale
        config.grid_scale = "INVALID"
        with pytest.raises(Exception, match="Unknown energy grid scale"):
            core.PriNCeRun(max_mass=1, photon_field=pf, cross_sections=cs)
        config.grid_scale = old_scale

    def test_without_secondaries(self, pf, cs):
        old_sec = config.secondaries
        config.secondaries = False
        run = core.PriNCeRun(max_mass=4, photon_field=pf, cross_sections=cs)
        # All species should be >= 100 (no secondaries)
        assert all(s >= 100 for s in run.spec_man.known_species)
        config.secondaries = old_sec

    def test_ignore_particles(self, pf, cs):
        old_ignore = config.ignore_particles
        config.ignore_particles = [20, 21, 0]
        run = core.PriNCeRun(max_mass=4, photon_field=pf, cross_sections=cs)
        for pid in config.ignore_particles:
            assert pid not in run.spec_man.known_species
        config.ignore_particles = old_ignore
