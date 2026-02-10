"""Additional tests for prince_cr.data module to increase coverage."""

import numpy as np
import pytest

import prince_cr.config as config

config.debug_level = 0

from prince_cr.data import (
    EnergyGrid,
    SpeciesManager,
    PRINCE_UNITS,
    spec_data,
    PrinceSpecies,
)


class TestEnergyGrid:
    def test_basic_construction(self):
        grid = EnergyGrid(3, 10, 8)
        assert grid.d > 0
        assert len(grid.grid) == grid.d
        assert len(grid.bins) == grid.d + 1
        assert len(grid.widths) == grid.d

    def test_grid_monotonic(self):
        grid = EnergyGrid(3, 10, 8)
        assert np.all(np.diff(grid.grid) > 0)
        assert np.all(np.diff(grid.bins) > 0)

    def test_widths_positive(self):
        grid = EnergyGrid(3, 10, 8)
        assert np.all(grid.widths > 0)


class TestPrinceUnits:
    def test_cm2sec_positive(self):
        assert PRINCE_UNITS.cm2sec > 0

    def test_has_expected_attrs(self):
        assert hasattr(PRINCE_UNITS, "cm2sec")


class TestSpecData:
    def test_proton_exists(self):
        assert 101 in spec_data
        assert "mass" in spec_data[101]

    def test_neutron_exists(self):
        assert 100 in spec_data
        assert "mass" in spec_data[100]

    def test_electron_exists(self):
        assert 20 in spec_data

    def test_particle_data_has_charge(self):
        assert "charge" in spec_data[101]
        assert spec_data[101]["charge"] == 1

    def test_particle_data_has_lifetime(self):
        assert "lifetime" in spec_data[100]


class TestSpeciesManager:
    def test_basic_construction(self):
        species = [100, 101, 402]
        sm = SpeciesManager(species, 10)
        assert sm.nspec == 3
        assert len(sm.known_species) == 3

    def test_ncoid2sref(self):
        species = [100, 101]
        sm = SpeciesManager(species, 10)
        for pid in species:
            assert pid in sm.ncoid2sref
            spec = sm.ncoid2sref[pid]
            assert isinstance(spec, PrinceSpecies)

    def test_species_refs(self):
        species = [100, 101, 402]
        sm = SpeciesManager(species, 10)
        assert len(sm.species_refs) == 3

    def test_add_grid(self):
        species = [100, 101]
        sm = SpeciesManager(species, 10)
        sm.add_grid("ph", 20)
        for spec in sm.species_refs:
            assert spec.lidx("ph") >= 0
            assert spec.uidx("ph") >= spec.lidx("ph")


class TestPrinceSpecies:
    def test_properties(self):
        species = [100, 101, 402]
        sm = SpeciesManager(species, 10)
        proton = sm.ncoid2sref[101]
        assert proton.A == 1
        assert proton.Z == 1
        assert proton.N == 0

    def test_he4_properties(self):
        species = [100, 101, 402]
        sm = SpeciesManager(species, 10)
        he4 = sm.ncoid2sref[402]
        assert he4.A == 4
        assert he4.Z == 2
        assert he4.N == 2

    def test_slice_indexing(self):
        species = [100, 101]
        sm = SpeciesManager(species, 10)
        for spec in sm.species_refs:
            sl = spec.sl
            assert sl.start >= 0
            assert sl.stop <= 20
