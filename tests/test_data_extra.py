"""Additional tests for prince_cr.data module to increase coverage."""

import numpy as np

from prince_cr.data import (
    EnergyGrid,
    SpeciesManager,
    PRINCE_UNITS,
    spec_data,
    PrinceSpecies,
)


# Common PDG IDs.
PDG_E_MINUS = 11
PDG_PROTON = 2212
PDG_NEUTRON = 2112
PDG_HE4 = 1000020040


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
        assert PDG_PROTON in spec_data
        assert "mass" in spec_data[PDG_PROTON]

    def test_neutron_exists(self):
        assert PDG_NEUTRON in spec_data
        assert "mass" in spec_data[PDG_NEUTRON]

    def test_electron_exists(self):
        assert PDG_E_MINUS in spec_data

    def test_particle_data_has_charge(self):
        assert "charge" in spec_data[PDG_PROTON]
        assert spec_data[PDG_PROTON]["charge"] == 1

    def test_particle_data_has_lifetime(self):
        assert "lifetime" in spec_data[PDG_NEUTRON]


class TestSpeciesManager:
    def test_basic_construction(self):
        species = [PDG_NEUTRON, PDG_PROTON, PDG_HE4]
        sm = SpeciesManager(species, 10)
        assert sm.nspec == 3
        assert len(sm.known_species) == 3

    def test_pdgid2sref(self):
        species = [PDG_NEUTRON, PDG_PROTON]
        sm = SpeciesManager(species, 10)
        for pid in species:
            assert pid in sm.pdgid2sref
            spec = sm.pdgid2sref[pid]
            assert isinstance(spec, PrinceSpecies)

    def test_species_refs(self):
        species = [PDG_NEUTRON, PDG_PROTON, PDG_HE4]
        sm = SpeciesManager(species, 10)
        assert len(sm.species_refs) == 3

    def test_add_grid(self):
        species = [PDG_NEUTRON, PDG_PROTON]
        sm = SpeciesManager(species, 10)
        sm.add_grid("ph", 20)
        for spec in sm.species_refs:
            assert spec.lidx("ph") >= 0
            assert spec.uidx("ph") >= spec.lidx("ph")


class TestPrinceSpecies:
    def test_properties(self):
        species = [PDG_NEUTRON, PDG_PROTON, PDG_HE4]
        sm = SpeciesManager(species, 10)
        proton = sm.pdgid2sref[PDG_PROTON]
        assert proton.A == 1
        assert proton.Z == 1
        assert proton.N == 0

    def test_he4_properties(self):
        species = [PDG_NEUTRON, PDG_PROTON, PDG_HE4]
        sm = SpeciesManager(species, 10)
        he4 = sm.pdgid2sref[PDG_HE4]
        assert he4.A == 4
        assert he4.Z == 2
        assert he4.N == 2

    def test_slice_indexing(self):
        species = [PDG_NEUTRON, PDG_PROTON]
        sm = SpeciesManager(species, 10)
        for spec in sm.species_refs:
            sl = spec.sl
            assert sl.start >= 0
            assert sl.stop <= 20
