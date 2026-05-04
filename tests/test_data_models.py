"""Tests for prince_cr.data module - PrinceDB, InterpolatorWrapper, and PrinceSpecies variants."""

import numpy as np
import pytest

from prince_cr.data import (
    EnergyGrid,
    SpeciesManager,
    PrinceSpecies,
    InterpolatorWrapper,
    db_handler,
)


# Common PDG IDs used throughout the tests.
PDG_GAMMA = 22
PDG_E_MINUS = 11
PDG_E_PLUS = -11
PDG_NU_E = 12
PDG_PI_PLUS = 211
PDG_PROTON = 2212
PDG_NEUTRON = 2112
PDG_HE4 = 1000020040
PDG_N14 = 1000070140


class TestInterpolatorWrapper:
    def test_basic_call(self):
        from scipy.interpolate import RegularGridInterpolator

        x = np.linspace(0, 5, 10)
        y = np.linspace(0, 5, 10)
        z = np.outer(x, y)
        rgi = RegularGridInterpolator(
            (x, y), z, method="linear", bounds_error=False, fill_value=0.0
        )
        wrapper = InterpolatorWrapper(rgi)
        result = wrapper(np.array([2.5]), np.array([2.5]))
        np.testing.assert_allclose(result, 2.5 * 2.5, atol=0.5)

    def test_scalar_call(self):
        from scipy.interpolate import RegularGridInterpolator

        x = np.linspace(0, 5, 10)
        y = np.linspace(0, 5, 10)
        z = np.outer(x, y)
        rgi = RegularGridInterpolator(
            (x, y), z, method="linear", bounds_error=False, fill_value=0.0
        )
        wrapper = InterpolatorWrapper(rgi)
        result = wrapper(2.5, 2.5)
        assert np.isfinite(result)

    def test_array_call(self):
        from scipy.interpolate import RegularGridInterpolator

        x = np.linspace(0, 5, 10)
        y = np.linspace(0, 5, 10)
        z = np.outer(x, y)
        rgi = RegularGridInterpolator(
            (x, y), z, method="linear", bounds_error=False, fill_value=0.0
        )
        wrapper = InterpolatorWrapper(rgi)
        xvals = np.array([1.0, 2.0, 3.0])
        yvals = np.array([1.0, 2.0, 3.0])
        result = wrapper(xvals, yvals)
        assert result.shape == xvals.shape


class TestPrinceDB:
    def test_version_exists(self):
        assert hasattr(db_handler, "version")
        assert isinstance(db_handler.version, (str, bytes))

    def test_photo_nuclear_db(self):
        result = db_handler.photo_nuclear_db("CRP2_TALYS")
        assert "energy_grid" in result
        assert "fragment_yields" in result
        assert "inel_mothers" in result
        assert "inelastic_cross_sctions" in result
        assert "mothers_daughters" in result

    def test_photo_meson_db(self):
        result = db_handler.photo_meson_db("SOPHIA")
        assert "energy_grid" in result
        assert "xbins" in result
        assert "fragment_yields" in result

    def test_check_subgroup_invalid(self):
        import h5py

        with h5py.File(db_handler.prince_db_fname, "r") as f:
            with pytest.raises(Exception, match="Unknown selections"):
                db_handler._check_subgroup_exists(f["photo_nuclear"], "NONEXISTENT")

    def test_ebl_spline(self):
        result = db_handler.ebl_spline("Gilmore2011", "fiducial")
        assert isinstance(result, InterpolatorWrapper)
        # Test that the wrapper is callable
        val = result(0.0, -12.0)
        assert np.isfinite(val)


class TestPrinceSpeciesVariants:
    """Test PrinceSpecies with different particle types to cover all branches."""

    def test_photon_species(self):
        """PDG 22: gamma (electromagnetic particle)"""
        species = [PDG_GAMMA, PDG_PROTON]
        sm = SpeciesManager(species, 10)
        photon = sm.pdgid2sref[PDG_GAMMA]
        assert photon.is_em is True
        assert photon.is_nucleus is False
        assert photon.has_redist is True

    def test_meson_species(self):
        """PDG 211: pi+ (meson)"""
        species = [PDG_PI_PLUS, PDG_PROTON]
        sm = SpeciesManager(species, 10)
        pion = sm.pdgid2sref[PDG_PI_PLUS]
        assert pion.is_hadron is True
        assert pion.is_meson is True
        assert pion.is_baryon is False
        assert pion.has_redist is True

    def test_lepton_species(self):
        """PDG 11: e- (lepton, em)"""
        species = [PDG_E_MINUS, PDG_PROTON]
        sm = SpeciesManager(species, 10)
        electron = sm.pdgid2sref[PDG_E_MINUS]
        assert electron.is_lepton is True
        assert electron.is_em is True
        assert electron.has_redist is True

    def test_positron_species(self):
        """PDG -11: e+ (lepton, em)"""
        species = [PDG_E_PLUS, PDG_PROTON]
        sm = SpeciesManager(species, 10)
        positron = sm.pdgid2sref[PDG_E_PLUS]
        assert positron.is_lepton is True
        assert positron.is_em is True

    def test_neutrino_species(self):
        """PDG 12: nu_e (lepton, not em)"""
        species = [PDG_NU_E, PDG_PROTON]
        sm = SpeciesManager(species, 10)
        nu = sm.pdgid2sref[PDG_NU_E]
        assert nu.is_lepton is True
        assert nu.is_em is False
        assert nu.is_nucleus is False

    def test_neutron_species(self):
        """PDG 2112: free neutron (baryon, nucleus)"""
        species = [PDG_NEUTRON, PDG_PROTON]
        sm = SpeciesManager(species, 10)
        neutron = sm.pdgid2sref[PDG_NEUTRON]
        assert neutron.is_hadron is True
        assert neutron.is_baryon is True
        assert neutron.is_nucleus is True
        assert neutron.A == 1
        assert neutron.Z == 0
        assert neutron.N == 1

    def test_heavy_nucleus(self):
        """PDG 1000070140: nitrogen-14"""
        species = [PDG_PROTON, PDG_N14]
        sm = SpeciesManager(species, 10)
        n14 = sm.pdgid2sref[PDG_N14]
        assert n14.is_nucleus is True
        assert n14.A == 14
        assert n14.Z == 7
        assert n14.N == 7

    def test_lbin_ubin(self):
        species = [PDG_NEUTRON, PDG_PROTON]
        sm = SpeciesManager(species, 10)
        proton = sm.pdgid2sref[PDG_PROTON]
        lbin = proton.lbin()
        ubin = proton.ubin()
        assert ubin - lbin == 11  # d + 1 = 10 + 1

    def test_calc_AZN_static(self):
        A, Z, N = PrinceSpecies.calc_AZN(PDG_HE4)
        assert A == 4
        assert Z == 2
        assert N == 2

        # K+ (PDG 321) is not a nucleus
        A, Z, N = PrinceSpecies.calc_AZN(321)
        assert A == 0
        assert Z == 0
        assert N == 0


class TestSpeciesManagerExtra:
    def test_repr(self):
        species = [PDG_NEUTRON, PDG_PROTON]
        sm = SpeciesManager(species, 10)
        r = repr(sm)
        assert "PDG id" in r
        assert "PriNCe idx" in r

    def test_redist_species(self):
        species = [PDG_NU_E, PDG_E_MINUS, PDG_NEUTRON, PDG_PROTON, PDG_HE4]
        sm = SpeciesManager(species, 10)
        # Non-nuclei and free p/n should have has_redist = True
        assert PDG_NU_E in sm.redist_species
        assert PDG_E_MINUS in sm.redist_species
        # Heavy nuclei (A>=2) are boost-conserving
        assert PDG_HE4 in sm.boost_conserv_species

    def test_sname_lookups(self):
        species = [PDG_NEUTRON, PDG_PROTON]
        sm = SpeciesManager(species, 10)
        for spec in sm.species_refs:
            assert spec.sname in sm.sname2princeidx
            assert spec.sname in sm.sname2sref

    def test_princeidx_lookups(self):
        species = [PDG_NEUTRON, PDG_PROTON]
        sm = SpeciesManager(species, 10)
        for idx in range(len(species)):
            assert idx in sm.princeidx2pdgid
            assert idx in sm.princeidx2sref
            assert idx in sm.princeidx2pname

    def test_add_grid_propagates(self):
        species = [PDG_NEUTRON, PDG_PROTON]
        sm = SpeciesManager(species, 10)
        sm.add_grid("ph", 20)
        for spec in sm.species_refs:
            assert "ph" in spec.grid_dims
            assert spec.grid_dims["ph"] == 20
            assert spec.lidx("ph") >= 0
            assert spec.uidx("ph") > spec.lidx("ph")


class TestEnergyGridExtra:
    def test_different_bins_per_decade(self):
        grid4 = EnergyGrid(3, 10, 4)
        grid8 = EnergyGrid(3, 10, 8)
        assert grid8.d > grid4.d

    def test_single_decade(self):
        grid = EnergyGrid(3, 4, 10)
        assert grid.d == 10

    def test_bins_bracket_grid(self):
        grid = EnergyGrid(3, 10, 8)
        assert grid.bins[0] < grid.grid[0]
        assert grid.bins[-1] > grid.grid[-1]
