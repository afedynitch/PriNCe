"""Tests for prince_cr.cr_sources module."""

import numpy as np
import pytest

from prince_cr.data import EnergyGrid, SpeciesManager
from prince_cr.cosmology import star_formation_rate, grb_rate_wp, agn_rate


class MockPrinceRun:
    """Mock PriNCeRun for testing source classes."""

    def __init__(self, species=None, d=88):
        if species is None:
            species = [100, 101, 402]
        self.cr_grid = EnergyGrid(3, 10, 8)
        self.spec_man = SpeciesManager(species, self.cr_grid.d)
        self.dim_states = self.cr_grid.d * self.spec_man.nspec


class TestEvolution:
    """Test the evolution() method of CosmicRaySource through a concrete subclass."""

    def _make_source(self, m="flat", species=None):
        from prince_cr.cr_sources import SimpleSource

        mock_run = MockPrinceRun(species=species)
        params = {}
        for s in mock_run.spec_man.known_species:
            if s >= 100:
                params[s] = (2.0, 1e10, 1.0)
        return SimpleSource(mock_run, params=params, m=m)

    def test_flat_evolution(self):
        src = self._make_source(m="flat")
        assert src.evolution(0.0) == 1.0
        assert src.evolution(1.0) == 1.0
        assert src.evolution(5.0) == 1.0

    def test_negative_z_raises(self):
        src = self._make_source(m="flat")
        with pytest.raises(Exception, match="negative z"):
            src.evolution(-1.0)

    def test_float_m(self):
        src = self._make_source(m=2.0)
        z = 1.0
        expected = (1 + z) ** 2.0 * star_formation_rate(z)
        np.testing.assert_allclose(src.evolution(z), expected, rtol=1e-10)

    def test_tuple_SFR(self):
        src = self._make_source(m=("SFR", 1.5))
        z = 1.0
        expected = (1 + z) ** 1.5 * star_formation_rate(z)
        np.testing.assert_allclose(src.evolution(z), expected, rtol=1e-10)

    def test_tuple_GRB(self):
        src = self._make_source(m=("GRB", 1.5))
        z = 1.0
        expected = (1 + z) ** 1.5 * grb_rate_wp(z)
        np.testing.assert_allclose(src.evolution(z), expected, rtol=1e-10)

    def test_tuple_AGN(self):
        src = self._make_source(m=("AGN", 1.5))
        z = 1.0
        expected = (1 + z) ** 1.5 * agn_rate(z)
        np.testing.assert_allclose(src.evolution(z), expected, rtol=1e-10)

    def test_tuple_TDE(self):
        src = self._make_source(m=("TDE", 5.0))
        z = 1.0
        expected = (1 + z) ** (5.0 - 3.0)
        np.testing.assert_allclose(src.evolution(z), expected, rtol=1e-10)

    def test_tuple_simple(self):
        src = self._make_source(m=("simple", 3.0))
        z = 1.0
        expected = (1 + z) ** 3.0
        np.testing.assert_allclose(src.evolution(z), expected, rtol=1e-10)

    def test_tuple_simple_flat_low_z(self):
        src = self._make_source(m=("simple_flat", 3.0))
        z = 0.5
        expected = (1 + z) ** 3.0
        np.testing.assert_allclose(src.evolution(z), expected, rtol=1e-10)

    def test_tuple_simple_flat_high_z(self):
        src = self._make_source(m=("simple_flat", 3.0))
        z = 2.0
        expected = (1 + 1) ** 3.0
        np.testing.assert_allclose(src.evolution(z), expected, rtol=1e-10)

    def test_tuple_simple_SFR_low_z(self):
        src = self._make_source(m=("simple_SFR", 5.0))
        z = 0.5
        expected = (1 + z) ** 5.0
        np.testing.assert_allclose(src.evolution(z), expected, rtol=1e-10)

    def test_tuple_simple_SFR_high_z(self):
        src = self._make_source(m=("simple_SFR", 5.0))
        z = 2.0
        expected = (1 + 1) ** 3.6 * (1 + z) ** (5.0 - 3.6)
        np.testing.assert_allclose(src.evolution(z), expected, rtol=1e-10)

    def test_unknown_evo_raises(self):
        src = self._make_source(m=999)
        with pytest.raises(Exception, match="Unknown source evo type"):
            src.evolution(0.0)


class TestSimpleSource:
    def test_injection_spectrum(self):
        from prince_cr.cr_sources import SimpleSource

        mock_run = MockPrinceRun()
        params = {101: (2.0, 1e10, 1.0)}
        src = SimpleSource(mock_run, params=params, m="flat")
        energy = np.logspace(3, 10, 50)
        result = src.injection_spectrum(101, energy, params[101])
        assert result.shape == energy.shape
        assert np.all(result > 0)

    def test_injection_rate(self):
        from prince_cr.cr_sources import SimpleSource

        mock_run = MockPrinceRun()
        params = {101: (2.0, 1e10, 1.0)}
        src = SimpleSource(mock_run, params=params, m="flat")
        result = src.injection_rate(0.0)
        assert result.shape == (mock_run.dim_states,)

    def test_injection_grid_precomputed(self):
        from prince_cr.cr_sources import SimpleSource

        mock_run = MockPrinceRun()
        params = {101: (2.0, 1e10, 1.0)}
        src = SimpleSource(mock_run, params=params, m="flat")
        assert np.any(src.injection_grid > 0)


class TestRigdityCutoffSource:
    def test_injection_spectrum(self):
        from prince_cr.cr_sources import RigdityCutoffSource

        mock_run = MockPrinceRun()
        params = {101: (2.0, 1e10, 1.0)}
        src = RigdityCutoffSource(mock_run, params=params, m="flat")
        energy = np.logspace(3, 10, 50)
        result = src.injection_spectrum(101, energy, params[101])
        assert result.shape == energy.shape
        assert np.all(result > 0)


class TestAugerFitSource:
    def test_injection_spectrum(self):
        from prince_cr.cr_sources import AugerFitSource

        mock_run = MockPrinceRun()
        params = {101: (2.0, 1e10, 1.0)}
        src = AugerFitSource(mock_run, params=params, m="flat")
        energy = np.logspace(3, 10, 50)
        result = src.injection_spectrum(101, energy, params[101])
        assert result.shape == energy.shape
        assert np.all(result > 0)


class TestRigidityFlexSource:
    def test_injection_spectrum(self):
        from prince_cr.cr_sources import RigidityFlexSource

        mock_run = MockPrinceRun()
        params = {101: (2.0, 1e10, 0.5, 1.0)}
        src = RigidityFlexSource(mock_run, params=params, m="flat")
        energy = np.logspace(3, 10, 50)
        result = src.injection_spectrum(101, energy, params[101])
        assert result.shape == energy.shape
        assert np.all(result > 0)


class TestSpectrumSource:
    def test_injection_spectrum(self):
        from prince_cr.cr_sources import SpectrumSource

        mock_run = MockPrinceRun()
        egrid = np.logspace(3, 10, 50)
        specgrid = egrid ** (-2.0)
        params = {101: (egrid, specgrid)}
        src = SpectrumSource(mock_run, params=params, m="flat")
        energy = np.logspace(3, 10, 30)
        result = src.injection_spectrum(101, energy, params[101])
        assert result.shape == energy.shape
        assert np.all(result >= 0)


class TestInjectionRateSingle:
    def test_basic(self):
        from prince_cr.cr_sources import SimpleSource

        mock_run = MockPrinceRun()
        params = {101: (2.0, 1e10, 1.0)}
        src = SimpleSource(mock_run, params=params, m="flat")
        energy = np.logspace(5, 10, 20)
        result = src.injection_rate_single(101, energy, 0.0)
        assert result.shape == energy.shape
        assert np.all(result > 0)
