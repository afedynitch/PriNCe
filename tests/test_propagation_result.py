"""Tests for prince_cr.solvers.propagation result classes."""

import numpy as np
import pytest

import prince_cr.config as config
from prince_cr.data import EnergyGrid, SpeciesManager
from prince_cr.solvers.propagation import UHECRPropagationResult

# Use the same grid parameters as conftest
_cr_lo, _cr_hi, _cr_bpd = config.cosmic_ray_grid


@pytest.fixture
def spec_man():
    species = [100, 101, 402]
    d = int((_cr_hi - _cr_lo) * _cr_bpd)
    return SpeciesManager(species, d)


@pytest.fixture
def egrid():
    return EnergyGrid(_cr_lo, _cr_hi, _cr_bpd).grid


@pytest.fixture
def result(spec_man, egrid):
    d = spec_man.grid_dims["default"]
    state = np.random.rand(spec_man.nspec * d) * 1e-20
    return UHECRPropagationResult(state, egrid, spec_man)


class TestUHECRPropagationResult:
    def test_init(self, result):
        assert result.state is not None
        assert result.egrid is not None
        assert result.spec_man is not None

    def test_known_species(self, result):
        ks = result.known_species
        assert 100 in ks
        assert 101 in ks
        assert 402 in ks

    def test_get_solution(self, result):
        e, sol = result.get_solution(101)
        assert e.shape == result.egrid.shape
        assert sol.shape == result.egrid.shape

    def test_get_solution_scale(self, result):
        e, sol = result.get_solution_scale(101, epow=3)
        assert len(e) == len(result.egrid)

    def test_add(self, result, spec_man, egrid):
        d = spec_man.grid_dims["default"]
        state2 = np.random.rand(spec_man.nspec * d) * 1e-20
        result2 = UHECRPropagationResult(state2, egrid, spec_man)
        combined = result + result2
        np.testing.assert_allclose(combined.state, result.state + result2.state)

    def test_add_different_egrid_raises(self, result, spec_man):
        d = spec_man.grid_dims["default"]
        state2 = np.random.rand(spec_man.nspec * d) * 1e-20
        egrid2 = EnergyGrid(_cr_lo + 1, _cr_hi, _cr_bpd).grid
        result2 = UHECRPropagationResult(state2, egrid2, spec_man)
        with pytest.raises(Exception, match="different energy grids"):
            result + result2

    def test_add_different_species_raises(self, result, egrid):
        d = int((_cr_hi - _cr_lo) * _cr_bpd)
        sm2 = SpeciesManager([100, 101], d)
        state2 = np.random.rand(sm2.nspec * d) * 1e-20
        result2 = UHECRPropagationResult(state2, egrid, sm2)
        with pytest.raises((Exception, ValueError)):
            result + result2

    def test_mul(self, result):
        scaled = result * 2.0
        np.testing.assert_allclose(scaled.state, result.state * 2.0)

    def test_mul_nonscalar_raises(self, result):
        with pytest.raises(Exception, match="scalar"):
            result * np.array([1.0, 2.0])

    def test_to_dict(self, result):
        d = result.to_dict()
        assert "egrid" in d
        assert "state" in d
        assert "known_spec" in d

    def test_from_dict(self, result):
        d = result.to_dict()
        restored = UHECRPropagationResult.from_dict(d)
        np.testing.assert_array_equal(restored.state, result.state)
        np.testing.assert_array_equal(restored.egrid, result.egrid)

    def test_get_solution_group(self, result):
        e, spec = result.get_solution_group([100, 101, 402], epow=3)
        assert len(e) > 0
        assert len(spec) == len(e)

    def test_get_solution_group_CR(self, result):
        e, spec = result.get_solution_group("CR", epow=3)
        assert len(e) > 0

    def test_get_solution_group_all(self, result):
        e, spec = result.get_solution_group("all", epow=3)
        assert len(e) > 0

    def test_get_lnA(self, result):
        e, avg, var = result.get_lnA([100, 101, 402])
        assert len(e) > 0
        assert len(avg) == len(e)
        assert len(var) == len(e)

    def test_get_energy_density(self, result):
        ed = result.get_energy_density(101)
        assert np.all(np.isfinite(ed))

    def test_check_id_grid_with_egrid(self, result):
        custom_egrid = np.logspace(3, 14, 50)
        ids, e = result._check_id_grid([101], custom_egrid)
        np.testing.assert_array_equal(e, custom_egrid)

    def test_check_id_grid_none(self, result):
        ids, e = result._check_id_grid([101], None)
        assert len(e) > 0

    def test_check_id_grid_nu(self, result):
        ids, e = result._check_id_grid("nu", None)
        # No neutrinos in this species list
        assert len(ids) == 0

    def test_check_id_grid_tuple(self, result):
        # Select species with A between 1 and 4
        def selector(s):
            return result.spec_man.ncoid2sref[s].A

        ids, e = result._check_id_grid((selector, 1, 4), None)
        assert len(ids) > 0
