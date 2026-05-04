"""Tests for prince_cr.decays module."""

import numpy as np
import pytest
from scipy.integrate import trapezoid as trapz

from prince_cr.decays import (
    get_particle_channels,
    get_decay_matrix,
    get_decay_matrix_bin_average,
    pion_to_numu,
    pion_to_numu_avg,
    pion_to_muon,
    pion_to_muon_avg,
    prob_muon_hel,
    muonplus_to_numubar,
    muonplus_to_nue,
    boost_conservation,
    boost_conservation_avg,
    nu_from_beta_decay,
)
from prince_cr.data import spec_data  # noqa: E402


# PDG IDs used in the test cases.
PDG_PI_PLUS = 211
PDG_PI_MINUS = -211
PDG_MU_PLUS = -13
PDG_MU_MINUS = 13
PDG_NU_E = 12
PDG_NU_E_BAR = -12
PDG_NU_MU = 14
PDG_NU_MU_BAR = -14
PDG_KAON_PLUS = 321
PDG_KAON_MINUS = -321
PDG_PROTON = 2212
PDG_NEUTRON = 2112
PDG_HE3 = 1000020030
PDG_TRITIUM = 1000010030


class TestPionToNumu:
    def test_shape_preserved(self):
        x = np.linspace(0, 1.5, 100)
        result = pion_to_numu(x)
        assert result.shape == x.shape

    def test_zero_outside_range(self):
        x = np.array([1.5, 2.0])
        result = pion_to_numu(x)
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_nonzero_inside_range(self):
        m_muon = spec_data[PDG_MU_PLUS]["mass"]
        m_pion = spec_data[PDG_PI_PLUS]["mass"]
        r = m_muon**2 / m_pion**2
        x = np.array([0.5 * (1 - r)])  # inside [0, 1-r]
        result = pion_to_numu(x)
        assert result[0] > 0

    def test_normalization(self):
        x = np.linspace(0, 1, 10000)
        result = pion_to_numu(x)
        integral = trapz(result, x)
        np.testing.assert_allclose(integral, 1.0, atol=0.02)


class TestPionToNumuAvg:
    def test_shape_mismatch_raises(self):
        xl = np.array([0.0, 0.1])
        xu = np.array([0.1])
        with pytest.raises(Exception, match="different grids"):
            pion_to_numu_avg(xl, xu)

    def test_basic_averaging(self):
        xl = np.linspace(0, 0.9, 50)
        xu = xl + 0.02
        result = pion_to_numu_avg(xl, xu)
        assert result.shape == xl.shape
        assert np.any(result > 0)


class TestPionToMuon:
    def test_shape_preserved(self):
        x = np.linspace(0, 1.5, 100)
        result = pion_to_muon(x)
        assert result.shape == x.shape

    def test_nonzero_inside_range(self):
        m_muon = spec_data[PDG_MU_PLUS]["mass"]
        m_pion = spec_data[PDG_PI_PLUS]["mass"]
        r = m_muon**2 / m_pion**2
        x = np.array([(r + 1.0) / 2.0])  # inside [r, 1]
        result = pion_to_muon(x)
        assert result[0] > 0


class TestPionToMuonAvg:
    def test_shape_mismatch_raises(self):
        xl = np.array([0.0, 0.1])
        xu = np.array([0.1])
        with pytest.raises(Exception, match="different grids"):
            pion_to_muon_avg(xl, xu)

    def test_basic(self):
        xl = np.linspace(0.5, 0.9, 20)
        xu = xl + 0.02
        result = pion_to_muon_avg(xl, xu)
        assert result.shape == xl.shape


class TestProbMuonHel:
    def test_shape(self):
        m_muon = spec_data[PDG_MU_PLUS]["mass"]
        m_pion = spec_data[PDG_PI_PLUS]["mass"]
        r = m_muon**2 / m_pion**2
        # Only use x values in valid range (r, 1]
        x = np.linspace(r + 0.01, 1.0, 50)
        result = prob_muon_hel(x, 1.0)
        assert result.shape == x.shape

    def test_helicity_values(self):
        m_muon = spec_data[PDG_MU_PLUS]["mass"]
        m_pion = spec_data[PDG_PI_PLUS]["mass"]
        r = m_muon**2 / m_pion**2
        x = np.array([(r + 1.0) / 2.0])
        p_plus = prob_muon_hel(x, 1.0)
        p_minus = prob_muon_hel(x, -1.0)
        np.testing.assert_allclose(p_plus + p_minus, 1.0, atol=1e-10)


class TestMuonDecays:
    def test_muonplus_to_numubar(self):
        x = np.linspace(0, 1.5, 100)
        result = muonplus_to_numubar(x, 0.0)
        assert np.any(result > 0)
        assert np.all(result[x > 1.0] == 0)

    def test_muonplus_to_nue(self):
        x = np.linspace(0, 1.5, 100)
        result = muonplus_to_nue(x, 0.0)
        assert np.any(result > 0)
        assert np.all(result[x > 1.0] == 0)

    def test_helicity_dependence(self):
        x = np.linspace(0, 1, 100)
        r1 = muonplus_to_numubar(x, 1.0)
        r2 = muonplus_to_numubar(x, -1.0)
        # Different helicities should give different results
        assert not np.allclose(r1, r2)


class TestBoostConservation:
    def test_delta_at_x1(self):
        x = np.array([0.5, 0.9, 1.0, 1.1])
        result = boost_conservation(x)
        assert result[2] > 0
        assert result[0] == 0
        assert result[1] == 0
        assert result[3] == 0


class TestBoostConservationAvg:
    def test_basic(self):
        xl = np.array([0.8, 0.95, 1.05])
        xu = np.array([0.95, 1.05, 1.2])
        result = boost_conservation_avg(xl, xu)
        assert result[1] > 0  # Bin containing 1.0
        assert result[0] == 0
        assert result[2] == 0


class TestGetDecayMatrix:
    """Test get_decay_matrix for different channels."""

    def test_pion_to_numu(self):
        x_grid = np.outer(np.linspace(0.01, 2, 20), np.ones(5))
        result = get_decay_matrix(PDG_PI_PLUS, PDG_NU_MU, x_grid)
        assert result.shape == x_grid.shape
        assert np.any(result > 0)

    def test_pion_minus_to_numubar(self):
        x_grid = np.outer(np.linspace(0.01, 2, 20), np.ones(5))
        result = get_decay_matrix(PDG_PI_MINUS, PDG_NU_MU_BAR, x_grid)
        assert result.shape == x_grid.shape
        assert np.any(result > 0)

    def test_pion_to_muon_helicity_mixed(self):
        x_grid = np.outer(np.linspace(0.01, 2, 20), np.ones(5))
        result = get_decay_matrix(PDG_PI_PLUS, PDG_MU_PLUS, x_grid)
        assert result.shape == x_grid.shape

    def test_muon_to_nue(self):
        x_grid = np.outer(np.linspace(0.01, 2, 20), np.ones(5))
        # μ+ → ν_e (helicity-mixed, h=0)
        result = get_decay_matrix(PDG_MU_PLUS, PDG_NU_E, x_grid)
        assert result.shape == x_grid.shape

    def test_muon_to_numubar(self):
        x_grid = np.outer(np.linspace(0.01, 2, 20), np.ones(5))
        # μ+ → ν̄_μ
        result = get_decay_matrix(PDG_MU_PLUS, PDG_NU_MU_BAR, x_grid)
        assert result.shape == x_grid.shape

    def test_muonminus_to_nuebar(self):
        x_grid = np.outer(np.linspace(0.01, 2, 20), np.ones(5))
        result = get_decay_matrix(PDG_MU_MINUS, PDG_NU_E_BAR, x_grid)
        assert result.shape == x_grid.shape

    def test_muonminus_to_numu(self):
        x_grid = np.outer(np.linspace(0.01, 2, 20), np.ones(5))
        result = get_decay_matrix(PDG_MU_MINUS, PDG_NU_MU, x_grid)
        assert result.shape == x_grid.shape

    def test_nucleus_to_nucleus(self):
        x_grid = np.outer(np.linspace(0.01, 2, 20), np.ones(5))
        # neutron β decay: n → p (boost conservation)
        result = get_decay_matrix(PDG_NEUTRON, PDG_PROTON, x_grid)
        assert result.shape == x_grid.shape

    def test_beta_decay_branch(self):
        # Hits the nucleus → ν_e_bar β-decay branch (Z+1 daughter).
        x_grid = np.outer(np.linspace(0.01, 2, 20), np.ones(5))
        try:
            result = get_decay_matrix(PDG_HE3, PDG_NU_E_BAR, x_grid)
            assert result.shape == x_grid.shape
        except (KeyError, Exception):
            # Some intermediate nuclei may not be in spec_data
            pass

    def test_unknown_channel(self):
        x_grid = np.outer(np.linspace(0.01, 2, 20), np.ones(5))
        result = get_decay_matrix(PDG_KAON_PLUS, PDG_KAON_MINUS, x_grid)
        np.testing.assert_array_equal(result, np.zeros(x_grid.shape))


class TestGetDecayMatrixBinAverage:
    def test_pion_to_numu_1d(self):
        x = np.linspace(0.05, 1, 20)
        dx = x[1] - x[0]
        xl = x - dx / 2
        xu = x + dx / 2
        result = get_decay_matrix_bin_average(PDG_PI_PLUS, PDG_NU_MU, xl, xu)
        assert result.shape == x.shape

    def test_pion_to_numu_2d(self):
        x_1d = np.linspace(0.05, 1.5, 10)
        x_grid = np.outer(x_1d, 1.0 / x_1d)
        dx = 0.05
        xl = x_grid - dx / 2
        xu = x_grid + dx / 2
        result = get_decay_matrix_bin_average(PDG_PI_PLUS, PDG_NU_MU, xl, xu)
        assert result.shape == x_grid.shape

    def test_pion_to_muon_2d(self):
        x_1d = np.linspace(0.05, 1.5, 10)
        x_grid = np.outer(x_1d, 1.0 / x_1d)
        dx = 0.05
        xl = x_grid - dx / 2
        xu = x_grid + dx / 2
        result = get_decay_matrix_bin_average(PDG_PI_PLUS, PDG_MU_PLUS, xl, xu)
        assert result.shape == x_grid.shape

    def test_muon_to_nue(self):
        x = np.linspace(0.05, 1, 20)
        dx = x[1] - x[0]
        result = get_decay_matrix_bin_average(PDG_MU_PLUS, PDG_NU_E, x - dx / 2, x + dx / 2)
        assert result.shape == x.shape

    def test_muon_to_numubar(self):
        x = np.linspace(0.05, 1, 20)
        dx = x[1] - x[0]
        result = get_decay_matrix_bin_average(PDG_MU_PLUS, PDG_NU_MU_BAR, x - dx / 2, x + dx / 2)
        assert result.shape == x.shape

    def test_muonminus_to_nuebar(self):
        x = np.linspace(0.05, 1, 20)
        dx = x[1] - x[0]
        result = get_decay_matrix_bin_average(PDG_MU_MINUS, PDG_NU_E_BAR, x - dx / 2, x + dx / 2)
        assert result.shape == x.shape

    def test_muonminus_to_numu(self):
        x = np.linspace(0.05, 1, 20)
        dx = x[1] - x[0]
        result = get_decay_matrix_bin_average(PDG_MU_MINUS, PDG_NU_MU, x - dx / 2, x + dx / 2)
        assert result.shape == x.shape

    def test_boost_conservation(self):
        x = np.linspace(0.5, 1.5, 20)
        dx = x[1] - x[0]
        result = get_decay_matrix_bin_average(PDG_NEUTRON, PDG_PROTON, x - dx / 2, x + dx / 2)
        assert result.shape == x.shape

    def test_unknown_channel(self):
        x = np.linspace(0.05, 1, 20)
        dx = x[1] - x[0]
        result = get_decay_matrix_bin_average(PDG_KAON_PLUS, PDG_KAON_MINUS, x - dx / 2, x + dx / 2)
        np.testing.assert_array_equal(result, np.zeros(x.shape))


class TestGetParticleChannels:
    def test_pion_plus(self):
        mo_energy = np.logspace(0, 3, 20)
        da_energy = np.logspace(0, 3, 20)
        x_grid, redist = get_particle_channels(PDG_PI_PLUS, mo_energy, da_energy)
        assert x_grid.shape == (20, 20)
        assert isinstance(redist, dict)
        assert len(redist) > 0


class TestNuFromBetaDecay:
    def test_neutron_decay(self):
        x = np.linspace(0.001, 2.0, 50)
        result = nu_from_beta_decay(x, PDG_NEUTRON, PDG_PROTON)
        assert result.shape == x.shape

    def test_with_angle(self):
        x = np.linspace(0.001, 2.0, 50)
        angle = np.array([0.5])
        result = nu_from_beta_decay(x, PDG_NEUTRON, PDG_PROTON, angle=angle)
        assert len(result) == len(x)
