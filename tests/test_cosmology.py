"""Tests for prince_cr.cosmology module."""

import numpy as np

import prince_cr.config as config
from prince_cr.cosmology import (
    H,
    star_formation_rate,
    grb_rate,
    grb_rate_wp,
    agn_rate,
)

config.debug_level = 0


class TestExpansionRate:
    def test_H_at_z0(self):
        result = H(0.0)
        expected = config.H_0s * np.sqrt(config.Omega_m + config.Omega_Lambda)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_H_increases_with_z(self):
        assert H(1.0) > H(0.0)
        assert H(2.0) > H(1.0)

    def test_H_array_input(self):
        z = np.array([0.0, 0.5, 1.0, 2.0])
        result = H(z)
        assert result.shape == z.shape
        assert np.all(np.diff(result) > 0)

    def test_H_custom_H0(self):
        result = H(0.0, H0=1.0)
        expected = np.sqrt(config.Omega_m + config.Omega_Lambda)
        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestStarFormationRate:
    def test_below_z_inhom(self):
        assert star_formation_rate(0.0, z_inhom=0.5) == 0.0

    def test_at_z0(self):
        result = star_formation_rate(0.0)
        np.testing.assert_allclose(result, 1.0, rtol=1e-10)

    def test_low_z_branch(self):
        # z <= 0.97
        z = 0.5
        expected = (1.0 + z) ** 3.44
        np.testing.assert_allclose(star_formation_rate(z), expected, rtol=1e-10)

    def test_mid_z_branch(self):
        # 0.97 < z <= 4.48
        z = 2.0
        expected = 10.0**1.09 * (1.0 + z) ** -0.26
        np.testing.assert_allclose(star_formation_rate(z), expected, rtol=1e-10)

    def test_high_z_branch(self):
        # z > 4.48
        z = 5.0
        expected = 10.0**6.66 * (1.0 + z) ** -7.8
        np.testing.assert_allclose(star_formation_rate(z), expected, rtol=1e-10)


class TestGRBRate:
    def test_grb_rate_at_z0(self):
        result = grb_rate(0.0)
        expected = (1 + 0) ** 1.4 * star_formation_rate(0.0)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_grb_rate_with_z_inhom(self):
        result = grb_rate(0.0, z_inhom=0.5)
        assert result == 0.0

    def test_grb_rate_at_z1(self):
        z = 1.0
        result = grb_rate(z)
        expected = (1 + z) ** 1.4 * star_formation_rate(z)
        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestGRBRateWP:
    def test_below_z_inhom(self):
        assert grb_rate_wp(0.0, z_inhom=0.5) == 0.0

    def test_low_z_branch(self):
        z = 1.0
        expected = (1 + z) ** 2.1
        np.testing.assert_allclose(grb_rate_wp(z), expected, rtol=1e-10)

    def test_high_z_branch(self):
        z = 4.0
        expected = (1 + 3) ** (2.1 + 1.4) * (1 + z) ** -1.4
        np.testing.assert_allclose(grb_rate_wp(z), expected, rtol=1e-10)


class TestAGNRate:
    def test_below_z_inhom(self):
        assert agn_rate(0.0, z_inhom=0.5) == 0.0

    def test_low_z_branch(self):
        z = 1.0
        expected = (1 + z) ** 5
        np.testing.assert_allclose(agn_rate(z), expected, rtol=1e-10)

    def test_mid_z_branch(self):
        z = 2.0
        expected = (1 + 1.7) ** 5
        np.testing.assert_allclose(agn_rate(z), expected, rtol=1e-10)

    def test_high_z_branch(self):
        z = 3.5
        expected = (1 + 1.7) ** 5 * 10 ** (2.7 - z)
        np.testing.assert_allclose(agn_rate(z), expected, rtol=1e-10)
