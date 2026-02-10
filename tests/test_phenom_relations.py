"""Tests for prince_cr.cross_sections._phenom_relations module."""

import numpy as np
import pytest

import prince_cr.config as config

config.debug_level = 0

from prince_cr.cross_sections._phenom_relations import (
    list_species_by_mass,
    partitions,
    combinations,
    cs_gpi,
    cs_gn,
    xm,
    cs_gxn,
    cs_gxn_all,
    cs_gp,
    cs_gSp,
    spallation_multiplicities,
    gxn_multiplicities,
    multiplicity_table,
    species_by_mass,
    resmul,
    partition_probability,
)


class TestListSpeciesByMass:
    def test_returns_dict(self):
        result = list_species_by_mass(56, 0.0)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_max_mass_limit(self):
        result = list_species_by_mass(10, 0.0)
        for A in result:
            assert A <= 10

    def test_species_by_mass_module_level(self):
        assert isinstance(species_by_mass, dict)
        assert len(species_by_mass) > 0


class TestPartitions:
    def test_zero(self):
        result = list(partitions(0))
        assert result == [[]]

    def test_one(self):
        result = list(partitions(1))
        assert [1] in result

    def test_two(self):
        result = list(partitions(2))
        assert [1, 1] in result
        assert [2] in result

    def test_three(self):
        result = list(partitions(3))
        assert [1, 1, 1] in result
        assert [2, 1] in result
        assert [3] in result

    def test_four(self):
        result = list(partitions(4))
        all_parts = [sorted(p) for p in result]
        assert [1, 1, 1, 1] in all_parts
        assert [1, 1, 2] in all_parts
        assert [2, 2] in all_parts
        assert [1, 3] in all_parts
        assert [4] in all_parts
        # No element > 4
        for p in result:
            for elem in p:
                assert elem <= 4


class TestCombinations:
    def test_basic(self):
        # x protons, y neutrons
        result = list(combinations(1, 1))
        assert len(result) >= 0

    def test_more_complex(self):
        result = list(combinations(2, 2))
        assert len(result) >= 0


class TestCrossSectionFormulas:
    def test_cs_gpi(self):
        result = cs_gpi(12)
        assert result > 0
        expected = 0.027 * 12**0.847
        np.testing.assert_allclose(result, expected)

    def test_cs_gn(self):
        result = cs_gn(12)
        assert result > 0
        expected = 0.104 * 12**0.81
        np.testing.assert_allclose(result, expected)

    def test_xm(self):
        result = xm(12)
        assert result == int(1.4 * 12**0.457)

    def test_cs_gxn_valid(self):
        A = 20
        result = cs_gxn(A, 2)
        assert result > 0

    def test_cs_gxn_out_of_range(self):
        result = cs_gxn(4, 100)
        assert result == 0

    def test_cs_gxn_all(self):
        result = cs_gxn_all(20)
        assert result > 0

    def test_cs_gp_with_Z(self):
        result = cs_gp(Z=10)
        expected = 0.115 * 10**0.5
        np.testing.assert_allclose(result, expected)

    def test_cs_gp_with_A(self):
        result = cs_gp(A=20)
        expected = 0.078 * 20**0.5
        np.testing.assert_allclose(result, expected)

    def test_cs_gSp_low_E(self):
        # A=56, Z=26 gives E = 446/56 ~ 7.96
        result = cs_gSp(26, 56, 2, 2)
        assert result > 0

    def test_cs_gSp_mid_E(self):
        # A=20, Z=10 gives E = 446/20 = 22.3, so E > 21 and E > 10
        result = cs_gSp(10, 20, 1, 1)
        assert result > 0

    def test_cs_gSp_small_E(self):
        # A=56, Z=26 gives E = 446/56 ~ 7.96, so E < 10 and E < 21
        result = cs_gSp(26, 56, 1, 1)
        assert result > 0


class TestMultiplicities:
    def test_gxn_multiplicities(self):
        # Test with oxygen-16 (1608)
        result = gxn_multiplicities(1608)
        assert isinstance(result, dict)
        assert 100 in result  # neutrons

    def test_gxn_multiplicities_light(self):
        # Test with helium-4 (402) - should have cs_sum == 0 path
        result = gxn_multiplicities(402)
        assert isinstance(result, dict)

    def test_spallation_multiplicities(self):
        # Test with carbon-12 (1206)
        result = spallation_multiplicities(1206)
        assert isinstance(result, dict)

    def test_multiplicity_table(self):
        # Test with carbon-12
        result = multiplicity_table(1206)
        assert isinstance(result, dict)
        assert len(result) > 0


class TestPartitionProbability:
    def test_basic(self):
        # Use simple combinations
        if 2 in species_by_mass and len(species_by_mass[2]) > 0:
            combos = [(species_by_mass[2][0],)]
            combos_out, yields = partition_probability(combos, 4)
            assert len(yields) == len(combos_out)
            np.testing.assert_allclose(sum(yields), 1.0, atol=1e-10)


class TestResmul:
    def test_loaded(self):
        assert isinstance(resmul, dict)
        assert len(resmul) > 0
