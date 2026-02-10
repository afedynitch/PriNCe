"""Tests for cross_sections base, SophiaSuperposition, and related classes."""

import numpy as np
import pytest

from prince_cr.util import get_AZN


class TestSophiaSuperposition:
    def test_nonel_proton(self, sophia):
        egrid, cs = sophia.nonel(101)
        assert len(egrid) > 0
        assert len(cs) > 0
        assert np.all(cs >= 0)

    def test_nonel_neutron(self, sophia):
        egrid, cs = sophia.nonel(100)
        assert len(egrid) > 0

    def test_nonel_nucleus(self, sophia):
        # He-4 using superposition
        egrid, cs = sophia.nonel(402)
        assert len(egrid) > 0
        assert np.all(cs >= 0)

    def test_incl_redistributed(self, sophia):
        # Proton inclusive to redistribution particle
        egrid, cs = sophia.incl(101, 101)
        assert len(egrid) > 0

    def test_incl_residual(self, sophia):
        # Proton -> residual
        egrid, cs = sophia.incl(402, 301)
        assert len(egrid) > 0

    def test_incl_out_of_range(self, sophia):
        # Daughter not in expected range
        egrid, cs = sophia.incl(402, 201)
        assert len(egrid) > 0

    def test_incl_diff_proton(self, sophia):
        # Proton inclusive differential to pion
        if (101, 2) in sophia.incl_diff_idcs:
            egrid, cs_diff = sophia.incl_diff(101, 2)
            assert len(egrid) > 0
            assert cs_diff.shape[1] == len(egrid)

    def test_incl_diff_neutron_daughter(self, sophia):
        if (100, 2) in sophia.incl_diff_idcs:
            egrid, cs_diff = sophia.incl_diff(100, 2)
            assert len(egrid) > 0

    def test_generate_incl_channels(self, sophia):
        channels = sophia.generate_incl_channels([101])
        assert len(channels) > 0

    def test_xcenters(self, sophia):
        xc = sophia.xcenters
        assert len(xc) > 0
        assert np.all(np.diff(xc) > 0)

    def test_xwidths(self, sophia):
        xw = sophia.xwidths
        assert len(xw) > 0
        assert np.all(xw > 0)

    def test_egrid_property(self, sophia):
        eg = sophia.egrid
        assert len(eg) > 0

    def test_set_range(self, sophia):
        # Save original range
        orig = sophia._range.copy()
        sophia.set_range(e_min=0.1, e_max=100.0)
        assert len(sophia._range) > 0
        # Restore
        sophia._range = orig

    def test_resp_property(self, sophia):
        """Test that resp property creates ResponseFunction."""
        resp = sophia.resp
        assert resp is not None
        assert hasattr(resp, "nonel_intp")

    def test_nonel_scale_A(self, sophia):
        egrid, cs = sophia.nonel_scale(101)
        A = get_AZN(101)[0]
        egrid2, cs2 = sophia.nonel(101)
        np.testing.assert_allclose(cs, cs2 / A)

    def test_nonel_scale_custom(self, sophia):
        egrid, cs = sophia.nonel_scale(101, scale=2.0)
        egrid2, cs2 = sophia.nonel(101)
        np.testing.assert_allclose(cs, cs2 * 2.0)

    def test_is_differential_redistributed(self, sophia):
        # Pion (< 101) should be differential
        assert sophia.is_differential(101, 2) is True

    def test_is_differential_boost_conserving(self, sophia):
        # Nucleus daughter > 101
        assert sophia.is_differential(402, 301) is False


class TestTabulatedCrossSection:
    def test_nonel(self, talys):
        egrid, cs = talys.nonel(101)
        assert len(egrid) > 0
        assert np.all(cs >= 0)

    def test_incl(self, talys):
        # Pick a known channel
        if (402, 201) in talys.incl_idcs or (402, 201) in talys._incl_tab:
            egrid, cs = talys.incl(402, 201)
            assert len(egrid) > 0

    def test_nonel_unknown_raises(self, talys):
        with pytest.raises(Exception, match="unknown"):
            talys.nonel(9999)

    def test_incl_unknown_raises(self, talys):
        with pytest.raises(Exception):
            talys.incl(9999, 9999)

    def test_incl_scale(self, talys):
        egrid, cs = talys.incl_scale(402, 201)
        assert len(egrid) > 0

    def test_incl_scale_custom(self, talys):
        egrid, cs = talys.incl_scale(402, 201, scale=2.0)
        egrid2, cs2 = talys.incl(402, 201)
        np.testing.assert_allclose(cs, cs2 * 2.0)

    def test_known_species(self, talys):
        assert len(talys.known_species) > 0
        assert 101 in talys.known_species

    def test_reactions_dict(self, talys):
        assert isinstance(talys.reactions, dict)
        assert len(talys.reactions) > 0

    def test_known_bc_channels(self, talys):
        assert isinstance(talys.known_bc_channels, list)

    def test_known_diff_channels(self, talys):
        assert isinstance(talys.known_diff_channels, list)

    def test_multiplicities(self, talys):
        """Test multiplicities method from base class."""
        if (402, 201) in talys.incl_idcs or (402, 201) in talys._incl_tab:
            egrid, mult = talys.multiplicities(402, 201)
            assert len(egrid) > 0
            assert len(mult) == len(egrid)


class TestCompositeCrossSection:
    def test_nonel(self, composite):
        egrid, cs = composite.nonel(101)
        assert len(egrid) > 0

    def test_incl(self, composite):
        egrid, cs = composite.incl(402, 201)
        assert len(egrid) > 0

    def test_resp(self, composite):
        resp = composite.resp
        assert resp is not None


class TestResponseFunction:
    def test_get_channel_nonel(self, sophia):
        resp = sophia.resp
        ygr, integral = resp.get_channel(101)
        assert len(ygr) > 0
        assert len(integral) > 0

    def test_get_channel_incl(self, sophia):
        resp = sophia.resp
        # Test an inclusive channel
        if sophia.incl_idcs:
            mo, da = sophia.incl_idcs[0]
            ygr, integral = resp.get_channel(mo, da)
            assert len(ygr) > 0

    def test_get_channel_scale(self, sophia):
        resp = sophia.resp
        ygr, cs = resp.get_channel_scale(101)
        assert len(ygr) > 0

    def test_get_channel_scale_custom(self, sophia):
        resp = sophia.resp
        ygr, cs = resp.get_channel_scale(101, scale=2.0)
        assert len(ygr) > 0

    def test_get_channel_scale_with_daughter(self, sophia):
        resp = sophia.resp
        if sophia.incl_idcs:
            mo, da = sophia.incl_idcs[0]
            ygr, cs = resp.get_channel_scale(mo, da)
            assert len(ygr) > 0

    def test_is_differential(self, sophia):
        resp = sophia.resp
        assert resp.is_differential(101, 2) is True

    def test_get_full_nonel(self, sophia):
        resp = sophia.resp
        ygr = np.logspace(-2, 3, 50)
        xgr = np.ones_like(ygr)
        result = resp.get_full(101, 101, ygr, xgr)
        assert result.shape == ygr.shape

    def test_get_full_incl(self, sophia):
        resp = sophia.resp
        if sophia.incl_idcs:
            mo, da = sophia.incl_idcs[0]
            ygr = np.logspace(-2, 3, 50)
            result = resp.get_full(mo, da, ygr)
            assert result.shape == ygr.shape

    def test_get_full_diff(self, sophia):
        resp = sophia.resp
        if sophia.incl_diff_idcs:
            mo, da = sophia.incl_diff_idcs[0]
            ygr = np.logspace(-2, 3, 50)
            xgr = 0.5 * np.ones_like(ygr)
            result = resp.get_full(mo, da, ygr, xgr)
            assert result.shape == ygr.shape
