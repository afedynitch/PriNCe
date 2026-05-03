"""Tests for cross_sections base accessors and the response pipeline,
exercised through FlukaPhotoNuclear (the only remaining cross-section
class)."""

import numpy as np
import pytest

from prince_cr.util import get_AZN


class TestFlukaAccessors:
    """Base-class accessor smoke tests, exercised through FlukaPhotoNuclear."""

    def test_nonel_proton(self, fluka):
        egrid, cs = fluka.nonel(101)
        assert len(egrid) > 0
        assert len(cs) > 0
        assert np.all(cs >= 0)

    def test_nonel_He4(self, fluka):
        egrid, cs = fluka.nonel(402)
        assert len(egrid) > 0
        assert np.all(cs >= 0)

    def test_nonel_unknown_raises(self, fluka):
        with pytest.raises(Exception, match="unknown"):
            fluka.nonel(9999)

    def test_xcenters(self, fluka):
        xc = fluka.xcenters
        assert len(xc) > 0
        assert np.all(np.diff(xc) > 0)

    def test_xwidths(self, fluka):
        xw = fluka.xwidths
        assert len(xw) > 0
        assert np.all(xw > 0)

    def test_egrid_property(self, fluka):
        eg = fluka.egrid
        assert len(eg) > 0

    def test_set_range(self, fluka):
        orig = fluka._range.copy()
        fluka.set_range(e_min=0.01, e_max=0.5)
        assert len(fluka._range) > 0
        # Restore so other tests still see the full grid
        fluka._range = orig

    def test_nonel_scale_A(self, fluka):
        egrid, cs = fluka.nonel_scale(101)
        A = get_AZN(101)[0]
        egrid2, cs2 = fluka.nonel(101)
        np.testing.assert_allclose(cs, cs2 / A)

    def test_nonel_scale_custom(self, fluka):
        egrid, cs = fluka.nonel_scale(101, scale=2.0)
        egrid2, cs2 = fluka.nonel(101)
        np.testing.assert_allclose(cs, cs2 * 2.0)

    def test_is_differential_redistributed(self, fluka):
        # Pion daughter (ncoid 2) is redistributed — ID <= redist_threshold_ID (101)
        assert fluka.is_differential(101, 2) is True

    def test_is_differential_boost_conserving(self, fluka):
        # He-4 → He-3: 302 > redist_threshold_ID and not in incl_diff_idcs
        assert fluka.is_differential(402, 302) is False


class TestResponseFunction:
    """ResponseFunction integration via FlukaPhotoNuclear."""

    def test_resp_property(self, fluka):
        resp = fluka.resp
        assert resp is not None
        assert hasattr(resp, "nonel_intp")

    def test_get_channel_nonel(self, fluka):
        resp = fluka.resp
        ygr, integral = resp.get_channel(101)
        assert len(ygr) > 0
        assert len(integral) > 0

    def test_get_channel_incl(self, fluka):
        resp = fluka.resp
        if fluka.incl_idcs:
            mo, da = fluka.incl_idcs[0]
            ygr, integral = resp.get_channel(mo, da)
            assert len(ygr) > 0

    def test_get_channel_scale(self, fluka):
        resp = fluka.resp
        ygr, cs = resp.get_channel_scale(101)
        assert len(ygr) > 0

    def test_get_channel_scale_custom(self, fluka):
        resp = fluka.resp
        ygr, cs = resp.get_channel_scale(101, scale=2.0)
        assert len(ygr) > 0
