"""Tests for prince_cr.photonfields module."""

import numpy as np
import pytest

import prince_cr.config as config

config.debug_level = 0

from prince_cr.photonfields import (
    PhotonField,
    CombinedPhotonField,
    FlatPhotonSpectrum,
    CMBPhotonSpectrum,
    EBLSplined2D,
    CIBFranceschini2D,
    CIBInoue2D,
    CIBGilmore2D,
    CIBDominguez2D,
    CIBFranceschiniZ0,
    CIBSteckerZ0,
)


class TestCMBPhotonSpectrum:
    def test_z0_positive(self):
        cmb = CMBPhotonSpectrum()
        E = np.logspace(-15, -9, 50)
        result = cmb.get_photon_density(E, 0.0)
        assert np.all(result >= 0)
        assert np.any(result > 0)

    def test_z0_shape(self):
        cmb = CMBPhotonSpectrum()
        E = np.logspace(-15, -9, 50)
        result = cmb.get_photon_density(E, 0.0)
        assert result.shape == E.shape

    def test_scales_with_redshift(self):
        cmb = CMBPhotonSpectrum()
        E = np.logspace(-14, -11, 20)
        n0 = cmb.get_photon_density(E, 0.0)
        n1 = cmb.get_photon_density(E, 1.0)
        # At higher z, the CMB should be denser
        assert np.sum(n1) > np.sum(n0)

    def test_high_energy_suppressed(self):
        cmb = CMBPhotonSpectrum()
        E = np.array([1e-6])  # Very high energy photon
        result = cmb.get_photon_density(E, 0.0)
        assert result[0] == 0.0  # Beyond exp range


class TestFlatPhotonSpectrum:
    def test_z0(self):
        flat = FlatPhotonSpectrum()
        E = np.logspace(-15, -9, 50)
        result = flat.get_photon_density(E, 0.0)
        np.testing.assert_allclose(result, np.ones_like(E) * 1e12)

    def test_redshift_scaling(self):
        flat = FlatPhotonSpectrum()
        E = np.logspace(-15, -9, 50)
        z = 1.0
        result = flat.get_photon_density(E, z)
        expected = (1.0 + z) ** 2 * 1e12
        np.testing.assert_allclose(result, expected * np.ones_like(E))


class TestCombinedPhotonField:
    def test_single_model(self):
        cpf = CombinedPhotonField([CMBPhotonSpectrum])
        E = np.logspace(-14, -11, 20)
        result = cpf.get_photon_density(E, 0.0)
        cmb = CMBPhotonSpectrum()
        expected = cmb.get_photon_density(E, 0.0)
        np.testing.assert_allclose(result, expected)

    def test_multiple_models(self):
        cpf = CombinedPhotonField([CMBPhotonSpectrum, FlatPhotonSpectrum])
        E = np.logspace(-14, -11, 20)
        result = cpf.get_photon_density(E, 0.0)
        cmb = CMBPhotonSpectrum()
        flat = FlatPhotonSpectrum()
        expected = cmb.get_photon_density(E, 0.0) + flat.get_photon_density(E, 0.0)
        np.testing.assert_allclose(result, expected)

    def test_tuple_args(self):
        cpf = CombinedPhotonField([(CIBInoue2D, "base")])
        E = np.logspace(-14, -11, 20)
        result = cpf.get_photon_density(E, 0.0)
        assert np.all(result >= 0)

    def test_add_model(self):
        cpf = CombinedPhotonField([CMBPhotonSpectrum])
        cpf.add_model(FlatPhotonSpectrum)
        E = np.logspace(-14, -11, 20)
        result = cpf.get_photon_density(E, 0.0)
        assert np.all(result > 0)

    def test_scalar_input(self):
        cpf = CombinedPhotonField([CMBPhotonSpectrum])
        result = cpf.get_photon_density(1e-13, 0.0)
        assert result.size >= 1


class TestCIBFranceschini2D:
    def test_init(self):
        cib = CIBFranceschini2D()
        assert cib.int2d is not None

    def test_get_photon_density(self):
        cib = CIBFranceschini2D()
        E = np.logspace(-12, -9, 20)
        result = cib.get_photon_density(E, 0.0)
        assert result.shape == E.shape

    def test_simple_scaling(self):
        cib = CIBFranceschini2D(simple_scaling=True)
        E = np.logspace(-12, -9, 20)
        result = cib.get_photon_density(E, 0.5)
        assert np.all(np.isfinite(result))


class TestCIBInoue2D:
    def test_base_model(self):
        cib = CIBInoue2D(model="base")
        E = np.logspace(-12, -9, 20)
        result = cib.get_photon_density(E, 0.0)
        assert np.all(result >= 0)

    def test_upper_model(self):
        cib = CIBInoue2D(model="upper")
        assert cib.int2d is not None

    def test_lower_model(self):
        cib = CIBInoue2D(model="lower")
        assert cib.int2d is not None

    def test_invalid_model(self):
        with pytest.raises(AssertionError):
            CIBInoue2D(model="invalid")


class TestCIBGilmore2D:
    def test_fiducial_model(self):
        cib = CIBGilmore2D(model="fiducial")
        E = np.logspace(-12, -9, 20)
        result = cib.get_photon_density(E, 0.0)
        assert result.shape == E.shape

    def test_fixed_model(self):
        cib = CIBGilmore2D(model="fixed")
        assert cib.int2d is not None

    def test_invalid_model(self):
        with pytest.raises(AssertionError):
            CIBGilmore2D(model="invalid")


class TestCIBDominguez2D:
    def test_base_model(self):
        cib = CIBDominguez2D(model="base")
        E = np.logspace(-12, -9, 20)
        result = cib.get_photon_density(E, 0.0)
        assert result.shape == E.shape

    def test_upper_lower_models(self):
        cib_up = CIBDominguez2D(model="upper")
        cib_lo = CIBDominguez2D(model="lower")
        assert cib_up.int2d is not None
        assert cib_lo.int2d is not None

    def test_invalid_model(self):
        with pytest.raises(AssertionError):
            CIBDominguez2D(model="invalid")


class TestCIBFranceschiniZ0:
    def test_init(self):
        cib = CIBFranceschiniZ0()
        assert len(cib.E) > 0
        assert len(cib.ngamma) > 0

    def test_z0_density(self):
        cib = CIBFranceschiniZ0()
        E = np.logspace(-12, -9, 20)
        result = cib.get_photon_density(E, 0)
        assert np.all(np.isfinite(result))

    def test_z_positive_raises(self):
        cib = CIBFranceschiniZ0()
        E = np.logspace(-12, -9, 20)
        with pytest.raises(Exception, match="z > 0"):
            cib.get_photon_density(E, 0.5)


class TestCIBSteckerZ0:
    def test_init(self):
        cib = CIBSteckerZ0()
        assert len(cib.E) > 0
        assert len(cib.ngamma) > 0

    def test_z0_density(self):
        cib = CIBSteckerZ0()
        E = np.logspace(-12, -9, 20)
        result = cib.get_photon_density(E, 0)
        assert np.all(np.isfinite(result))

    def test_z_positive_raises(self):
        cib = CIBSteckerZ0()
        E = np.logspace(-12, -9, 20)
        with pytest.raises(Exception, match="z > 0"):
            cib.get_photon_density(E, 0.5)
