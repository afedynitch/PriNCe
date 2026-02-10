"""Additional tests for prince_cr.interaction_rates module."""

import numpy as np
import pytest

import prince_cr.config as config

config.debug_level = 0
config.max_mass = 14

from prince_cr import core, cross_sections, photonfields


@pytest.fixture(scope="module")
def prince_run():
    pf = photonfields.CombinedPhotonField(
        [photonfields.CMBPhotonSpectrum, photonfields.CIBGilmore2D]
    )
    cs = cross_sections.CompositeCrossSection(
        [
            (0.0, cross_sections.TabulatedCrossSection, ("CRP2_TALYS",)),
            (0.14, cross_sections.SophiaSuperposition, ()),
        ]
    )
    return core.PriNCeRun(max_mass=4, photon_field=pf, cross_sections=cs)


class TestPhotoNuclearInteractionRate:
    def test_photon_vector(self, prince_run):
        pv = prince_run.int_rates.photon_vector(0.0)
        assert pv.shape == prince_run.ph_grid.grid.shape
        assert np.all(pv >= 0)

    def test_photon_vector_different_z(self, prince_run):
        pv0 = prince_run.int_rates.photon_vector(0.0)
        pv1 = prince_run.int_rates.photon_vector(1.0)
        assert not np.allclose(pv0, pv1)

    def test_update_rates(self, prince_run):
        result = prince_run.int_rates._update_rates(0.5, force_update=True)
        assert result is True

    def test_update_rates_cache(self, prince_run):
        prince_run.int_rates._update_rates(0.3, force_update=True)
        result = prince_run.int_rates._update_rates(0.3, force_update=False)
        assert result is False

    def test_get_hadr_jacobian(self, prince_run):
        jac = prince_run.int_rates.get_hadr_jacobian(0.5, force_update=True)
        assert jac is not None
        assert jac.shape[0] > 0

    def test_get_hadr_jacobian_with_scale(self, prince_run):
        jac = prince_run.int_rates.get_hadr_jacobian(
            0.5, scale_fac=2.0, force_update=True
        )
        assert jac is not None

    def test_coupling_mat_exists(self, prince_run):
        from scipy.sparse import issparse

        assert issparse(prince_run.int_rates.coupling_mat)

    def test_single_interaction_length(self, prince_run):
        egrid, length = prince_run.int_rates.single_interaction_length(
            101, 0.0, pfield=prince_run.photon_field
        )
        assert len(egrid) > 0
        assert len(length) > 0


class TestContinuousAdiabaticLossRate:
    def test_loss_vector(self, prince_run):
        lv = prince_run.adia_loss_rates_grid.loss_vector(0.0)
        assert lv.shape == (prince_run.dim_states,)
        assert np.all(lv >= 0)

    def test_loss_vector_z1(self, prince_run):
        lv = prince_run.adia_loss_rates_grid.loss_vector(1.0)
        assert np.all(lv >= 0)

    def test_loss_vector_custom_energy(self, prince_run):
        energy = np.ones(prince_run.dim_states) * 1e6
        lv = prince_run.adia_loss_rates_grid.loss_vector(0.0, energy=energy)
        assert lv.shape == energy.shape

    def test_loss_vector_bins(self, prince_run):
        lv = prince_run.adia_loss_rates_bins.loss_vector(0.0)
        assert lv.shape == (prince_run.dim_bins,)

    def test_single_loss_length(self, prince_run):
        egrid, length = prince_run.adia_loss_rates_grid.single_loss_length(101, 0.0)
        assert len(egrid) > 0
        assert len(length) > 0
        assert np.all(np.isfinite(length))


class TestContinuousPairProductionLossRate:
    def test_loss_vector(self, prince_run):
        lv = prince_run.pair_loss_rates_grid.loss_vector(0.0)
        assert lv.shape == (prince_run.dim_states,)

    def test_loss_vector_bins(self, prince_run):
        lv = prince_run.pair_loss_rates_bins.loss_vector(0.0)
        assert lv.shape == (prince_run.dim_bins,)

    def test_photon_vector(self, prince_run):
        pv = prince_run.pair_loss_rates_grid.photon_vector(0.0)
        assert pv.shape == prince_run.pair_loss_rates_grid.photon_grid.shape

    def test_single_loss_length(self, prince_run):
        egrid, length = prince_run.pair_loss_rates_grid.single_loss_length(
            101, 0.0, pfield=prince_run.photon_field
        )
        assert len(egrid) > 0
        assert len(length) > 0

    def test_phi_function(self, prince_run):
        xi = np.logspace(np.log10(2.01), 5, 100)
        phi = prince_run.pair_loss_rates_grid._phi(xi)
        assert phi.shape == xi.shape
        assert np.all(np.isfinite(phi))

    def test_scale_vec_positive_for_nuclei(self, prince_run):
        sv = prince_run.pair_loss_rates_grid.scale_vec
        # Check that at least some entries are positive (for nuclei)
        assert np.any(sv > 0)
