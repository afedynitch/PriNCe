"""Tests for prince_cr.config module."""

import prince_cr.config as config


class TestConfigConstants:
    def test_H0_positive(self):
        assert config.H_0 > 0
        assert config.H_0s > 0

    def test_omega_sum(self):
        # Omega_m + Omega_Lambda should be approximately 1
        assert 0 < config.Omega_m < 1
        assert 0 < config.Omega_Lambda < 1

    def test_E_CMB_positive(self):
        assert config.E_CMB > 0

    def test_cosmic_ray_grid_format(self):
        assert len(config.cosmic_ray_grid) == 3
        log_emin, log_emax, bins = config.cosmic_ray_grid
        assert log_emin < log_emax
        assert bins > 0

    def test_photon_grid_format(self):
        assert len(config.photon_grid) == 3
        log_emin, log_emax, bins = config.photon_grid
        assert log_emin < log_emax
        assert bins > 0

    def test_grid_scale(self):
        assert config.grid_scale in ["E", "logE"]

    def test_semi_lagr_method(self):
        assert isinstance(config.semi_lagr_method, str)

    def test_data_dir_exists(self):
        import os

        assert os.path.isdir(config.data_dir)

    def test_db_fname_exists(self):
        import os

        assert os.path.isfile(os.path.join(config.data_dir, config.db_fname))

    def test_linear_algebra_backend(self):
        assert config.linear_algebra_backend in ["MKL", "scipy", "cupy", "CUPY"]

    def test_has_cuda_bool(self):
        assert isinstance(config.has_cuda, bool)

    def test_has_cupy_bool(self):
        assert isinstance(config.has_cupy, bool)

    def test_has_mkl_bool(self):
        assert isinstance(config.has_mkl, bool)

    def test_kernel_config_valid(self):
        assert config.kernel_config in ["CUDA", "MKL", "numpy"]

    def test_debug_level_is_number(self):
        assert isinstance(config.debug_level, (int, float))

    def test_tau_dec_threshold(self):
        assert config.tau_dec_threshold >= 0

    def test_max_mass(self):
        assert config.max_mass > 0

    def test_redist_threshold_ID(self):
        assert isinstance(config.redist_threshold_ID, int)

    def test_x_cut(self):
        assert config.x_cut >= 0
        assert config.x_cut_proton >= 0

    def test_update_rates_z_threshold(self):
        assert config.update_rates_z_threshold > 0

    def test_MKL_threads(self):
        assert config.MKL_threads > 0

    def test_base_path(self):
        import os

        assert os.path.isdir(config.base_path)
