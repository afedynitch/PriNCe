# Test whether interaction rates are correctly created.

from prince_cr import core


class TestIntRates:
    def test_kernel_m4(self, pf, cs):
        prince_run = core.PriNCeRun(max_mass=4, photon_field=pf, cross_sections=cs)
        ph_dim = prince_run.ph_grid.d
        assert prince_run.int_rates._batch_matrix.shape[1] == ph_dim
        assert prince_run.int_rates._batch_matrix.shape[0] > 0
        assert prince_run.int_rates._batch_rows.shape[0] > 0
        assert prince_run.int_rates._batch_cols.shape[0] > 0
        assert prince_run.int_rates._batch_vec.shape[0] > 0

    def test_kernel_m1(self, pf, cs):
        prince_run = core.PriNCeRun(max_mass=1, photon_field=pf, cross_sections=cs)
        ph_dim = prince_run.ph_grid.d
        assert prince_run.int_rates._batch_matrix.shape[1] == ph_dim
        assert prince_run.int_rates._batch_matrix.shape[0] > 0
        assert prince_run.int_rates._batch_rows.shape[0] > 0
        assert prince_run.int_rates._batch_cols.shape[0] > 0
        assert prince_run.int_rates._batch_vec.shape[0] > 0

    def test_kernel_m14(self, prince_run_m14):
        ph_dim = prince_run_m14.ph_grid.d
        assert prince_run_m14.int_rates._batch_matrix.shape[1] == ph_dim
        assert prince_run_m14.int_rates._batch_matrix.shape[0] > 0
        assert prince_run_m14.int_rates._batch_rows.shape[0] > 0
        assert prince_run_m14.int_rates._batch_cols.shape[0] > 0
        assert prince_run_m14.int_rates._batch_vec.shape[0] > 0
