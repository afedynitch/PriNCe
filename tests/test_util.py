"""Tests for prince_cr.util module."""

import numpy as np
import pytest

import prince_cr.config as config

from prince_cr.util import (
    convert_to_namedtuple,
    get_interp_object,
    get_2Dinterp_object,
    RectBivariateSplineNoExtrap,
    RectBivariateSplineLogData,
    caller_name,
    info,
    get_AZN,
    bin_widths,
    AdditiveDictionary,
    PrinceBDF,
    PrinceProgressBar,
)


class TestConvertToNamedtuple:
    def test_basic(self):
        d = {"a": 1, "b": 2}
        nt = convert_to_namedtuple(d)
        assert nt.a == 1
        assert nt.b == 2

    def test_custom_name(self):
        d = {"x": 10}
        nt = convert_to_namedtuple(d, name="MyTuple")
        assert type(nt).__name__ == "MyTuple"
        assert nt.x == 10


class TestGetInterpObject:
    def test_basic_interpolation(self):
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        spl = get_interp_object(x, y)
        result = spl(5.0)
        np.testing.assert_allclose(result, np.sin(5.0), atol=0.05)

    def test_shape_mismatch_raises(self):
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 10, 40)
        with pytest.raises(Exception, match="identical shapes"):
            get_interp_object(x, y)

    def test_custom_kwargs(self):
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        spl = get_interp_object(x, y, k=3, ext="zeros")
        assert spl(5.0) is not None

    def test_extrapolation_to_zero(self):
        x = np.linspace(1, 10, 50)
        y = np.ones(50)
        spl = get_interp_object(x, y)
        result = spl(20.0)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)


class TestGet2DInterpObject:
    def test_basic_2d_interpolation(self):
        x = np.linspace(0, 5, 10)
        y = np.linspace(0, 5, 10)
        z = np.outer(x, y)
        interp = get_2Dinterp_object(x, y, z)
        result = interp(np.array([2.5]), np.array([2.5]))
        assert result.shape[0] > 0

    def test_shape_mismatch_raises(self):
        x = np.linspace(0, 5, 10)
        y = np.linspace(0, 5, 8)
        z = np.ones((10, 10))
        with pytest.raises(Exception, match="do not match"):
            get_2Dinterp_object(x, y, z)

    def test_with_grid_kwarg(self):
        x = np.linspace(0, 5, 10)
        y = np.linspace(0, 5, 10)
        z = np.outer(x, y)
        interp = get_2Dinterp_object(x, y, z)
        result = interp(np.array([2.5]), np.array([2.5]), grid=True)
        assert result is not None


class TestRectBivariateSplineNoExtrap:
    def test_call_without_grid(self):
        x = np.linspace(0, 5, 10)
        y = np.linspace(0, 5, 10)
        z = np.outer(x, y)
        spl = RectBivariateSplineNoExtrap(x, y, z)
        result = spl(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert result.shape == (2, 2)

    def test_call_with_grid(self):
        x = np.linspace(0, 5, 10)
        y = np.linspace(0, 5, 10)
        z = np.outer(x, y)
        spl = RectBivariateSplineNoExtrap(x, y, z)
        result = spl(np.array([1.0, 2.0]), np.array([1.0, 2.0]), grid=True)
        assert result is not None


class TestRectBivariateSplineLogData:
    def test_basic(self):
        x = np.logspace(0, 2, 10)
        y = np.logspace(0, 2, 10)
        z = np.outer(np.log10(x), np.log10(y))
        spl = RectBivariateSplineLogData(x, y, z)
        result = spl(np.array([10.0]), np.array([10.0]))
        assert result.shape[0] > 0


class TestCallerName:
    def test_returns_string(self):
        name = caller_name(skip=1)
        assert isinstance(name, str)

    def test_high_skip_returns_empty(self):
        name = caller_name(skip=100)
        assert name == ""

    def test_module_info(self):
        old_val = config.print_module
        config.print_module = True
        name = caller_name(skip=1)
        config.print_module = old_val
        assert isinstance(name, str)


class TestInfo:
    def test_silent_at_low_debug(self):
        old = config.debug_level
        config.debug_level = 0
        # Should not raise
        info(1, "test message")
        config.debug_level = old

    def test_prints_at_high_debug(self, capsys):
        old = config.debug_level
        config.debug_level = 10
        info(1, "test message")
        config.debug_level = old
        captured = capsys.readouterr()
        assert "test message" in captured.out

    def test_blank_caller(self, capsys):
        old = config.debug_level
        config.debug_level = 10
        info(1, "hello", blank_caller=True)
        config.debug_level = old
        captured = capsys.readouterr()
        assert "hello" in captured.out

    def test_no_caller(self, capsys):
        old = config.debug_level
        config.debug_level = 10
        info(1, "world", no_caller=True)
        config.debug_level = old
        captured = capsys.readouterr()
        assert "world" in captured.out

    def test_condition_false(self, capsys):
        old = config.debug_level
        config.debug_level = 10
        info(1, "should not appear", condition=False)
        config.debug_level = old
        captured = capsys.readouterr()
        assert "should not appear" not in captured.out

    def test_override_debug_fcn(self, capsys):
        old_level = config.debug_level
        old_override = config.override_debug_fcn
        old_max = config.override_max_level
        config.debug_level = 10
        config.override_debug_fcn = ["test_override_debug_fcn"]
        config.override_max_level = 10
        info(1, "override message")
        config.debug_level = old_level
        config.override_debug_fcn = old_override
        config.override_max_level = old_max
        captured = capsys.readouterr()
        assert "override message" in captured.out


class TestGetAZN:
    def test_proton(self):
        A, Z, N = get_AZN(101)
        assert A == 1
        assert Z == 1
        assert N == 0

    def test_neutron(self):
        A, Z, N = get_AZN(100)
        assert A == 1
        assert Z == 0
        assert N == 1

    def test_helium4(self):
        A, Z, N = get_AZN(402)
        assert A == 4
        assert Z == 2
        assert N == 2

    def test_iron56(self):
        A, Z, N = get_AZN(5626)
        assert A == 56
        assert Z == 26
        assert N == 30

    def test_below_100(self):
        A, Z, N = get_AZN(50)
        assert A == 0
        assert Z == 0
        assert N == 0


class TestBinWidths:
    def test_uniform_bins(self):
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        widths = bin_widths(edges)
        np.testing.assert_array_equal(widths, [1.0, 1.0, 1.0])

    def test_nonuniform_bins(self):
        edges = np.array([0.0, 1.0, 3.0, 6.0])
        widths = bin_widths(edges)
        np.testing.assert_array_equal(widths, [1.0, 2.0, 3.0])


class TestAdditiveDictionary:
    def test_new_key(self):
        d = AdditiveDictionary()
        d["a"] = 5
        assert d["a"] == 5

    def test_existing_key_adds(self):
        d = AdditiveDictionary()
        d["a"] = 5
        d["a"] = 3
        assert d["a"] == 8

    def test_tuple_value(self):
        d = AdditiveDictionary()
        d["a"] = ("keep", 5)
        d["a"] = ("ignore", 3)
        assert d["a"] == ("keep", 8)


class TestPrinceProgressBar:
    def test_no_bar(self):
        with PrinceProgressBar(bar_type=None, nsteps=10) as pbar:
            pbar.update()

    def test_false_bar(self):
        with PrinceProgressBar(bar_type=False, nsteps=10) as pbar:
            pbar.update()

    def test_default_bar(self):
        with PrinceProgressBar(bar_type="default", nsteps=10) as pbar:
            pbar.update()


class TestPrinceBDF:
    def test_simple_ode(self):
        """Test BDF solver on dy/dt = -y, y(0)=1 => y(1) = e^-1"""
        from scipy.sparse import csr_matrix

        def fun(t, y):
            return -y

        # Create a simple sparsity pattern
        sparsity = csr_matrix(np.ones((1, 1)))

        solver = PrinceBDF(
            fun,
            0.0,
            np.array([1.0]),
            1.0,
            max_step=0.1,
            atol=1e-8,
            rtol=1e-8,
            jac_sparsity=sparsity,
        )

        while solver.status == "running":
            solver.step()

        np.testing.assert_allclose(solver.y[0], np.exp(-1), rtol=1e-4)

    def test_vectorized_ode(self):
        """Test BDF solver with vectorized=True"""
        from scipy.sparse import csr_matrix

        def fun(t, y):
            return -y

        sparsity = csr_matrix(np.ones((2, 2)))

        solver = PrinceBDF(
            fun,
            0.0,
            np.array([1.0, 2.0]),
            1.0,
            max_step=0.1,
            atol=1e-6,
            rtol=1e-6,
            jac_sparsity=sparsity,
            vectorized=True,
        )

        while solver.status == "running":
            solver.step()

        np.testing.assert_allclose(solver.y[0], np.exp(-1), rtol=1e-3)
        np.testing.assert_allclose(solver.y[1], 2 * np.exp(-1), rtol=1e-3)
