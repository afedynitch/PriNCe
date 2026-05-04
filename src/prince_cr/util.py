"""This module contains utility functions, which fulfill common puposes
in different modules of this project."""

import inspect

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
import prince_cr.config as config


def convert_to_namedtuple(dictionary, name="GenericNamedTuple"):
    """Converts a dictionary to a named tuple."""
    from collections import namedtuple

    return namedtuple(name, list(dictionary.keys()))(**dictionary)


def get_interp_object(xgrid, ygrid, **kwargs):
    """Returns simple standard interpolation object.

    Default type of interpolation is a spline of order
    one without extrapolation (extrapolation to zero).

    Args:
        xgrid (numpy.array): x values of function
        ygrid (numpy.array): y values of function
    """
    if xgrid.shape != ygrid.shape:
        raise Exception(
            "xgrid and ygrid args need identical shapes: {0} != {1}".format(
                xgrid.shape, ygrid.shape
            )
        )

    if "k" not in kwargs:
        kwargs["k"] = 1
    if "ext" not in kwargs:
        kwargs["ext"] = "zeros"

    return InterpolatedUnivariateSpline(xgrid, ygrid, **kwargs)
    # if xwidths is not None:
    #     return np.tile(xwidths,len(ygrid)).reshape(len(xwidths),len(ygrid))*res
    # else:
    #     return res


def get_2Dinterp_object(xgrid, ygrid, zgrid, xbins=None, **kwargs):
    """Returns simple standard interpolation object for 2-dimentsional
    distribution.

    Default type of interpolation is a spline of order
    one without extrapolation (extrapolation to zero).

    Args:
        xgrid (numpy.array): x values of function
        ygrid (numpy.array): y values of function
    """
    if (xgrid.shape[0], ygrid.shape[0]) != zgrid.shape:
        raise Exception(
            "x and y grid do not match z grid shape: {0} != {1}".format(
                (xgrid.shape, ygrid.shape), zgrid.shape
            )
        )

    if "kx" not in kwargs:
        kwargs["kx"] = 1
    if "ky" not in kwargs:
        kwargs["ky"] = 1
    if "s" not in kwargs:
        kwargs["s"] = 0.0
    return RectBivariateSplineNoExtrap(xgrid, ygrid, zgrid, xbins, **kwargs)


class RectBivariateSplineNoExtrap(RectBivariateSpline):
    """Same as RectBivariateSpline but makes sure, that extrapolated data is alway 0"""

    def __init__(self, xgrid, ygrid, zgrid, xbins=None, *args, **kwargs):
        self.xbins = xbins
        RectBivariateSpline.__init__(self, xgrid, ygrid, zgrid, *args, **kwargs)
        xknots, yknots = self.get_knots()
        self.xmin, self.xmax = np.min(xknots), np.max(xknots)
        self.ymin, self.ymax = np.min(yknots), np.max(yknots)

    def __call__(self, x, y, **kwargs):
        if "grid" not in kwargs:
            x, y = np.meshgrid(x, y)
            kwargs["grid"] = False

            result = RectBivariateSpline.__call__(self, x, y, **kwargs)
            # result = np.where((x < self.xmax) & (x > self.xmin), result, 0.)
            # result[np.isnan(result)] = 0.
            # return result.T
            return np.where(np.isnan(result), 0.0, result).T
        else:
            result = RectBivariateSpline.__call__(self, x, y, **kwargs)
            # result = np.where((x <= xmax) & (x >= xmin), result, 0.)
            # result[np.isnan(result)] = 0.
            # return result
            return np.where(np.isnan(result), 0.0, result)


class RectBivariateSplineLogData(RectBivariateSplineNoExtrap):
    """Same as RectBivariateSpline but data is internally interpoled as log(data)"""

    def __init__(self, x, y, z, *args, **kwargs):
        x = np.log10(x)
        y = np.log10(y)

        info(2, "Spline created")
        RectBivariateSplineNoExtrap.__init__(self, x, y, z, *args, **kwargs)

    def __call__(self, x, y, **kwargs):
        x = np.log10(x)
        y = np.log10(y)

        result = RectBivariateSplineNoExtrap.__call__(self, x, y, **kwargs)
        return result


def caller_name(skip=2):
    """Get a name of a caller in the format module.class.method

    `skip` specifies how many levels of stack to skip while getting caller
    name. skip=1 means "who calls me", skip=2 "who calls my caller" etc.
    An empty string is returned if skipped levels exceed stack height.abs

    From https://gist.github.com/techtonik/2151727
    """

    stack = inspect.stack()
    start = 0 + skip

    if len(stack) < start + 1:
        return ""

    parentframe = stack[start][0]

    name = []

    if config.print_module:
        module = inspect.getmodule(parentframe)
        # `modname` can be None when frame is executed directly in console
        if module:
            name.append(module.__name__ + ".")

    # detect classname
    if "self" in parentframe.f_locals:
        # I don't know any way to detect call from the object method
        # there seems to be no way to detect static method call - it will
        # be just a function call

        name.append(parentframe.f_locals["self"].__class__.__name__ + "::")

    codename = parentframe.f_code.co_name
    if codename != "<module>":  # top level usually
        name.append(codename + "(): ")  # function or a method
    else:
        name.append(": ")  # If called from module scope

    del parentframe
    return "".join(name)


def info(min_dbg_level, *message, **kwargs):
    """Print to console if `min_debug_level <= config.debug_level`

    The fuction determines automatically the name of caller and appends
    the message to it. Message can be a tuple of strings or objects
    which can be converted to string using `str()`.

    Args:
        min_dbg_level (int): Minimum debug level in config for printing
        message (tuple): Any argument or list of arguments that casts to str
        condition (bool): Print only if condition is True
        blank_caller (bool): blank the caller name (for multiline output)
        no_caller (bool): don't print the name of the caller
    """
    condition = kwargs.pop("condition", min_dbg_level <= config.debug_level)
    # Dont' process the if the function if nothing will happen
    if not (condition or config.override_debug_fcn):
        return

    blank_caller = kwargs.pop("blank_caller", False)
    no_caller = kwargs.pop("no_caller", False)
    if config.override_debug_fcn and min_dbg_level < config.override_max_level:
        fcn_name = caller_name(skip=2).split("::")[-1].split("():")[0]
        if fcn_name in config.override_debug_fcn:
            min_dbg_level = 0

    if condition and min_dbg_level <= config.debug_level:
        message = [str(m) for m in message]
        cname = caller_name() if not no_caller else ""
        if blank_caller:
            cname = len(cname) * " "
        print(cname + " ".join(message))


_PDG_NUCLEUS_PREFIX = 1000000000
_PDG_PROTON = 2212
_PDG_NEUTRON = 2112


def is_nucleus(pdg_id):
    """True iff ``pdg_id`` represents a nucleus (free p/n or A>=2).

    PDG conventions: free proton 2212, free neutron 2112, all other nuclei
    encoded as ``10LZZZAAAI`` (we only consider ground-state isomers,
    L = I = 0, so the form is ``1000000000 + Z*10000 + A*10``).
    """
    pdg_id = int(pdg_id)
    if pdg_id == _PDG_PROTON or pdg_id == _PDG_NEUTRON:
        return True
    if pdg_id // _PDG_NUCLEUS_PREFIX != 1:
        return False
    rest = pdg_id - _PDG_NUCLEUS_PREFIX
    Z = (rest // 10000) % 1000
    A = (rest // 10) % 1000
    return Z <= A and A >= 1


def make_nucleus_pdg(A, Z, isomer=0):
    """Compose a nuclear PDG ID from mass / charge.

    Free proton and neutron return their canonical PDG codes (2212, 2112);
    all other ground-state nuclei use ``10LZZZAAAI`` with L = isomer = 0.
    """
    A = int(A)
    Z = int(Z)
    if A == 1 and Z == 1:
        return _PDG_PROTON
    if A == 1 and Z == 0:
        return _PDG_NEUTRON
    return _PDG_NUCLEUS_PREFIX + Z * 10000 + A * 10 + int(isomer)


def get_AZN(pdg_id):
    """Return ``(A, Z, N)`` for a nucleus PDG ID; ``(0, 0, 0)`` otherwise.

    Args:
        pdg_id (int): PDG Monte Carlo number.

    Returns:
        (int, int, int): mass number A, charge number Z, neutron number N.
    """
    pdg_id = int(pdg_id)
    if pdg_id == _PDG_PROTON:
        return 1, 1, 0
    if pdg_id == _PDG_NEUTRON:
        return 1, 0, 1
    if pdg_id // _PDG_NUCLEUS_PREFIX != 1:
        return 0, 0, 0
    rest = pdg_id - _PDG_NUCLEUS_PREFIX
    Z = (rest // 10000) % 1000
    A = (rest // 10) % 1000
    return A, Z, A - Z


def bin_widths(bin_edges):
    """Computes and returns bin widths from given edges."""
    edg = np.array(bin_edges)

    return np.abs(edg[1:, ...] - edg[:-1, ...])


class AdditiveDictionary(dict):
    """This dictionary subclass adds values if keys are
    are already present instead of overwriting. For value tuples
    only the second argument is added and the first kept to its
    original value."""

    def __setitem__(self, key, value):
        if key not in self:
            super(AdditiveDictionary, self).__setitem__(key, value)
        elif isinstance(value, tuple):
            super(AdditiveDictionary, self).__setitem__(
                key, (self[key][0], value[1] + self[key][1])
            )
        else:
            super(AdditiveDictionary, self).__setitem__(key, self[key] + value)


class PrinceProgressBar(object):
    """This is a wrapper around tqdm to process some prince
    argument handling, making it optional, for notebooks and
    python scripts using the bar_type argument."""

    def __init__(self, bar_type=None, nsteps=None):
        if bar_type is None or bar_type is False:
            self.pbar = None
        elif bar_type == "notebook":
            from tqdm import tqdm_notebook as tqdm

            self.pbar = tqdm(total=nsteps)
            self.pbar.update()
        else:
            from tqdm import tqdm

            self.pbar = tqdm(total=nsteps)
            self.pbar.update()

    def __enter__(self):
        return self

    def update(self):
        if self.pbar is not None:
            self.pbar.update()

    def __exit__(self, type, value, traceback):
        if self.pbar is not None:
            self.pbar.close()
