import numpy as np

import prince_cr.config as config
from prince_cr.util import (
    BilinearGrid2D,
    get_2Dinterp_object,
    get_interp_object,
    info,
    get_AZN,
)

from .base import _is_redistributed


class ResponseFunction(object):
    """Redistribution Function based on Crossection model

    The response function is the angular average crosssection
    """

    def __init__(self, cross_section):
        self.cross_section = cross_section

        self.xcenters = cross_section.xcenters

        # Copy indices from CrossSection Model
        self.nonel_idcs = cross_section.nonel_idcs
        self.incl_idcs = cross_section.incl_idcs
        self.incl_diff_idcs = cross_section.incl_diff_idcs
        # O(1) membership companion (see `CrossSectionBase`).
        self._incl_diff_idcs_set = set(self.incl_diff_idcs)

        # Dictionary of reponse function interpolators
        self.nonel_intp = {}
        self.incl_intp = {}
        self.incl_diff_intp = {}
        self.incl_diff_intp_integral = {}

        # Batched view of `incl_diff_intp_integral` (Rec #3): all
        # 2D-integral channels share the same `(xcenters, ygr)` grid by
        # construction, so we stack them along a trailing axis to enable
        # one-shot bilinear evaluation in `interaction_rates._init_matrices`.
        # `incl_diff_integral_keys` is the channel ordering used in the
        # stack's last axis; consumers compose
        # ``incl_diff_integral_stack_intp(...)[..., i_for_channel]``.
        self.incl_diff_integral_keys = []
        self.incl_diff_integral_stack_intp = None

        self._precompute_interpolators()

    def is_differential(self, mother, daughter):
        """True if (mother, daughter) requires a redistribution kernel.

        Mirrors `CrossSectionBase.is_differential` but reads only the
        public state copied at __init__ (`incl_diff_idcs`).
        """
        return (
            _is_redistributed(daughter)
            or (mother, daughter) in self._incl_diff_idcs_set
        )

    def get_full(self, mother, daughter, ygrid, xgrid=None):
        """Return the full response function :math:`f(y) + g(y) + h(x,y)`
        on the ygrid. xgrid is ignored if `h(x,y)` not in the channel.
        """
        if xgrid is not None and ygrid.shape != xgrid.shape:
            raise Exception("ygrid and xgrid do not have the same shape!!")
        if get_AZN(mother)[0] < get_AZN(daughter)[0]:
            info(
                3,
                "WARNING: channel {:} -> {:} with daughter heavier than mother!".format(
                    mother, daughter
                ),
            )

        res = np.zeros(ygrid.shape)

        if (mother, daughter) in self.incl_intp:
            res += self.incl_intp[(mother, daughter)](ygrid)
        elif (mother, daughter) in self.incl_diff_intp:
            # incl_diff_res = self.incl_diff_intp[(mother, daughter)](
            #    xgrid, ygrid, grid=False)
            # if mother == 101:
            #    incl_diff_res = np.where(xgrid < 0.9, incl_diff_res, 0.)
            # res += incl_diff_res
            # if not(mother == daughter):
            res += self.incl_diff_intp[(mother, daughter)].ev(xgrid, ygrid)

        if mother == daughter and mother in self.nonel_intp:
            # nonel cross section leads to absorption, therefore the minus
            if xgrid is None:
                res -= self.nonel_intp[mother](ygrid)
            else:
                diagonal = xgrid == 1.0
                res[diagonal] -= self.nonel_intp[mother](ygrid[diagonal])

        return res

    def get_channel(self, mother, daughter=None):
        """Reponse function :math:`f(y)` or :math:`g(y)` as
        defined in the note.

        Returns :math:`f(y)` or :math:`g(y)` if a daughter
        index is provided. If the inclusive channel has a redistribution,
        :math:`h(x,y)` will be returned

        Args:
            mother (int): mother nucleus(on)
            daughter (int, optional): daughter nucleus(on)

        Returns:
            (numpy.array) Reponse function on self._ygrid_tab
        """
        from scipy import integrate

        cs_model = self.cross_section
        egrid, cross_section = None, None

        if daughter is not None:
            if (mother, daughter) in self.incl_diff_idcs:
                egrid, cross_section = cs_model.incl_diff(mother, daughter)
            elif (mother, daughter) in self.incl_idcs:
                egrid, cross_section = cs_model.incl(mother, daughter)
            else:
                raise Exception(
                    "Unknown inclusive channel {:} -> {:} for this model".format(
                        mother, daughter
                    )
                )
        else:
            egrid, cross_section = cs_model.nonel(mother)

        # note that cumulative_trapezoid works also for 2d-arrays and will integrate along axis = 1
        integral = integrate.cumulative_trapezoid(egrid * cross_section, x=egrid)
        ygrid = egrid[1:] / 2.0

        return ygrid, integral / (2 * ygrid**2)

    def get_channel_scale(self, mother, daughter=None, scale="A"):
        """Returns the reponse function scaled by `scale`.

        Convenience funtion for plotting, where it is important to
        compare the cross section/response function per nucleon.

        Args:
            mother (int): Mother nucleus(on)
            scale (float): If `A` then nonel/A is returned, otherwise
                           scale can be any float.

        Returns:
            (numpy.array, numpy.array): Tuple of Energy grid in GeV,
                                        scale * inclusive cross section
                                        in :math:`cm^{-2}`
        """

        ygr, cs = self.get_channel(mother, daughter)

        if scale == "A":
            scale = 1.0 / get_AZN(mother)[0]

        return ygr, scale * cs

    def _precompute_interpolators(self):
        """Interpolate each response function and store interpolators.

        Uses :func:`prince_cr.util.get_interp_object` as interpolator.
        This might result in too many knots and can be subject to
        future optimization.
        """

        info(2, "Computing interpolators for response functions")

        # All 1D channels (nonel + boost-conserving incl) share egrid, hence
        # ygr = egrid[1:]/2. Compute their response f(y) in ONE vectorized
        # cumulative_trapezoid (the get_channel formula f = cumtrapz(egrid*σ)
        # /(2 y²)) instead of a per-channel get_channel call, then build the
        # k=1 interpolators from the stack and STORE the stacks so
        # interaction_rates._init_matrices can batch the antiderivative sampling.
        cs = self.cross_section
        egrid = cs.egrid
        ygr = egrid[1:] / 2.0
        from scipy.integrate import cumulative_trapezoid as _ctrap

        def _resp_stack(keys, getter):
            if not keys:
                return np.zeros((0, ygr.size))
            sig = np.array([getter(k)[1] for k in keys])     # (n_ch, n_E), shared egrid
            return _ctrap(egrid * sig, x=egrid, axis=1) / (2.0 * ygr**2)

        nonel_keys = list(self.nonel_idcs)
        incl_keys = list(self.incl_idcs)
        nonel_f = _resp_stack(nonel_keys, lambda mo: cs.nonel(mo))
        incl_f = _resp_stack(incl_keys, lambda k: cs.incl(*k))

        # stacks for the batched antiderivative in interaction_rates._init_matrices
        self.response_ygrid = ygr
        self.nonel_keys, self.nonel_f_stack = nonel_keys, nonel_f
        self.incl_keys, self.incl_f_stack = incl_keys, incl_f

        # Per-channel k=1 interpolators are needed ONLY by the reference matrix
        # path (config.fast_response_build=False) and by get_full(). Building
        # 50k+ spline objects is pure overhead on the fast path, where
        # _init_matrices consumes the stacks via the response-integral operator,
        # so skip them there. (get_full() therefore requires the reference path.)
        self.nonel_intp = {}
        self.incl_intp = {}
        if not getattr(config, "fast_response_build", True):
            info(5, "Nonelastic response functions f(y)")
            for i, mo in enumerate(nonel_keys):
                self.nonel_intp[mo] = get_interp_object(ygr, nonel_f[i])
            info(5, "Inclusive (boost conserving) response functions g(y)")
            for i, key in enumerate(incl_keys):
                self.incl_intp[key] = get_interp_object(ygr, incl_f[i])

        info(5, "Inclusive (redistributed) response functions h(y)")
        self.incl_diff_intp = {}
        # Collect per-channel integral arrays so we can stack them after
        # the loop. All channels share the same `(xcenters, ygr)` grid
        # because `xcenters` is fixed and the FLUKA tabulation gives
        # every channel the same energy grid; cross-checked at m=56.
        from scipy.integrate import cumulative_trapezoid

        stacked_integrals = []
        stacked_keys = []
        shared_ygr = None
        for mother, daughter in self.incl_diff_idcs:
            ygr, rfunc = self.get_channel(mother, daughter)
            # Use the lighter `BilinearGrid2D` (RegularGridInterpolator)
            # instead of `RectBivariateSplineNoExtrap` for the
            # incl_diff path: bilinear-on-regular-grid is bit-exact in
            # the domain and skips FITPACK's knot fit. The 2D-spline
            # one is preserved for `incl_diff_intp` since `get_full`
            # treats it differently and external callers may assume
            # the spline surface there.
            self.incl_diff_intp[(mother, daughter)] = get_2Dinterp_object(
                self.xcenters, ygr, rfunc, self.cross_section.xbins
            )

            integral = cumulative_trapezoid(rfunc, ygr, axis=1, initial=0)
            integral = cumulative_trapezoid(integral, self.xcenters, axis=0, initial=0)

            self.incl_diff_intp_integral[(mother, daughter)] = BilinearGrid2D(
                self.xcenters, ygr, integral, self.cross_section.xbins
            )

            # Accumulate the integral on the shared grid for batched eval.
            # The first channel sets `shared_ygr`; subsequent channels
            # must match. (This is true for FLUKA — all channels share
            # the same egrid — but assert defensively.)
            if shared_ygr is None:
                shared_ygr = ygr
            elif (
                ygr.shape != shared_ygr.shape
                or not np.array_equal(ygr, shared_ygr)
            ):
                # Bail out of the batched path: keep the per-channel
                # interpolators and leave the stack intp as None so the
                # consumer falls back to the per-channel `.ev` loop.
                self.incl_diff_integral_keys = []
                self.incl_diff_integral_stack_intp = None
                stacked_integrals = None
                continue
            if stacked_integrals is not None:
                stacked_integrals.append(integral)
                stacked_keys.append((mother, daughter))

        if stacked_integrals and shared_ygr is not None:
            # Shape: (n_channels, n_x, n_y). RGI takes values shape
            # (n_x, n_y, n_channels) for trailing-axis batching.
            stack = np.stack(stacked_integrals, axis=-1)
            self.incl_diff_integral_keys = stacked_keys
            self.incl_diff_integral_stack_intp = BilinearGrid2D(
                self.xcenters, shared_ygr, stack, self.cross_section.xbins
            )
