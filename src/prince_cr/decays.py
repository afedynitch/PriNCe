"""The module contains everything to handle cross section interfaces."""

import numpy as np
from scipy.integrate import trapezoid as trapz

from prince_cr.util import info, get_AZN, is_nucleus
from prince_cr.data import spec_data


# PDG codes referenced in the channel switch below. Names mirror the PDG MC
# numbering scheme.
_PDG_PI_PLUS = 211
_PDG_PI_MINUS = -211
_PDG_PI_ZERO = 111
_PDG_GAMMA = 22
_PDG_MU_PLUS = -13
_PDG_MU_MINUS = 13
_PDG_NU_E = 12
_PDG_NU_E_BAR = -12
_PDG_NU_MU = 14
_PDG_NU_MU_BAR = -14
_PDG_PIONS = (_PDG_PI_PLUS, _PDG_PI_MINUS)
_PDG_MUONS = (_PDG_MU_PLUS, _PDG_MU_MINUS)
_PDG_E_NEUTRINOS = (_PDG_NU_E, _PDG_NU_E_BAR)
_PDG_MU_NEUTRINOS = (_PDG_NU_MU, _PDG_NU_MU_BAR)


def _tabulated_decay_avg(dndx, tab_xbins, x_lower, x_upper):
    """True bin-average of a tabulated dN/dx over ``[x_lower, x_upper]``.

    `dndx` is per-decay dN/dx on `tab_xbins` (n_x bins). Both FLUKA and
    Pythia decay tables share PriNCe's per-nucleon x convention so
    `tab_xbins` are the same edges as `prince_cr.cross_sections.xbins`.

    Integrates the histogram's cumulative mass C(x) = ∫dN/dx dx (piecewise
    linear in x, exact for a histogram) and returns (C(x_up)−C(x_lo))/Δx —
    number-conserving on ANY target grid that covers the support. The
    previous implementation point-evaluated a log-x interpolation of the
    density at the bin midpoint, which loses essentially all the mass of
    delta-like kernels (e.g. the π⁰→γγ rest-frame spike at x=½: the 8/dec
    transport grid caught ~0.2% of the photon number). Bins outside the
    tabulated support average to 0.
    """
    mass = np.concatenate([[0.0], np.cumsum(dndx * np.diff(tab_xbins))])
    lo = np.asarray(x_lower, dtype=float).ravel()
    up = np.asarray(x_upper, dtype=float).ravel()
    C_lo = np.interp(lo, tab_xbins, mass, left=0.0, right=mass[-1])
    C_up = np.interp(up, tab_xbins, mass, left=0.0, right=mass[-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        avg = np.where(up > lo, (C_up - C_lo) / (up - lo), 0.0)
    return avg.reshape(np.shape(x_lower))


def _nucleus_charge_step(nucleus_pdg, dZ):
    """Return the PDG ID of a nucleus with the same A but Z shifted by ``dZ``.

    Used by β±-decay channels: the daughter has the same mass number but the
    charge changes by ±1. PDG nuclear codes encode Z with a stride of 10000
    (``10LZZZAAAI``); free p/n have to be promoted to the nuclear form.
    """
    A, Z, _ = get_AZN(nucleus_pdg)
    from prince_cr.util import make_nucleus_pdg

    return make_nucleus_pdg(A, Z + dZ)


def get_particle_channels(mo, mo_energy, da_energy):
    """
    Loops over a all daughers for a given mother and generates
    a list of redistribution matrices on the grid:
     np.outer( da_energy , 1 / mo_energy )

    Args:
      mo (int): PDG id of the mother particle
      mo_energy (float): energy grid of the mother particle
      da_energy (float): energy grid of the daughter particle (same for all daughters)
    Returns:
      list of np.array: list of redistribution functions on on xgrid
    """
    info(10, "Generating decay redistribution for", mo)
    dbentry = spec_data[mo]
    x_grid = np.outer(da_energy, (1 / mo_energy))

    redist = {}
    for branching, daughters in dbentry["branchings"]:
        for da in daughters:
            # daughter is a nucleus → boost conservation (Lorentz factor preserved)
            if is_nucleus(da):
                res = np.zeros(x_grid.shape)
                res[x_grid == 1.0] = 1.0
            else:
                res = get_decay_matrix(mo, da, x_grid)
            redist[da] = branching * res

    return x_grid, redist


def get_decay_matrix(mo, da, x_grid):
    """
    Selects the correct redistribution for the given decay channel.
    If the channel is unknown a zero grid is returned instead of raising an error

    Args:
      mo (int): PDG id of the mother
      da (int): PDG id of the daughter
      x_grid (float): grid in x = E_da / E_mo on which to return the result
                      (If x is a 2D matrix only the last column is computed
                      and then repeated over the matrix assuming that the
                      main diagonal is always x = 1)
    Returns:
      float: redistribution on the grid mo_energy / da_energy
    """

    info(10, "Generating decay redistribution for", mo, da)

    # --------------------------------
    # pi0 → gamma gamma (analytic lab-frame box; see pi0_to_gamma)
    # --------------------------------
    if mo == _PDG_PI_ZERO and da == _PDG_GAMMA:
        return pi0_to_gamma(x_grid)

    # --------------------------------
    # pi+ → nu_mu, pi- → nu_mubar
    # --------------------------------
    if mo == _PDG_PI_PLUS and da == _PDG_NU_MU:
        return pion_to_numu(x_grid)
    if mo == _PDG_PI_MINUS and da == _PDG_NU_MU_BAR:
        return pion_to_numu(x_grid)

    # --------------------------------
    # pi± → μ± (helicity-mixed; helicity collapsed to 0 in the PDG scheme)
    # --------------------------------
    if mo == _PDG_PI_PLUS and da == _PDG_MU_PLUS:
        return pion_to_muon(x_grid)
    if mo == _PDG_PI_MINUS and da == _PDG_MU_MINUS:
        return pion_to_muon(x_grid)

    # --------------------------------
    # μ → ν (helicity 0; chromo's Pythia post-decay produces helicity-mixed
    # muons, so we evaluate the standard distributions at h=0).
    # --------------------------------
    if mo == _PDG_MU_PLUS and da == _PDG_NU_E:
        return muonplus_to_nue(x_grid, 0.0)
    if mo == _PDG_MU_PLUS and da == _PDG_NU_MU_BAR:
        return muonplus_to_numubar(x_grid, 0.0)
    if mo == _PDG_MU_MINUS and da == _PDG_NU_E_BAR:
        return muonplus_to_nue(x_grid, 0.0)
    if mo == _PDG_MU_MINUS and da == _PDG_NU_MU:
        return muonplus_to_numubar(x_grid, 0.0)

    # --------------------------------
    # neutrinos from beta decays of nuclei
    # --------------------------------
    if is_nucleus(mo) and da == _PDG_NU_E:
        # ν_e produced when the daughter has Z-1: electron-capture / β+-style.
        daughter = _nucleus_charge_step(mo, -1)
        info(10, "nu_e from β-style decay", mo, daughter, da)
        return nu_from_beta_decay(x_grid, mo, daughter)
    if is_nucleus(mo) and da == _PDG_NU_E_BAR:
        # ν̄_e produced when the daughter has Z+1: β-decay.
        daughter = _nucleus_charge_step(mo, +1)
        info(10, "nubar_e from β-style decay", mo, daughter, da)
        return nu_from_beta_decay(x_grid, mo, daughter)
    # nucleus → nucleus (β-decay boost conservation)
    if is_nucleus(mo) and is_nucleus(da):
        info(10, "beta decay boost conservation", mo, da)
        return boost_conservation(x_grid)

    info(
        5,
        "Called with unknown channel {:} to {:}, returning an empty redistribution".format(
            mo, da
        ),
    )
    # no known channel, return zeros
    return np.zeros(x_grid.shape)


def get_decay_matrix_bin_average(mo, da, x_lower, x_upper):
    """
    Selects the correct redistribution for the given decay channel.
    If the channel is unknown a zero grid is returned instead of raising an error

    Args:
      mo (int): index of the mother
      da (int): index of the daughter
      x_grid (float): grid in x = E_da / E_mo on which to return the result

    Returns:
      float: redistribution on the grid mo_energy / da_energy
    """

    # TODO: Some of the distribution are not averaged yet.
    # The error is small for smooth distributions though
    info(10, "Generating decay redistribution for", mo, da)

    x_grid = (x_upper + x_lower) / 2

    # remember shape, but only calculate for last column, as x repeats in each column

    shape = x_grid.shape

    if len(shape) == 2:
        x_grid = x_grid[:, -1]
        x_upper = x_upper[:, -1]
        x_lower = x_lower[:, -1]

    result = None

    # --------------------------------
    # pi0 → gamma gamma : analytic lab-frame box, OVERRIDES the tabulated kernel.
    # The tabulated π⁰→γγ entry is the rest-frame spike at x=½ (see the comment in
    # _tabulated_decay_avg); used as a lab-frame redistribution it pins every γ at
    # half the π⁰ energy and loses the broad low-E box tail (verified vs AM3:
    # native π⁰→γγ caught ~8% of the photon number below 1e14 eV). The correct
    # lab-frame distribution is analytic (massless daughter, ultra-relativistic
    # parent) so we use it directly. See pi0_to_gamma_avg.
    # --------------------------------
    if mo == _PDG_PI_ZERO and da == _PDG_GAMMA:
        result = pi0_to_gamma_avg(x_lower, x_upper)

    # --------------------------------
    # Tabulated FLUKA / Pythia (preferred — populated at module load by
    # data._merge_tabulated_decays from /decays/FLUKA_DECAY_2025/ and
    # /decays/PYTHIA_HADRON_2025/). Both groups share PriNCe's per-nucleon
    # x convention: nuclei -> x = E_d_rest/m_N, hadrons -> x = E_d_rest/m_mo.
    # --------------------------------
    from prince_cr.data import _TABULATED_DECAY_DX, _TABULATED_DECAY_XBINS
    if result is not None:
        pass
    elif (mo, da) in _TABULATED_DECAY_DX and _TABULATED_DECAY_XBINS is not None:
        result = _tabulated_decay_avg(
            _TABULATED_DECAY_DX[(mo, da)], _TABULATED_DECAY_XBINS,
            x_lower, x_upper,
        )

    # --------------------------------
    # pi+ → nu_mu, pi- → nu_mubar
    # --------------------------------
    elif mo == _PDG_PI_PLUS and da == _PDG_NU_MU:
        result = pion_to_numu_avg(x_lower, x_upper)
    elif mo == _PDG_PI_MINUS and da == _PDG_NU_MU_BAR:
        result = pion_to_numu_avg(x_lower, x_upper)

    # --------------------------------
    # pi± → μ± (helicity-mixed, h=0)
    # --------------------------------
    elif mo == _PDG_PI_PLUS and da == _PDG_MU_PLUS:
        result = pion_to_muon_avg(x_lower, x_upper)
    elif mo == _PDG_PI_MINUS and da == _PDG_MU_MINUS:
        result = pion_to_muon_avg(x_lower, x_upper)

    # --------------------------------
    # μ → ν (h=0, see comment above)
    # --------------------------------
    elif mo == _PDG_MU_PLUS and da == _PDG_NU_E:
        result = muonplus_to_nue(x_grid, 0.0)
    elif mo == _PDG_MU_PLUS and da == _PDG_NU_MU_BAR:
        result = muonplus_to_numubar(x_grid, 0.0)
    elif mo == _PDG_MU_MINUS and da == _PDG_NU_E_BAR:
        result = muonplus_to_nue(x_grid, 0.0)
    elif mo == _PDG_MU_MINUS and da == _PDG_NU_MU:
        result = muonplus_to_numubar(x_grid, 0.0)

    # --------------------------------
    # neutrinos from beta decays
    # --------------------------------
    # TODO: The following beta decay to neutrino distr need to be averaged analyticaly
    # TODO: Also the angular averaging is done numerically still
    elif is_nucleus(mo) and da == _PDG_NU_E:
        daughter = _nucleus_charge_step(mo, -1)
        info(10, "nu_e from β-style decay", mo, daughter, da)
        result = nu_from_beta_decay(x_grid, mo, daughter)
    elif is_nucleus(mo) and da == _PDG_NU_E_BAR:
        daughter = _nucleus_charge_step(mo, +1)
        info(10, "nubar_e from β-style decay", mo, daughter, da)
        result = nu_from_beta_decay(x_grid, mo, daughter)
    elif is_nucleus(mo) and is_nucleus(da):
        info(10, "beta decay boost conservation", mo, da)
        result = boost_conservation_avg(x_lower, x_upper)
    else:
        info(
            5,
            "Called with unknown channel {:} to {:}, returning an empty redistribution".format(
                mo, da
            ),
        )
        # no known channel, return zeros
        result = np.zeros(x_grid.shape)

    # now fill this into diagonals of matrix
    if len(shape) == 2:
        #'filling matrix'
        res_mat = np.zeros(shape)
        for idx, val in enumerate(result[::-1]):
            np.fill_diagonal(res_mat[:, idx:], val)
        result = res_mat

    return result


#: PER-PHOTON density of the π⁰ → γγ channel box (dN/dx = 1 on [0,1]).
#: CRITICAL: the chain reducer's ``new_dec_diff_tab`` is an AdditiveDictionary
#: (util.py), so each of the TWO daughters in the (111, [22,22]) branching loop
#: (_recurse_into_daughters) ACCUMULATES — the box is applied twice. The kernel
#: must therefore be PER-PHOTON (∫dN/dx=1, ∫x dN/dx=½); the two accumulated
#: applications give 2 photons total carrying ∫x = 1 → E_π0 (energy-conserving).
#: An earlier value of 2.0 (mistaking the AdditiveDictionary for overwrite)
#: DOUBLED the π⁰→γγ photon energy (γ_reduced = 2×π⁰_raw), breaking c·J energy
#: conservation (Σx 1.15→0.94) and inflating the in-source γ ~2× vs AM3. See
#: lesson pi0-gamma-box-double-count. (Verified it87: γ→π⁰ energy at mult=1.)
_PI0_GAMMA_MULTIPLICITY = 1.0


def pi0_to_gamma(x):
    """Lab-frame PER-PHOTON energy distribution of one γ from π⁰ → γγ.

    Two-body decay into massless daughters: in the π⁰ rest frame each photon
    carries E* = m_π0/2; boosting an ultra-relativistic π⁰ (β→1, as for the
    photo-meson secondaries) spreads each lab photon uniformly over
    x = E_γ/E_π0 ∈ [0, 1] (the r→0 limit of `pion_to_numu`). Per photon the
    density is dN/dx = 1 on [0,1] (∫x dN/dx = ½). The reducer's
    AdditiveDictionary accumulates the two [22,22] daughters → 2 photons,
    ∫x = 1 → E_π0 total (energy-conserving). Replaces the tabulated rest-frame
    spike at x=½, which used as a lab redistribution pins every γ at E_π0/2 and
    loses the low-E box tail.
    """
    res = np.zeros(x.shape)
    res[np.logical_and(0.0 < x, x <= 1.0)] = _PI0_GAMMA_MULTIPLICITY
    return res


def pi0_to_gamma_avg(x_lower, x_upper):
    """Bin-averaged `pi0_to_gamma`: box on [0,1] with per-photon density 1."""
    if x_lower.shape != x_upper.shape:
        raise Exception("different grids for xmin, xmax provided")

    bins_width = x_upper - x_lower
    res = np.zeros(x_lower.shape)
    xmin, xmax = 0.0, 1.0

    # upper bin edge not contained (x_lower < 1 < x_upper)
    cond = np.where(np.logical_and(x_lower < xmax, x_upper > xmax))
    res[cond] = _PI0_GAMMA_MULTIPLICITY * (xmax - x_lower[cond]) / bins_width[cond]

    # bins fully contained in [0, 1]
    cond = np.where(np.logical_and(xmin <= x_lower, x_upper <= xmax))
    res[cond] = _PI0_GAMMA_MULTIPLICITY

    return res


def pion_to_numu(x):
    """
    Energy distribution of a numu from the decay of pi

    Args:
      x (float): energy fraction transferred to the secondary
    Returns:
      float: probability density at x
    """
    res = np.zeros(x.shape)

    m_muon = spec_data[_PDG_MU_PLUS]["mass"]
    m_pion = spec_data[_PDG_PI_PLUS]["mass"]
    r = m_muon**2 / m_pion**2
    xmin = 0.0
    xmax = 1 - r

    cond = np.where(np.logical_and(xmin < x, x <= xmax))
    res[cond] = 1 / (1 - r)
    return res


def pion_to_numu_avg(x_lower, x_upper):
    """
    Energy distribution of a numu from the decay of pi

    Args:
      x_lower,x_lower (float): energy fraction transferred to the secondary, lower/upper bin edge
    Returns:
      float: average probability density in bins (xmin,xmax)
    """
    if x_lower.shape != x_upper.shape:
        raise Exception("different grids for xmin, xmax provided")

    bins_width = x_upper - x_lower
    res = np.zeros(x_lower.shape)

    m_muon = spec_data[_PDG_MU_PLUS]["mass"]
    m_pion = spec_data[_PDG_PI_PLUS]["mass"]
    r = m_muon**2 / m_pion**2
    xmin = 0.0
    xmax = 1 - r

    # lower bin edged not contained
    cond = np.where(np.logical_and(xmin > x_lower, xmin < x_upper))
    res[cond] = 1 / (1 - r) * (x_upper[cond] - xmin) / bins_width[cond]

    # upper bin edge not contained
    cond = np.where(np.logical_and(x_lower < xmax, x_upper > xmax))
    res[cond] = 1 / (1 - r) * (xmax - x_lower[cond]) / bins_width[cond]

    # bins fully contained
    cond = np.where(np.logical_and(xmin <= x_lower, x_upper <= xmax))
    res[cond] = 1 / (1 - r)

    return res


def pion_to_muon(x):
    """
    Energy distribution of a muon from the decay of pi

    Args:
      x (float): energy fraction transferred to the secondary
    Returns:
      float: probability density at x
    """
    res = np.zeros(x.shape)

    m_muon = spec_data[_PDG_MU_PLUS]["mass"]
    m_pion = spec_data[_PDG_PI_PLUS]["mass"]
    r = m_muon**2 / m_pion**2
    xmin = r
    xmax = 1.0

    cond = np.where(np.logical_and(xmin < x, x <= xmax))
    res[cond] = 1 / (1 - r)
    return res


def pion_to_muon_avg(x_lower, x_upper):
    """
    Energy distribution of a numu from the decay of pi

    Args:
      x_lower,x_lower (float): energy fraction transferred to the secondary, lower/upper bin edge
    Returns:
      float: average probability density in bins (xmin,xmax)
    """
    if x_lower.shape != x_upper.shape:
        raise Exception("different grids for xmin, xmax provided")

    bins_width = x_upper - x_lower
    res = np.zeros(x_lower.shape)

    m_muon = spec_data[_PDG_MU_PLUS]["mass"]
    m_pion = spec_data[_PDG_PI_PLUS]["mass"]
    r = m_muon**2 / m_pion**2
    xmin = r
    xmax = 1.0

    # lower bin edged not contained
    cond = np.where(np.logical_and(xmin > x_lower, xmin < x_upper))
    res[cond] = 1 / (1 - r) * (x_upper[cond] - xmin) / bins_width[cond]

    # upper bin edge not contained
    cond = np.where(np.logical_and(x_lower < xmax, x_upper > xmax))
    res[cond] = 1 / (1 - r) * (xmax - x_lower[cond]) / bins_width[cond]

    # bins fully contained
    cond = np.where(np.logical_and(xmin <= x_lower, x_upper <= xmax))
    res[cond] = 1 / (1 - r)

    return res


def prob_muon_hel(x, h):
    """
    Probability for muon+ from pion+ decay to have helicity h
    the result is only valid for x > r

    Args:
      h (int): helicity +/- 1
    Returns:
      float: probability for this helicity
    """

    m_muon = spec_data[_PDG_MU_PLUS]["mass"]
    m_pion = spec_data[_PDG_PI_PLUS]["mass"]

    r = m_muon**2 / m_pion**2

    # helicity expectation value
    hel = 2 * r / (1 - r) / x - (1 + r) / (1 - r)

    res = np.zeros(x.shape)
    cond = np.where(np.logical_and(x > r, x <= 1))
    res[cond] = (1 + hel * h) / 2  # this result is only correct for x > r
    return res


def muonplus_to_numubar(x, h):
    """
    Energy distribution of a numu_bar from the decay of muon+ with helicity h
    (For muon- decay calc with h = -h due to CP invariance)

    Args:
      x (float): energy fraction transferred to the secondary
      h (float): helicity of the muon

    Returns:
      float: probability density at x
    """
    p1 = np.poly1d([4.0 / 3.0, -3.0, 0.0, 5.0 / 3.0])
    p2 = np.poly1d([-8.0 / 3.0, 3.0, 0.0, -1.0 / 3.0])

    res = np.zeros(x.shape)
    cond = x <= 1.0
    res[cond] = p1(x[cond]) + h * p2(x[cond])
    return res


def muonplus_to_nue(x, h):
    """
    Energy distribution of a n from the decay of muon+ with helicity h
    (For muon- decay calc with h = -h due to CP invariance)

    Args:
      x (float): energy fraction transferred to the secondary
      h (float): helicity of the muon

    Returns:
      float: probability density at x
    """
    p1 = np.poly1d([4.0, -6.0, 0.0, 2.0])
    p2 = np.poly1d([-8.0, 18.0, -12.0, 2.0])

    res = np.zeros(x.shape)
    cond = x <= 1.0
    res[cond] = p1(x[cond]) + h * p2(x[cond])
    return res


def boost_conservation(x):
    """Returns an x=1 distribution for ejected nucleons"""
    dist = np.zeros_like(x)
    # dist[(x == np.max(x)) & (x > 0.9)] = 1.*20.
    dist[x == 1.0] = 1.0 / 0.115
    return dist


def boost_conservation_avg(x_lower, x_upper):
    """Returns an x=1 distribution for ejected nucleons"""
    dist = np.zeros_like(x_lower)

    # boost conservation is a delta peak at x = 1
    # if is is contained in the bin, the the value
    # to 1 / width, else set it to zero
    cond = np.where(np.logical_and(x_lower < 1.0, x_upper > 1.0))
    bins_width = x_upper[cond] - x_lower[cond]
    dist[cond] = 1.0 / bins_width
    return dist


def nu_from_beta_decay(x_grid, mother, daughter, Gamma=200, angle=None):
    """
    Energy distribution of a neutrinos from beta-decays of mother to daughter
    The res frame distrution is boosted to the observers frame and then angular averaging is done numerically

    Args:
      x_grid (float): energy fraction transferred to the secondary
      mother (int): id of mother
      daughter (int): id of daughter
      Gamma (float): Lorentz factor of the parent particle, default: 200
                     For large Gamma this should not play a role, as the decay is scale invariant
      angle (float): collision angle, if None this will be averaged over 2 pi
    Returns:
      float: probability density on x_grid
    """
    import warnings

    info(10, "Calculating neutrino energy from beta decay", mother, daughter)

    mass_el = spec_data[11]["mass"]
    mass_mo = spec_data[mother]["mass"]
    mass_da = spec_data[daughter]["mass"]

    Z_mo = spec_data[mother]["charge"]
    Z_da = spec_data[daughter]["charge"]

    A_mo, _, _ = get_AZN(mother)

    if mother == 2112 and daughter == 2212:
        # n → p + e^- + ν_e^bar (free neutron β-decay): the masses are
        # already nucleon masses, so the q-value is mn - mp - me.
        qval = mass_mo - mass_da - mass_el
    elif Z_da == Z_mo - 1:  # beta+ decay
        qval = mass_mo - mass_da - 2 * mass_el
    elif Z_da == Z_mo + 1:  # beta- decay
        qval = mass_mo - mass_da
    else:
        raise Exception(
            "Not an allowed beta decay channel: {:} -> {:}".format(mother, daughter)
        )

    # substitute this to the energy grid
    E0 = qval + mass_el
    # NOTE: we subsitute into energy per nucleon here
    Emo = Gamma * mass_mo / A_mo
    E = x_grid * Emo

    # print '------','beta decay','------'
    # print mother
    # print E0
    # print A_mo
    # print Emo

    if angle is None:
        # ctheta = np.linspace(-1, 1, 1000)
        # we use here logspace, as high resolution is mainly needed at small energies
        # otherwise the solution will oscillate at low energy
        ctheta = np.unique(
            np.concatenate(
                (
                    np.logspace(-8, 0, 1000) - 1,
                    1 - np.logspace(0, -8, 1000),
                )
            )
        )
    else:
        ctheta = angle

    boost = Gamma * (1 - ctheta)
    Emax = E0 * boost

    E_mesh, boost_mesh = np.meshgrid(E, boost, indexing="ij")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = (
            E_mesh**2
            / boost_mesh**5
            * (Emax - E_mesh)
            * np.sqrt((E_mesh - Emax) ** 2 - boost_mesh**2 * mass_el**2)
        )
    res[E_mesh > Emax] = 0.0
    res = np.nan_to_num(res)

    if np.all(res == 0):
        info(
            10,
            "Differential distribution is all zeros for",
            mother,
            daughter,
            "No angle averaging performed!",
        )
    elif angle is None:
        # now average over angle
        res = trapz(res, x=ctheta, axis=1)
        res = res / trapz(res, x=x_grid)
    else:
        res = res[:, 0]
        res = res / trapz(res, x=x_grid)

    return res

