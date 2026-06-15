"""Source-frame photon fields.

PriNCe's main `prince_cr.photonfields` module hosts cosmological photon
backgrounds (CMB + EBL). The classes here implement *in-source* photon
fields — the radiation environment inside a single dissipation zone of
an astrophysical source. They subclass the same `PhotonField` ABC, so
any call site that takes a `pfield=` override accepts them without
changes; the only convention shift is that the energy axis is the fluid
rest frame and the `z` argument is ignored (a source zone has no
cosmological redshift evolution of its own — the source is taken at a
fixed epoch, and its emission is later transported through the EBL by
the propagation half of PriNCe).

Units throughout match the rest of `prince_cr`:

- photon energy in GeV (fluid rest frame),
- photon number density in :math:`\\mathrm{GeV}^{-1}\\,\\mathrm{cm}^{-3}`,
- `z` accepted for interface compatibility but unused.
"""

import numpy as np
from scipy.integrate import quad

from prince_cr.data import PRINCE_UNITS
from prince_cr.photonfields import PhotonField


class SourceBrokenPowerLaw(PhotonField):
    """Broken power-law photon spectrum in a relativistic source zone.

    Implements Eq. 1 of Guo, Qian, Wu 2025 (Phys. Rev. D 112, 063022):

    .. math::

       n(E_\\gamma) = n_0 \\times
       \\begin{cases}
         (E_\\gamma / E_{\\gamma,b})^{-\\alpha}, & E_\\gamma \\le E_{\\gamma,b} \\\\
         (E_\\gamma / E_{\\gamma,b})^{-\\beta},  & E_\\gamma >    E_{\\gamma,b}
       \\end{cases}

    with both :math:`E_\\gamma` and :math:`E_{\\gamma,b}` in the *fluid rest
    frame*. The normalisation :math:`n_0` is fixed so that the photon
    energy density in the fluid frame equals
    :math:`U_\\gamma = L_{\\gamma,\\mathrm{iso}} / (4\\pi \\Gamma^2 R^2 c)`,
    i.e. the observer-frame isotropic luminosity boosted into the
    co-moving zone.

    Parameters
    ----------
    L_gamma_iso_erg_s : float
        Observer-frame isotropic-equivalent photon luminosity integrated
        over the allowed energy band (erg/s). Guo+ adopt
        :math:`10^{52}` for the prompt-radiation (PR) phase,
        :math:`10^{49}` for the extended-emission (EE) phase, and
        :math:`10^{47}` for the plateau-emission (PE) phase.
    E_gamma_b_ob_eV : float
        Observer-frame spectral-break energy (eV). Converted to the
        fluid frame internally via :math:`E_{\\gamma,b} =
        E_{\\gamma,b}^{\\rm ob}/\\Gamma`.
    alpha : float
        Low-energy power-law index (Guo+ adopt 0.5).
    beta : float
        High-energy power-law index (Guo+ adopt 2.0).
    Gamma : float
        Bulk Lorentz factor of the dissipation zone.
    R_cm : float
        Dissipation radius (cm).
    E_gamma_min_eV, E_gamma_max_eV : float, optional
        Fluid-frame minimal/maximal photon energies (eV). Defaults
        match Guo+: 0.1 eV and 1 MeV.

    Notes
    -----
    The frame convention diverges from PriNCe's CMB/EBL fields:
    `get_photon_density(E, z)` interprets ``E`` as a *fluid-rest-frame*
    energy in GeV and ignores ``z``. Callers using this field for
    rate calculations should likewise pass fluid-frame CR energies.
    """

    frame = "fluid-rest"

    def __init__(
        self,
        L_gamma_iso_erg_s,
        E_gamma_b_ob_eV,
        alpha,
        beta,
        Gamma,
        R_cm,
        E_gamma_min_eV=0.1,
        E_gamma_max_eV=1.0e6,
    ):
        self.L_gamma_iso_erg_s = float(L_gamma_iso_erg_s)
        self.E_gamma_b_ob_eV = float(E_gamma_b_ob_eV)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.Gamma = float(Gamma)
        self.R_cm = float(R_cm)

        # Fluid-frame energy bounds in GeV.
        self.E_min_GeV = float(E_gamma_min_eV) * 1.0e-9
        self.E_max_GeV = float(E_gamma_max_eV) * 1.0e-9
        # Fluid-frame break: observer-frame value divided by Gamma.
        self.E_b_GeV = self.E_gamma_b_ob_eV * 1.0e-9 / self.Gamma

        # Fluid-frame photon energy density [GeV / cm^3].
        # U_gamma = L_iso / (4 pi Gamma^2 R^2 c)
        U_gamma_erg_cm3 = self.L_gamma_iso_erg_s / (
            4.0 * np.pi * self.Gamma**2 * self.R_cm**2 * PRINCE_UNITS.c
        )
        self.U_gamma_GeV_cm3 = U_gamma_erg_cm3 * PRINCE_UNITS.erg2GeV

        # Normalisation: n_0 such that ∫ E n(E) dE = U_gamma.
        self.n0 = self.U_gamma_GeV_cm3 / self._unit_energy_integral()

    # -- internals -----------------------------------------------------

    def _unit_shape(self, E_GeV):
        """Unnormalised n(E)/n0 on the fluid-frame energy grid."""
        E = np.asarray(E_GeV, dtype=float)
        out = np.zeros_like(E)
        in_range = (E >= self.E_min_GeV) & (E <= self.E_max_GeV)
        low = in_range & (E <= self.E_b_GeV)
        high = in_range & (E > self.E_b_GeV)
        # Avoid divide-by-zero for off-grid evaluations.
        eb = self.E_b_GeV
        with np.errstate(divide="ignore", invalid="ignore"):
            out[low] = (E[low] / eb) ** (-self.alpha)
            out[high] = (E[high] / eb) ** (-self.beta)
        return out

    def _unit_energy_integral(self):
        """∫ E (n/n0) dE on [E_min, E_max], analytic per-segment."""

        def seg(E_lo, E_hi, gamma):
            # ∫ E * (E/E_b)^{-gamma} dE = E_b^gamma * ∫ E^{1-gamma} dE
            exponent = 2.0 - gamma
            eb = self.E_b_GeV
            if abs(exponent) < 1e-12:
                return (eb**gamma) * np.log(E_hi / E_lo)
            return (
                (eb**gamma)
                * (E_hi**exponent - E_lo**exponent)
                / exponent
            )

        # Low-energy segment [E_min, min(E_b, E_max)] with index alpha.
        E_lo = self.E_min_GeV
        E_hi_low = min(self.E_b_GeV, self.E_max_GeV)
        I_low = seg(E_lo, E_hi_low, self.alpha) if E_hi_low > E_lo else 0.0

        # High-energy segment [max(E_b, E_min), E_max] with index beta.
        E_lo_high = max(self.E_b_GeV, self.E_min_GeV)
        E_hi_high = self.E_max_GeV
        I_high = seg(E_lo_high, E_hi_high, self.beta) if E_hi_high > E_lo_high else 0.0

        return I_low + I_high

    # -- API -----------------------------------------------------------

    def get_photon_density(self, E, z=0.0):
        """Fluid-frame photon number density in GeV^-1 cm^-3.

        Parameters
        ----------
        E : array_like
            Photon energy in GeV (fluid rest frame).
        z : float, optional
            Ignored — kept for interface parity with cosmological
            `PhotonField` subclasses.
        """
        return self.n0 * self._unit_shape(np.atleast_1d(E))

    def __repr__(self):
        return (
            f"SourceBrokenPowerLaw(L_iso={self.L_gamma_iso_erg_s:.2e} erg/s, "
            f"E_b_ob={self.E_gamma_b_ob_eV:.2e} eV, "
            f"alpha={self.alpha}, beta={self.beta}, "
            f"Gamma={self.Gamma}, R={self.R_cm:.2e} cm)"
        )
