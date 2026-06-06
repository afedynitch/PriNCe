"""EM interaction rates for the PRINCE transport solver (active γ/e± species).

Parallel rate path to ``PhotoNuclearInteractionRate``: builds a sparse Jacobian
(units 1/cm, same convention as the nuclear path — see
``interaction_rates.single_interaction_length``) carrying the electromagnetic
couplings, summed into M(z) by the solver's ``_refresh_z_caches``. Because the
solver rebuilds rates per z-step (evaluating the photon field at z), the EM
cascade evolves self-consistently in redshift with NO fixed-z / (1+z)
kernel-frame approximation — see methods/em-cascade-in-transport.

Milestone status:
  M1 (here): γ diagonal sink  -dτ_gg/dl  (γγ pair-production absorption).
  M2 (TODO): γ→e± pair injection + e±→γ IC off-diagonals.
"""

import numpy as np
import scipy.sparse as sp

from prince_cr.cascade.opacity import _kernel_per_length
from prince_cr.util import info


class EMInteractionRate(object):
    """Electromagnetic Jacobian (1/cm) for active EM species in transport.

    Mirrors the public contract of ``PhotoNuclearInteractionRate``:
    ``get_hadr_jacobian(z, scale_fac, force_update, pfield) -> csr_matrix`` of
    shape ``(dim_states, dim_states)``, refreshed in place per z.

    Args:
        prince_run: the :class:`~prince_cr.core.PriNCeRun` handle.
        include_pair, include_ic: enable the M2 off-diagonal channels
            (not yet implemented; reserved).
    """

    def __init__(
        self,
        prince_run,
        include_pair=False,
        include_ic=False,
        eps_min=1e-15,
        eps_max=1e-7,
        n_eps=256,
        n_mu=128,
    ):
        self.prince_run = prince_run
        self.spec_man = prince_run.spec_man
        self.egrid = prince_run.cr_grid.grid  # GeV; γ has A=1 → grid == E_γ
        self.dim_states = prince_run.dim_states
        self.include_pair = include_pair
        self.include_ic = include_ic
        if include_pair or include_ic:
            raise NotImplementedError(
                "EM pair/IC off-diagonals are M2 — not yet implemented."
            )
        self.eps = np.logspace(np.log10(eps_min), np.log10(eps_max), n_eps)
        self.mu = np.linspace(-1.0, 1.0, n_mu)
        self._zcache = None
        self._build_sparsity()

    @property
    def photon_field(self):
        return self.prince_run.photon_field

    def _build_sparsity(self):
        """γ-block diagonal sparsity pattern (M1)."""
        if 22 not in self.spec_man.pdgid2sref:
            raise ValueError("Photon (PDG 22) not in the species set.")
        g = self.spec_man.pdgid2sref[22]
        rows = np.arange(g.sl.start, g.sl.stop)
        assert rows.size == self.egrid.size
        self._gamma_rows = rows
        data = np.zeros(rows.size, dtype=np.float64)
        self.coupling_mat = sp.csr_matrix(
            (data, (rows, rows)), shape=(self.dim_states, self.dim_states)
        )
        self.coupling_mat.sort_indices()
        info(1, "EMInteractionRate: γ sink on {0} bins".format(rows.size))

    def _gamma_sink_per_length(self, z):
        """-dτ_gg/dl per γ energy bin [1/cm] (the absorption rate)."""
        field = self.photon_field
        rate = np.empty(self.egrid.size, dtype=np.float64)
        for i, E in enumerate(self.egrid):
            rate[i] = _kernel_per_length(E, z, field, self.eps, self.mu)
        return rate  # >= 0

    def get_hadr_jacobian(self, z, scale_fac=1.0, force_update=False, pfield=None):
        """Sparse EM Jacobian (1/cm) at redshift z. Diagonal γ loss only (M1)."""
        if force_update or self._zcache != z:
            sink = self._gamma_sink_per_length(z)  # 1/cm, >= 0
            # CSR data is in (sorted row) order == γ grid order == sink order.
            self.coupling_mat.data[:] = -sink  # diagonal loss term
            self._zcache = z
        if scale_fac != 1.0:
            out = self.coupling_mat.copy()
            out.data *= scale_fac
            return out
        return self.coupling_mat
