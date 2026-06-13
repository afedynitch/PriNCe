"""Multi-RHS ETD2 solver: a column-k solve must equal a single-RHS solve of
source k, both CR-only and with the co-evolved EM cascade. Guards the K-axis
broadcasting in the cascade injection (the EM cascade transfer matmul and the
Bethe-Heitler e± injection must broadcast over the RHS axis)."""
import os

import numpy as np
import pytest

import prince_cr.config as config

_SMOKE = os.path.expanduser(
    "~/work/devel/UH-UHECR-Fluka-Prince/runs/2026-05-04_pfu-v1-smoke"
)
_BACKENDS = ["scipy"] + (["CUPY"] if getattr(config, "has_cupy", False) else [])


def _multi_vs_single(em, backend):
    config.fluka_db_path = _SMOKE
    config.fluka_db_fname = "prince_db_v1_smoke.h5"
    config.x_cut = 1e-4
    config.x_cut_proton = 1e-2
    config.tau_dec_threshold = np.inf
    config.em_grid_bins_dec = 16
    config.linear_algebra_backend = backend
    config.use_cuda_graphs = False

    from prince_cr import cross_sections as pxs
    from prince_cr import photonfields as pf
    from prince_cr.core import PriNCeRun
    from prince_cr.cr_sources import SimpleSource
    from prince_cr.solvers import (
        MultiRHSPropagationSolverETD2,
        UHECRPropagationSolverETD2,
    )

    field = pf.CombinedPhotonField([pf.CMBPhotonSpectrum, pf.CIBGilmore2D])
    kw = dict(max_mass=4, photon_field=field, cross_sections=pxs.FlukaPhotoNuclear())
    if em:
        kw.update(enable_em_cascade=True, enable_em_decoupled_grid=True)
    run = PriNCeRun(**kw)
    sm = run.spec_man
    E = run.cr_grid.grid
    psl = sm.pdgid2sref[2212].sl
    norms, gammas = [1.0, 2.0], [2.5, 2.3]
    zi, zf, dz = 0.2, 0.0, 1e-2

    def mk_src(nrm, gm):
        s = SimpleSource(run, norm=nrm, params={2212: (gm, 2e11, 1.0)})
        s.injection_grid[psl.start:psl.stop][E < 1e8] = 0.0
        return s

    singles = []
    for nrm, gm in zip(norms, gammas):
        s = UHECRPropagationSolverETD2(initial_z=zi, final_z=zf, prince_run=run)
        s.add_source_class(mk_src(nrm, gm))
        s.solve(dz=dz, progressbar=False)
        singles.append(np.asarray(s.state))

    m = MultiRHSPropagationSolverETD2(initial_z=zi, final_z=zf, prince_run=run, K=len(norms))
    for k, (nrm, gm) in enumerate(zip(norms, gammas)):
        m[k].add_source_class(mk_src(nrm, gm))
    m.solve(dz=dz, progressbar=False)
    Sm = np.asarray(m.state)
    assert Sm.shape == (run.dim_states, len(norms))

    worst = 0.0
    for k in range(len(norms)):
        a, b = singles[k], Sm[:, k]
        assert np.all(np.isfinite(b))
        mm = np.abs(a) > np.abs(a).max() * 1e-8
        worst = max(worst, np.abs(b[mm] / a[mm] - 1.0).max())
    return worst


@pytest.mark.skipif(not os.path.isdir(_SMOKE), reason="FLUKA smoke DB not available")
@pytest.mark.parametrize("backend", _BACKENDS)
def test_multi_rhs_cr_only(backend):
    assert _multi_vs_single(em=False, backend=backend) < 1e-10


@pytest.mark.skipif(not os.path.isdir(_SMOKE), reason="FLUKA smoke DB not available")
@pytest.mark.parametrize("backend", _BACKENDS)
def test_multi_rhs_em_cascade(backend):
    # co-evolved EM cascade column-k must equal the single-RHS solve of source k
    # (K-axis broadcasting in the cascade transfer + BH injection).
    assert _multi_vs_single(em=True, backend=backend) < 1e-6
