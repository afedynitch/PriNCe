"""Production-grid (max_mass=56) MKL vs scipy bench.

Single PriNCeRun build (~3 min on Zen 2) shared across all backend
runs. Reports wall, parity vs scipy, speedup.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

import prince_cr.config as cfg

cfg.fluka_db_path = "/home/anatoli/work/devel/UH-UHECR-Fluka-Prince/runs/2026-05-04_pfu-v1-prod"
cfg.fluka_db_fname = "prince_db_v1.h5"
cfg.cosmic_ray_grid = (3, 14, 8)
cfg.photon_grid = (-15, -6, 8)
cfg.x_cut = 1e-4
cfg.x_cut_proton = 1e-2
cfg.tau_dec_threshold = float("inf")
cfg.max_mass = 56
cfg.debug_level = 0

from prince_cr import core, cross_sections, photonfields
from prince_cr.solvers import UHECRPropagationSolverETD2
from prince_cr.cr_sources import AugerFitSource

PARAMS = {
    2212: (0.96, 10**9.68, 20.0),
    1000020040: (0.96, 10**9.68, 50.0),
    1000070140: (0.96, 10**9.68, 30.0),
    1000260560: (0.96, 10**9.68, 10.0),
}


def run_one(prince_run, backend, threads=None):
    cfg.linear_algebra_backend = backend
    if backend == "MKL":
        cfg.set_mkl_threads(threads)
    s = UHECRPropagationSolverETD2(
        initial_z=1.0, final_z=0.0, prince_run=prince_run,
        enable_pairprod_losses=True, enable_adiabatic_losses=True,
        enable_injection_jacobian=True, enable_partial_diff_jacobian=True,
    )
    s.add_source_class(AugerFitSource(prince_run, norm=1e-50, params=PARAMS))
    t0 = time.perf_counter()
    s.solve(dz=1e-3)
    elapsed = time.perf_counter() - t0
    state = s.state.copy()
    s.close()
    return state, elapsed


def main():
    print("Building PriNCeRun(max_mass=56)…", flush=True)
    t0 = time.perf_counter()
    pf = photonfields.CombinedPhotonField(
        [photonfields.CMBPhotonSpectrum, photonfields.CIBGilmore2D]
    )
    cs = cross_sections.FlukaPhotoNuclear()
    pr = core.PriNCeRun(max_mass=56, photon_field=pf, cross_sections=cs)
    print(f"  built in {time.perf_counter()-t0:.1f}s, dim_states={pr.dim_states}",
          flush=True)

    # warmup scipy then time
    print("\nscipy reference (warmup + timed)…", flush=True)
    _ = run_one(pr, "scipy")
    ref, t_sci = run_one(pr, "scipy")
    print(f"scipy: {t_sci:.3f}s norm={np.linalg.norm(ref):.4e}", flush=True)

    for nt in (1, 4, 8, 16, 24, 32):
        s, t = run_one(pr, "MKL", threads=nt)
        L2 = np.linalg.norm(s - ref) / np.linalg.norm(ref)
        print(f"MKL CSR t={nt:3d}: {t:.3f}s ({t_sci/t:5.2f}x vs scipy)  L2={L2:.2e}",
              flush=True)


if __name__ == "__main__":
    main()
