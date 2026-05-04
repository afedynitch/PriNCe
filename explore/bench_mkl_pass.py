"""Stage 1 MKL-pass smoke + benchmark.

Skirts conftest.py's hard-coded `/ceph/sharedfs/...` FLUKA db path (broken
on this host) by setting ``config.fluka_db_path`` from a local default
that resolves on SATORI's `/home/anatoli/work/devel/...` view. Runs the
same ETD2 solve under each backend in a single Python process.

Usage:
    python explore/bench_mkl_pass.py [--max-mass=14] [--threads=8,16,32]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


def _setup_paths_and_config(max_mass: int, fluka_db_dir: str, fluka_db_fname: str):
    # Pull config in first so we can mutate before any heavy import.
    import prince_cr.config as config

    config.debug_level = 1
    # Production grid (8 bins/decade) trimmed by max_mass.
    config.cosmic_ray_grid = (3, 14, 8)
    config.photon_grid = (-15, -6, 8)
    config.x_cut = 1e-4
    config.x_cut_proton = 1e-2
    config.tau_dec_threshold = np.inf
    config.max_mass = max_mass
    config.fluka_db_path = fluka_db_dir
    config.fluka_db_fname = fluka_db_fname
    return config


def _build_run(config):
    from prince_cr import core, cross_sections, photonfields

    pf = photonfields.CombinedPhotonField(
        [photonfields.CMBPhotonSpectrum, photonfields.CIBGilmore2D]
    )
    cs = cross_sections.FlukaPhotoNuclear()
    return core.PriNCeRun(max_mass=config.max_mass, photon_field=pf, cross_sections=cs)


def _solve_one(prince_run, backend: str, threads: int, dz: float, label: str):
    import prince_cr.config as config
    from prince_cr.cr_sources import AugerFitSource
    from prince_cr.solvers import UHECRPropagationSolverETD2

    config.linear_algebra_backend = backend
    if backend.lower() == "mkl" and config.has_mkl:
        config.set_mkl_threads(threads)

    solver = UHECRPropagationSolverETD2(
        initial_z=1.0,
        final_z=0.0,
        prince_run=prince_run,
        enable_pairprod_losses=True,
        enable_adiabatic_losses=True,
        enable_injection_jacobian=True,
        enable_partial_diff_jacobian=True,
    )
    solver.add_source_class(
        AugerFitSource(
            prince_run,
            norm=1e-50,
            params={
                2212: (0.96, 10**9.68, 20.0),
                1000020040: (0.96, 10**9.68, 50.0),
                1000070140: (0.96, 10**9.68, 30.0),
            },
        )
    )

    t0 = time.perf_counter()
    solver.solve(dz=dz, verbose=False, summary=False, progressbar=False)
    elapsed = time.perf_counter() - t0
    state = solver.state.copy()
    try:
        solver.close()
    except Exception:
        pass
    print(f"[{label}] backend={backend} threads={threads} wall={elapsed:.3f}s "
          f"state-norm={np.linalg.norm(state):.6e}")
    return state, elapsed


def _compare(ref: np.ndarray, got: np.ndarray, label: str):
    nz = ref != 0.0
    if not nz.any():
        print(f"[{label}] reference all-zero — nothing to compare")
        return
    rel = np.abs(got[nz] - ref[nz]) / np.abs(ref[nz])
    L2 = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    print(f"[{label}] L2 relative error vs scipy: {L2:.3e} | "
          f"per-bin rel max: {rel.max():.3e} | finite: {np.isfinite(got).all()}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-mass", type=int, default=14)
    ap.add_argument("--threads", default="8,16,32",
                    help="comma-separated MKL_NUM_THREADS sweep values")
    ap.add_argument("--dz", type=float, default=1e-3)
    ap.add_argument("--fluka-db-dir", default=os.environ.get(
        "PRINCE_FLUKA_DB_DIR",
        "/home/anatoli/work/devel/UH-UHECR-Fluka-Prince/runs/2026-05-04_pfu-v1-prod"))
    ap.add_argument("--fluka-db-fname", default=os.environ.get(
        "PRINCE_FLUKA_DB_FNAME", "prince_db_v1.h5"))
    args = ap.parse_args(argv)

    config = _setup_paths_and_config(args.max_mass, args.fluka_db_dir, args.fluka_db_fname)
    print(f"FLUKA db: {os.path.join(config.fluka_db_path, config.fluka_db_fname)}")
    print(f"has_mkl={config.has_mkl}  has_cupy={config.has_cupy}  "
          f"backend(default)={config.linear_algebra_backend}")

    print("\n[1/3] Building PriNCeRun (one-shot, shared across backends)…")
    t0 = time.perf_counter()
    prince_run = _build_run(config)
    print(f"      built in {time.perf_counter()-t0:.1f}s, "
          f"dim_states={prince_run.dim_states}")

    print("\n[2/3] scipy reference run…")
    ref_state, ref_t = _solve_one(prince_run, "scipy", 1, args.dz, "scipy")

    if not config.has_mkl:
        print("\nNo MKL available — skipping MKL passes.")
        return

    print("\n[3/3] MKL CSR thread sweep…")
    for nt in (int(t) for t in args.threads.split(",") if t.strip()):
        os.environ["MKL_NUM_THREADS"] = str(nt)
        # Default backend uses CSR (no mkl_bsr_blocksize set).
        if hasattr(config, "mkl_bsr_blocksize"):
            config.mkl_bsr_blocksize = None
        state, t = _solve_one(prince_run, "MKL", nt, args.dz, f"MKL CSR t={nt}")
        _compare(ref_state, state, f"MKL CSR t={nt}")
        print(f"      speedup vs scipy: {ref_t/t:.2f}x")

    print("\n[3b] MKL BSR(blocksize=6) at fastest CSR thread count…")
    config.mkl_bsr_blocksize = 6
    state, t = _solve_one(prince_run, "MKL", 16, args.dz, "MKL BSR(6) t=16")
    _compare(ref_state, state, "MKL BSR(6) t=16")
    print(f"      speedup vs scipy: {ref_t/t:.2f}x")
    config.mkl_bsr_blocksize = None


if __name__ == "__main__":
    main()
