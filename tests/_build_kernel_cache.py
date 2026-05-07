"""Build a cached PriNCeRun pickle for solver benchmarking.

Run this once to produce the kernel cache referenced by tests/conftest.py
(via the PRINCE_KERNEL_CACHE env var).

Usage:
    python tests/_build_kernel_cache.py [output_path] [--max-mass N] [--db-path DIR] [--db-fname NAME]

Defaults:
    output_path: ../PriNCe-examples/prince_run_talys.ppo (relative to repo root)
    max_mass:    56  (production cap; ~3 min build, multi-GB pickle)
    db_path:     $PRINCE_FLUKA_DB or ~/work/devel/UH-UHECR-Fluka-Prince/runs/2026-05-04_pfu-v1-prod
    db_fname:    prince_db_v1.h5

For the test fixture (max_mass=16) use:
    python tests/_build_kernel_cache.py tests/data/prince_run_m16.ppo --max-mass 16
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path

import numpy as np

import prince_cr.config as cfg
from prince_cr import core, cross_sections, photonfields


def main(out_path: Path, max_mass: int, db_path: Path, db_fname: str) -> None:
    cfg.x_cut = 1e-4
    cfg.x_cut_proton = 1e-2
    cfg.tau_dec_threshold = np.inf

    print("Building photon field (CMB + CIBGilmore2D)...")
    pf = photonfields.CombinedPhotonField(
        [photonfields.CMBPhotonSpectrum, photonfields.CIBGilmore2D]
    )

    print(f"Building cross sections (FLUKA, db={db_path}/{db_fname})...")
    cfg.fluka_db_path = str(db_path)
    cfg.fluka_db_fname = db_fname
    cs = cross_sections.FlukaPhotoNuclear()

    t0 = time.time()
    print(f"Building PriNCeRun(max_mass={max_mass})...")
    run = core.PriNCeRun(max_mass=max_mass, photon_field=pf, cross_sections=cs)
    print(f"  ... done in {time.time() - t0:.1f}s")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Pickling to {out_path} ...")
    t0 = time.time()
    with open(out_path, "wb") as fh:
        pickle.dump(run, fh, protocol=-1)
    sz = out_path.stat().st_size
    if sz >= 1e9:
        print(f"  ... done in {time.time() - t0:.1f}s ({sz / 1e9:.2f} GB)")
    else:
        print(f"  ... done in {time.time() - t0:.1f}s ({sz / 1e6:.2f} MB)")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    default_out = repo_root.parent.parent / "PriNCe-examples" / "prince_run_talys.ppo"
    default_db_path = os.environ.get(
        "PRINCE_FLUKA_DB",
        os.path.expanduser(
            "~/work/devel/UH-UHECR-Fluka-Prince/runs/2026-05-04_pfu-v1-prod"
        ),
    )

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("output_path", nargs="?", default=str(default_out))
    p.add_argument("--max-mass", type=int, default=56)
    p.add_argument("--db-path", default=default_db_path)
    p.add_argument("--db-fname", default="prince_db_v1.h5")
    args = p.parse_args()

    main(
        out_path=Path(args.output_path),
        max_mass=args.max_mass,
        db_path=Path(args.db_path),
        db_fname=args.db_fname,
    )
