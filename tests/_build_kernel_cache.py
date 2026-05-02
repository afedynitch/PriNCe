"""Build a cached PriNCeRun pickle for solver benchmarking.

Run this once to produce the kernel cache referenced by tests/conftest.py
(via the PRINCE_KERNEL_CACHE env var). Takes ~3 minutes; pickle is multi-GB.

Usage:
    python tests/_build_kernel_cache.py [output_path]

Default output path: ../PriNCe-examples/prince_run_talys.ppo (relative to repo root).
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np

import prince_cr.config as cfg
from prince_cr import core, cross_sections, photonfields


def main(out_path: Path) -> None:
    cfg.x_cut = 1e-4
    cfg.x_cut_proton = 1e-2
    cfg.tau_dec_threshold = np.inf

    print("Building photon field (CMB + CIBGilmore2D)...")
    pf = photonfields.CombinedPhotonField(
        [photonfields.CMBPhotonSpectrum, photonfields.CIBGilmore2D]
    )

    print("Building cross sections (Talys + Sophia superposition)...")
    cs = cross_sections.CompositeCrossSection(
        [
            (0.0, cross_sections.TabulatedCrossSection, ("CRP2_TALYS",)),
            (0.14, cross_sections.SophiaSuperposition, ()),
        ]
    )

    t0 = time.time()
    print("Building PriNCeRun(max_mass=56)...")
    run = core.PriNCeRun(max_mass=56, photon_field=pf, cross_sections=cs)
    print(f"  ... done in {time.time() - t0:.1f}s")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Pickling to {out_path} ...")
    t0 = time.time()
    with open(out_path, "wb") as fh:
        pickle.dump(run, fh, protocol=-1)
    print(f"  ... done in {time.time() - t0:.1f}s ({out_path.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    default_out = repo_root.parent.parent / "PriNCe-examples" / "prince_run_talys.ppo"
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else default_out
    main(out)
