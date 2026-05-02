"""Profile PriNCeRun construction.

Measures wall-clock and memory for each stage of the cross-section/kernel pipeline:

  1. Cross-section table loading + reduction (`_optimize_and_generate_index`)
  2. Response-function spline construction (`_precompute_interpolators`)
  3. Batch-matrix construction (`PhotoNuclearInteractionRate._init_matrices`)
  4. Coupling matrix init (`_init_coupling_mat`)

Also reports:
  - number of species, bc channels, diff channels
  - batch_matrix shape and bytes
  - total interpolator footprint via pickle dump

Usage:
    python explore/profile_kernel_init.py --max-mass 14
    python explore/profile_kernel_init.py --max-mass 56
"""

from __future__ import annotations

import argparse
import gc
import io
import pickle
import resource
import sys
import time

import numpy as np


def rss_mb():
    """Current resident-set-size in MB (macOS reports bytes, Linux KB)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return r / 1024**2
    return r / 1024  # Linux: KB


def pickle_size(obj):
    buf = io.BytesIO()
    pickle.dump(obj, buf, protocol=-1)
    return buf.tell()


class Stage:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        gc.collect()
        self.t0 = time.perf_counter()
        self.r0 = rss_mb()
        return self

    def __exit__(self, *exc):
        self.dt = time.perf_counter() - self.t0
        self.dr = rss_mb() - self.r0
        print(f"[{self.label}] {self.dt:6.2f}s  RSS_delta={self.dr:+7.1f} MB  RSS={rss_mb():7.1f} MB")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-mass", type=int, default=14)
    ap.add_argument("--cr-grid", type=str, default="3,14,8",
                    help="cosmic_ray_grid as 'log10Emin,log10Emax,bins/decade'")
    ap.add_argument("--ph-grid", type=str, default="-15,-6,8")
    ap.add_argument("--full-pickle", action="store_true",
                    help="also pickle whole PriNCeRun and report size")
    args = ap.parse_args()

    import prince_cr.config as cfg

    cfg.debug_level = 0
    cfg.cosmic_ray_grid = tuple(int(x) if i == 2 else float(x)
                                for i, x in enumerate(args.cr_grid.split(",")))
    cfg.photon_grid = tuple(int(x) if i == 2 else float(x)
                            for i, x in enumerate(args.ph_grid.split(",")))
    cfg.x_cut = 1e-4
    cfg.x_cut_proton = 1e-2
    cfg.tau_dec_threshold = np.inf
    cfg.max_mass = args.max_mass

    print(f"=== profile max_mass={args.max_mass} cr_grid={cfg.cosmic_ray_grid} "
          f"ph_grid={cfg.photon_grid} ===")

    from prince_cr import core, cross_sections, photonfields

    with Stage("photon_field"):
        pf = photonfields.CombinedPhotonField(
            [photonfields.CMBPhotonSpectrum, photonfields.CIBGilmore2D]
        )

    with Stage("cross_sections.__init__ (table load + reduction)") as st_cs:
        cs = cross_sections.CompositeCrossSection(
            [
                (0.0, cross_sections.TabulatedCrossSection, ("CRP2_TALYS",)),
                (0.14, cross_sections.SophiaSuperposition, ()),
            ]
        )

    print(f"  known_species:       {len(cs.known_species)}")
    print(f"  known_bc_channels:   {len(cs.known_bc_channels)}")
    print(f"  known_diff_channels: {len(cs.known_diff_channels)}")

    # Force response-function build (lazy)
    with Stage("ResponseFunction (.resp first access)"):
        _ = cs.resp
        # Touch to force interpolator construction
        _ = len(cs.resp.nonel_intp)
        _ = len(cs.resp.incl_intp)
        _ = len(cs.resp.incl_diff_intp)
        _ = len(cs.resp.incl_diff_intp_integral)

    print(f"  nonel_intp:               {len(cs.resp.nonel_intp)} splines")
    print(f"  incl_intp:                {len(cs.resp.incl_intp)} splines")
    print(f"  incl_diff_intp:           {len(cs.resp.incl_diff_intp)} 2D splines")
    print(f"  incl_diff_intp_integral:  {len(cs.resp.incl_diff_intp_integral)} 2D splines (antideriv)")

    sz_nonel = pickle_size(cs.resp.nonel_intp) / 1e6
    sz_incl = pickle_size(cs.resp.incl_intp) / 1e6
    sz_diff = pickle_size(cs.resp.incl_diff_intp) / 1e6
    sz_diff_int = pickle_size(cs.resp.incl_diff_intp_integral) / 1e6
    print(f"  pickle sizes (MB): nonel={sz_nonel:.1f}  incl={sz_incl:.1f}  "
          f"incl_diff={sz_diff:.1f}  incl_diff_integral={sz_diff_int:.1f}")

    # Now build PriNCeRun (this triggers PhotoNuclearInteractionRate, which is what
    # we mostly care about)
    with Stage("PriNCeRun.__init__ (incl. _init_matrices)"):
        run = core.PriNCeRun(max_mass=args.max_mass, photon_field=pf, cross_sections=cs)

    ir = run.int_rates
    bm = ir._batch_matrix
    cm = ir.coupling_mat
    print(f"  dim_states:        {run.dim_states}")
    print(f"  cr_grid.d:         {run.cr_grid.d}")
    print(f"  ph_grid.d:         {run.ph_grid.d}")
    print(f"  batch_matrix:      shape={bm.shape}  bytes={bm.nbytes/1e6:.1f} MB")
    print(f"  batch_rows:        {ir._batch_rows.shape}  bytes={ir._batch_rows.nbytes/1e6:.1f} MB")
    print(f"  batch_cols:        {ir._batch_cols.shape}  bytes={ir._batch_cols.nbytes/1e6:.1f} MB")
    print(f"  coupling_mat nnz:  {cm.nnz}  data_bytes={cm.data.nbytes/1e6:.1f} MB")

    # rough breakdown
    bc_rows = run.cr_grid.d * len(cs.known_bc_channels)
    diff_rows = bm.shape[0] - bc_rows
    print(f"  rows attributable to bc channels:   {bc_rows}")
    print(f"  rows attributable to diff channels: {diff_rows}")
    print(f"  diff rows / dcr^2 (avg per channel): "
          f"{diff_rows / max(1, len(cs.known_diff_channels)) / run.cr_grid.d**2:.3f}")

    if args.full_pickle:
        with Stage("pickle whole PriNCeRun"):
            sz = pickle_size(run) / 1e9
        print(f"  full pickle: {sz:.2f} GB")


if __name__ == "__main__":
    main()
