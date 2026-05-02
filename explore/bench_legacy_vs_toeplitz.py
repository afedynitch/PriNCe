"""Benchmark the legacy and toeplitz kernel-construction paths side-by-side.

Builds the same `PhotoNuclearInteractionRate` twice — once with
`config.kernel_method = "legacy"`, once with `"toeplitz"` — and reports:

  - wall-clock time of `_init_matrices`
  - peak RSS during init
  - persistent `_batch_matrix` size

The cross sections + photon field are built once and reused, so the timed
window is exactly the kernel-construction step.

Usage:
    python explore/bench_legacy_vs_toeplitz.py --max-mass 14
    python explore/bench_legacy_vs_toeplitz.py --max-mass 56
"""

from __future__ import annotations

import argparse
import gc
import resource
import sys
import time

import numpy as np


def rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return r / 1024**2
    return r / 1024


def reset_peak_rss():
    """RUSAGE_SELF.ru_maxrss is monotonic per process, so we can't reset it.

    We just record the value before each phase and report deltas.
    """
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-mass", type=int, default=14)
    ap.add_argument("--cr-grid", type=str, default="3,14,8")
    ap.add_argument("--ph-grid", type=str, default="-15,-6,8")
    ap.add_argument("--methods", type=str, default="legacy,toeplitz",
                    help="comma-separated list of methods to benchmark")
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

    print(f"=== bench max_mass={args.max_mass} cr_grid={cfg.cosmic_ray_grid} "
          f"ph_grid={cfg.photon_grid} ===")

    from prince_cr import core, cross_sections, photonfields, interaction_rates

    print("\nBuilding photon field + cross sections (shared across runs)...")
    t0 = time.perf_counter()
    pf = photonfields.CombinedPhotonField(
        [photonfields.CMBPhotonSpectrum, photonfields.CIBGilmore2D]
    )
    cs = cross_sections.CompositeCrossSection(
        [
            (0.0, cross_sections.TabulatedCrossSection, ("CRP2_TALYS",)),
            (0.14, cross_sections.SophiaSuperposition, ()),
        ]
    )
    # Force the response splines (lazy build)
    _ = cs.resp
    print(f"  setup done in {time.perf_counter() - t0:.1f}s  "
          f"(species={len(cs.known_species)}, bc={len(cs.known_bc_channels)}, "
          f"diff={len(cs.known_diff_channels)})")

    results = {}
    for method in args.methods.split(","):
        method = method.strip()
        gc.collect()
        rss_before = rss_mb()

        cfg.kernel_method = method

        # Patch the IR to time only `_init_matrices`. PriNCeRun calls
        # _estimate_batch_matrix → _init_matrices → _init_coupling_mat in __init__.
        # We want just the middle one.
        IR = interaction_rates.PhotoNuclearInteractionRate
        original = IR._init_matrices
        timing = {}

        def timed_init(self, _orig=original, _timing=timing):
            gc.collect()
            r0 = rss_mb()
            t0 = time.perf_counter()
            _orig(self)
            t1 = time.perf_counter()
            r1 = rss_mb()
            _timing["dt"] = t1 - t0
            _timing["rss_delta"] = r1 - r0
            _timing["rss_after"] = r1

        IR._init_matrices = timed_init
        try:
            print(f"\n[{method}] building PriNCeRun...")
            run = core.PriNCeRun(max_mass=args.max_mass,
                                  photon_field=pf, cross_sections=cs)
        finally:
            IR._init_matrices = original

        bm = run.int_rates._batch_matrix
        results[method] = {
            "init_dt": timing["dt"],
            "init_rss_delta": timing["rss_delta"],
            "init_rss_after": timing["rss_after"],
            "batch_rows": bm.shape[0],
            "batch_bytes": bm.nbytes,
            "rss_total_after": rss_mb(),
            "rss_growth": rss_mb() - rss_before,
        }
        print(f"[{method}] _init_matrices: {timing['dt']:.2f}s  "
              f"RSS_delta={timing['rss_delta']:+.1f} MB  "
              f"batch_matrix={bm.shape[0]}×{bm.shape[1]} ({bm.nbytes/1e6:.1f} MB)")
        # Drop the run before next iteration
        del run
        gc.collect()

    # Summary
    print("\n=== summary ===")
    fmt = "{:>10}  {:>12}  {:>14}  {:>16}  {:>14}"
    print(fmt.format("method", "init [s]", "RSS Δ [MB]", "batch rows", "batch [MB]"))
    print(fmt.format("------", "--------", "----------", "----------", "----------"))
    for m, r in results.items():
        print(fmt.format(m, f"{r['init_dt']:.2f}",
                          f"{r['init_rss_delta']:+.1f}",
                          f"{r['batch_rows']:,}",
                          f"{r['batch_bytes']/1e6:.1f}"))

    if "legacy" in results and "toeplitz" in results:
        sp = results["legacy"]["init_dt"] / results["toeplitz"]["init_dt"]
        mem = results["legacy"]["init_rss_delta"] / max(
            results["toeplitz"]["init_rss_delta"], 1e-9)
        print(f"\nspeedup (legacy/toeplitz init time): {sp:.2f}×")
        print(f"transient RSS reduction: legacy needed "
              f"{results['legacy']['init_rss_delta']:.0f} MB, "
              f"toeplitz needed {results['toeplitz']['init_rss_delta']:.0f} MB  "
              f"({mem:.2f}× less)")


if __name__ == "__main__":
    main()
