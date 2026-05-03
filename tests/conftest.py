"""Shared test configuration and fixtures.

Sets reduced grid sizes and provides session-scoped fixtures for expensive
objects (cross sections, PriNCeRun) to keep memory usage low enough for
CI runners and parallel test execution.
"""

import os
import pickle
import time
from pathlib import Path

import numpy as np

import prince_cr.config as config

# ---------------------------------------------------------------------------
# Global test configuration — applied before any test module is imported.
# ---------------------------------------------------------------------------
config.debug_level = 0

# Reduced grids: 20 CR bins (vs 88), 16 photon bins (vs 72)
config.cosmic_ray_grid = (6, 11, 4)
config.photon_grid = (-12, -8, 4)

# Common physics settings used across the test suite
config.x_cut = 1e-4
config.x_cut_proton = 1e-2
config.tau_dec_threshold = np.inf
config.max_mass = 14

# FLUKA db lives in a sibling repo during v0 development. Production users
# either copy the file into data_dir or override this path themselves.
config.fluka_db_path = "/Users/anatoli/devel_mac/prince-fluka-utils"
config.fluka_db_fname = "prince_db_v0.h5"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
import pytest  # noqa: E402

from prince_cr import core, cross_sections, photonfields  # noqa: E402


@pytest.fixture(scope="session")
def pf():
    """Session-scoped combined photon field."""
    return photonfields.CombinedPhotonField(
        [photonfields.CMBPhotonSpectrum, photonfields.CIBGilmore2D]
    )


@pytest.fixture(scope="session")
def sophia():
    """Session-scoped SOPHIA cross sections."""
    return cross_sections.SophiaSuperposition()


@pytest.fixture(scope="session")
def talys():
    """Session-scoped TALYS cross sections."""
    return cross_sections.TabulatedCrossSection("CRP2_TALYS")


@pytest.fixture(scope="session")
def composite(talys, sophia):
    """Session-scoped composite cross section (TALYS + SOPHIA)."""
    return cross_sections.CompositeCrossSection(
        [
            (0.0, cross_sections.TabulatedCrossSection, ("CRP2_TALYS",)),
            (0.14, cross_sections.SophiaSuperposition, ()),
        ]
    )


# Alias for tests that use the name "cs"
@pytest.fixture(scope="session")
def cs(composite):
    """Alias for composite cross section."""
    return composite


@pytest.fixture(scope="session")
def prince_run_m4(pf, cs):
    """Session-scoped PriNCeRun with max_mass=4."""
    return core.PriNCeRun(max_mass=4, photon_field=pf, cross_sections=cs)


@pytest.fixture(scope="session")
def prince_run_m14(pf, cs):
    """Session-scoped PriNCeRun with max_mass=14."""
    return core.PriNCeRun(max_mass=14, photon_field=pf, cross_sections=cs)


# ---------------------------------------------------------------------------
# Production-size cached kernel for solver benchmarking
# ---------------------------------------------------------------------------
# Build the .ppo with: python tests/_build_kernel_cache.py <output_path>
# Then point PRINCE_KERNEL_CACHE at it. Falls back to a session-scoped build
# if not set, which is slow (~3 min) but correct.
@pytest.fixture(scope="session")
def cached_prince_run():
    """Production-size PriNCeRun (max_mass=56), loaded from pickle if available."""
    cache_path = os.environ.get("PRINCE_KERNEL_CACHE")
    if cache_path and Path(cache_path).exists():
        t0 = time.time()
        with open(cache_path, "rb") as fh:
            run = pickle.load(fh)
        print(f"[cached_prince_run] Loaded from {cache_path} in {time.time() - t0:.2f}s")
        return run

    print(
        "[cached_prince_run] PRINCE_KERNEL_CACHE not set or missing; "
        "building kernel from scratch (~3 min). Run "
        "tests/_build_kernel_cache.py to cache it."
    )
    # Production-grid build matching tests/_build_kernel_cache.py
    saved = (config.cosmic_ray_grid, config.photon_grid, config.max_mass)
    try:
        # Use defaults from prince_cr.config (production grid)
        import importlib
        import prince_cr.config as _cfg

        importlib.reload(_cfg)
        _cfg.x_cut = 1e-4
        _cfg.x_cut_proton = 1e-2
        _cfg.tau_dec_threshold = np.inf
        pf_full = photonfields.CombinedPhotonField(
            [photonfields.CMBPhotonSpectrum, photonfields.CIBGilmore2D]
        )
        cs_full = cross_sections.CompositeCrossSection(
            [
                (0.0, cross_sections.TabulatedCrossSection, ("CRP2_TALYS",)),
                (0.14, cross_sections.SophiaSuperposition, ()),
            ]
        )
        run = core.PriNCeRun(max_mass=56, photon_field=pf_full, cross_sections=cs_full)
    finally:
        config.cosmic_ray_grid, config.photon_grid, config.max_mass = saved
    return run


# ---------------------------------------------------------------------------
# Wall-clock helper for solver benchmarking
# ---------------------------------------------------------------------------
class _Bench:
    def __init__(self, label):
        self.label = label
        self.t0 = None
        self.elapsed = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0
        print(f"[bench] {self.label}: {self.elapsed:.3f}s")
        return False


@pytest.fixture
def bench_propagation():
    """Return a context manager that times a block and stores elapsed time.

    Usage:
        with bench_propagation("BDF") as b:
            solver.solve()
        assert b.elapsed < 60.0
    """
    return _Bench


# ---------------------------------------------------------------------------
# Tight-cache ETD2 reference state for solver regression / validation
# ---------------------------------------------------------------------------
BASELINE_FILE = Path(__file__).resolve().parent / "data" / "baseline_state.npz"

AUGER_BASELINE_PARAMS = {
    101: (0.96, 10**9.68, 20.0),
    402: (0.96, 10**9.68, 50.0),
    1407: (0.96, 10**9.68, 30.0),
}


@pytest.fixture(scope="session")
def baseline_state(cached_prince_run):
    """Tight-cache ETD2 state at z=0 for the Auger fit case.

    Cached to disk under ``tests/data/baseline_state.npz``. Regenerate by
    deleting the file or setting ``PRINCE_REGEN_BASELINE=1``.
    """
    from prince_cr.cr_sources import AugerFitSource
    from prince_cr.solvers import UHECRPropagationSolverETD2

    regen = os.environ.get("PRINCE_REGEN_BASELINE") == "1"
    if BASELINE_FILE.exists() and not regen:
        data = np.load(BASELINE_FILE)
        state = data["state"]
        if state.size == cached_prince_run.dim_states:
            return state
    BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)

    solver = UHECRPropagationSolverETD2(
        initial_z=1.0,
        final_z=0.0,
        prince_run=cached_prince_run,
        enable_pairprod_losses=True,
        enable_adiabatic_losses=True,
        enable_injection_jacobian=True,
        enable_partial_diff_jacobian=True,
    )
    # Tight cache threshold for the reference; default 0.01 leaves ~1% systematic.
    solver.recomp_z_threshold = 1e-3
    solver.add_source_class(
        AugerFitSource(cached_prince_run, norm=1e-50, params=AUGER_BASELINE_PARAMS)
    )
    t0 = time.perf_counter()
    solver.solve(dz=2e-4, verbose=False, summary=False, progressbar=False)
    print(f"[baseline] ETD2 thr=1e-3 dz=2e-4 in {time.perf_counter() - t0:.2f}s")
    state = solver.state.copy()
    np.savez(BASELINE_FILE, state=state, thr=1e-3, dz=2e-4)
    return state
