"""Shared test configuration and fixtures.

Sets reduced grid sizes and provides session-scoped fixtures for expensive
objects (cross sections, PriNCeRun) to keep memory usage low enough for
CI runners and parallel test execution.
"""

import os

# Cap per-process BLAS / threadpool sizes BEFORE numpy/scipy/MKL get
# imported below. pytest-xdist's ``-n auto`` spawns one worker per
# logical CPU; without these caps each worker opens its own
# ``cpu_count``-sized OpenBLAS/MKL/OMP pool, so the host ends up
# running ~cpu_count^2 contended threads (server lockup on a 48-core
# box). One thread per worker is the safe default; user can override
# from the calling environment.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import pickle  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import prince_cr.config as config  # noqa: E402

# Belt-and-suspenders: if some BLAS backend already initialised its
# threadpool from a default-on bonus path before this module ran (e.g.
# imported transitively by another conftest), cap it now via
# threadpoolctl. Tolerated as best-effort — threadpoolctl is in the
# project's runtime extras but not strictly required by tests.
try:
    from threadpoolctl import threadpool_limits, threadpool_info

    threadpool_limits(limits=1)
    # Verbose-mode probe: surfaces the cap (or a missed pool) before tests
    # start. Cheap; runs once per process.
    if os.environ.get("PRINCE_TEST_VERBOSE_THREADS") == "1":
        for tp in threadpool_info():
            print(f"[threads] {tp['user_api']} via {tp['internal_api']}: "
                  f"num_threads={tp['num_threads']}")
except Exception:
    pass

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

# FLUKA smoke db ships v2-sparse format under the bookkeeping repo's run
# directory. Override fluka_db_path / fluka_db_fname to retarget elsewhere.
_FLUKA_SMOKE_DB_DIR = os.path.expanduser(
    "~/work/devel/UH-UHECR-Fluka-Prince/runs/2026-05-04_pfu-v1-smoke"
)
config.fluka_db_path = _FLUKA_SMOKE_DB_DIR
config.fluka_db_fname = "prince_db_v1_smoke.h5"

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
def fluka():
    """Session-scoped FLUKA photo-nuclear cross sections."""
    return cross_sections.FlukaPhotoNuclear()


# Alias for tests that use the name "cs"
@pytest.fixture(scope="session")
def cs(fluka):
    """Alias for the FLUKA cross section."""
    return fluka


@pytest.fixture(scope="session")
def prince_run_m4(pf, cs):
    """Session-scoped PriNCeRun with max_mass=4."""
    return core.PriNCeRun(max_mass=4, photon_field=pf, cross_sections=cs)


@pytest.fixture(scope="session")
def prince_run_m14(pf, cs):
    """Session-scoped PriNCeRun with max_mass=14."""
    return core.PriNCeRun(max_mass=14, photon_field=pf, cross_sections=cs)


# Light-medium nuclei cap (oxygen-16). Pickled to tests/data/prince_run_m16.ppo
# on first build; reloads in <1 s. ~10 s build vs the multi-GB Fe-56 cache.
# Suitable as the default reference for end-to-end solver tests; opts out via
# PRINCE_REGEN_FIXTURE=1.
_M16_CACHE = Path(__file__).resolve().parent / "data" / "prince_run_m16.ppo"


@pytest.fixture(scope="session")
def prince_run_m16(pf, cs):
    """Session-scoped PriNCeRun with max_mass=16 (H..O cap), pickled."""
    if _M16_CACHE.exists() and not os.environ.get("PRINCE_REGEN_FIXTURE"):
        with _M16_CACHE.open("rb") as fh:
            return pickle.load(fh)
    run = core.PriNCeRun(max_mass=16, photon_field=pf, cross_sections=cs)
    _M16_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with _M16_CACHE.open("wb") as fh:
        pickle.dump(run, fh, protocol=-1)
    return run


# ---------------------------------------------------------------------------
# Production-size cached kernel for solver benchmarking
# ---------------------------------------------------------------------------
# Build the .ppo with: python tests/_build_kernel_cache.py [--max-mass N]
# <output_path>. Then point PRINCE_KERNEL_CACHE at it. Falls back to a
# session-scoped build if not set, which is slow (~3 min at max_mass=56)
# but correct.
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
        _cfg.fluka_db_path = _FLUKA_SMOKE_DB_DIR
        _cfg.fluka_db_fname = "prince_db_v1_smoke.h5"
        pf_full = photonfields.CombinedPhotonField(
            [photonfields.CMBPhotonSpectrum, photonfields.CIBGilmore2D]
        )
        cs_full = cross_sections.FlukaPhotoNuclear()
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
    2212:       (0.96, 10**9.68, 20.0),  # proton
    1000020040: (0.96, 10**9.68, 50.0),  # He-4
    1000070140: (0.96, 10**9.68, 30.0),  # N-14
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
