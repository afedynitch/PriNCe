"""PriNCe configuration module.

Module-level globals here hold project-wide *defaults*. Per-run overrides
live on :class:`BackendConfig`, which is attached to every
:class:`prince_cr.core.PriNCeRun` as ``prince_run.backend``. The solver
and the interaction-rate machinery read backend dispatch through that
handle (``prince_run.backend.linear_algebra_backend`` etc.) so a single
process can run multiple :class:`PriNCeRun` instances with different
backend settings without mutating module globals.

Two categories of state still live on this module:

* ``debug_level``, grids, cosmological parameters — read once at
  :class:`PriNCeRun` construction; safe as globals.
* ``has_cupy``, ``has_mkl``, ``mkl_path``, ``mkl`` — autodetect /
  process-level resource handles. Not user-facing dispatch knobs.
"""

import os
import os.path as path
import platform
import sys
import importlib.util
import pathlib
from dataclasses import dataclass, field, fields
from typing import Optional

import numpy as np

base_path = path.dirname(path.abspath(__file__))

#: Debug flag for verbose printing, 0 silences PriNCe entirely
debug_level = 1
#: Printout debug info only for functions in this list (just give the name,
#: "get_solution" for instance) Warning, this option slows down initialization
#: by a lot. Use only when needed.
override_debug_fcn = []
#: Override debug printout for debug levels < value for the functions above
override_max_level = 10
#: Print module name in debug output
print_module = False

# =================================================================
# Paths and library locations
# =================================================================

#: Directory where the data files for the calculation are stored
data_dir = path.join(base_path, "data")

#: PrinceDB file name (legacy; serves EBL_models in production after the
#: FLUKA wiring lands)
db_fname = "prince_db_05.h5"

#: FLUKA-derived photo-nuclear database file name. Built by
#: ``prince-fluka-utils`` (separate repo). Path defaults to ``data_dir``;
#: override ``prince_cr.config.fluka_db_path`` to point at the source repo.
fluka_db_fname = "prince_db_v0.h5"

#: Directory containing ``fluka_db_fname``. No auto-download for v0 — set
#: this to the prince-fluka-utils repo root or copy the file into
#: ``data_dir``.
fluka_db_path = data_dir

#: SOPHIA photo-meson database, repacked into PDG numbering by
#: ``scripts/repack_sophia_pdg.py`` (in the UH-UHECR project repo) from the
#: legacy ``photo_nuclear/SOPHIA`` tables. Lives in ``data_dir``. Consumed by
#: ``cross_sections.SophiaSuperposition`` as an alternative to the FLUKA
#: photo-meson model.
sophia_db_fname = "prince_db_sophia_pdg.h5"

#: Directory containing ``sophia_db_fname``; defaults to ``data_dir``.
sophia_db_path = data_dir

#: EM-cascade engine. True -> kinetic single-scatter cascade
#: (``cascade.kinetic_cascade_transfer``; per-scatter IC + electron tracking,
#: reproduces the multi-generation E_X..E_abs plateau — validated vs Kalashev
#: Fig 2). False -> legacy cooled single-step (``cascade_transfer_matrix``),
#: which undershoots the plateau; kept for comparison.
em_kinetic_cascade = True

#: Model file for redistribution functions (from SOPHIA or similar)
redist_fname = "sophia_redistribution_logbins.npy"


# =========================================================================
# Physics configuration
# =========================================================================

#: Cosmological parameters

#: Hubble constant
H_0 = 70.5  # km s^-1 Mpc^-1
H_0s = 2.28475e-18  # s^-1

#: Omega_m
Omega_m = 0.27

#: Omega_Lambda
Omega_Lambda = 0.73

#: CMB energy kB*T0 [GeV]
E_CMB = 2.34823e-13

# ===========================================================================
# Grids
# ===========================================================================

#: Cosmic ray energy grid (defines system size for solver)
#: Number of bins in multiples of 4 recommended for maximal vectorization
#: efficiency for 256 bit AVX or similar
#: Format (log10(E_min), log10(E_max), nbins/decade of energy)
cosmic_ray_grid = (3, 14, 8)
#: Photon grid of target field, only for calculation of rates
photon_grid = (-15, -6, 8)

#: Scale of the energy grid
#: 'E': logarithmic in energy E_i = E_min * (Delta)^i
#: 'logE': linear grid in x = log_10(E): x_i = x_min + i * Delta
grid_scale = "E"

# ===========================================================================
# Model options
# ===========================================================================

#: Threshold lifetime value for explicit transport of particles of this type. It
#: means that if a particle is unstable with lifetime smaller than this threshold,
#: it will be decayed until all final state particles of this chain are stable.
#: In other words: short intermediate states will be integrated out
tau_dec_threshold = np.inf  # All unstable particles decay
# tau_dec_threshold = 0.  # None unstable particles decay
# tau_dec_threshold = 850. # This value is for stable neutrons

#: Gate the cross-section chain reducer so unstable mothers reach the state
#: vector. When True, the lifetime-based drop and the chain-walk fold-out in
#: ``_DecayChainReducer`` are skipped; decay is then handled explicitly by the
#: solver's Λ operator (``enable_decay=True`` on ``UHECRPropagationSolverETD2``;
#: see methods/explicit-decay-kernel-design.md). Unknown mothers (no entry in
#: ``spec_data``) are still dropped — only the lifetime gate flips.
enable_explicit_decay = False

#: Cut on energy redistribution functions
#: Resitribution below this x value are set to 0.
#: "x_cut" : 0.,
#: "x_cut_proton" : 0.,
x_cut = 1e-4
x_cut_proton = 1e-1

#: cut on photon energy, cross section above y = E_cr e_ph / m_cr does not contribute
y_cut = np.inf

# Build equation system up to a maximal nuclear mass of
max_mass = np.inf

# Limit energy range loaded from HDF5 cross section tables.
# Set to (e_min, e_max) tuple to load only data within this range,
# reducing memory usage. Units match the energy grid in the database.
# None means load the full range.
cross_section_e_range = None

# Include secondaries like photons and neutrinos
secondaries = True
# List of specific particles to ignore (PDG codes). e- = 11, e+ = -11.
ignore_particles = [
    11,
    -11,
]  # (we ignore electrons / positrons; their physics is not fully implemented)

# ===========================================================================
# Parameters of numerical integration
# ===========================================================================

# Update rates at not more frequently than this value in z
update_rates_z_threshold = 0.01

# #Number of MKL threads (for sparse matrix multiplication the performance
# #advantage from using more than a few threads is limited by memory bandwidth)
MKL_threads = 16

# Sparse matrix-vector product from "CUPY"|"MKL"|"scipy"
linear_algebra_backend = "CUPY"

# MKL Sparse BLAS block size for the photo-hadronic ``M_off`` matrix.
# ``None`` keeps it CSR; an integer ≥ 2 stores it as BSR with that
# block size (auto-padding the matrix). Per-op BSR(bs=2) is ~5 % faster
# than CSR-no-opt at production grid, but refreshing the BSR's flattened
# block layout from PriNCe's 1D CSR ``M_off.data`` costs ~11 ms via
# fancy-index scatter (vs ~1 ms ``np.copyto`` for CSR). At W ≈ 10 SpMVs
# per cache window the scatter dominates, so default is ``None``. The
# CSR→BSR index map in :class:`prince_cr.mkl_sparse.MklSparseMatrix` is
# preserved for hosts or update cadences (W ≥ ~30) where BSR pays off.
mkl_bsr_blocksize = None

# Float dtype for the cupy backend's ETD2 hot path (per-step SpMV, κ
# multiplication, exp/φ chain). Default ``"float64"``: PriNCe's stiff
# multi-decade-dynamic-range propagation accumulates a *systematic*
# (not random) bias at fp32 over ~100-1000 steps that visibly distorts
# the propagated CR spectrum — under-suppresses the low-E flux and
# over-suppresses the GZK shoulder, mimicking missing continuous losses.
# Switch to ``"float32"`` only for benchmarking on hardware that
# throttles fp64 (e.g., consumer RTX Ampere with 1/64 fp64 throughput)
# and where you have validated the fp32 spectrum against an fp64 / MKL
# reference for your config.
cupy_dtype = "float64"

# Float dtype for the per-cache-window dense matvec that rebuilds the
# photo-hadronic rate matrix (``_batch_matrix @ photon_vector``).
# Applies to every backend (scipy, MKL, cupy): the matvec is
# bandwidth-bound at this shape (rows ≫ cols), so the dtype halves the
# byte traffic and ~doubles throughput on every BLAS we run.
# Default ``"float32"`` gives the *mixed-precision* pipeline: SGEMM /
# DGEMV at the rate-cache rebuild, with the result cast up to the
# solver dtype (fp64) before it reaches the ETD2 hot path (per-step
# SpMV, κ multiplication, exp/φ chain — where the fp32 systematic
# bias originates and where keeping fp64 buys correctness). On cupy /
# RTX 3090 the mixed pipeline runs ~8× faster than full fp64 at
# L2rel ~3e-5 vs MKL; on Apple Accelerate the dense matvec alone is
# ~2× faster (`_batch_matrix` is 263 MB at max_mass=24 → bandwidth-
# bound dgemv → sgemv) translating to ~10–15 % wall save on host
# scipy. Set to ``"float64"`` to force fp64 GEMM (parity baseline),
# or to ``None`` to follow the solver dtype. The cast is one
# nnz-sized copy per cache window (~10 ETD2 steps), so the per-step
# cost is negligible vs the matvec itself.
xs_dtype = "float32"

# Capture the per-step ETD2 kernel sequence into a CUDA Graph and replay
# it across each cache window (~10 steps), re-capturing on every
# ``_refresh_z_caches`` (the cuSPARSE descriptors close over ``M_off`` /
# ``D_off`` device pointers that change at each window). Enable only when
# ``has_cupy`` is True and the cupy backend is selected; the launcher
# falls back to the eager fused-kernel cupy path otherwise.
use_cuda_graphs = True


# When True AND ``linear_algebra_backend == "MKL"``, route the rate-
# cache-rebuild dense matvec through MKL CBLAS DGEMV so it shares MKL's
# threadpool with the Sparse BLAS path. **Default is False** because on
# AMD Zen 2 + Intel MKL 2026, ``cblas_dgemv`` is dispatched as a serial
# kernel regardless of ``mkl_set_num_threads`` (MKL's L2 BLAS heuristic
# treats DGEMV as memory-bandwidth-bound and skips parallelization);
# OpenBLAS DGEMV at the same shape parallelizes ~14× faster end-to-end.
# The clean alternative on Zen is :func:`set_thread_count` (pairs
# OpenBLAS and MKL through threadpoolctl). Preserve this flag for hosts
# where MKL DGEMV does parallelize (e.g. Intel CPUs).
use_mkl_dense_matvec = False


# =================================================================
# Backend dispatch — per-run handle (BackendConfig)
# =================================================================
#
# A small dataclass that gathers the dispatch knobs the solver and
# interaction-rate machinery need to consult per-run. PriNCeRun creates
# one of these in __init__ (defaults pulled from this module's globals)
# and exposes it as ``prince_run.backend``. Code that has a PriNCeRun
# handle should read ``prince_run.backend.<knob>`` rather than the
# module global; the latter survives only as a default and as a
# back-compat surface for notebooks that mutate it directly.

@dataclass
class BackendConfig:
    """Per-run backend dispatch settings.

    Defaults are pulled from the :mod:`prince_cr.config` module globals
    of the same names at construction time. Mutate on a specific run
    (e.g. ``prince_run.backend.linear_algebra_backend = "MKL"``) to
    flip backends without touching the global; create a new solver
    instance afterwards so it picks up the new setting.
    """

    linear_algebra_backend: str = ""
    use_cuda_graphs: bool = False
    use_mkl_dense_matvec: bool = False
    cupy_dtype: str = "float64"
    xs_dtype: Optional[str] = "float32"
    mkl_bsr_blocksize: Optional[int] = None
    mkl_threads: int = 16
    update_rates_z_threshold: float = 0.01

    @classmethod
    def from_globals(cls) -> "BackendConfig":
        """Build a BackendConfig with values copied from the module globals.

        Used by :class:`PriNCeRun` when no explicit ``backend=`` is
        provided. The copy means later mutations of the module globals
        do not affect previously-built runs.
        """
        return cls(
            linear_algebra_backend=linear_algebra_backend,
            use_cuda_graphs=use_cuda_graphs,
            use_mkl_dense_matvec=use_mkl_dense_matvec,
            cupy_dtype=cupy_dtype,
            xs_dtype=xs_dtype,
            mkl_bsr_blocksize=mkl_bsr_blocksize,
            mkl_threads=MKL_threads,
            update_rates_z_threshold=update_rates_z_threshold,
        )


# Check for CUPY library for GPU support
try:
    import cupy

    has_cupy = True
    mempool = cupy.get_default_memory_pool()
    mempool.free_all_blocks()
except ModuleNotFoundError:
    print("CUPY not found for GPU support. Degrading to MKL.")
    if linear_algebra_backend.lower() == "cupy":
        linear_algebra_backend = "MKL"
    has_cupy = False

#: Autodetect best solver
#: determine shared library extension and MKL path
pf = platform.platform()

prefix = pathlib.Path(sys.prefix)
if "Linux" in pf:
    mkl_libs = list((prefix / "lib").glob("libmkl_rt*"))
    mkl_path = mkl_libs[0] if mkl_libs else prefix / "lib" / "libmkl_rt.so"
elif "macOS" in pf:
    mkl_path = prefix / "lib" / "libmkl_rt.dylib"
    has_accelerate = True
else:
    # Windows or unknown OS: search for mkl_rt*.dll in Library/bin and lib
    mkl_path = None
    mkl_dirs = [prefix / "Library" / "bin", prefix / "lib"]
    mkl_candidates = []
    for d in mkl_dirs:
        if d.exists():
            mkl_candidates.extend(d.glob("mkl_rt*.dll"))
    if mkl_candidates:
        mkl_path = mkl_candidates[0]
    else:
        # fallback to default path
        mkl_path = prefix / "Library" / "bin" / "mkl_rt.dll"

# mkl library handler
mkl = None

has_mkl = bool(mkl_path.is_file())

# Look for cupy module
has_cuda = importlib.util.find_spec("cupy") is not None

# CUDA is usually fastest, then MKL. Fallback to numpy.
if has_cuda:
    kernel_config = "CUDA"
elif has_mkl:
    kernel_config = "MKL"
else:
    kernel_config = "numpy"
if debug_level >= 2:
    print(f"Auto-detected {kernel_config} solver.")


def _load_mkl():
    """Lazily load ``libmkl_rt`` exactly once and cache it on ``mkl``.

    Splitting the load from :func:`set_mkl_threads` matters because the
    ``MklSparseMatrix`` wrappers in :mod:`prince_cr.mkl_sparse` pin their
    own reference to the cdll handle. Re-loading the library on every
    thread-count change would leave already-built wrappers tied to a
    stale ``cdll`` while the global ``mkl`` pointed at a fresh one — a
    subtle source of cross-handle bugs. Pinning the global to a single
    cdll for the lifetime of the process keeps every wrapper looking at
    the same symbol table. Pattern lifted from MCEq.
    """
    global mkl
    if mkl is not None or not has_mkl:
        return
    from ctypes import cdll

    mkl = cdll.LoadLibrary(os.fspath(mkl_path))


def set_mkl_threads(nthreads):
    """Set the MKL thread count (loads ``libmkl_rt`` on first call).

    Idempotent on the library side: only ``mkl_set_num_threads`` is
    called on subsequent invocations. The cached cdll handle is
    preserved, so handles in ``MklSparseMatrix`` wrappers stay valid
    across thread-count changes.

    Note: this only configures MKL's threadpool. numpy/scipy in the
    default venv link against bundled OpenBLAS, which keeps a
    *separate* threadpool that defaults to all physical cores. When
    both pools are active simultaneously (e.g. ETD2 doing MKL Sparse
    SpMV and OpenBLAS DGEMV in the same step) they oversubscribe Zen 2
    cores. Use :func:`set_thread_count` instead for the common case
    where you want both pools sized to the same N.
    """
    global MKL_threads
    from ctypes import byref, c_int

    _load_mkl()
    MKL_threads = nthreads
    if mkl is not None:
        mkl.mkl_set_num_threads(byref(c_int(nthreads)))
        if debug_level >= 5:
            print(f"MKL threads limited to {nthreads}")


def set_thread_count(nthreads):
    """Size both MKL and the underlying numpy/scipy BLAS pool to ``nthreads``.

    On hosts where numpy is bundled-OpenBLAS (the default venv on
    this project), this is what you want when running ETD2 with the
    MKL backend: the per-step SpMV uses MKL's pool and the dense
    cache-rebuild matvec uses OpenBLAS. Sized together, total threads
    stay bounded; sized independently, the two pools fight over physical
    cores under load.

    Effect:

    * Calls :func:`set_mkl_threads(nthreads)`.
    * Calls ``threadpoolctl.threadpool_limits(nthreads, "blas")`` so
      OpenBLAS / MKL inside numpy / scipy / accelerate libraries all
      respect the same cap.

    No-op on the BLAS side if :mod:`threadpoolctl` is missing — falls
    back to the historical behaviour with a warning.
    """
    set_mkl_threads(nthreads)
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=int(nthreads), user_api="blas")
        if debug_level >= 5:
            print(f"BLAS (OpenBLAS/MKL inside numpy) threads limited to {nthreads}")
    except ImportError:
        import warnings

        warnings.warn(
            "threadpoolctl not installed — set_thread_count fell back to "
            "set_mkl_threads only; OpenBLAS thread count unaffected. "
            "`pip install threadpoolctl` to fix.",
            stacklevel=2,
        )


if has_mkl:
    set_mkl_threads(MKL_threads)

if not has_mkl and linear_algebra_backend.lower() == "mkl":
    print("MKL runtime not found. Degrading to scipy.")
    linear_algebra_backend = "scipy"


def _download_file(url, outfile):
    """Downloads the PriNCe database from github release binaries."""

    from tqdm import tqdm
    import requests
    import math

    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)

    # Total size in bytes.
    total_size = int(r.headers.get("content-length", 0))
    block_size = 1024 * 1024
    wrote = 0
    with open(outfile, "wb") as f:
        for data in tqdm(
            r.iter_content(block_size),
            total=math.ceil(total_size // block_size),
            unit="MB",
            unit_scale=True,
        ):
            wrote = wrote + len(data)
            f.write(data)
    if total_size != 0 and wrote != total_size:
        raise Exception("ERROR, something went wrong")


# Download database file from github
base_url = "https://github.com/joheinze/PriNCe/releases/download/"
release_tag = "v0.5_alpha_release/"
url = base_url + release_tag + db_fname
if not path.isfile(path.join(data_dir, db_fname)):
    print("Downloading for PriNCe database file {0}.".format(db_fname))
    if debug_level >= 2:
        print(url)
    _download_file(url, path.join(data_dir, db_fname))
else:
    import h5py

    try:
        with h5py.File(path.join(data_dir, db_fname), "r") as prince_db:
            db_version = prince_db.attrs["version"]
    except (OSError, KeyError, Exception):
        print(f"Database file {db_fname} corrupted. Retrying download.")
        _download_file(url, path.join(data_dir, db_fname))
    finally:
        with h5py.File(path.join(data_dir, db_fname), "r") as prince_db:
            db_version = prince_db.attrs["version"]
        if debug_level >= 2:
            print(f"Using database file version {db_version}.")

# if path.isfile(path.join(data_dir, '...previous db name...')):
#     import os
#     print('Removing previous database {0}.'.format('...previous db name...'))
#     os.unlink(path.join(data_dir, '...previous db name...'))
