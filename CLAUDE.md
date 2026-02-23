# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About PriNCe

PriNCe (**Pr**opagation **i**ncluding **N**uclear **C**ascade **e**quations) is a scientific Python package for solving ultra-high energy cosmic ray (UHECR) transport equations on cosmological scales. The code propagates cosmic rays through photon backgrounds (CMB, CIB) while accounting for photo-hadronic interactions, pair production, adiabatic losses, and nuclear disintegration.

## Development Commands

### Testing
```bash
# Run all tests with pytest
pytest

# Run a specific test file
pytest tests/test_propagation.py

# Run a specific test function
pytest tests/test_propagation.py::test_propagation

# Run tests with coverage
pytest --cov=prince_cr --cov-report=html

# Run tests in parallel (configured in pyproject.toml)
pytest -n auto
```

### Installation
```bash
# Install in development mode
pip install -e .

# Install with test dependencies
pip install -e ".[test]"
```

### Code Quality
```bash
# Format code with black
black src/prince_cr tests

# Lint with ruff
ruff check src/prince_cr tests
```

## Architecture Overview

### Core Data Flow

1. **PriNCeRun (core.py)**: Main orchestrator class that initializes and coordinates all components
   - Sets up energy grids (cosmic ray and photon)
   - Initializes cross sections, photon fields, interaction rates
   - Manages species list and system dimensions

2. **Solver Stack (solvers/propagation.py)**:
   - `UHECRPropagationSolver`: Base solver with derivative computation
   - `UHECRPropagationSolverBDF`: Backward differentiation formula integrator (recommended)
   - `UHECRPropagationSolverEULER`: Simple Euler stepper for testing
   - `UHECRPropagationResult`: Container for propagation results with analysis methods

3. **Physics Modules**:
   - **cross_sections/**: Photo-nuclear and photo-meson cross sections
     - `base.py`: Base classes and tabulated cross sections
     - `disintegration.py`: Nuclear disintegration channels
     - `photo_meson.py`: SOPHIA superposition model
   - **interaction_rates.py**: Computes interaction rates from cross sections × photon density
   - **photonfields.py**: CMB and CIB (cosmic infrared background) models
   - **decays.py**: Particle decay chains and redistribution functions
   - **cosmology.py**: Cosmological Hubble parameter H(z)

4. **Data Management (data.py)**:
   - `SpeciesManager`: Tracks particle species and provides index mapping
   - `EnergyGrid`: Logarithmic energy grid with bins and widths
   - `PrinceDB`: HDF5 database interface for cross section tables

### Key Design Patterns

**Jacobian Matrix Construction**: The interaction rates module builds a sparse CSR matrix representing all photo-nuclear interactions. The matrix is recomputed only when redshift changes significantly (controlled by `config.update_rates_z_threshold`).

**Semi-Lagrangian Solver**: Continuous energy losses (adiabatic expansion, pair production) are handled separately from discrete interactions using semi-Lagrangian interpolation methods. Multiple interpolation schemes are available via `config.semi_lagr_method` (linear, quadratic, cubic, 4th/5th order Lagrange weights).

**Batch Matrix Computation**: Interaction rates are computed in batches across the photon energy grid, then integrated to produce the final rate matrix. This design optimizes memory access patterns.

**Linear Algebra Backends**: The code supports three backends for sparse matrix operations:
- `"scipy"`: Standard numpy/scipy (always available)
- `"MKL"`: Intel MKL for CPU acceleration (auto-detected)
- `"cupy"`: GPU acceleration via CuPy (optional)

Backend selection is in `config.linear_algebra_backend`.

## Configuration System

The `config.py` module uses module-level globals for configuration. Key settings:

- **Grids**: `cosmic_ray_grid`, `photon_grid` - tuple format (log10_Emin, log10_Emax, nbins_per_decade)
- **Physics**: `tau_dec_threshold` (decay lifetime cutoff), `max_mass` (maximum nuclear mass), `secondaries` (include photons/neutrinos)
- **Numerics**: `update_rates_z_threshold`, `semi_lagr_method`, `linear_algebra_backend`
- **Debug**: `debug_level` (0=silent, higher=more verbose)

The configuration is read at import time and can be modified before creating a `PriNCeRun` instance.

## Testing Approach

Tests are located in `tests/` with one test file per module. The current branch `improve_tests` is modernizing the test suite:

- Tests use pytest fixtures for shared setup (photon fields, cross sections, PriNCeRun instances)
- Integration tests in `test_propagation.py` verify end-to-end propagation
- Unit tests validate individual components (cross sections, rates, grids, etc.)

**Important**: Many tests require the HDF5 database file which is auto-downloaded on first import. Tests may be slow due to physics computations.

## Data Files

- `src/prince_cr/data/prince_db_05.h5`: Main database with cross section tables (auto-downloaded from GitHub releases)
- `src/prince_cr/data/particle_data.ppo`: Particle properties (mass, lifetime, branching ratios)
- `src/prince_cr/data/sophia_redistribution_logbins.npy`: Energy redistribution functions

The database download is handled automatically by `config.py` on first import.

## Common Pitfalls

1. **Energy vs Energy-per-nucleon**: The solver works in energy-per-nucleon. Use `get_solution_scale()` to convert to total energy.

2. **Redshift Convention**: Integration runs from `initial_z` (high) to `final_z` (low), i.e., backward in time from emission to observation.

3. **Species IDs**: Particles use NCo IDs - proton is 101, neutron is 100, nuclei are A*100+Z. Secondaries (photons, neutrinos) are <100.

4. **Index Mapping**: `SpeciesManager` provides `lidx()`/`uidx()` for grid indices and `lbin()`/`ubin()` for bin edges. These are critical for slicing state vectors correctly.

5. **Matrix Updates**: The Jacobian is cached and only updated when `|z - z_cached| > threshold`. Forgetting this can cause stale rates in custom solvers.

## Platform Notes

This codebase is cross-platform (Windows, Linux, macOS). The config automatically detects available linear algebra backends (MKL, CuPy) and selects the fastest available. On Windows, MKL libraries are typically in `Library/bin/mkl_rt.dll`.
