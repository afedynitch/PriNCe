# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About PriNCe

PriNCe (**Pr**opagation **i**ncluding **N**uclear **C**ascade **e**quations) is a scientific Python package for solving ultra-high energy cosmic ray (UHECR) transport equations on cosmological scales. The code propagates cosmic rays through photon backgrounds (CMB, CIB) while accounting for photo-hadronic interactions, pair production, adiabatic losses, and nuclear disintegration.

## Mathematical Model

PriNCe solves the cosmic-ray transport equation discretized over species and energy bins. The state vector `n` has length `dim_states = cr_grid.d * spec_man.nspec` and is laid out species-major: each species occupies a contiguous block of CR-energy bins, with `SpeciesManager.lidx()/uidx()` returning the slice for a given species. Typical production sizes are ~150 species × ~88 CR-energy bins ≈ 13k DoF; test fixtures use much smaller grids.

The continuous PDE is integrated in **redshift `z`**, running backward in time from `initial_z` (high) to `final_z` (low). The Jacobian of comoving distance with respect to redshift is `dl/dz = -1 / ((1+z) H(z) · cm2sec)` (`UHECRPropagationSolver.dldz`, propagation.py:256–257), with `H(z) = H₀ √(Ω_m (1+z)³ + Ω_Λ)` (cosmology.py:10–21).

The discretized RHS, in compact form (`eqn_deriv_standard`, propagation.py:398–424):

```
dn/dz = J(z) · n           # photo-hadronic rate matrix (CSR)
      + Q(z)               # external injection (sources)
      + (dl/dz) · D · (κ(z) ⊙ n)    # continuous losses (adiabatic + pair production)
```

- `J(z)` is the sparse interaction matrix: diagonal entries are loss rates, off-diagonal entries are inclusive yields (gain channels). Built in `interaction_rates.py:285–325`.
- `Q(z) = (dl/dz) · cm2sec · Σ source_k.injection_rate(z)` (propagation.py:259–266). Linear in source list, independent of `n`.
- `κ(z) = κ_adia(z) + κ_pair(z)` is the per-bin energy-loss rate vector (interaction_rates.py:393–555).
- `D` is the sparse finite-difference operator on the log-energy grid (`partial_diff.py:486–639`). It is the **default and only path actually exercised** in production: `enable_partial_diff_jacobian=True` is the default for `UHECRPropagationSolver`.
- `⊙` denotes elementwise product.

The full system has the structure `dn/dz = L(z) n + Q(z)` with `L = J + (dl/dz) · D · diag(κ)` — fully linear in `n` plus an inhomogeneous source. This linearity is what makes exponential integrators (e.g. ETD2) viable.

**Note on the semi-Lagrangian code**: `SemiLagrangianSolver` (`partial_diff.py:5–483`) and `UHECRPropagationSolver.semi_lagrangian()` (propagation.py:284–396) implement an alternative split-step approach (advance `J·n + Q` over `dz`, then apply an interpolated bin-shift for energy losses). This path is only reachable from `UHECRPropagationSolverEULER.solve()` when `enable_partial_diff_jacobian=False` — neither is the default. `config.semi_lagr_method` (`5th_order`, `cubic`, etc.) only affects this dead-by-default branch. Treat this code as legacy and not part of the production data flow.

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
   - `UHECRPropagationSolver` (lines 168–515): Base class. Owns the rate matrix, sources, continuous-loss operators, and the RHS function `eqn_deriv_standard` (lines 398–424). Subclasses pick the integrator.
   - `UHECRPropagationSolverBDF` (lines 518–636): Wraps `scipy.integrate.BDF` via `PrinceBDF` (`util.py:243–419`), which subclasses BDF and overrides `_step_impl` to cap step size without forcing Jacobian rebuilds (avoids oscillations at high z). Default tolerances: `rtol=1e-10`, `atol=1e40` (effectively `rtol`-only). Stepping loop: `while r.status == "running": r.step()` — fully scipy-driven.
   - `UHECRPropagationSolverEULER` (lines 638–720): Forward-Euler with explicit `dz` steps. Always pairs with the semi-Lagrangian split-step for continuous losses.
   - `UHECRPropagationResult` (lines 13–166): Result container exposing `get_solution()`, `get_solution_scale()` (energy-per-nucleon → total energy), `get_solution_group()`, `get_lnA()`, `get_energy_density()`.

   Solver choice is by class instantiation (no config flag). Standard usage: build a `PriNCeRun`, instantiate a solver with `initial_z`/`final_z` and the enable-flags, attach sources via `add_source_class()`, call `solve(dz=...)`. The end-to-end pattern is in `tests/test_propagation.py:14–47`.

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

**Jacobian Matrix Construction**: `interaction_rates.py` builds a sparse CSR matrix `coupling_mat` whose sparsity pattern is **fixed at initialization** (`_init_coupling_mat`, lines 285–325) — it encodes which species couple to which via photo-hadronic channels and never changes. When the redshift changes by more than `config.update_rates_z_threshold` (default `0.01`), only the values are refreshed via a single matrix–vector product against the photon-density vector:
```python
np.dot(self._batch_matrix, self.photon_vector(z), out=self._batch_vec)
self.coupling_mat.data = scale_fac * self._batch_vec
```
(`_update_rates` / `_update_coupling_mat`, lines 326–362). The `indices`/`indptr` arrays are never touched, so any custom solver must preserve this contract — it relies on lexicographically pre-sorted row/column indices established at init. Switching matrix backends (scipy/MKL/cupy) changes how SpMV is computed but not this update mechanism.

**Continuous-Loss FD Stencil (default path)**: The default `enable_partial_diff_jacobian=True` route uses `DifferentialOperator` (`partial_diff.py:486–639`), a sparse 4th-order one-sided FD operator on log-E. Centered rows use `[-3, -10, 18, -6, 1]/(12·h)` at log-E offsets `[-1, 0, 1, 2, 3]` — asymmetric, upwind-biased toward higher energies (correct for losses moving flux down in E). The first/last three rows use 3- and 4-point one-sided stencils (only 2nd-order at the very edges). The operator is left-multiplied by `diag(1/E)` to convert `d/dlog(E)` → `d/dE`, then `block_diag`-replicated `nspec` times so a single SpMV `D · (κ⊙n)` handles all species simultaneously.

**Legacy semi-Lagrangian path (not used by default)**: `SemiLagrangianSolver` (`partial_diff.py:5–483`) provides Lagrange-interpolation methods (`linear`, `quadratic`, `cubic`, `4th_order`, `5th_order`, `finite_diff`, etc.) selectable via `config.semi_lagr_method`. These all route through `get_shifted_state()` which divides by the bin-width gradient — exact only for power-law-within-bin spectra. This branch is only reachable via `UHECRPropagationSolverEULER` with `enable_partial_diff_jacobian=False` and is treated as legacy.

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

5. **Matrix Updates**: The Jacobian is cached and only updated when `|z - z_cached| > threshold`. Forgetting this can cause stale rates in custom solvers. The update path overwrites `coupling_mat.data` in place — never reassign the matrix or rebuild it, or you break the sparsity-pattern contract relied on by all SpMV backends.

6. **`tau_dec_threshold` is a "resonance"-style approximation**: `config.tau_dec_threshold` (config.py:85) folds species with lifetimes below the threshold directly into their parents' interaction redistributions at cross-section build time (`cross_sections/base.py:283`, `photo_meson.py:352`). Default `np.inf` keeps all unstable species in the system. Lowering it removes short-lived species from the state vector — analogous to MCEq's `force_resonance` — at the cost of approximating their decay as instantaneous. This is a stiffness-mitigation hack; with sufficiently stable integrators it can be relaxed.

7. **Source vs. initial state**: `solve()` always starts from `initial_state = np.zeros(dim_states)`. Cosmic rays come into the system through injection sources (`add_source_class`), not initial conditions. There is no public API for setting a non-zero starting spectrum without subclassing.

## Platform Notes

This codebase is cross-platform (Windows, Linux, macOS). The config automatically detects available linear algebra backends (MKL, CuPy) and selects the fastest available. On Windows, MKL libraries are typically in `Library/bin/mkl_rt.dll`.
