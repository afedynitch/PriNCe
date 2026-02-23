"""Shared test configuration and fixtures.

Sets reduced grid sizes and provides session-scoped fixtures for expensive
objects (cross sections, PriNCeRun) to keep memory usage low enough for
CI runners and parallel test execution.
"""

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
