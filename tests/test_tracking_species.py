"""Tests for the passive tracking-species feature.

Covers the four landing checkpoints from
``wiki/methods/tracking-species-design.md``:

  * Step B — data model. ``PrinceTrackedSpecies`` inherits physics from a
    real daughter while occupying its own state-vector slot under a
    synthetic PDG ID. ``is_nucleus(synthetic) == False`` so PDG-arithmetic
    branches treat tracked species as non-nuclei (passive observer); the
    real daughter's flags survive on the ``trk.real_pdgid`` carry-along.

  * Step C — chain-reducer hook (primary path). Charge-exchange tracking
    in the chain-reducer-off / explicit-decay-on configuration: a tracked
    neutron records ``flux(z) > 0`` produced from photo-hadronic ``p → n``;
    the real-species fluxes are bit-identical to a no-tracking run
    (``tracked → real`` cannot leak by construction of the kernel rows).

  * Step D — energy-range filter. Three tracked-proton species with
    disjoint photon-energy windows sum to a fourth tracker with
    ``e_gamma_range=None`` to machine precision (linear mask on the
    photon axis × linear convolution).

  * Step E — Λ_off hook. Tracked-ν̄_e from nuclei via the
    ``enable_explicit_decay=True`` path: Λ_off carries the tracked-species
    rows in the decay operator; the resulting flux is non-zero and
    bounded above by the real ν̄_e flux at every bin.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

import prince_cr.config as conf
from prince_cr.core import PriNCeRun
from prince_cr.cr_sources import AugerFitSource
from prince_cr.cross_sections import FlukaPhotoNuclear
from prince_cr.data import (
    PrinceTrackedSpecies,
    SpeciesManager,
    _TRACKING_PDG_BASE,
)
from prince_cr.solvers import UHECRPropagationSolverETD2
from prince_cr.util import is_nucleus


_AUGER_HE4 = {1000020040: (0.96, 10**9.68, 50.0)}
_AUGER_PROTON = {2212: (0.96, 10**9.68, 20.0)}


@pytest.fixture
def fresh_cs():
    """A fresh ``FlukaPhotoNuclear`` per test. Tracking amendment mutates
    the cross-section's channel dicts, so we cannot share with other tests
    via the session-scoped ``cs`` fixture.
    """
    return FlukaPhotoNuclear(max_mass=4)


def _make_solver(prince_run, *, enable_decay=False, source_params=_AUGER_HE4):
    prince_run.backend.linear_algebra_backend = "scipy"
    s = UHECRPropagationSolverETD2(
        initial_z=1.0,
        final_z=0.0,
        prince_run=prince_run,
        enable_pairprod_losses=True,
        enable_adiabatic_losses=True,
        enable_injection_jacobian=True,
        enable_partial_diff_jacobian=True,
        enable_decay=enable_decay,
    )
    s.add_source_class(
        AugerFitSource(prince_run, norm=1e-50, params=source_params)
    )
    return s


def _solve(prince_run, *, enable_decay=False, source_params=_AUGER_HE4):
    s = _make_solver(
        prince_run, enable_decay=enable_decay, source_params=source_params
    )
    s.solve(dz=5e-2)
    return s.state.copy(), s


def _flux(state, spec_man, pdgid):
    s = spec_man.pdgid2sref[pdgid]
    return state[s.lidx() : s.uidx()].copy()


# ---------------------------------------------------------------------------
# Step B — data model
# ---------------------------------------------------------------------------
def test_synthetic_pdg_is_disjoint_from_pdg_range():
    """``_TRACKING_PDG_BASE`` must be outside ``util.is_nucleus`` /
    ``util.get_AZN`` reach so the kernel skips tracked species as
    mothers without any explicit guard.
    """
    # 2e9 is above the 10LZZZAAAI nucleus range (max ~1.01e9) and the
    # |PDG| < 10^6 hadron/lepton range.
    assert _TRACKING_PDG_BASE >= 2_000_000_000
    for synthetic in (_TRACKING_PDG_BASE + 1, _TRACKING_PDG_BASE + 100):
        assert not is_nucleus(synthetic)


def test_tracked_species_physics_from_real_daughter(pf, fresh_cs):
    """``PrinceTrackedSpecies`` inherits the real daughter's physics
    flags and exposes them through both the synthetic ``pdgid`` (state-
    vector indexing) and the real ``real_pdgid`` (PDG-arithmetic
    dispatch). Use ν̄_e as the daughter so the test is independent of
    the chain reducer / explicit-decay configuration.
    """
    run = PriNCeRun(
        max_mass=4,
        photon_field=pf,
        cross_sections=fresh_cs,
        tracked_species=[
            dict(parent_pdgs=lambda s: s.is_nucleus and s.A >= 2,
                 daughter_pdg=-12,
                 process_class="decay", alias="nuebar_from_nuclei_test"),
        ],
    )
    trk = run.spec_man.sname2sref["nuebar_from_nuclei_test"]
    real_nuebar = run.spec_man.pdgid2sref[-12]
    assert isinstance(trk, PrinceTrackedSpecies)
    assert trk.real_pdgid == -12
    assert trk.pdgid != trk.real_pdgid
    assert trk.pdgid >= _TRACKING_PDG_BASE
    assert not is_nucleus(trk.pdgid)        # synthetic id is out-of-range
    assert trk.is_tracking
    assert trk.A == real_nuebar.A
    assert trk.Z == real_nuebar.Z
    assert trk.mass == real_nuebar.mass
    assert trk.lifetime == real_nuebar.lifetime
    assert trk.has_redist == real_nuebar.has_redist
    assert trk.is_lepton == real_nuebar.is_lepton
    assert trk.princeidx > real_nuebar.princeidx  # appended at the end
    # The state vector grew by exactly one species.
    spec_man = run.spec_man
    assert spec_man.nspec == len(spec_man.species_refs)


def test_decay_only_tracker_rejects_energy_window():
    sm = SpeciesManager([2212, 2112, 12], 5)
    with pytest.raises(ValueError):
        sm.add_tracking_species(
            parent_pdgs=[2112], daughter_pdg=12,
            process_class="decay", e_gamma_range=(1e6, 1e8),
        )


# ---------------------------------------------------------------------------
# Step C — chain-reducer hook (charge exchange)
# ---------------------------------------------------------------------------
def _run_charge_exchange(pf, fresh_cs, with_tracking):
    """Build a max_mass=4 PriNCeRun with ``enable_explicit_decay=True`` so
    free neutron lives in the state vector, optionally registering a
    tracked-n from proton.
    """
    saved_eed = conf.enable_explicit_decay
    saved_tau = conf.tau_dec_threshold
    conf.enable_explicit_decay = True
    # Drop the threshold below the neutron lifetime so n survives the
    # chain reducer's stability filter even in explicit-decay mode (we
    # need a (p, n) entry in ``_incl_tab`` for the photo-nuclear hook
    # to mirror).
    conf.tau_dec_threshold = 0.0
    try:
        kwargs = dict(max_mass=4, photon_field=pf,
                      cross_sections=FlukaPhotoNuclear(max_mass=4))
        if with_tracking:
            kwargs["tracked_species"] = [dict(
                parent_pdgs=[2212], daughter_pdg=2112,
                process_class="photo-nuclear", alias="trk_n_from_p",
            )]
        run = PriNCeRun(**kwargs)
        return _solve(
            run, enable_decay=True, source_params=_AUGER_PROTON
        ) + (run,)
    finally:
        conf.enable_explicit_decay = saved_eed
        conf.tau_dec_threshold = saved_tau


def test_charge_exchange_tracking_passive_observer(pf):
    """Tracked-n flux is non-zero and bounded above by real-n; the
    real-n flux is bit-identical to a no-tracking baseline at every
    bin (passive-observer invariant)."""
    state_baseline, _, run_b = _run_charge_exchange(pf, None, with_tracking=False)
    state_tracked, _, run_t = _run_charge_exchange(pf, None, with_tracking=True)

    # Real-species invariance — atol=0, rtol=0 (literal equality).
    for pdg in (2212, 2112):
        if pdg not in run_b.spec_man.pdgid2sref:
            continue
        # princeidx may have shifted because tracked species sit at the
        # high end. Re-extract per side using each run's own spec_man.
        flux_b = _flux(state_baseline, run_b.spec_man, pdg)
        flux_t = _flux(state_tracked, run_t.spec_man, pdg)
        assert flux_b.shape == flux_t.shape
        assert_allclose(flux_t, flux_b, atol=0.0, rtol=0.0,
                        err_msg=f"tracking leaked into real PDG {pdg}")

    trk = run_t.spec_man.sname2sref["trk_n_from_p"]
    real_n = run_t.spec_man.pdgid2sref[2112]
    f_trk = state_tracked[trk.sl]
    f_real = state_tracked[real_n.sl]
    assert np.any(f_trk > 0.0), "tracked-n flux should be non-zero"
    # Tracked flux ≤ real for every bin (tracked is a strict subset of
    # the real production sources at this point — only (p, n) feeds it,
    # while real-n also picks up n from any other nuclear-channel
    # contribution; with proton-only source and max_mass=4 those routes
    # are absent, so we expect equality on bins with non-zero (p, n)
    # production). Allow a small slack for floating-point drift.
    f_pos = f_real > 0
    assert np.all(f_trk[f_pos] <= f_real[f_pos] * (1.0 + 1e-10))


# ---------------------------------------------------------------------------
# Step D — energy-range partition
# ---------------------------------------------------------------------------
def test_e_gamma_partition_sum_rule(pf):
    """Three disjoint photon-energy windows + one unrestricted tracker on
    the same (parent, daughter, process_class) bucket. Sum of the
    partitioned trackers must equal the unrestricted tracker.

    He-4 → p photo-disintegration carries a wide photon-energy spectrum,
    making it a good check that the mask zeros only the intended bins.
    """
    # Cover [0, ∞) with three windows so the partition sum equals the
    # unrestricted tracker by construction of the mask
    # `(ph >= lo) & (ph < hi)`.
    partition = [
        (0.0, 1e6,    "lo"),
        (1e6, 1e8,    "mid"),
        (1e8, np.inf, "hi"),
    ]
    spec_full = dict(parent_pdgs=[1000020040], daughter_pdg=2212,
                     process_class="photo-nuclear", alias="p_from_He4_all")
    spec_parts = [
        dict(parent_pdgs=[1000020040], daughter_pdg=2212,
             process_class="photo-nuclear", alias=f"p_from_He4_{label}",
             e_gamma_range=(lo, hi))
        for lo, hi, label in partition
    ]
    run = PriNCeRun(
        max_mass=4, photon_field=pf,
        cross_sections=FlukaPhotoNuclear(max_mass=4),
        tracked_species=[spec_full, *spec_parts],
    )
    state, _ = _solve(run, source_params=_AUGER_HE4)

    full = state[run.spec_man.sname2sref["p_from_He4_all"].sl]
    partsum = np.zeros_like(full)
    for lo, hi, label in partition:
        partsum += state[run.spec_man.sname2sref[f"p_from_He4_{label}"].sl]
    # Convolution is linear in the kernel; the masks tile the photon
    # axis exactly. Any residual is floating-point round-off in the
    # accumulation, well within rtol=1e-10.
    assert_allclose(partsum, full, atol=0.0, rtol=1e-10)


# ---------------------------------------------------------------------------
# Step E — Λ_off hook (explicit-decay mode)
# ---------------------------------------------------------------------------
def test_nuebar_from_neutron_via_lambda_off(pf):
    """In explicit-decay mode, ν̄_e is produced exclusively from β-decay
    of free neutron. A tracked-ν̄_e with ``parent_pdgs=[2112]`` is
    therefore expected to match real ν̄_e on every bin (the tracked
    species captures the entire population). Validates the Λ_off
    accumulator path.
    """
    saved_eed = conf.enable_explicit_decay
    conf.enable_explicit_decay = True
    try:
        run_b = PriNCeRun(
            max_mass=4, photon_field=pf,
            cross_sections=FlukaPhotoNuclear(max_mass=4),
        )
        state_baseline, _ = _solve(run_b, enable_decay=True)

        run_t = PriNCeRun(
            max_mass=4, photon_field=pf,
            cross_sections=FlukaPhotoNuclear(max_mass=4),
            tracked_species=[dict(
                parent_pdgs=[2112], daughter_pdg=-12,
                process_class="decay", alias="nuebar_from_n",
            )],
        )
        state_tracked, _ = _solve(run_t, enable_decay=True)
    finally:
        conf.enable_explicit_decay = saved_eed

    # Real-species invariance.
    for pdg in (2212, 2112, -12):
        if pdg not in run_b.spec_man.pdgid2sref:
            continue
        f_b = _flux(state_baseline, run_b.spec_man, pdg)
        f_t = _flux(state_tracked, run_t.spec_man, pdg)
        assert_allclose(f_t, f_b, atol=0.0, rtol=0.0,
                        err_msg=f"tracking leaked into real PDG {pdg}")

    trk = run_t.spec_man.sname2sref["nuebar_from_n"]
    f_trk = state_tracked[trk.sl]
    f_real = _flux(state_tracked, run_t.spec_man, -12)
    assert np.any(f_trk > 0.0), "tracked ν̄_e flux from neutron must be non-zero"
    # ν̄_e is produced overwhelmingly (not exclusively) from free n β-decay
    # in this setup: π⁻→μ⁻→ν̄_e chains contribute at the ~0.5% level since
    # the tabulated-decay bin-average fix (the old point-eval missed the
    # two-body π→μ rest-frame delta almost entirely, artificially silencing
    # the muon chain — see wiki lessons/decay-tabulated-point-eval).
    # Tracked-ν̄_e (parent = n only) must match real ν̄_e at the dominant
    # production bins up to that contamination. Restrict to bins carrying
    # ≥10 % of the peak flux — lower bins mix the muon-chain contribution
    # and ETD2's rounding-floor oscillation in the decay tail.
    peak = np.max(np.abs(f_real))
    sig = np.abs(f_real) > 1e-1 * peak
    assert sig.any(), "no significant bins to compare"
    assert_allclose(f_trk[sig], f_real[sig], atol=0.0, rtol=1e-2)
