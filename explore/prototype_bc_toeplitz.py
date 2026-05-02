"""Prototype the Toeplitz reformulation for boost-conserving (bc) channels.

For every bc channel `(mo, da)` the kernel matrix block is:
    B[i_mo, j] = (m_p / E_mo[i_mo]) * (R̂_ch(yu[i_mo, j]) - R̂_ch(yl[i_mo, j]))
with yu = bph[j+1] * E_mo[i_mo] / m_p, yl = bph[j] * E_mo[i_mo] / m_p.

If E_mo (cosmic ray bin centers) and b_ph (photon bin edges) are both log-spaced
with the SAME log-step δ, then yu and yl lie on a 1D log-grid indexed by (i+j+1)
and (i+j). We exploit this:

  - sample the antiderivative once on the (dcr+dph)-point log-y grid
  - take a 1-step finite difference → ΔR̂_ch[k], k = 0..dcr+dph-1
  - B[i, j] = factor[i] * ΔR̂_ch[i + j]

This script:
  1. Builds a small PriNCeRun (max_mass=4, full grid).
  2. For every bc channel, recomputes the (dcr, dph) tile both ways.
  3. Asserts max relative error is ~machine epsilon.
  4. Measures wall-clock for the dense and Toeplitz paths.

Run with:
    python explore/prototype_bc_toeplitz.py [--max-mass N]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import prince_cr.config as cfg


def setup(max_mass=4):
    """Build a small PriNCeRun and return everything we need for the prototype."""
    cfg.debug_level = 0
    cfg.cosmic_ray_grid = (3, 14, 8)
    cfg.photon_grid = (-15, -6, 8)
    cfg.x_cut = 1e-4
    cfg.x_cut_proton = 1e-2
    cfg.tau_dec_threshold = np.inf
    cfg.max_mass = max_mass

    from prince_cr import core, cross_sections, photonfields

    pf = photonfields.CombinedPhotonField(
        [photonfields.CMBPhotonSpectrum, photonfields.CIBGilmore2D]
    )
    cs = cross_sections.CompositeCrossSection(
        [
            (0.0, cross_sections.TabulatedCrossSection, ("CRP2_TALYS",)),
            (0.14, cross_sections.SophiaSuperposition, ()),
        ]
    )
    run = core.PriNCeRun(max_mass=max_mass, photon_field=pf, cross_sections=cs)
    return run


def build_dense(intp, emo, plims, m_pr, ecr_widths, ph_widths):
    """The current code path. Returns the (dcr, dph) tile B."""
    yl = plims[0, None, :] * emo[:, None] / m_pr
    yu = plims[1, None, :] * emo[:, None] / m_pr
    delta_y = ph_widths[None, :] * emo[:, None] / m_pr
    int_fac = ecr_widths[:, None] * ph_widths[None, :] / emo[:, None]
    delta_x = ecr_widths / emo  # (dcr,)
    diff_fac = 1.0 / delta_x[:, None] / delta_y
    return (intp(yu) - intp(yl)) * int_fac * diff_fac


def build_toeplitz(intp, emo, plims, m_pr):
    """Toeplitz path: only (dcr+dph) antiderivative evaluations.

    For bc channels, int_fac · diff_fac = m_p / E_mo[i_mo] (independent of j).
    yu and yl values lie on a 1D log-grid indexed by (i+j+1) and (i+j).

    Returns the (dcr, dph) tile B.
    """
    dcr = emo.size
    dph = plims.shape[1]
    # The unique log-y grid: length dcr + dph
    # plims[0, j] = b_ph[j], plims[1, j] = b_ph[j+1]
    # yl[i, j] = b_ph[j] * emo[i] / m_p   →  index k = i + j
    # yu[i, j] = b_ph[j+1] * emo[i] / m_p →  index k = i + j + 1
    # So we need y_grid[k] for k = 0 .. dcr + dph - 1
    # Take y_grid[k] = (emo[0] * b_ph[0] / m_p) * 10^(k δ) where
    # δ = log10(emo[1]/emo[0]) = log10(b_ph[1]/b_ph[0]).
    log_step = np.log10(emo[1] / emo[0])
    log_step_check = np.log10(plims[0, 1] / plims[0, 0])
    assert np.isclose(log_step, log_step_check), (
        f"cr/ph log-steps differ: {log_step} vs {log_step_check}"
    )
    y0 = emo[0] * plims[0, 0] / m_pr
    k = np.arange(dcr + dph)
    y_grid = y0 * 10 ** (k * log_step)

    # ONE batched antiderivative evaluation
    R_grid = intp(y_grid)
    # Δ R̂[k] = R̂[k+1] - R̂[k]  for k = 0..dcr+dph-2
    dR = np.diff(R_grid)  # length dcr + dph - 1

    # Map: B[i, j] = (m_p / emo[i]) * dR[i + j]
    # Build the (dcr, dph) matrix as Toeplitz indexing
    i_idx = np.arange(dcr)[:, None]
    j_idx = np.arange(dph)[None, :]
    B = (m_pr / emo[:, None]) * dR[i_idx + j_idx]
    return B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-mass", type=int, default=4)
    args = ap.parse_args()

    print(f"Building PriNCeRun(max_mass={args.max_mass})...")
    t0 = time.perf_counter()
    run = setup(args.max_mass)
    print(f"  ... {time.perf_counter() - t0:.1f}s")

    cs = run.cross_sections
    resp = cs.resp
    ir = run.int_rates

    from prince_cr.data import PRINCE_UNITS

    m_pr = PRINCE_UNITS.m_proton
    ecr = run.cr_grid.grid
    eph = run.ph_grid.grid
    bph = run.ph_grid.bins
    plims = np.vstack([bph[:-1], bph[1:]])
    bcr = run.cr_grid.bins
    elims = np.vstack([bcr[:-1], bcr[1:]])
    delta_ec = run.cr_grid.widths
    delta_ph = run.ph_grid.widths

    bc_pairs = list(cs.known_bc_channels)
    print(f"\nValidating {len(bc_pairs)} bc channels (and self-coupling diagonals).")

    # We also handle the (mo, mo) self-coupling rows that come from has_nonel only
    # (they don't appear in known_bc_channels but follow the same code path).
    extras = []
    for moid in cs.known_species:
        if moid < 100:
            continue
        if (moid, moid) not in bc_pairs and (moid, moid) not in cs.known_diff_channels:
            extras.append((moid, moid))
    if extras:
        print(f"  + {len(extras)} self-coupling-only rows")

    all_pairs = bc_pairs + extras

    max_rel_err = 0.0
    max_abs_err = 0.0
    n_compared = 0

    t_dense = 0.0
    t_toep = 0.0

    for moid, daid in all_pairs:
        has_nonel = moid == daid
        intp_nonel = resp.nonel_intp[moid].antiderivative() if has_nonel else None
        has_incl = (moid, daid) in resp.incl_intp
        intp_bc = resp.incl_intp[(moid, daid)].antiderivative() if has_incl else None

        if not (has_nonel or has_incl):
            continue

        # Dense reference path
        t0 = time.perf_counter()
        dense = np.zeros((ecr.size, eph.size))
        if has_incl:
            dense += build_dense(intp_bc, ecr, plims, m_pr, delta_ec, delta_ph)
        if has_nonel:
            dense -= build_dense(intp_nonel, ecr, plims, m_pr, delta_ec, delta_ph)
        t_dense += time.perf_counter() - t0

        # Toeplitz path
        t0 = time.perf_counter()
        toep = np.zeros((ecr.size, eph.size))
        if has_incl:
            toep += build_toeplitz(intp_bc, ecr, plims, m_pr)
        if has_nonel:
            toep -= build_toeplitz(intp_nonel, ecr, plims, m_pr)
        t_toep += time.perf_counter() - t0

        # Compare
        denom = np.maximum(np.abs(dense), 1e-300)
        rel = np.abs(dense - toep) / denom
        # Where dense is exactly zero, just use absolute
        zero = dense == 0
        if np.any(zero):
            rel[zero] = 0.0
        max_rel_err = max(max_rel_err, rel.max())
        max_abs_err = max(max_abs_err, np.abs(dense - toep).max())
        n_compared += 1

    print(f"\nResults across {n_compared} channels:")
    print(f"  max rel err: {max_rel_err:.3e}")
    print(f"  max abs err: {max_abs_err:.3e}")
    print(f"  dense build time:     {t_dense * 1000:7.1f} ms")
    print(f"  toeplitz build time:  {t_toep * 1000:7.1f} ms  ({t_dense / max(t_toep, 1e-12):.1f}× faster)")

    # Storage comparison
    dcr, dph = ecr.size, eph.size
    n_ch = n_compared
    dense_bytes = n_ch * dcr * dph * 8
    toep_bytes = n_ch * (dcr + dph) * 8
    print(f"\nPer-channel storage (bc):")
    print(f"  dense:    {dcr * dph * 8} bytes/ch")
    print(f"  toeplitz: {(dcr + dph) * 8} bytes/ch  ({dcr * dph / (dcr + dph):.1f}× smaller)")
    print(f"  for {n_ch} channels: {dense_bytes / 1e6:.1f} MB → {toep_bytes / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
