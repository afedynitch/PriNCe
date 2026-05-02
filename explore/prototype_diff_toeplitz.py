"""Prototype the 2D Toeplitz reformulation for differential (diff) channels.

For each diff channel, the kernel block is:
    B[i_mo, i_da, j] = factor(i_mo, i_da) · ΔΔR̂_ch(xu, xl, yu, yl)

where:
    factor(i_mo, i_da) = (m_p / E_mo[i_mo]) · (Δec[i_mo] / Δec[i_da])
    ΔΔR̂(xu,xl,yu,yl) = R̂(xu,yu) - R̂(xl,yu) - R̂(xu,yl) + R̂(xl,yl)
    yu/yl = b_ph[j+1 / j] * E_mo[i_mo] / m_p
    xu/xl = b_cr[i_da+1 / i_da] / E_mo[i_mo]

If both grids are log-spaced with the same log-step δ, these reduce to:
    log_y indexed by (i_mo + j) ∈ [0, dcr+dph-1]
    log_x indexed by (i_da - i_mo) ∈ [-(dcr-1), dcr-1]

So the 2D antiderivative needs only (dcr+dph) × (2 dcr - 1) evaluations per channel
instead of 4 · dcr² · dph.

Verification:
  - same grid construction as `_init_matrices`
  - apply x_cut filter as in the original code
  - subtract nonel on diagonal as in the original code
  - compare element-by-element against the dense path

Run with:
    python explore/prototype_diff_toeplitz.py [--max-mass N]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import prince_cr.config as cfg


def setup(max_mass=4):
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
    return core.PriNCeRun(max_mass=max_mass, photon_field=pf, cross_sections=cs)


def build_dense(intp_diff_integral, intp_nonel, ecr, bcr, bph, delta_ec, delta_ph,
                m_pr, x_cut_low, has_nonel):
    """Reproduce exactly what `_init_matrices` does in the diff branch.

    Returns (res, cuts) where:
      res    - shape (dcr, dcr, dph) full tile, with x_cut filter applied (zeros outside)
      cuts   - shape (dcr, dcr) boolean mask of valid (i_mo, i_da) entries
    """
    dcr = ecr.size
    dph = bph.size - 1
    elims = np.vstack([bcr[:-1], bcr[1:]])
    plims = np.vstack([bph[:-1], bph[1:]])

    emo = ecr[:, None, None]
    eda = ecr[None, :, None]
    target_shape = np.ones((dcr, dcr, dph))

    xl = elims[0, None, :, None] / emo * target_shape
    xu = elims[1, None, :, None] / emo * target_shape
    delta_x = delta_ec[None, :, None] / emo

    yl = plims[0, None, None, :] * emo / m_pr * target_shape
    yu = plims[1, None, None, :] * emo / m_pr * target_shape
    delta_y = delta_ph[None, None, :] * emo / m_pr

    int_fac = (delta_ec[:, None, None] * delta_ph[None, None, :] / emo) * target_shape
    diff_fac = 1.0 / delta_x / delta_y

    # cut on x (the same way the original code does)
    cuts = np.logical_and(xl >= x_cut_low, xl <= 1)
    cuts = cuts[:, :, 0]   # cuts depend only on (i_mo, i_da), not j

    res_full = np.zeros((dcr, dcr, dph))

    # 4-corner antiderivative differences (only for cut entries)
    res = intp_diff_integral.ev(xu[cuts], yu[cuts])
    res -= intp_diff_integral.ev(xl[cuts], yu[cuts])
    res -= intp_diff_integral.ev(xu[cuts], yl[cuts])
    res += intp_diff_integral.ev(xl[cuts], yl[cuts])
    res *= diff_fac[cuts] * int_fac[cuts]
    res[res < 0] = 0.0

    res_full[cuts] = res

    # Nonel subtraction on the diagonal i_mo == i_da
    if has_nonel:
        intp_nonel_antid = intp_nonel.antiderivative()
        diag_idx = np.arange(dcr)
        # Use only valid (cuts[i, i]) entries
        diag_valid = cuts[diag_idx, diag_idx]
        i_keep = diag_idx[diag_valid]
        if i_keep.size > 0:
            yu_d = yu[i_keep, i_keep, :]
            yl_d = yl[i_keep, i_keep, :]
            df = diff_fac[i_keep, i_keep, :]
            inf = int_fac[i_keep, i_keep, :]
            sub = (intp_nonel_antid(yu_d) - intp_nonel_antid(yl_d)) * df * inf
            res_full[i_keep, i_keep, :] -= sub

    return res_full, cuts


def build_toeplitz(intp_diff_integral, intp_nonel, ecr, bcr, bph, delta_ec, delta_ph,
                   m_pr, x_cut_low, has_nonel):
    """2D Toeplitz path."""
    dcr = ecr.size
    dph = bph.size - 1

    log_step_e = np.log10(ecr[1] / ecr[0])
    log_step_p = np.log10(bph[1] / bph[0])
    log_step_b = np.log10(bcr[1] / bcr[0])
    assert np.isclose(log_step_e, log_step_p), \
        f"cr/ph log-step mismatch: {log_step_e} vs {log_step_p}"
    assert np.isclose(log_step_e, log_step_b), \
        f"cr center/bin log-step mismatch: {log_step_e} vs {log_step_b}"

    log_step = log_step_e

    # 1D y-grid: y[k] = ecr[0] * bph[0] / m_p * 10^(k δ), k = 0..dcr+dph-1
    y0 = ecr[0] * bph[0] / m_pr
    k_y = np.arange(dcr + dph)
    y_grid = y0 * 10 ** (k_y * log_step)

    # 1D x-grid: x[k] = bcr[0] / ecr[0] * 10^(k δ), k = 0..2*dcr-1
    # Range of (i_da - i_mo + j) for j∈{0,1}: [-(dcr-1), dcr] → length 2 dcr
    x0 = bcr[0] / ecr[0]
    k_x = np.arange(2 * dcr)
    # (i_da - i_mo + 0) corresponds to xl, (+1) to xu
    # Map: (i_da - i_mo) ∈ [-(dcr-1), dcr-1] → k = i_da - i_mo + (dcr - 1) → range [0, 2 dcr - 2]
    # We need k+1 too, so we sample k = 0..2 dcr - 1 (length 2 dcr).
    x_grid = x0 * 10 ** ((k_x - (dcr - 1)) * log_step)

    # Sample the 2D antiderivative on the (y_grid, x_grid) outer product
    # intp_diff_integral.ev(x, y) takes 1D arrays and returns 1D pointwise result.
    # We want a 2D grid.
    Y, X = np.meshgrid(y_grid, x_grid, indexing="ij")  # both shape (dcr+dph, 2dcr)
    R = intp_diff_integral.ev(X.ravel(), Y.ravel()).reshape(Y.shape)

    # 4-corner difference: ΔΔR̂[a, b] = R[a+1, b+1] - R[a+1, b] - R[a, b+1] + R[a, b]
    # a in [0, dcr+dph-2], b in [0, 2dcr-2]
    ddR = R[1:, 1:] - R[1:, :-1] - R[:-1, 1:] + R[:-1, :-1]
    # shape (dcr+dph-1, 2dcr-1)

    # Build factor(i_mo, i_da):
    #   m_pr/emo[i_mo] · delta_ec[i_mo]/delta_ec[i_da]
    fac_mo = m_pr / ecr  # (dcr,)
    fac_ratio = delta_ec[:, None] / delta_ec[None, :]   # (dcr, dcr) ; per (i_mo, i_da)
    factor_mo_da = fac_mo[:, None] * fac_ratio  # (dcr, dcr)  per (i_mo, i_da)

    # Index map:
    #   a = i_mo + j  ∈ [0, dcr+dph-2]
    #   b = i_da - i_mo + (dcr - 1)  ∈ [0, 2 dcr - 2]
    i_mo = np.arange(dcr)[:, None, None]
    i_da = np.arange(dcr)[None, :, None]
    j = np.arange(dph)[None, None, :]
    a = i_mo + j     # (dcr, 1, dph) broadcast → effectively (dcr, dcr, dph)
    b = i_da - i_mo + (dcr - 1)  # (dcr, dcr, 1)

    # Broadcast to full (dcr, dcr, dph) shape
    a_full = np.broadcast_to(a, (dcr, dcr, dph))
    b_full = np.broadcast_to(b, (dcr, dcr, dph))

    res_full = factor_mo_da[:, :, None] * ddR[a_full, b_full]
    # Clip negative values to match dense path (res[res<0] = 0 done before nonel)
    res_full = np.where(res_full < 0, 0.0, res_full)

    # Apply x_cut filter (zero out)
    elims_l = bcr[:-1]
    xl = elims_l[None, :] / ecr[:, None]   # (dcr, dcr)
    cuts2d = np.logical_and(xl >= x_cut_low, xl <= 1)
    res_full = np.where(cuts2d[:, :, None], res_full, 0.0)

    # Nonel subtraction on diagonal — same path, same indexing
    if has_nonel:
        # Use the bc Toeplitz trick for the 1D nonel antiderivative
        intp_nonel_antid = intp_nonel.antiderivative()
        R_nonel = intp_nonel_antid(y_grid)  # (dcr+dph,)
        dR_nonel = np.diff(R_nonel)         # (dcr+dph-1,)

        # On the diagonal: i_mo == i_da → b = dcr - 1, factor reduces to m_p/ecr[i_mo]
        # B_nonel[i, j] = (m_p/ecr[i]) · dR_nonel[i + j]
        i_idx = np.arange(dcr)[:, None]
        j_idx = np.arange(dph)[None, :]
        B_nonel = (m_pr / ecr[:, None]) * dR_nonel[i_idx + j_idx]  # (dcr, dph)

        diag_valid = cuts2d[np.arange(dcr), np.arange(dcr)]
        for i in np.where(diag_valid)[0]:
            res_full[i, i, :] -= B_nonel[i]

    return res_full, cuts2d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-mass", type=int, default=4)
    ap.add_argument("--n-channels", type=int, default=20,
                    help="limit number of diff channels for fast iteration")
    args = ap.parse_args()

    print(f"Building PriNCeRun(max_mass={args.max_mass})...")
    t0 = time.perf_counter()
    run = setup(args.max_mass)
    print(f"  ... {time.perf_counter() - t0:.1f}s")

    cs = run.cross_sections
    resp = cs.resp

    from prince_cr.data import PRINCE_UNITS

    m_pr = PRINCE_UNITS.m_proton
    ecr = run.cr_grid.grid
    bcr = run.cr_grid.bins
    bph = run.ph_grid.bins
    delta_ec = run.cr_grid.widths
    delta_ph = run.ph_grid.widths

    diff_pairs = list(cs.known_diff_channels)
    if args.n_channels > 0:
        diff_pairs = diff_pairs[:args.n_channels]
    print(f"\nValidating {len(diff_pairs)} diff channels.")

    max_rel_err = 0.0
    max_abs_err = 0.0
    t_dense = 0.0
    t_toep = 0.0

    for moid, daid in diff_pairs:
        intp_diff_int = resp.incl_diff_intp_integral[(moid, daid)]
        has_nonel = (moid == daid) and (moid in resp.nonel_intp)
        intp_nonel = resp.nonel_intp.get(moid)
        x_cut_low = cfg.x_cut_proton if daid == 101 else cfg.x_cut

        t0 = time.perf_counter()
        dense, _ = build_dense(intp_diff_int, intp_nonel, ecr, bcr, bph,
                                delta_ec, delta_ph, m_pr, x_cut_low, has_nonel)
        t_dense += time.perf_counter() - t0

        t0 = time.perf_counter()
        toep, _ = build_toeplitz(intp_diff_int, intp_nonel, ecr, bcr, bph,
                                  delta_ec, delta_ph, m_pr, x_cut_low, has_nonel)
        t_toep += time.perf_counter() - t0

        denom = np.maximum(np.abs(dense), 1e-30)
        rel = np.abs(dense - toep) / denom
        rel[dense == 0] = 0.0
        max_rel_err = max(max_rel_err, rel.max())
        max_abs_err = max(max_abs_err, np.abs(dense - toep).max())

    print(f"\nResults across {len(diff_pairs)} channels:")
    print(f"  max rel err: {max_rel_err:.3e}")
    print(f"  max abs err: {max_abs_err:.3e}")
    print(f"  dense build time:    {t_dense * 1000:8.1f} ms")
    print(f"  toeplitz build time: {t_toep * 1000:8.1f} ms  ({t_dense / max(t_toep, 1e-12):.1f}× faster)")

    dcr, dph = ecr.size, bph.size - 1
    n = len(diff_pairs)
    dense_bytes = n * dcr * dcr * dph * 8
    toep_bytes = n * (dcr + dph) * (2 * dcr - 1) * 8
    print(f"\nPer-channel storage (diff):")
    print(f"  dense:    {dcr * dcr * dph * 8:,} bytes/ch")
    print(f"  toeplitz: {(dcr + dph) * (2 * dcr - 1) * 8:,} bytes/ch  "
          f"({(dcr * dcr * dph) / ((dcr + dph) * (2 * dcr - 1)):.1f}× smaller)")
    print(f"  for {n} channels: {dense_bytes / 1e6:.1f} MB → {toep_bytes / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
