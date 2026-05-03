"""Regression test for `CrossSectionBase._reduce_channels`.

After a fresh `CompositeCrossSection` build, captures a fingerprint of
`_incl_tab` and `_incl_diff_tab` (post decay-chain reduction) and compares
against a golden snapshot on disk.

The fingerprint is per-channel `(shape, sum, min, max, ssum)`. Bit-exact
equality is expected for refactors that preserve iteration order and
arithmetic ops; rtol=0 / atol=0 catches even last-bit drift. Loosen if a
refactor legitimately reorders accumulator updates.

Regenerate with ``PRINCE_REGEN_REDUCE=1`` or by deleting the golden file
at ``tests/data/cs_reduced_tabs.pkl``. The first run with no golden writes
the file and skips.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pytest

GOLDEN_PATH = Path(__file__).resolve().parent / "data" / "cs_reduced_tabs.pkl"


def _build_fresh_cs():
    """Rebuild the FlukaPhotoNuclear cross section from scratch.

    The session-scoped `fluka` fixture caches across the test session;
    instantiating here lets us verify that a fresh build still reduces
    to the same channel set / values.
    """
    from prince_cr import cross_sections

    return cross_sections.FlukaPhotoNuclear()


def _array_fingerprint(arr):
    """Compact deterministic summary of an ndarray for regression diffing."""
    a = np.asarray(arr, dtype=np.float64)
    return {
        "shape": tuple(a.shape),
        "sum": float(a.sum()),
        "min": float(a.min()) if a.size else 0.0,
        "max": float(a.max()) if a.size else 0.0,
        "ssum": float((a * a).sum()),
    }


def _entry_fingerprint(value):
    """Tab values are either ndarrays or `(egrid, matrix)` tuples."""
    if isinstance(value, tuple):
        egrid, arr = value
        return ("tuple", _array_fingerprint(egrid), _array_fingerprint(arr))
    return ("array", _array_fingerprint(value))


def _snapshot_tabs(cs):
    incl = {k: _entry_fingerprint(v) for k, v in cs._incl_tab.items()}
    diff = {k: _entry_fingerprint(v) for k, v in cs._incl_diff_tab.items()}
    return {"incl": incl, "diff": diff}


def _assert_fingerprints_close(a, e, label):
    assert a[0] == e[0], f"{label}: kind mismatch (actual={a[0]} expected={e[0]})"
    for i in range(1, len(e)):
        a_i, e_i = a[i], e[i]
        assert a_i["shape"] == e_i["shape"], (
            f"{label}: shape mismatch {a_i['shape']} vs {e_i['shape']}"
        )
        for k in ("sum", "min", "max", "ssum"):
            np.testing.assert_allclose(
                a_i[k],
                e_i[k],
                rtol=1e-12,
                atol=0,
                err_msg=f"{label}: '{k}' drift",
            )


def _assert_tabs_match(actual, expected, label):
    actual_keys = set(actual.keys())
    expected_keys = set(expected.keys())
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    assert not missing, f"{label}: missing channels {sorted(missing)[:10]}"
    assert not extra, f"{label}: unexpected channels {sorted(extra)[:10]}"
    for key in sorted(expected_keys):
        _assert_fingerprints_close(actual[key], expected[key], f"{label}[{key}]")


def test_reduce_channels_regression():
    """Fresh FlukaPhotoNuclear build reduces to a stable channel set/values."""
    cs = _build_fresh_cs()
    snap = _snapshot_tabs(cs)

    regen = os.environ.get("PRINCE_REGEN_REDUCE") == "1"
    if regen or not GOLDEN_PATH.exists():
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(GOLDEN_PATH, "wb") as fh:
            pickle.dump(snap, fh, protocol=pickle.HIGHEST_PROTOCOL)
        pytest.skip(f"wrote golden snapshot to {GOLDEN_PATH}")

    with open(GOLDEN_PATH, "rb") as fh:
        golden = pickle.load(fh)

    _assert_tabs_match(snap["incl"], golden["incl"], "_incl_tab")
    _assert_tabs_match(snap["diff"], golden["diff"], "_incl_diff_tab")
