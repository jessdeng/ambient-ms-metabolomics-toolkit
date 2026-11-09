"""Tests for the step-10 spectral-evidence helpers (synthetic spectra, known answers)."""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from validation.evidence import baseline_corrected_area, feature_evidence, feature_index, normalised_intensity


def _synthetic(peak_height, base=10.0, apex=200.1):
    """45 spectra (5 conditions x 3 colonies x 3 technical); one Gaussian peak on a flat baseline."""
    mz = np.round(np.arange(195.0, 205.0, 0.1), 1)
    y = np.repeat(list('ABCDE'), 9); g = np.repeat([f'k{i}' for i in range(15)], 3)
    X = np.full((45, mz.size), base)
    shape = np.exp(-0.5 * ((mz - apex) / 0.35) ** 2)
    X += np.asarray(peak_height)[:, None] * shape[None, :]
    lo = np.arange(195.0, 205.0, 0.5); Xb = np.stack([X[:, (mz >= a) & (mz < a + 0.5)].sum(1) for a in lo], 1)
    gm = pd.DataFrame(dict(bin_lo=lo, signal_mz=lo + 0.25, signal_group=np.arange(lo.size)))
    gm.loc[(gm.bin_lo == 199.5) | (gm.bin_lo == 200.0), 'signal_group'] = -1
    gm.loc[gm.signal_group == -1, 'signal_mz'] = 200.0
    return dict(raw_mz=mz, X_raw=X, X_binned=Xb, mz_binned=lo + 0.25, y=y, groups=g, group_map=gm)


def test_net_area_recovers_peak_above_flat_baseline():
    h = np.repeat([100.0, 200, 300, 400, 500], 9)
    d = _synthetic(h)
    S = baseline_corrected_area(d, 199.75, [(196.0, 197.5), (203.0, 204.5)])
    mz = d['raw_mz']; inb = (mz >= 199.5) & (mz < 200.5)
    expected_net = h * np.exp(-0.5 * ((mz[inb] - 200.1) / 0.35) ** 2).sum()
    assert np.allclose(S.net_raw, expected_net, rtol=1e-3)           # flank points carry ~no peak tail
    assert np.allclose(S.gross_raw - S.net_raw, 10.0 * inb.sum(), rtol=1e-3)
    assert (S.base_frac < 1).all() and (S.net_raw > 0).all()


def test_no_peak_gives_nonpositive_net_and_unit_baseline_share():
    d = _synthetic(np.zeros(45))
    S = baseline_corrected_area(d, 199.75, [(196.0, 197.5), (203.0, 204.5)])
    assert np.allclose(S.net_raw, 0) and np.allclose(S.base_frac, 1)


def test_feature_evidence_apex_detection_and_variance_partition():
    rng = np.random.default_rng(3)
    cond = np.repeat([100.0, 200, 400, 800, 1600], 9)
    colony = np.repeat(rng.lognormal(0, 0.05, 15), 3)
    tech = rng.lognormal(0, 0.02, 45)
    d = _synthetic(cond * colony * tech)
    j = feature_index(d, 199.75); assert j == int(np.argmin(np.abs(d['mz_binned'] - 199.75)))
    out, prof = feature_evidence(d, 199.75)
    assert abs(out['apex_median'] - 200.1) < 0.051 and out['detected_frac'] == 1
    assert out['frac_condition_plate'] > 0.9 and out['highest_condition'] == 'E' and out['lowest_condition'] == 'A'
    assert out['colony_pattern_spearman_tic_vs_raw'] > 0.9
    assert np.allclose(normalised_intensity(d, j, 'tic'), d['X_binned'][:, j] / d['X_raw'].sum(1))
