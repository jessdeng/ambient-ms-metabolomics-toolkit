"""Tests for evidence-based signal grouping (shared/feature_groups.py)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from standard.preprocessing import bin_features
from shared.feature_groups import build_signal_groups, aggregate_to_groups, flag_relations

RAW_MZ = np.round(np.arange(150.0, 170.0001, 0.1), 1)
GROUPS = np.repeat(np.arange(10), 3)          # 10 colonies x 3 technical replicates


def _peak(c, height, fwhm=0.7):
    s = fwhm / 2.3548
    return height * np.exp(-0.5 * ((RAW_MZ - c) / s) ** 2)


BACKGROUND_IONS = (151.0, 153.0, 155.0, 157.0, 163.0, 165.0, 167.0, 169.0)


def _spectra(make, seed=0):
    """Spectra with independent background ions dominating the TIC (as in real
    profiles), Gaussian baseline noise, and a colony-level abundance for the
    signal under test."""
    rng = np.random.default_rng(seed)
    col = rng.lognormal(0, 0.6, 10)[GROUPS]                  # colony-level abundance
    out = []
    for k in range(len(GROUPS)):
        bg = sum(_peak(c, 5e5 * rng.lognormal(0, 0.3)) for c in BACKGROUND_IONS)
        out.append(make(rng, col[k]) + bg + np.abs(rng.normal(50, 15, RAW_MZ.size)))
    return np.vstack(out), rng


def _decision(pairs, lo_mz):
    sub = pairs[(pairs.mz_i > lo_mz - 0.3) & (pairs.mz_i < lo_mz + 0.3)]
    return None if sub.empty else sub.decision.iloc[0]


def test_split_peak_is_merged_and_isotope_is_flagged_not_merged():
    def make(rng, a):
        return (_peak(160.0, 1e5 * a * rng.lognormal(0, 0.05)) +      # split across 160.0 edge
                _peak(161.0, 1.2e4 * a * rng.lognormal(0, 0.05)))     # +1 isotopologue
    raw, _ = _spectra(make)
    Xb, mzb = bin_features(raw, RAW_MZ, 0.5)
    m, pairs = build_signal_groups(Xb, mzb, raw, RAW_MZ, GROUPS, mz0=150.0)
    assert _decision(pairs, 159.75) == 'merged'
    g = m.loc[(m.mz_bin > 159.6) & (m.mz_bin < 160.3), 'signal_group'].unique()
    assert len(g) == 1
    assert abs(m.loc[m.signal_group == g[0], 'signal_mz'].iloc[0] - 160.0) < 0.15
    Xg, gids, smz = aggregate_to_groups(Xb, m)
    rel = flag_relations(smz, Xg, GROUPS)
    iso = rel[rel.relation.str.contains('isotopologue')]
    assert ((iso.mz_i - 160.0).abs() < 0.15).any() and ((iso.mz_j - 161.0).abs() < 0.2).any()
    g161 = m.loc[(m.mz_bin > 160.6) & (m.mz_bin < 161.3), 'signal_group'].unique()
    assert set(g161).isdisjoint(set(g))                       # isotope kept separate


def test_drifting_covarying_pair_is_not_merged():
    # Strongly varying abundance (so the two bins co-vary) plus per-spectrum apex
    # drift of +/-0.2 Da (so the split ratio is unstable): must NOT be merged.
    def make(rng, a):
        return _peak(160.0 + rng.uniform(-0.2, 0.2), 1e5 * a ** 3)
    raw, _ = _spectra(make, seed=1)
    Xb, mzb = bin_features(raw, RAW_MZ, 0.5)
    m, pairs = build_signal_groups(Xb, mzb, raw, RAW_MZ, GROUPS, mz0=150.0)
    assert _decision(pairs, 159.75) == 'co-varying, unresolved'
    g = m.loc[(m.mz_bin > 159.6) & (m.mz_bin < 160.3), 'signal_group'].unique()
    assert len(g) == 2


def test_independent_neighbours_are_not_candidates():
    def make(rng, a):
        return _peak(159.6, 1e5 * rng.lognormal(0, 0.8)) + _peak(160.4, 1e5 * rng.lognormal(0, 0.8))
    raw, _ = _spectra(make, seed=2)
    Xb, mzb = bin_features(raw, RAW_MZ, 0.5)
    _, pairs = build_signal_groups(Xb, mzb, raw, RAW_MZ, GROUPS, mz0=150.0)
    assert _decision(pairs, 159.75) is None


def test_nominal_bins_keep_unit_resolution_peak_in_one_bin():
    from standard.preprocessing import bin_features_nominal
    mz = np.round(np.arange(150.0, 170.0001, 0.1), 1)
    for centre in (160.0, 160.2, 159.9):
        x = np.exp(-0.5 * ((mz - centre) / (0.8 / 2.3548)) ** 2)
        Xb, lab = bin_features_nominal(x[None, :], mz, offset=0.6)
        assert lab[np.argmax(Xb[0])] == 160.0
        assert Xb[0, lab == 160.0][0] / Xb.sum() > 0.8     # Gaussian, FWHM 0.8, apex <= 0.2 Da off-centre
    Xb, lab = bin_features_nominal(np.ones((2, len(mz))), mz, offset=0.6)
    assert np.allclose(Xb.sum(1), len(mz)) and np.all(np.diff(lab) == 1)
