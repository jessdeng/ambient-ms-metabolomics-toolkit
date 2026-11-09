"""Tests for rank-based agreement (validated/agreement.py)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from validation.agreement import (jaccard, expected_jaccard_random, pairwise_agreement,
                                 vip_threshold_analysis, family_structure)


def test_expected_random_jaccard_matches_simulation():
    rng = np.random.default_rng(0); p, n = 500, 50
    sim = np.mean([jaccard(rng.choice(p, n, replace=False), rng.choice(p, n, replace=False)) for _ in range(4000)])
    assert abs(expected_jaccard_random(n, p) - sim) < 0.003


def test_pairwise_agreement_identity_and_reversal():
    rng = np.random.default_rng(1); v = rng.normal(size=300)
    df = pairwise_agreement({'a': v, 'b': v.copy(), 'c': -v}, ns=(10,))
    ab = df[(df.method_a == 'a') & (df.method_b == 'b')].iloc[0]
    ac = df[(df.method_a == 'a') & (df.method_b == 'c')].iloc[0]
    assert ab.jaccard_top10 == 1 and abs(ab.spearman - 1) < 1e-12
    assert ac.jaccard_top10 == 0 and abs(ac.spearman + 1) < 1e-12


def test_vip_threshold_fraction():
    vip = np.r_[np.full(10, 2.0), np.full(90, 0.5)]
    imp = np.r_[np.arange(10, 0, -1), np.zeros(90)].astype(float)
    r = vip_threshold_analysis(vip, {'m': imp}, ns=(10, 20)).iloc[0]
    assert r.retention_rate == 0.1 and r.frac_top10_inside_vip_gt_threshold == 1.0
    assert r.frac_top20_inside_vip_gt_threshold == 0.5


def test_family_structure_recovers_planted_families():
    rng = np.random.default_rng(2); base = rng.normal(size=(2, 400))
    imps = {f'{f}{i}': base[k] + 0.2 * rng.normal(size=400) for k, f in enumerate('XY') for i in range(3)}
    fam = {m: m[0] for m in imps}
    s, M = family_structure(pairwise_agreement(imps, ns=(25,)), fam)
    assert s['ari_empirical_vs_a_priori'] == 1.0 and s['mean_within_family'] > s['mean_between_family']
