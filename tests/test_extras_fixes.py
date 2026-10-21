"""Regression tests for the corrected extras functions and the Figure 3 script."""
import os, re, sys
import numpy as np
import pytest
ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path[:0] = [os.path.join(ROOT, 'src'), ROOT]
from standard import extras
from standard.extras import VIPFilter, run_reproducibility_report
from shared.classifier_comparison_standard import lda_classify, make_preprocessor


def test_reproducibility_report_binary_and_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(extras.config, 'TOP_N_FEATURES', 10, raising=False)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 40)); y = np.repeat(['a', 'b'], 15); X[y == 'a', :4] += 1.5
    df = run_reproducibility_report(X, y, np.arange(40.0), 'test', str(tmp_path))
    rows = df.set_index('Model')
    assert rows.loc['Ridge', 'Shared features'] == 10                    # binary coef_ handled
    assert not rows.loc['Ridge', 'Seed-dependent'] and 'seed-independent' in rows.loc['Ridge', 'Top 10 overlap (%)']
    assert 'Gradient Boosting' in rows.index and rows.loc['Random Forest', 'Seed-dependent']


def test_nested_vip_filter_removes_selection_bias():
    rng = np.random.default_rng(1)
    y = np.repeat(list('ABC'), 12); g = np.repeat([f'k{i}' for i in range(12)], 3)
    X = rng.lognormal(size=(36, 400))                                  # no class signal at all
    steps = make_preprocessor(normalization='tic', log_transform='log10', scaling='autoscale', variance_percentile=0,
                              abundance_percentile=0, prevalence_threshold=0, snr_floor_enabled=False)
    nested = list(steps) + [('vip_filter', VIPFilter(threshold=1.0))]
    acc_nested = lda_classify(X, y, n_splits=3, groups=g, prep_steps=nested, n_repeats=3)[0].mean()
    # leaky reference: select on all data first, then cross-validate
    from standard.preprocessing import preprocess
    from standard.pipeline import compute_vip_1comp
    Xp = preprocess(X, 'tic', 'log10', 'autoscale'); keep = compute_vip_1comp(Xp, y) > 1.0
    scal = make_preprocessor(normalization='none', log_transform='none', scaling='none', variance_percentile=0,
                             abundance_percentile=0, prevalence_threshold=0, snr_floor_enabled=False)
    acc_leaky = lda_classify(Xp[:, keep] - Xp[:, keep].min() + 1, y, n_splits=3, groups=g, prep_steps=scal, n_repeats=3)[0].mean()
    assert acc_nested < acc_leaky and acc_nested < 0.6


def test_vip_filter_keeps_minimum_features():
    X = np.random.default_rng(2).normal(size=(12, 5)); y = np.repeat([0, 1, 2], 4)
    f = VIPFilter(threshold=100.0, min_features=2).fit(X, y)
    assert f.transform(X).shape[1] == 2


def test_figure3_script_maps_all_vote_counts():
    src = open(os.path.join(ROOT, 'src', 'scripts', 'vip_vs_nmethods.py')).read()
    m = re.search(r"df\['n_methods'\]\.map\((\{[^}]*\})\)", src)
    assert m and set(eval(m.group(1))) == {1, 2, 3, 4, 5, 6}
