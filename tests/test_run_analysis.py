"""End to end on the bundled demo experiment (in a temporary copy of data/)."""
import os, shutil, sys
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path[:0] = [os.path.join(ROOT, 'src'), ROOT]


def test_demo_run_writes_the_expected_tables(tmp_path, monkeypatch):
    shutil.copytree(os.path.join(ROOT, 'data', 'sample_experiment'), tmp_path / 'data' / 'sample_experiment')
    from standard import run_analysis
    monkeypatch.setattr(run_analysis, '_ROOT', str(tmp_path))
    monkeypatch.setenv('TOOLKIT_EXPERIMENT', 'sample_experiment')
    monkeypatch.setattr(run_analysis.config, 'RF_SEEDS', 2)
    monkeypatch.setattr(run_analysis.config, 'GB_SEEDS', 2)
    monkeypatch.setattr(run_analysis.config, 'N_CV_REPEATS', 1)
    monkeypatch.setattr(run_analysis.config, 'N_PERMUTATIONS', 9)
    monkeypatch.setattr(run_analysis.config, 'N_JOBS', 1)
    run_analysis.main()
    out = tmp_path / 'results' / 'sample_experiment'
    for f in ('feature_rankings', 'candidates', 'ranker_concordance', 'signal_groups',
              'pca_explained_variance', 'classifier_accuracy', 'feature_overlap',
              'permutation_test'):
        assert (out / f'{f}_sample_experiment.csv').exists(), f
    assert (out / 'run_manifest.json').exists()
    for f in ('classifier_comparison', 'corroboration', 'ranker_agreement', 'condition_abundance',
              'vip_scores', 'spectrum_features', 'pca'):
        assert (out / f'{f}_sample_experiment.png').exists(), f
    summary = (out / 'summary_sample_experiment.txt').read_text()
    assert 'Method comparison' in summary and 'Permutation test' in summary
    ranks = pd.read_csv(out / 'feature_rankings_sample_experiment.csv')
    cand = pd.read_csv(out / 'candidates_sample_experiment.csv')
    assert {'screen_F', 'screen_q', 'vip_score', 'rf_score', 'frac_condition_plate',
            'colonies_consistent', 'technical_cv_pct', 'signal_group'} <= set(ranks.columns)
    assert {'n_methods', 'consensus', 'selected_by', 'agg_worst_family_rank', 'in_svm_top',
            'in_lr_top', 'in_gb_top', 'mean_margin'} <= set(ranks.columns)
    assert ranks.n_methods.between(0, 6).all()
    for m in ('vip', 'ridge', 'lr', 'svm', 'rf', 'gb'):                # no top list is padded
        assert (ranks.loc[ranks[f'in_{m}_top'], f'{m}_score'] > 0).all(), m
    assert ranks.in_gb_top.sum() < 50 and 'fewer than 50 bins have a non-zero score' in summary
    assert len(cand) == (ranks.candidate | ranks.consensus).sum()
    assert {'apex_mz', 'baseline_share', 'related_ions'} <= set(cand.columns)
    overlap = pd.read_csv(out / 'feature_overlap_sample_experiment.csv')
    assert ((overlap.n_methods >= 2) | overlap.consensus).all()
    perm = pd.read_csv(out / 'permutation_test_sample_experiment.csv')
    assert perm.p_value.between(1 / 10, 1).all() and len(perm) == 5
    assert not any(c in ranks.columns for c in ('p_value', 'q_value_BH'))


def test_all_experiments_writes_the_cross_experiment_outputs(tmp_path, monkeypatch):
    for name in ('exp_a', 'exp_b'):
        shutil.copytree(os.path.join(ROOT, 'data', 'sample_experiment'), tmp_path / 'data' / name)
    shutil.copytree(os.path.join(ROOT, 'data', 'sample_experiment'), tmp_path / 'data' / 'sample_experiment')
    from standard import run_analysis
    monkeypatch.setattr(run_analysis, '_ROOT', str(tmp_path))
    monkeypatch.setenv('TOOLKIT_EXPERIMENT', 'all')
    for key, value in (('RF_SEEDS', 2), ('GB_SEEDS', 2), ('N_JOBS', 1), ('N_CV_REPEATS', 1),
                       ('RUN_PERMUTATION', False)):
        monkeypatch.setattr(run_analysis.config, key, value)
    assert run_analysis._experiments() == ['exp_a', 'exp_b']          # the bundled demo is skipped
    run_analysis.main()
    res = tmp_path / 'results'
    table = pd.read_csv(res / 'classifier_accuracy_all_experiments.csv', index_col=0)
    assert list(table.columns) == ['exp_a', 'exp_b'] and len(table) == 6
    assert (res / 'candidates_all_experiments.csv').exists()
    assert (res / 'pca_overview.png').exists() and (res / 'vip_vs_nmethods.png').exists()
    assert not (res / 'exp_a' / 'permutation_test_exp_a.csv').exists()
