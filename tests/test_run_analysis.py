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
    monkeypatch.setattr(run_analysis.config, 'N_CV_REPEATS', 1)
    run_analysis.main()
    out = tmp_path / 'results' / 'sample_experiment'
    for f in ('feature_rankings', 'candidates', 'ranker_concordance', 'signal_groups',
              'pca_explained_variance', 'classifier_accuracy'):
        assert (out / f'{f}_sample_experiment.csv').exists(), f
    assert (out / 'run_manifest.json').exists()
    assert (out / 'classifier_comparison_sample_experiment.png').exists()
    ranks = pd.read_csv(out / 'feature_rankings_sample_experiment.csv')
    cand = pd.read_csv(out / 'candidates_sample_experiment.csv')
    assert {'screen_F', 'screen_q', 'vip_score', 'rf_score', 'frac_condition_plate',
            'colonies_consistent', 'technical_cv_pct', 'signal_group'} <= set(ranks.columns)
    assert len(cand) == ranks.candidate.sum() and {'apex_mz', 'baseline_share', 'related_ions'} <= set(cand.columns)
    assert not any(c in ranks.columns for c in ('p_value', 'q_value_BH', 'n_methods'))
