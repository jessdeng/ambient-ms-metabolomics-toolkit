"""Tests for manifest-based colony grouping (audit addition)."""
import os, sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from shared.grouping import load_manifest_groups, resolve_groups

NAMES = ['Amber/AmbB5T1.csv', 'Amber/AmberB5T2.csv', 'Amber/AmberB5T3.csv',
         'White/W_B6T1.csv', 'White/W_B5T2.csv', 'White/W_B5T3.csv']
Y = ['Amber'] * 3 + ['White'] * 3


def _manifest(tmp_path, rows=None):
    rows = rows or [(n, c, 'B5') for n, c in zip(NAMES, Y)]
    p = tmp_path / 'metadata_manifest.csv'
    pd.DataFrame(rows, columns=['file_path', 'class', 'colony']).to_csv(p, index=False)
    return str(p)


def test_manifest_merges_misnamed_replicates(tmp_path):
    g = load_manifest_groups(_manifest(tmp_path), NAMES, y_labels=Y, expected_tech_reps=3)
    assert sorted(set(g)) == ['Amber::B5', 'White::B5']


def test_filename_rule_splits_the_same_colonies(tmp_path):
    from shared.grouping import make_groups
    assert len(set(make_groups(np.array(Y), NAMES, validate=False))) == 4


def test_class_mismatch_raises(tmp_path):
    rows = [(n, 'Amber', 'B5') for n in NAMES]
    with pytest.raises(ValueError, match='class mismatch'):
        load_manifest_groups(_manifest(tmp_path, rows), NAMES, y_labels=Y)


def test_missing_file_raises(tmp_path):
    with pytest.raises(ValueError, match='not in manifest'):
        load_manifest_groups(_manifest(tmp_path), NAMES + ['White/extra.csv'])


def test_tech_rep_count_enforced(tmp_path):
    rows = [(n, c, 'B5') for n, c in zip(NAMES, Y)]
    rows[0] = (NAMES[0], 'Amber', 'B6')
    with pytest.raises(ValueError, match='technical'):
        load_manifest_groups(_manifest(tmp_path, rows), NAMES, y_labels=Y, expected_tech_reps=3)


def test_resolve_groups_falls_back_without_manifest(tmp_path):
    g = resolve_groups(np.array(Y), NAMES, str(tmp_path), 'metadata_manifest.csv', verbose=False)
    assert len(set(g)) == 4          # filename rule, unchanged behaviour
    _manifest(tmp_path)
    g2 = resolve_groups(np.array(Y), NAMES, str(tmp_path), 'metadata_manifest.csv', verbose=False)
    assert len(set(g2)) == 2


def test_load_experiment_skips_root_manifest(tmp_path):
    from standard.preprocessing import load_experiment
    for cls in ('A', 'B'):
        for w in ('W1', 'W2'):
            d = tmp_path / cls; d.mkdir(exist_ok=True)
            pd.DataFrame({'mz': [100.0, 100.1, 100.2], 'int': [1.0, 2.0, 3.0]}).to_csv(d / f'{cls}_{w}T1.csv', index=False)
    pd.DataFrame({'file_path': ['A/x.csv'], 'class': ['A'], 'colony': ['W1']}).to_csv(tmp_path / 'metadata_manifest.csv', index=False)
    X, y, names, mz = load_experiment(str(tmp_path), validate=False)
    assert len(names) == 4 and all('/' in n for n in names)
