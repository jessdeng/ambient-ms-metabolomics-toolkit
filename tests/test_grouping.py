"""Condition and colony of every spectrum (manifest, or folder layout)."""
import os, sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from shared.grouping import parse_path, resolve_design, write_manifest_template


def test_parse_path_layouts():
    assert parse_path('Green/S5_Green_A1T3.csv') == ('Green', 'S5_Green_A1')
    assert parse_path('Green/A1_T2.txt') == ('Green', 'A1')
    assert parse_path('Strain/Green/colonyB/inj_03.csv') == ('Strain/Green', 'colonyB')
    assert parse_path('Green/A1.csv') == ('Green', 'A1')       # no T<n>: whole stem
    with pytest.raises(ValueError):
        parse_path('lonely.csv')


NAMES = [f'{c}/{k}T{t}.csv' for c in ('A', 'B') for k in ('c1', 'c2') for t in (1, 2)]


def test_folder_design_and_checks(tmp_path):
    cond, col = resolve_design(str(tmp_path), NAMES)
    assert list(cond[:4]) == ['A'] * 4 and len(set(col)) == 4 and col[0] == 'A::c1'
    with pytest.raises(ValueError, match='fewer than 2 colonies'):
        resolve_design(str(tmp_path), [n for n in NAMES if 'c2' not in n or n.startswith('A')])
    with pytest.raises(ValueError, match='without 3 spectra'):
        resolve_design(str(tmp_path), NAMES, expected_tech_reps=3)
    with pytest.warns(UserWarning, match='its own colony'):
        resolve_design(str(tmp_path), [f'A/x{i}.csv' for i in range(2)] + [f'B/y{i}.csv' for i in range(2)])


def test_manifest_overrides_folder_layout(tmp_path):
    man = pd.DataFrame({'file_path': NAMES, 'class': ['A'] * 4 + ['B'] * 4,
                        'biological_replicate_group': ['k1', 'k1', 'k2', 'k2', 'k3', 'k3', 'k4', 'k4']})
    man.to_csv(tmp_path / 'metadata_manifest.csv', index=False)
    cond, col = resolve_design(str(tmp_path), NAMES)
    assert col[0] == 'A::k1' and col[-1] == 'B::k4'
    with pytest.raises(ValueError, match='missing from'):
        resolve_design(str(tmp_path), NAMES + ['B/new.csv'])


def test_manifest_template_roundtrip(tmp_path):
    for n in NAMES:
        (tmp_path / os.path.dirname(n)).mkdir(exist_ok=True)
        (tmp_path / n).write_text('mz,int\n100,1\n')
    df = write_manifest_template(str(tmp_path))
    assert set(df.technical_replicate) == {1, 2}
    cond, col = resolve_design(str(tmp_path), NAMES)
    assert len(set(col)) == 4
