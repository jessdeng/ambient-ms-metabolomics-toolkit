"""
Tests for runtime.py — out-of-the-box experiment resolution (C-3) and the
run-provenance manifest (Section 3).
"""
import os
import sys
import json

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from shared.runtime import resolve_experiment_dir, write_run_manifest  # noqa: E402


def _make_experiment(base, name, with_data=True):
    d = os.path.join(base, 'data', name)
    os.makedirs(d, exist_ok=True)
    if with_data:
        with open(os.path.join(d, 's1.csv'), 'w') as f:
            f.write('mz,intensity\n100,5\n')
    return d


# --------------------------------------------------------------------------- #
# C-3: resolve_experiment_dir                                                   #
# --------------------------------------------------------------------------- #
def test_resolve_uses_configured_when_present(tmp_path):
    base = str(tmp_path)
    _make_experiment(base, 'my_real_exp')
    _make_experiment(base, 'sample_experiment')
    d, name, fallback = resolve_experiment_dir(base, 'my_real_exp')
    assert name == 'my_real_exp' and fallback is False
    assert d.endswith(os.path.join('data', 'my_real_exp'))


def test_resolve_falls_back_on_placeholder(tmp_path):
    base = str(tmp_path)
    _make_experiment(base, 'sample_experiment')
    d, name, fallback = resolve_experiment_dir(base, 'your_experiment_folder')
    assert name == 'sample_experiment' and fallback is True


def test_resolve_falls_back_on_missing_folder(tmp_path):
    base = str(tmp_path)
    _make_experiment(base, 'sample_experiment')
    d, name, fallback = resolve_experiment_dir(base, 'does_not_exist')
    assert fallback is True and name == 'sample_experiment'


def test_resolve_falls_back_when_configured_folder_has_no_data(tmp_path):
    base = str(tmp_path)
    _make_experiment(base, 'empty_exp', with_data=False)
    _make_experiment(base, 'sample_experiment')
    _d, name, fallback = resolve_experiment_dir(base, 'empty_exp')
    assert fallback is True and name == 'sample_experiment'


def test_resolve_raises_when_no_data_and_no_sample(tmp_path):
    base = str(tmp_path)
    os.makedirs(os.path.join(base, 'data'), exist_ok=True)
    with pytest.raises(FileNotFoundError):
        resolve_experiment_dir(base, 'your_experiment_folder')


# --------------------------------------------------------------------------- #
# Section 3: write_run_manifest                                                 #
# --------------------------------------------------------------------------- #
def test_manifest_written_with_expected_schema(tmp_path):
    base = str(tmp_path)
    out_dir = os.path.join(base, 'results')
    manifest, path = write_run_manifest(
        out_dir, base, config_module=None,
        experiment_name='sample_experiment', experiment_dir='/x/data/sample',
        pipeline='standard')

    assert os.path.exists(path)
    for key in ('generated_at', 'experiment', 'pipeline', 'python_version',
                'platform', 'packages', 'git', 'config_json', 'resolved_config'):
        assert key in manifest
    assert manifest['experiment'] == 'sample_experiment'
    assert manifest['pipeline'] == 'standard'
    # Core scientific packages are reported with a version (or 'not installed').
    for pkg in ('numpy', 'pandas', 'scipy', 'scikit-learn'):
        assert pkg in manifest['packages']

    with open(path) as f:               # round-trips as valid JSON
        on_disk = json.load(f)
    assert on_disk['packages'] == manifest['packages']


def test_manifest_captures_config_json(tmp_path):
    base = str(tmp_path)
    with open(os.path.join(base, 'config.json'), 'w') as f:
        json.dump({'EXPERIMENT': 'foo', 'RANDOM_SEED': 7}, f)
    manifest, _ = write_run_manifest(os.path.join(base, 'results'), base)
    assert manifest['config_json'] == {'EXPERIMENT': 'foo', 'RANDOM_SEED': 7}


def test_manifest_resolved_config_filters_to_uppercase_scalars(tmp_path):
    import types
    cfg = types.SimpleNamespace(RANDOM_SEED=42, EXPERIMENT='x',
                                _private='hidden', lower=1, OBJ=object())
    manifest, _ = write_run_manifest(os.path.join(str(tmp_path), 'r'), str(tmp_path),
                                     config_module=cfg)
    rc = manifest['resolved_config']
    assert rc['RANDOM_SEED'] == 42 and rc['EXPERIMENT'] == 'x'
    assert '_private' not in rc and 'lower' not in rc and 'OBJ' not in rc
