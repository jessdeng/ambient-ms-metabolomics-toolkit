"""
Tests for consolidate_data.py — discovery and the .gitignore lockdown logic
(the privacy boundary between synthetic demo data and private paper data).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from shared.consolidate_data import (  # noqa: E402
    discover_experiments, update_gitignore, _has_data_files,
    _BLOCK_START, _BLOCK_END, SAMPLE_EXPERIMENT,
)


def _exp(base, *parts, data=True):
    d = os.path.join(base, *parts)
    os.makedirs(d, exist_ok=True)
    if data:
        with open(os.path.join(d, 'f.csv'), 'w') as f:
            f.write('mz,intensity\n100,5\n')
    return d


def test_discover_skips_sample_and_empty(tmp_path):
    base = str(tmp_path)
    _exp(base, 'data', 'experiment_a')
    _exp(base, 'data', 'experiment_b')
    _exp(base, 'data', SAMPLE_EXPERIMENT)          # must be skipped
    _exp(base, 'data', 'empty_dir', data=False)    # no data -> skipped
    root, found = discover_experiments(base)
    names = sorted(n for n, _ in found)
    assert names == ['experiment_a', 'experiment_b']


def test_discover_accepts_data_dir_directly(tmp_path):
    base = str(tmp_path)
    _exp(base, 'experiment_a')                # source IS the data dir (no nested data/)
    root, found = discover_experiments(base)
    assert [n for n, _ in found] == ['experiment_a']


def test_has_data_files(tmp_path):
    d = _exp(str(tmp_path), 'x')
    assert _has_data_files(d) is True
    empty = os.path.join(str(tmp_path), 'y'); os.makedirs(empty)
    assert _has_data_files(empty) is False


def test_gitignore_adds_patterns_and_clean_negation(tmp_path):
    base = str(tmp_path)
    # Pre-existing broad rule, like the real repo.
    with open(os.path.join(base, '.gitignore'), 'w') as f:
        f.write('data/*\n!data/.gitkeep\n')
    names = ['experiment_a', 'experiment_b']
    added, path = update_gitignore(base, names, dry_run=False)
    text = open(path).read()
    lines = text.splitlines()

    # patterns present, anchored, no trailing junk
    assert '/data/experiment_a' in lines
    assert '/data/experiment_b' in lines
    # the sample negation is on its OWN line with NO inline comment
    assert f'!/data/{SAMPLE_EXPERIMENT}/' in lines
    block = lines[lines.index(_BLOCK_START):lines.index(_BLOCK_END) + 1]
    for ln in block:
        if ln.startswith('/data/') or ln.startswith('!/data/'):
            assert '#' not in ln               # no inline comments on patterns


def test_gitignore_idempotent_and_unions(tmp_path):
    base = str(tmp_path)
    update_gitignore(base, ['experiment_a'], dry_run=False)
    update_gitignore(base, ['experiment_a'], dry_run=False)          # repeat -> no dup
    added2, path = update_gitignore(base, ['experiment_b'], dry_run=False)  # add another
    lines = open(path).read().splitlines()
    assert lines.count(_BLOCK_START) == 1 and lines.count(_BLOCK_END) == 1
    assert lines.count('/data/experiment_a') == 1          # not duplicated
    assert '/data/experiment_b' in lines                   # union preserved a + b
    assert '/data/experiment_a' in lines


def test_gitignore_dry_run_writes_nothing(tmp_path):
    base = str(tmp_path)
    update_gitignore(base, ['experiment_a'], dry_run=True)
    assert not os.path.exists(os.path.join(base, '.gitignore'))
