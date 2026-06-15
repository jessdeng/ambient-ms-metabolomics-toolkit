"""
Regression tests for leak-free CV grouping (grouping.py).

These lock the C-1 fix: flat (3-tier, filename-prefix) AND nested (4-tier,
replicate-folder) layouts must both produce ≥2 biological groups per class with
technical replicates never split across groups, and a deficient layout must RAISE
a loud ValueError (no silent warn-and-continue).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from shared.grouping import (  # noqa: E402
    parse_sample, make_groups, validate_grouping, permute_labels_by_group,
    check_batch_confound, generate_metadata_template, _extract_timestamp,
)


# --------------------------------------------------------------------------- #
# parse_sample: bottom-up depth rule                                            #
# --------------------------------------------------------------------------- #
def test_flat_layout_uses_filename_prefix():
    cls, grp, src = parse_sample('Green/S5_Green_A1T1.csv')
    assert cls == 'Green'
    assert grp == 'S5_Green_A1'      # trailing T1 stripped -> colony prefix
    assert src == 'prefix'


def test_flat_layout_groups_technical_replicates_together():
    names = ['Green/S5_Green_A1T1.csv',
             'Green/S5_Green_A1T2.csv',
             'Green/S5_Green_A1T3.csv']
    ids = {parse_sample(n)[1] for n in names}
    assert ids == {'S5_Green_A1'}    # all three collapse to one biological rep


def test_nested_layout_uses_replicate_folder():
    cls, grp, src = parse_sample('Light/repA/inj_03.csv')
    assert cls == 'Light'
    assert grp == 'repA'
    assert src == 'folder'


def test_deep_nested_joins_class_above_replicate():
    cls, grp, src = parse_sample('StrainX/Light/repA/inj_03.csv')
    assert cls == 'StrainX/Light'    # everything above the replicate folder
    assert grp == 'repA'
    assert src == 'folder'


def test_tech_rep_token_not_overstripped():
    # 'EXTRACT3' ends in T+digit but T follows a letter -> must NOT be stripped.
    _cls, grp, _src = parse_sample('CondA/EXTRACT3.csv')
    assert grp == 'EXTRACT3'


def test_bare_filename_raises():
    with pytest.raises(ValueError):
        parse_sample('lonely_file.csv')


# --------------------------------------------------------------------------- #
# make_groups + validation                                                      #
# --------------------------------------------------------------------------- #
def _flat_dataset():
    names, labels = [], []
    for cond in ['Green', 'Amber']:
        for colony in ['A1', 'B1', 'C1']:
            for t in [1, 2, 3]:
                names.append(f'{cond}/S5_{cond}_{colony}T{t}.csv')
                labels.append(cond)
    return np.array(labels), names


def test_make_groups_flat_three_colonies_per_class():
    y, names = _flat_dataset()
    g = make_groups(y, names)
    per_class = {c: len(set(g[y == c])) for c in np.unique(y)}
    assert per_class == {'Green': 3, 'Amber': 3}


def test_make_groups_no_replicate_crosses_group_boundary():
    y, names = _flat_dataset()
    g = make_groups(y, names)
    # Every (label, colony) maps to exactly one group string.
    for n, grp in zip(names, g):
        assert grp.count('::') == 1


def test_validate_raises_on_single_group_per_class():
    # One prefix per class -> 1 biological group each -> must raise.
    names = [f'{c}/only_sample_T{t}.csv' for c in ['A', 'B'] for t in [1, 2, 3]]
    y = np.array([c for c in ['A', 'B'] for _ in [1, 2, 3]])
    with pytest.raises(ValueError) as exc:
        make_groups(y, names, validate=True)
    msg = str(exc.value)
    assert 'A' in msg and 'B' in msg          # names the offending classes
    assert 'StratifiedGroupKFold' in msg


def test_validate_can_be_skipped():
    names = [f'{c}/only_sample_T{t}.csv' for c in ['A', 'B'] for t in [1, 2, 3]]
    y = np.array([c for c in ['A', 'B'] for _ in [1, 2, 3]])
    g = make_groups(y, names, validate=False)   # no raise
    assert len(g) == len(names)


def test_custom_tech_rep_pattern():
    # Replicate marker '_rep<digits>' instead of 'T<digits>'.
    names = ['Cond/sampleA_rep1.csv', 'Cond/sampleA_rep2.csv']
    ids = {parse_sample(n, tech_rep_pattern=r'_rep\d+$')[1] for n in names}
    assert ids == {'sampleA'}


# --------------------------------------------------------------------------- #
# A1: group-level (colony) label permutation                                    #
# --------------------------------------------------------------------------- #
def _grouped_dataset():
    y, names = _flat_dataset()           # 2 classes x 3 colonies x 3 tech reps
    groups = make_groups(y, names)
    return np.asarray(y), groups


def test_permutation_keeps_technical_replicates_together():
    y, groups = _grouped_dataset()
    rng = np.random.default_rng(0)
    yp = permute_labels_by_group(y, groups, rng)
    # Every group must still map to exactly ONE permuted label.
    for g in np.unique(groups):
        assert len(set(yp[groups == g])) == 1


def test_permutation_preserves_groups_per_class_counts():
    y, groups = _grouped_dataset()
    rng = np.random.default_rng(1)
    yp = permute_labels_by_group(y, groups, rng)
    # The multiset of per-group labels is preserved -> #groups per class constant.
    def groups_per_class(labels):
        return {c: len(set(groups[labels == c])) for c in np.unique(labels)}
    assert groups_per_class(yp) == groups_per_class(y)


def test_permutation_is_reproducible_with_seed():
    y, groups = _grouped_dataset()
    a = permute_labels_by_group(y, groups, np.random.default_rng(42))
    b = permute_labels_by_group(y, groups, np.random.default_rng(42))
    assert np.array_equal(a, b)


def test_permutation_actually_shuffles():
    # With 6 colonies a fixed seed should produce at least one different mapping.
    y, groups = _grouped_dataset()
    yp = permute_labels_by_group(y, groups, np.random.default_rng(7))
    assert not np.array_equal(yp, y)


def test_permutation_rejects_multiclass_group():
    # A group that spans two classes is a programming error -> raise.
    y = np.array(['A', 'A', 'B'])
    groups = np.array(['g1', 'g1', 'g1'])   # one group, mixed labels
    with pytest.raises(ValueError):
        permute_labels_by_group(y, groups, np.random.default_rng(0))


# --------------------------------------------------------------------------- #
# B4: batch / confound check                                                    #
# --------------------------------------------------------------------------- #
def test_confound_check_clean_design_no_warnings():
    y, names = _flat_dataset()              # 3 colonies/class, balanced
    g = make_groups(y, names)
    assert check_batch_confound(y, g, names=names, verbose=False) == []


def test_confound_check_flags_single_group_dominance():
    # class A: 8 spectra from one colony, 1 from another -> dominance warning.
    names, y = [], []
    for i in range(8):
        names.append(f'A/A_big_T{i}.csv'); y.append('A')
    names.append('A/A_small_T1.csv'); y.append('A')
    for colony in ['c1', 'c2', 'c3']:
        for t in [1, 2, 3]:
            names.append(f'B/B_{colony}_T{t}.csv'); y.append('B')
    y = np.array(y)
    g = make_groups(y, names, validate=False)
    warns = check_batch_confound(y, g, names=names, verbose=False)
    assert any('one biological group' in w for w in warns)


def test_confound_check_flags_thin_replication():
    # 2 colonies/class (< default min_groups_warn=3) -> thin-replication warning.
    names, y = [], []
    for cls in ['A', 'B']:
        for colony in ['c1', 'c2']:
            for t in [1, 2, 3]:
                names.append(f'{cls}/{cls}_{colony}_T{t}.csv'); y.append(cls)
    y = np.array(y)
    g = make_groups(y, names)
    warns = check_batch_confound(y, g, names=names, verbose=False)
    assert any('limited power' in w for w in warns)


# --------------------------------------------------------------------------- #
# C1: metadata manifest generator                                               #
# --------------------------------------------------------------------------- #
def test_extract_timestamp():
    assert _extract_timestamp('S5_Green_20240517_A1T1.csv') == '2024-05-17'
    assert _extract_timestamp('run_2024-05-17T14-32-08.txt') == '2024-05-17T14:32:08'
    assert _extract_timestamp('plate_20240517_143208.csv') == '2024-05-17T14:32:08'
    assert _extract_timestamp('S5_Green_A1T1.csv') == ''


def test_generate_metadata_template_flat(tmp_path):
    # Build a flat experiment: 2 conditions x 2 colonies x 2 tech reps.
    import csv
    for cond in ['Green', 'Amber']:
        d = tmp_path / cond
        d.mkdir()
        for colony in ['A1', 'B1']:
            for t in [1, 2]:
                with open(d / f'S5_{cond}_{colony}T{t}.csv', 'w', newline='') as f:
                    w = csv.writer(f); w.writerow(['mz', 'int']); w.writerow([100, 1])

    df, out_path = generate_metadata_template(str(tmp_path))
    assert os.path.basename(out_path) == 'metadata_manifest.csv'
    assert set(['sample_id', 'file_path', 'class', 'biological_replicate_group',
                'technical_replicate', 'acquisition_order',
                'acquisition_order_source', 'injection_batch']).issubset(df.columns)
    assert len(df) == 8
    # Green/A1 T1 & T2 share one biological replicate group.
    grn_a1 = df[df['file_path'].str.contains('Green_A1')]
    assert grn_a1['biological_replicate_group'].nunique() == 1
    # injection_batch is a blank placeholder; order falls to user entry here.
    assert (df['injection_batch'] == '').all()
    assert (df['acquisition_order_source'] == 'TODO_user_entry').all()


def test_generate_metadata_template_infer_order(tmp_path):
    d = tmp_path / 'CondA'
    d.mkdir()
    import csv
    for colony in ['c1', 'c2']:
        with open(d / f'sample_{colony}_T1.csv', 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['mz', 'int']); w.writerow([100, 1])
    df, _ = generate_metadata_template(str(tmp_path), infer_order=True)
    assert list(df['acquisition_order']) == [1, 2]
    assert (df['acquisition_order_source'] == 'filename_sort_proxy').all()
