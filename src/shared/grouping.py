"""
Which condition and which colony each spectrum belongs to.

The colony (biological replicate) is the experimental unit; its technical
replicates are repeated sprays of the same colony. The single source of truth is
``metadata_manifest.csv`` in the experiment folder (columns file_path, class,
colony). Without a manifest the folder layout is used:

* nested  <condition>/<colony>/<file>   -> colony = the file's folder
* flat    <condition>/<colony>T<n>.csv  -> colony = file name minus 'T<n>'

Create a manifest to check or correct the result:
    python src/shared/grouping.py data/MY_EXPERIMENT
"""

import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

_TECH_REP = re.compile(r'(?<=[0-9_\-.\s])[Tt]\d+$')     # 'A1T3' -> 'A1'


def parse_path(rel_path):
    """(condition, colony) from a path relative to the experiment folder."""
    parts = [p for p in rel_path.replace('\\', '/').split('/') if p not in ('', '.')]
    if len(parts) < 2:
        raise ValueError(f"{rel_path!r}: put spectra under a condition folder "
                         "(data/<experiment>/<condition>/...)")
    *dirs, fname = parts
    if len(dirs) >= 2:
        return '/'.join(dirs[:-1]), dirs[-1]
    stem = re.sub(r'\.(csv|txt)$', '', fname, flags=re.I).strip()
    colony = _TECH_REP.sub('', stem).rstrip(' _-.') or stem
    return dirs[0], colony


def resolve_design(experiment_dir, names, manifest_name='metadata_manifest.csv',
                   expected_tech_reps=None, min_colonies=2, allow_unequal_colonies=False):
    """Return (condition, colony_id) arrays aligned with ``names``.

    colony_id is '<condition>::<colony>' so it is unique across conditions.
    Raises if a file is missing from the manifest, if a condition has fewer than
    ``min_colonies`` colonies, if a colony does not have ``expected_tech_reps``
    spectra (when given), or if the colonies do not all have the same number of
    spectra (unless ``allow_unequal_colonies``). Unequal colonies usually mean
    one colony named two ways ('AmbB5T1' next to 'AmberB5T2'): its sprays would
    then be split between training and test folds and counted as two colonies.
    """
    path = os.path.join(experiment_dir, manifest_name) if manifest_name else None
    if path and os.path.exists(path):
        man = pd.read_csv(path, dtype=str)
        if 'colony' not in man and 'biological_replicate_group' in man:
            man = man.rename(columns={'biological_replicate_group': 'colony'})
        missing_cols = {'file_path', 'class', 'colony'} - set(man.columns)
        if missing_cols:
            raise ValueError(f"{manifest_name} lacks columns {sorted(missing_cols)}")
        man = man.set_index(man['file_path'].str.replace('\\', '/', regex=False))
        absent = [n for n in names if n not in man.index]
        if absent:
            raise ValueError(f"{len(absent)} file(s) missing from {manifest_name}, e.g. {absent[:3]}")
        cond = np.array([str(man.loc[n, 'class']).strip() for n in names])
        col = np.array([str(man.loc[n, 'colony']).strip() for n in names])
        print(f"  Design from {manifest_name}")
    else:
        cond, col = map(np.array, zip(*(parse_path(n) for n in names)))
        print("  Design from folder layout (no manifest)")
    colony_id = np.array([f"{c}::{k}" for c, k in zip(cond, col)])

    if len(set(colony_id)) == len(colony_id) and len(names) > 1:
        warnings.warn("Every spectrum became its own colony: no technical replicates were "
                      "recognised. Check the file names or add a metadata manifest.", stacklevel=2)
    per_cond = {c: len(set(colony_id[cond == c])) for c in np.unique(cond)}
    thin = {c: n for c, n in per_cond.items() if n < min_colonies}
    if thin:
        raise ValueError(f"Conditions with fewer than {min_colonies} colonies: {thin}. "
                         "Colony-level statistics need replicate colonies.")
    if expected_tech_reps is not None:
        ids, counts = np.unique(colony_id, return_counts=True)
        off = {i: int(n) for i, n in zip(ids, counts) if n != expected_tech_reps}
        if off:
            raise ValueError(f"Colonies without {expected_tech_reps} spectra: {off}")
    if not allow_unequal_colonies:
        ids, counts = np.unique(colony_id, return_counts=True)
        usual = int(np.bincount(counts).argmax())
        off = {str(i): int(n) for i, n in zip(ids, counts) if n != usual}
        if off:
            raise ValueError(
                f"{len(off)} colonies do not have the usual {usual} spectra: {off}. If these are "
                "one colony named two ways, fix the file names or write a metadata manifest "
                "(python src/shared/grouping.py data/<experiment>, then correct its 'colony' "
                "column). If the colonies really have different numbers of spectra, set "
                "ALLOW_UNEQUAL_COLONIES to true.")
    return cond, colony_id


def write_manifest_template(experiment_dir, out_name='metadata_manifest.csv'):
    """Write the design inferred from the folder layout for the user to check."""
    rows = []
    for dirpath, dirnames, filenames in os.walk(experiment_dir):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith('.'))
        for f in sorted(filenames):
            if f.lower().endswith(('.csv', '.txt')) and f.lower() != out_name and not f.startswith('.'):
                rel = os.path.relpath(os.path.join(dirpath, f), experiment_dir).replace(os.sep, '/')
                cond, col = parse_path(rel)
                rows.append(dict(file_path=rel, **{'class': cond}, colony=col))
    df = pd.DataFrame(rows).sort_values('file_path')
    df['technical_replicate'] = df.groupby(['class', 'colony']).cumcount() + 1
    out = os.path.join(experiment_dir, out_name)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows -> {out}. Check 'class' and 'colony' for every file.")
    return df


if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if len(sys.argv) != 2:
        raise SystemExit("usage: python src/shared/grouping.py data/<experiment>")
    write_manifest_template(sys.argv[1])
