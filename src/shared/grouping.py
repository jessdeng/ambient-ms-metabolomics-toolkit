"""
grouping.py — biological-replicate grouping for leak-free cross-validation
==========================================================================
Cross-validation must keep the technical replicates of one biological sample
together (same colony/well/injection), otherwise a near-identical replicate
lands in both train and test and accuracy is inflated to ~1.0 (pseudoreplication).

The only thing the pipeline needs is: *which files are replicates of the same
sample?* This module answers that purely from the DIRECTORY STRUCTURE — the
filenames are never parsed for metadata.

Topology contract
------------------
``load_experiment()`` returns each sample's file path RELATIVE to the experiment
directory (POSIX form, e.g. ``'Amber/colony_A/inj_03.csv'``). The folder that
directly contains a file is that file's biological sample; every file sharing
that folder is a technical replicate of the same sample. ``make_groups`` reads
the containing-folder name straight off that relative path.

Modes (``mode`` argument; default 'auto'):
  'auto'      (default) — biological sample id = name of the folder that
                          directly contains the file (directory topology).
                          All replicates in a folder share one CV group.
  'per_file'            — treat every file as its own independent biological
                          sample (correct only if there are NO technical
                          replicates). Produces optimistic CV when replicates
                          actually exist.
  'regex'               — escape hatch: apply a custom regex to the relative
                          path; the first capture group is the sample id.

There is also ``groups_from_csv()`` for a fully explicit mapping when the
directory layout cannot express the grouping, and ``assign_replicate_indices()``
to recover the 1-based technical-replicate number of each file (alphabetical
within its folder) for metadata/auditing.
"""

import os
import re
import posixpath
import numpy as np

_EXT_RE = re.compile(r'\.(csv|txt)$', re.IGNORECASE)


def _posix(name):
    """Normalise any path to forward slashes."""
    return str(name).replace('\\', '/')


def _stem(name):
    """Filename without directory or .csv/.txt extension."""
    return _EXT_RE.sub('', posixpath.basename(_posix(name))).strip()


def _sample_folder(name):
    """Biological-sample id = name of the directory that directly contains the
    file. ``name`` is a path relative to the experiment dir, as produced by
    load_experiment(). Returns None when the name carries no directory
    component (so the caller can fall back instead of collapsing every file
    into one group). Filenames are never parsed."""
    parent = posixpath.dirname(_posix(name))
    if not parent:
        return None
    return posixpath.basename(parent)


def make_groups(y_labels, names, mode='auto', regex=None, verbose=False):
    """Build a CV group label per sample: '<class>::<sample_id>'.

    Topology-based (default 'auto'): the sample id is the name of the folder
    that directly contains the file — i.e. the directory structure, NOT the
    filename. All technical replicates inside one folder therefore share a
    group and can never straddle the train/test split.

    ``names`` must be the file paths RELATIVE to the experiment dir, exactly as
    returned by load_experiment(). The return value is unchanged: an ndarray of
    '<class>::<sample_id>' strings, one per sample, ready for StratifiedGroupKFold.

    Backward compatible: ``make_groups(y_labels, names)`` keeps working.
    """
    y_labels = np.asarray(y_labels)
    custom = re.compile(regex) if (mode == 'regex' and regex) else None

    groups = []
    missing = 0
    for lab, nm in zip(y_labels, names):
        if mode == 'per_file':
            token = _stem(nm)
        elif custom is not None:
            mm = custom.search(_posix(nm))
            token = mm.group(1) if (mm and mm.groups()) else _stem(nm)
        else:  # 'auto' — directory topology
            token = _sample_folder(nm)
            if token is None:
                # Name had no folder component; fall back to per-file so CV can
                # still run, and flag it so the user can pass proper paths.
                token = _stem(nm)
                missing += 1
        groups.append(f"{lab}::{token}")

    if missing and mode == 'auto':
        print(f"  [warning] {missing} sample name(s) carried no directory "
              f"component, so their biological-sample id fell back to the "
              f"filename. Pass the relative paths returned by load_experiment() "
              f"so grouping stays purely topology-based.")

    groups = np.array(groups)
    if verbose:
        summarize_groups(y_labels, names, groups, mode)
    return groups


def assign_replicate_indices(names):
    """Return the 1-based technical-replicate index of each file, assigned by
    sorting filenames alphabetically WITHIN their containing folder.

    Filenames are used only for ordering — never parsed for a replicate number.
    The returned list is aligned with ``names``. This is metadata/auditing
    information; the leak-free CV grouping itself does not need it (replicates
    are grouped by folder, not by index).
    """
    by_folder = {}
    for i, nm in enumerate(names):
        norm = _posix(nm)
        folder = posixpath.dirname(norm)
        by_folder.setdefault(folder, []).append((posixpath.basename(norm), i))

    idx = [0] * len(names)
    for items in by_folder.values():
        for rep, (_fname, i) in enumerate(sorted(items), start=1):
            idx[i] = rep
    return idx


def summarize_groups(y_labels, names, groups, mode='auto'):
    """Print how files were grouped, and warn about the two failure modes that
    silently break cross-validation: no grouping at all, or <2 groups in a class."""
    y_labels = np.asarray(y_labels)
    n_files  = len(groups)
    n_groups = len(set(groups))
    print(f"  Sample grouping (mode='{mode}'): {n_files} files -> "
          f"{n_groups} biological sample group(s)")

    # Show a couple of examples so the user can eyeball correctness.
    seen = {}
    for nm, g in zip(names, groups):
        seen.setdefault(g, []).append(_stem(nm))
    for g, members in list(seen.items())[:3]:
        label = g.split('::', 1)[-1]
        print(f"    e.g. '{label}'  <-  {', '.join(members[:4])}"
              + (" ..." if len(members) > 4 else ""))

    if n_groups == n_files and mode != 'per_file':
        print("  [warning] No replicates were detected — every file became its "
              "own group. If your files include technical replicates of the same "
              "sample, cross-validation accuracy will be OPTIMISTIC. Each sample "
              "folder should contain all of that sample's replicate files "
              "(experiment/<condition>/<sample>/rep1, rep2, ...).")

    # Per-class group count — grouped CV needs >= 2 biological groups per class.
    classes = np.unique(y_labels)
    deficient = []
    for c in classes:
        g_in_c = len(set(groups[y_labels == c]))
        if g_in_c < 2:
            deficient.append((c, g_in_c))
    if deficient:
        msg = ', '.join(f"{c} ({n})" for c, n in deficient)
        print(f"  [warning] These classes have < 2 biological sample groups: "
              f"{msg}. Grouped cross-validation cannot estimate generalisation "
              f"for them and will raise an error — you need at least two "
              f"independent sample folders per class.")


def groups_from_csv(names, csv_path, file_col='file', group_col='sample'):
    """Explicit grouping from a metadata CSV (filename -> sample id). Use when the
    directory layout cannot express the grouping. Matches on the file's basename
    and falls back to the filename for any file missing from the table."""
    import pandas as pd
    table = pd.read_csv(csv_path)
    lookup = dict(zip(table[file_col].astype(str), table[group_col].astype(str)))
    return np.array([lookup.get(_stem(nm), _stem(nm)) for nm in names])
