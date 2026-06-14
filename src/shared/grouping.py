"""
grouping.py — biological-replicate grouping for leak-free cross-validation
==========================================================================
Cross-validation must keep the technical replicates of one biological sample
together (same colony/well/injection), otherwise a near-identical replicate
lands in both train and test and accuracy is inflated to ~1.0 (pseudoreplication).

Grouping is derived **purely from the directory structure**. Filenames are never
parsed for metadata, and there is no regex/filename fallback.

Topology contract
-----------------
``load_experiment()`` returns each sample's file path RELATIVE to the experiment
directory (POSIX form, e.g. ``'Amber/colony_A/inj_03.csv'``). The rules are:

  * The folder that DIRECTLY contains a file is that file's biological sample;
    the folder's name is the sample id.
  * Every file sharing that folder is a technical replicate of the same sample,
    so they all receive the same CV group and never split across folds.

``make_groups`` reads the containing-folder name straight off the relative path.
If a path carries no directory component it raises — it does not guess from the
filename. ``assign_replicate_indices`` recovers the 1-based technical-replicate
number of each file (alphabetical within its folder) for metadata/auditing.
"""

import re
import posixpath
import numpy as np

_EXT_RE = re.compile(r'\.(csv|txt)$', re.IGNORECASE)


def _posix(name):
    """Normalise any path to forward slashes."""
    return str(name).replace('\\', '/')


def _stem(name):
    """Filename without directory or .csv/.txt extension (display only)."""
    return _EXT_RE.sub('', posixpath.basename(_posix(name))).strip()


def _sample_folder(name):
    """Biological-sample id = name of the directory that directly contains the
    file. ``name`` is a path relative to the experiment dir, as produced by
    load_experiment(). Returns None when the path has no directory component."""
    parent = posixpath.dirname(_posix(name))
    if not parent:
        return None
    return posixpath.basename(parent)


def make_groups(y_labels, names, verbose=False):
    """Build a CV group label per sample: '<class>::<sample_id>'.

    The sample id is the name of the folder that directly contains the file —
    i.e. the directory structure, never the filename. All technical replicates
    inside one folder therefore share a group and can never straddle the
    train/test split.

    ``names`` must be the file paths RELATIVE to the experiment dir, exactly as
    returned by load_experiment(). A name without a directory component is a
    programming/topology error and raises ValueError — there is deliberately no
    fallback to filename parsing.

    Returns an ndarray of '<class>::<sample_id>' strings, one per sample, ready
    for StratifiedGroupKFold.
    """
    y_labels = np.asarray(y_labels)
    if len(y_labels) != len(names):
        raise ValueError(
            f"y_labels ({len(y_labels)}) and names ({len(names)}) length mismatch."
        )

    groups = []
    bad = []
    for lab, nm in zip(y_labels, names):
        folder = _sample_folder(nm)
        if folder is None:
            bad.append(str(nm))
            continue
        groups.append(f"{lab}::{folder}")

    if bad:
        raise ValueError(
            "make_groups requires directory-topology paths (relative to the "
            "experiment dir, e.g. '<condition>/<sample>/<file>.csv'). These "
            f"names carried no folder and cannot be grouped without parsing the "
            f"filename: {bad[:5]}{' ...' if len(bad) > 5 else ''}. Pass the "
            "names returned by load_experiment()."
        )

    groups = np.array(groups)
    if verbose:
        summarize_groups(y_labels, names, groups)
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


def summarize_groups(y_labels, names, groups):
    """Print how files were grouped, and warn about the two failure modes that
    silently break cross-validation: no grouping at all, or <2 groups in a class."""
    y_labels = np.asarray(y_labels)
    n_files  = len(groups)
    n_groups = len(set(groups))
    print(f"  Sample grouping (directory topology): {n_files} files -> "
          f"{n_groups} biological sample group(s)")

    # Show a couple of examples so the user can eyeball correctness.
    seen = {}
    for nm, g in zip(names, groups):
        seen.setdefault(g, []).append(_stem(nm))
    for g, members in list(seen.items())[:3]:
        label = g.split('::', 1)[-1]
        print(f"    e.g. '{label}'  <-  {', '.join(members[:4])}"
              + (" ..." if len(members) > 4 else ""))

    if n_groups == n_files:
        print("  [warning] Every file became its own group — no folder contains "
              "more than one file, so no technical replicates were detected. If "
              "replicates exist, place all replicate files of a sample in that "
              "sample's folder (experiment/<condition>/<sample>/rep1, rep2, ...).")

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
