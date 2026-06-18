"""
grouping.py — biological-replicate grouping for leak-free cross-validation
==========================================================================
Cross-validation must keep the technical replicates of one biological sample
together (same colony/well/injection); otherwise a near-identical replicate
lands in both train and test and accuracy is inflated to ~1.0 (pseudoreplication).

This module derives the CV group from the file's location using a **bottom-up,
relative-depth** rule so that BOTH common directory layouts work without forcing
everyone into one rigid structure. ``experiment_dir`` is whatever folder is
handed to ``load_experiment``; paths below are RELATIVE to it.

Supported layouts (relative to ``experiment_dir``)
--------------------------------------------------
1. **Nested / "4-tier" layout** — ``<...>/<condition>/<biological_replicate>/file``
   (≥2 directory levels above the file). Read bottom-up:

     * the file's IMMEDIATE parent folder is the biological replicate → CV group;
     * everything ABOVE that folder is joined to form the class label
       (so ``Strain/Condition/RepA/f.csv`` → class ``"Strain/Condition"``,
       group ``"RepA"``, and ``Condition/RepA/f.csv`` → class ``"Condition"``,
       group ``"RepA"``);
     * every file inside the replicate folder is a technical replicate and shares
       the group, so replicates never straddle a fold.

2. **Flat / "3-tier" layout** — ``<condition>/file`` (exactly 1 directory level
   above the file). There is no biological-replicate FOLDER, so the replicate is
   recovered from the FILE-NAME PREFIX:

     * the class is the condition folder;
     * the biological replicate id is the file stem with a trailing
       technical-replicate token (default ``T<number>``, e.g. ``A1T3``) stripped,
       so ``S5_Green_A1T1/2/3.csv`` all collapse to replicate ``"S5_Green_A1"``;
     * if no technical-replicate token is found, the whole stem is used, i.e.
       each file becomes its own biological entity (still leak-free).

The technical-replicate token is configurable via ``tech_rep_pattern``.

Strict pre-flight validation
----------------------------
``make_groups`` (and ``validate_grouping``) count the unique biological groups per
class and **raise** ``ValueError`` — listing the offending classes and their
folders/prefixes — if any class has fewer than two. This fails loudly at parse
time instead of letting ``StratifiedGroupKFold`` blow up several steps downstream.
"""

import os
import re
import warnings
import posixpath
import numpy as np

_EXT_RE = re.compile(r'\.(csv|txt)$', re.IGNORECASE)

# Best-effort acquisition timestamp embedded in a filename, e.g.
# '20240517', '2024-05-17', '2024-05-17_14-32-08', '20240517T143208'.
_TIMESTAMP_RE = re.compile(
    r'(20\d{2})[-_]?(\d{2})[-_]?(\d{2})'        # date  YYYY MM DD
    r'(?:[ _T-]?(\d{2})[-_:]?(\d{2})(?:[-_:]?(\d{2}))?)?'   # optional HH MM SS
)

# --- Technical-replicate token stripping (flat-layout prefix rule) --------------
# A biological-replicate id is recovered from a filename by ITERATIVELY removing
# trailing *technical* tokens: injection index ('_01'), polarity tag ('_NEG'),
# batch/run/rep markers ('_batch1', '_rep2', '_inj3'), and the classic 'T<n>'
# suffix ('A1T3'). Stripping is aggressive on purpose so that common mass-spec
# names such as '7860_8e6_NEG_01' collapse to the true biological id '7860_8e6'
# (all 10 injections then share ONE group). Every delimited token must be preceded
# by a separator ('_', '-', '.', or space) so embedded digits — e.g. the '6' in
# '8e6', or the well number in 'A1' — are never clipped.
# IMPORTANT — default strips only UNAMBIGUOUSLY TECHNICAL tokens. Words like
# 'rep', 'replicate', 'batch', 'run', 'sample' are deliberately NOT in the default
# set: in biology they usually denote the BIOLOGICAL replicate / experimental
# block, which is exactly the unit grouped CV must KEEP SEPARATE. Stripping them by
# default would silently collapse a valid 3-biological-replicate design into a fake
# n=1 study. Users whose 'repN' really is a technical replicate can opt in by
# passing an explicit tech_rep_pattern= (see module docstring / README).
_DELIM_TOKEN_RE = re.compile(
    r'[_\-.\s]+'                                                    # required delimiter
    r'(?:'
    r'(?:injection|inj|acq(?:uisition)?|tech(?:nical)?|scan)\.?\d+'  # technical markers
    r'|neg|pos'                                                     # polarity tag
    r'|\d+'                                                         # bare injection index
    r')$',
    re.IGNORECASE,
)

# Opt-in extension: pass this as tech_rep_pattern= to ALSO strip rep/replicate/
# batch/run/sample markers when you know they are technical, not biological.
AGGRESSIVE_TECH_REP_RE = re.compile(
    r'[_\-.\s]+'
    r'(?:(?:rep|replicate|run|batch|sample|injection|inj|acq(?:uisition)?'
    r'|tech(?:nical)?|scan)\.?\d+|r\d+|b\d+|neg|pos|\d+)$',
    re.IGNORECASE,
)
# 'T<n>' may follow a digit (e.g. 'A1T3') OR a delimiter; handled separately so a
# well id like 'A1' keeps its trailing digit once the T-token is removed.
_TREP_TOKEN_RE = re.compile(r'(?<=[0-9_\-.\s])[Tt]\d+$')

# Back-compat alias: callers/tests that import this name still resolve. The new,
# aggressive multi-token behaviour is the default inside _strip_tech_tokens().
DEFAULT_TECH_REP_RE = _TREP_TOKEN_RE

_SEP_STRIP = ' _-.'


def _compile_pattern(tech_rep_pattern):
    """Accept None (use built-in aggressive set), a compiled regex, or a string."""
    if tech_rep_pattern is None:
        return None
    if isinstance(tech_rep_pattern, str):
        return re.compile(tech_rep_pattern)
    return tech_rep_pattern


def _strip_tech_tokens(stem, tech_rep_pattern=None, max_tokens=12):
    """Strip trailing technical-replicate tokens from a filename stem.

    ``tech_rep_pattern=None`` (default) applies the built-in token set iteratively
    (T<n> first, then a delimited injection/polarity/batch/rep token). Passing a
    custom compiled/str pattern OVERRIDES the set and strips only that token
    (still iteratively). Stripping never empties the stem and never removes a
    non-delimited embedded number.
    """
    user_pat = _compile_pattern(tech_rep_pattern)
    delim_pat = user_pat if user_pat is not None else _DELIM_TOKEN_RE
    out = stem
    for _ in range(max_tokens):
        # 'T<n>' is unambiguously technical, so it is always stripped first,
        # regardless of whether a custom delimited pattern was supplied.
        m = _TREP_TOKEN_RE.search(out) or delim_pat.search(out)
        if not m or m.start() == 0:          # nothing to strip / would empty stem
            break
        candidate = out[:m.start()].rstrip(_SEP_STRIP)
        if not candidate:
            break
        out = candidate
    return out


def _posix(name):
    """Normalise any path to forward slashes."""
    return str(name).replace('\\', '/')


def _stem(name):
    """Filename without directory or .csv/.txt extension."""
    return _EXT_RE.sub('', posixpath.basename(_posix(name))).strip()


def _rel_parts(name):
    """Relative path split into clean components (drops '' and '.')."""
    return [seg for seg in _posix(name).split('/') if seg not in ('', '.')]


def parse_sample(rel_path, tech_rep_pattern=None):
    """Resolve one sample's ``(class_label, group_id, source)`` from its path.

    ``rel_path`` must be RELATIVE to the experiment directory, as produced by
    ``load_experiment`` (e.g. ``'Green/S5_Green_A1T1.csv'`` or
    ``'Condition/RepA/inj_03.csv'``). The bottom-up rule:

      * ≥2 directory levels  -> 'folder' mode: group = immediate parent folder,
        class = everything above it (slash-joined).
      * exactly 1 dir level  -> 'prefix' mode: class = the single folder,
        group = file stem with the trailing technical-replicate token removed.
      * 0 directory levels   -> raise (a bare filename carries no class/topology).

    Returns
    -------
    (class_label, group_id, source) where source is 'folder' or 'prefix'.
    """
    parts = _rel_parts(rel_path)
    if len(parts) < 2:
        raise ValueError(
            f"Cannot group {rel_path!r}: it has no directory component, so its "
            "class and biological replicate are undefined. Place data files under "
            "at least a condition folder (data/<condition>/file.csv) and pass the "
            "paths returned by load_experiment()."
        )

    *dirs, fname = parts

    if len(dirs) >= 2:
        # Nested layout: immediate parent = biological replicate; class is the
        # joined path above it (captures Condition, or Strain/Condition, ...).
        class_label = '/'.join(dirs[:-1])
        group_id = dirs[-1]
        return class_label, group_id, 'folder'

    # Flat layout (exactly one directory level): recover the replicate from the
    # filename prefix by iteratively stripping trailing technical-replicate tokens.
    class_label = dirs[0]
    stem = _stem(fname)
    stripped = _strip_tech_tokens(stem, tech_rep_pattern)
    group_id = stripped if stripped else stem
    return class_label, group_id, 'prefix'


def make_groups(y_labels, names, tech_rep_pattern=None, verbose=False,
                validate=True, min_groups_per_class=2):
    """Build a CV group label per sample: ``'<class>::<biological_replicate>'``.

    The biological replicate is derived from the directory topology (nested
    layout) or the filename prefix (flat layout) via :func:`parse_sample`, so all
    technical replicates of one biological sample share a group and can never
    straddle the train/test split. The class portion uses the label in
    ``y_labels`` (so externally merged/renamed classes are respected).

    Parameters
    ----------
    y_labels : array-like (n_samples,)
        Class label per sample, aligned with ``names``.
    names : sequence of str
        File paths RELATIVE to the experiment dir, as returned by
        ``load_experiment()``.
    tech_rep_pattern : None | str | compiled regex
        Trailing technical-replicate token used only in the flat-layout prefix
        rule. ``None`` uses ``DEFAULT_TECH_REP_RE`` (``T<digits>``).
    validate : bool
        If True (default), run :func:`validate_grouping` and raise when any class
        has fewer than ``min_groups_per_class`` biological groups.

    Returns
    -------
    np.ndarray of '<class>::<replicate>' strings, ready for StratifiedGroupKFold.
    """
    y_labels = np.asarray(y_labels)
    if len(y_labels) != len(names):
        raise ValueError(
            f"y_labels ({len(y_labels)}) and names ({len(names)}) length mismatch."
        )

    groups = []
    for lab, nm in zip(y_labels, names):
        _cls, group_id, _src = parse_sample(nm, tech_rep_pattern)
        groups.append(f"{lab}::{group_id}")
    groups = np.array(groups)

    # Explicit fallback warning: if stripping recovered NO shared replicate
    # prefix, every file is its own group and the anti-pseudoreplication machinery
    # is effectively disabled. Tell the user loudly so a naming mismatch does not
    # silently inflate downstream cross-validated accuracy.
    n_files = len(groups)
    n_unique = len(set(groups.tolist()))
    if n_files > 1 and n_unique == n_files:
        warnings.warn(
            f"Replicate grouping produced ONE GROUP PER FILE ({n_files} files -> "
            f"{n_unique} groups): no technical replicates were detected, so every "
            "spectrum is being treated as an independent biological sample. If your "
            "files are technical replicates of fewer biological samples, their "
            "names do not match the replicate-token rule. Verify your filename "
            "structure, or pass an explicit tech_rep_pattern= to load_experiment()/"
            "make_groups(). Proceeding as fully-independent samples can INFLATE "
            "cross-validated accuracy.",
            stacklevel=2,
        )

    if verbose:
        summarize_groups(y_labels, names, groups)
    if validate:
        validate_grouping(y_labels, groups, names,
                          min_groups_per_class=min_groups_per_class)
    return groups


def validate_grouping(y_labels, groups, names=None, min_groups_per_class=2):
    """Strict pre-flight check: every class needs ≥ ``min_groups_per_class``
    independent biological groups, or StratifiedGroupKFold cannot make folds.

    Raises ``ValueError`` (never warns-and-continues) listing each offending
    class, its biological group ids, and a few example files, with a concrete fix.
    """
    y_labels = np.asarray(y_labels)
    groups = np.asarray(groups)
    names_arr = np.asarray(names, dtype=object) if names is not None else None

    problems = []
    for c in np.unique(y_labels):
        in_c = (y_labels == c)
        uniq = sorted(set(groups[in_c]))
        if len(uniq) < min_groups_per_class:
            examples = list(names_arr[in_c][:6]) if names_arr is not None else []
            problems.append((c, uniq, examples))

    if not problems:
        return

    lines = []
    for c, uniq, examples in problems:
        ids = ', '.join(g.split('::', 1)[-1] for g in uniq) or '(none)'
        lines.append(f"  - class '{c}': only {len(uniq)} biological group(s) "
                     f"[{ids}]")
        if examples:
            shown = ', '.join(str(e) for e in examples)
            more = ' ...' if len(examples) >= 6 else ''
            lines.append(f"      files: {shown}{more}")

    raise ValueError(
        "Leak-free grouped cross-validation (StratifiedGroupKFold) requires at "
        f"least {min_groups_per_class} independent biological groups per class, "
        "but these do not have enough:\n"
        + "\n".join(lines)
        + "\n\nHow to fix one of:\n"
        "  1. Add more biological replicates for the affected class(es).\n"
        "  2. Nested layout: give each biological replicate its own sub-folder "
        "(data/<condition>/<replicate>/files...).\n"
        "  3. Flat layout: ensure each replicate's technical-replicate files share "
        "a common filename prefix and differ only by a trailing 'T<n>' token "
        "(e.g. A1T1, A1T2, A1T3 -> replicate 'A1'); adjust tech_rep_pattern if "
        "your naming differs.\n"
        "  4. Merge or drop the deficient class before cross-validation."
    )


def permute_labels_by_group(y_labels, groups, rng):
    """Permute class labels at the GROUP (biological-replicate) level.

    This is the correct null for a grouped/pseudoreplicated design. Every group
    keeps a single label, so all of its technical replicates move together (they
    are never scrambled independently), and the multiset of per-group labels is
    shuffled — which preserves the number of groups per class. The result is a
    valid restricted permutation that keeps each group single-class, so the
    downstream StratifiedGroupKFold split stays well-defined under the null.

    Parameters
    ----------
    y_labels : array-like (n_samples,)
        Original class label per sample.
    groups : array-like (n_samples,)
        CV group id per sample, as built by :func:`make_groups`. Each group must
        be single-class (it is, by construction: ``'<class>::<replicate>'``).
    rng : np.random.Generator
        Seeded generator (``np.random.default_rng(seed)``) for reproducibility.

    Returns
    -------
    np.ndarray (n_samples,) — sample-aligned permuted labels.
    """
    y_labels = np.asarray(y_labels)
    groups = np.asarray(groups)
    uniq = np.unique(groups)

    grp_label = np.empty(uniq.shape[0], dtype=y_labels.dtype)
    for i, g in enumerate(uniq):
        labs = set(y_labels[groups == g])
        if len(labs) != 1:
            raise ValueError(
                f"Group {g!r} spans multiple classes {sorted(labs)}; cannot "
                "permute at the group level. Rebuild groups with make_groups() "
                "so every group is single-class."
            )
        grp_label[i] = next(iter(labs))

    permuted = rng.permutation(grp_label)               # shuffle group->class map
    mapping = {g: permuted[i] for i, g in enumerate(uniq)}
    return np.array([mapping[g] for g in groups])


def assign_replicate_indices(names):
    """Return the 1-based technical-replicate index of each file within its
    biological group (metadata/auditing only — the CV grouping does not need it).

    Files are bucketed by their resolved biological group (folder or filename
    prefix), then ordered alphabetically within the bucket. Falls back silently
    for any path that cannot be parsed (index 0).
    """
    by_group = {}
    for i, nm in enumerate(names):
        try:
            _cls, group_id, _src = parse_sample(nm)
        except ValueError:
            group_id = _posix(nm)
        by_group.setdefault(group_id, []).append((_posix(nm), i))

    idx = [0] * len(names)
    for items in by_group.values():
        for rep, (_path, i) in enumerate(sorted(items), start=1):
            idx[i] = rep
    return idx


def summarize_groups(y_labels, names, groups):
    """Print how files were grouped (informational; validate_grouping enforces)."""
    y_labels = np.asarray(y_labels)
    n_files = len(groups)
    n_groups = len(set(groups))
    print(f"  Sample grouping: {n_files} files -> {n_groups} biological "
          f"group(s)")

    seen = {}
    for nm, g in zip(names, groups):
        seen.setdefault(g, []).append(_stem(nm))
    for g, members in list(seen.items())[:3]:
        label = g.split('::', 1)[-1]
        print(f"    e.g. '{label}'  <-  {', '.join(members[:4])}"
              + (" ..." if len(members) > 4 else ""))

    if n_groups == n_files:
        print("  [note] Every file became its own group — no biological "
              "replicate has >1 technical replicate (no folder or shared filename "
              "prefix groups them). If replicates exist, nest them in a per-"
              "replicate folder or give them a shared prefix + 'T<n>' suffix.")


# --------------------------------------------------------------------------- #
# B4: batch / acquisition-order confound check                                  #
# --------------------------------------------------------------------------- #
def check_batch_confound(y_labels, groups, names=None, dominance_threshold=0.8,
                         min_groups_warn=3, verbose=True, enforce=False):
    """Structural check for technical confounding in the class/replicate layout.

    Grouped CV stops technical replicates leaking, but it cannot detect a *design*
    confound — e.g. a class that is really one biological replicate measured many
    times, or a condition that was acquired as a single block. This flags two such
    risks (it never raises; it returns/prints advisory warnings):

      1. **Single-replicate dominance.** If one biological group supplies
         >= ``dominance_threshold`` of a class's spectra, that "class effect" may
         be one colony's idiosyncrasy (or one acquisition block), not biology.
      2. **Thin replication.** A class with fewer than ``min_groups_warn`` biological
         groups gives little power to separate biology from batch.

    Acquisition order / injection batch themselves are not encoded in the spectra,
    so this routine cannot test run-order drift directly — generate the metadata
    manifest (:func:`generate_metadata_template`), fill in ``acquisition_order`` /
    ``injection_batch``, and confirm class is not predictable from them.

    Returns
    -------
    list[str] — advisory messages (empty if no risk detected).
    """
    y_labels = np.asarray(y_labels)
    groups = np.asarray(groups)
    warnings_out = []

    # -- Hard enforcement: an n=1 design cannot be validated by grouped CV --------
    # A class with a single biological group spread across several technical files
    # would let cross-validation report a meaningless (typically ~1.0) score, since
    # train and test folds then contain near-identical injections of the SAME
    # sample. Fail loudly here instead of emitting a fake metric downstream.
    if enforce:
        offenders = []
        for c in np.unique(y_labels):
            in_c = (y_labels == c)
            n_grp = len(set(groups[in_c].tolist()))
            if n_grp < 2:
                offenders.append((c, n_grp, int(in_c.sum())))
        if offenders:
            lines = [
                f"  - class '{c}': only {n_grp} independent biological group(s) "
                f"across {n_samp} file(s)"
                for c, n_grp, n_samp in offenders
            ]
            raise ValueError(
                "Grouped cross-validation requires at least 2 independent "
                "biological groups per class to validate predictive power, but "
                "these classes do not have enough:\n"
                + "\n".join(lines)
                + "\n\nEach listed class contains only ONE independent biological "
                "sample measured as multiple technical injections, so any apparent "
                "class separation is confounded with that single sample's identity "
                "or acquisition batch — a CV score here would be meaningless "
                "(typically a fake ~1.0). To proceed: add real biological "
                "replicates, OR — if these files genuinely are separate biological "
                "samples — adjust the replicate-token rule (tech_rep_pattern=) so "
                "they are not collapsed into one group."
            )

    for c in np.unique(y_labels):
        in_c = (y_labels == c)
        g_in_c = groups[in_c]
        uniq, counts = np.unique(g_in_c, return_counts=True)
        n_groups = len(uniq)
        n_samples = int(in_c.sum())
        dom_frac = counts.max() / n_samples if n_samples else 0.0

        if n_groups < min_groups_warn:
            warnings_out.append(
                f"class '{c}': only {n_groups} biological group(s) — limited power "
                f"to distinguish biology from batch/technical drift.")
        if n_groups > 1 and dom_frac >= dominance_threshold:
            dom_id = uniq[np.argmax(counts)].split('::', 1)[-1]
            warnings_out.append(
                f"class '{c}': {dom_frac*100:.0f}% of spectra come from one "
                f"biological group ('{dom_id}') — the class effect may be that "
                f"single replicate/acquisition block, not the condition.")

    if verbose and warnings_out:
        print("  [confound check] potential technical confounding:")
        for w in warnings_out:
            print(f"    - {w}")
        print("    -> fill acquisition_order / injection_batch in the metadata "
              "manifest and verify class is not predictable from them.")
    return warnings_out


# --------------------------------------------------------------------------- #
# C1: MSI-aligned metadata manifest generator                                   #
# --------------------------------------------------------------------------- #
def _extract_timestamp(text):
    """Return a normalised timestamp string found in `text`, or '' if none."""
    m = _TIMESTAMP_RE.search(str(text))
    if not m:
        return ''
    y, mo, d, hh, mm, ss = m.groups()
    stamp = f"{y}-{mo}-{d}"
    if hh and mm:
        stamp += f"T{hh}:{mm}" + (f":{ss}" if ss else "")
    return stamp


def generate_metadata_template(experiment_dir, out_path=None, tech_rep_pattern=None,
                               infer_order=False):
    """Crawl an experiment directory and write an MSI-aligned metadata manifest.

    Discovers every .csv/.txt spectrum under ``experiment_dir`` (any depth),
    resolves each file's class and biological-replicate group with the same
    bottom-up topology rule used for cross-validation (:func:`parse_sample`), and
    writes ``metadata_manifest.csv`` into the experiment folder. This gives the
    sample/biological metadata table the MSI minimum-reporting standards expect
    (Sumner et al. 2007), pre-filled where it can be inferred and clearly marked
    where the user must complete it (acquisition order, injection batch).

    Columns
    -------
    sample_id, file_path, class, biological_replicate_group, technical_replicate,
    acquisition_order, acquisition_order_source, injection_batch.

    ``acquisition_order`` is pre-populated only from an unambiguous timestamp in
    the filename. With ``infer_order=True`` it instead falls back to the file's
    rank in a deterministic name sort, tagged ``filename_sort_proxy`` so it is
    never mistaken for a true run order — leave it for manual entry otherwise.
    ``injection_batch`` is always left blank for the user (it cannot be inferred
    from the spectra). Run-order / batch are exactly the covariates B4 cannot
    check automatically, so completing them here is what enables a drift check.

    Parameters
    ----------
    experiment_dir : str
    out_path : str or None
        Destination CSV. Defaults to ``<experiment_dir>/metadata_manifest.csv``.
    tech_rep_pattern : None | str | regex   passed to parse_sample.
    infer_order : bool   fill acquisition_order from name-sort rank if no timestamp.

    Returns
    -------
    (pandas.DataFrame, out_path)
    """
    import pandas as pd

    experiment_dir = os.path.abspath(experiment_dir)
    discovered = []
    for dirpath, dirnames, filenames in os.walk(experiment_dir):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith('.'))
        for fname in sorted(filenames):
            if fname.lower().endswith(('.csv', '.txt')) and not fname.startswith('.'):
                ap = os.path.join(dirpath, fname)
                rel = os.path.relpath(ap, experiment_dir).replace(os.sep, '/')
                discovered.append(rel)
    discovered.sort()
    if not discovered:
        raise ValueError(
            f"No .csv/.txt data files found anywhere under {experiment_dir!r}.")

    rep_idx = assign_replicate_indices(discovered)
    rows = []
    for rel, trep in zip(discovered, rep_idx):
        cls, group_id, _src = parse_sample(rel, tech_rep_pattern)
        ts = _extract_timestamp(rel)
        if ts:
            order, order_src = ts, 'filename_timestamp'
        else:
            order, order_src = '', 'TODO_user_entry'
        rows.append({
            'sample_id': _stem(rel),
            'file_path': rel,
            'class': cls,
            'biological_replicate_group': group_id,
            'technical_replicate': trep,
            'acquisition_order': order,
            'acquisition_order_source': order_src,
            'injection_batch': '',          # placeholder — user fills in
        })

    df = pd.DataFrame(rows)

    if infer_order and (df['acquisition_order'] == '').all():
        # Deterministic name-sort rank as an explicit, clearly-labelled PROXY.
        df = df.sort_values('file_path').reset_index(drop=True)
        df['acquisition_order'] = np.arange(1, len(df) + 1)
        df['acquisition_order_source'] = 'filename_sort_proxy'

    if out_path is None:
        out_path = os.path.join(experiment_dir, 'metadata_manifest.csv')
    df.to_csv(out_path, index=False, encoding='utf-8')
    print(f"  Wrote metadata manifest ({len(df)} files) -> {out_path}")
    return df, out_path


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m shared.grouping <experiment_dir> [--infer-order]")
        raise SystemExit(2)
    _exp = sys.argv[1]
    _infer = '--infer-order' in sys.argv[2:]
    generate_metadata_template(_exp, infer_order=_infer)
