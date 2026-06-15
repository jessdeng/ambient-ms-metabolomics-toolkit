#!/usr/bin/env python3
"""
consolidate_data.py — map LOCAL real datasets into data/ WITHOUT ever committing them
=====================================================================================
Keeps a hard wall between the open-source toolkit and private paper data.

  * The synthetic ``data/sample_experiment/`` ships with the repo and IS committed,
    so anyone can run the pipeline out-of-the-box. This script never touches it.
  * Your real experiments (e.g. 'experiment_a', 'experiment_b', 'experiment_c', 'experiment_d')
    live only on your machine. This script symlinks (default) or copies them into
    ``data/`` for local analysis and appends their names to ``.gitignore`` so Git
    can never track or push them.

Usage
-----
    # Map your real data snapshot into data/ (symlinks; nothing is committed)
    python -m shared.consolidate_data "/path/to/ambient-ms-metabolomics-toolkit-main"

    # Copy instead of symlink (self-contained, uses more disk)
    python -m shared.consolidate_data "/path/to/snapshot" --copy

    # Preview without changing anything
    python -m shared.consolidate_data "/path/to/snapshot" --dry-run

    # Only lock down .gitignore (don't create links/copies)
    python -m shared.consolidate_data "/path/to/snapshot" --no-link

The source may be either the snapshot repo (which contains a ``data/`` folder) or a
``data/`` directory directly — both are handled.
"""

import argparse
import os
import shutil
import subprocess
import sys

SAMPLE_EXPERIMENT = 'sample_experiment'
_BLOCK_START = '# >>> consolidate_data: local real datasets — DO NOT COMMIT >>>'
_BLOCK_END = '# <<< consolidate_data <<<'


# --------------------------------------------------------------------------- #
# Discovery                                                                     #
# --------------------------------------------------------------------------- #
def repo_root():
    """Repository root = two levels above this file (src/shared/ -> repo)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _has_data_files(directory):
    """True if `directory` contains at least one .csv/.txt anywhere below it."""
    for _dp, _dn, filenames in os.walk(directory):
        if any(f.lower().endswith(('.csv', '.txt')) and not f.startswith('.')
               for f in filenames):
            return True
    return False


def experiments_root(source):
    """Accept either a snapshot repo (has data/) or a data/ dir directly."""
    source = os.path.abspath(os.path.expanduser(source))
    nested = os.path.join(source, 'data')
    return nested if os.path.isdir(nested) else source


def discover_experiments(source):
    """Return (root, [(name, abs_path), ...]) for real experiment folders.

    Skips hidden folders and the synthetic ``sample_experiment``, and only
    includes folders that actually contain spectra (.csv/.txt below them).
    """
    root = experiments_root(source)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Source directory not found: {root!r}")
    found = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if name.startswith('.') or name == SAMPLE_EXPERIMENT or not os.path.isdir(path):
            continue
        if _has_data_files(path):
            found.append((name, os.path.abspath(path)))
    return root, found


# --------------------------------------------------------------------------- #
# Linking / copying                                                             #
# --------------------------------------------------------------------------- #
def map_experiment(src, dst, mode, dry_run):
    """Symlink or copy one experiment folder into data/. Never overwrites.

    Returns one of: 'exists' (left untouched), 'planned' (dry-run),
    'linked', 'copied'.
    """
    if os.path.lexists(dst):
        return 'exists'                      # never overwrite or delete
    if dry_run:
        return 'planned'
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if mode == 'copy':
        shutil.copytree(src, dst)
        return 'copied'
    try:
        os.symlink(src, dst, target_is_directory=True)
    except OSError as exc:                    # e.g. Windows without privilege
        raise OSError(
            f"Could not create symlink {dst!r} -> {src!r} ({exc}). "
            f"Re-run with --copy to copy the data instead."
        )
    return 'linked'


# --------------------------------------------------------------------------- #
# .gitignore lockdown                                                           #
# --------------------------------------------------------------------------- #
def _pattern_for(name, data_subdir='data'):
    # Anchored to repo root; no trailing slash so it matches a real directory
    # OR a symlink of that name (and everything beneath a real dir).
    return f'/{data_subdir}/{name}'


def update_gitignore(root, names, dry_run, data_subdir='data'):
    """Append real-dataset ignore patterns inside a managed block. Idempotent.

    Existing patterns in the block are preserved and unioned with `names`, so the
    protection only ever grows. A negation keeps the synthetic sample tracked even
    if a broad ``data/*`` rule exists elsewhere. Returns (newly_added, path).
    """
    path = os.path.join(root, '.gitignore')
    text = ''
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            text = f.read()
    lines = text.splitlines()

    # Locate an existing managed block, if any.
    try:
        i0 = lines.index(_BLOCK_START)
        i1 = lines.index(_BLOCK_END)
        block_old = lines[i0:i1 + 1]
        before, after = lines[:i0], lines[i1 + 1:]
    except ValueError:
        block_old, before, after = [], lines, []

    prev_patterns = [ln.strip() for ln in block_old
                     if ln.strip().startswith(f'/{data_subdir}/')
                     and not ln.strip().startswith('!')]
    want = [_pattern_for(n, data_subdir) for n in names]
    all_patterns = sorted(set(prev_patterns) | set(want))
    newly_added = [w for w in want if w not in prev_patterns]

    # NOTE: .gitignore has NO inline comments — '#' is only a comment at the start
    # of a line. Every comment below is therefore on its own line.
    guard = f'!/{data_subdir}/{SAMPLE_EXPERIMENT}/'
    block = [
        _BLOCK_START,
        '# Private experimental data mapped by consolidate_data.py — never commit.',
        *all_patterns,
        '# Keep the synthetic demo dataset tracked even if data/* is ignored above:',
        guard,
        _BLOCK_END,
    ]

    new_lines = list(before)
    if new_lines and new_lines[-1].strip() != '':
        new_lines.append('')
    new_lines += block + after
    new_text = '\n'.join(new_lines).rstrip('\n') + '\n'

    if not dry_run and new_text != text:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_text)
    return newly_added, path


# --------------------------------------------------------------------------- #
# Verification                                                                  #
# --------------------------------------------------------------------------- #
def verify_ignored(root, names, data_subdir='data'):
    """Use `git check-ignore` to prove real data is ignored and the sample is not.

    Returns (ok, details). `ok` is False if any real folder is NOT ignored or the
    sample IS ignored. Silently returns ok=True with a note if Git is unavailable.
    """
    def _ignored(rel):
        try:
            r = subprocess.run(['git', '-C', root, 'check-ignore', '-q', rel],
                               capture_output=True, timeout=5)
        except Exception:
            return None                       # git not available
        if r.returncode == 128:
            return None                       # not a git repo
        return r.returncode == 0              # 0 => ignored

    details, ok = [], True
    for name in names:
        rel = f'{data_subdir}/{name}'
        ig = _ignored(rel)
        if ig is None:
            return True, [('git-unavailable', 'skipped git check-ignore verification')]
        details.append((rel, 'IGNORED' if ig else 'NOT IGNORED (!)'))
        ok = ok and ig

    sample_rel = f'{data_subdir}/{SAMPLE_EXPERIMENT}'
    sig = _ignored(sample_rel)
    if sig is not None:
        details.append((sample_rel, 'tracked (correct)' if not sig
                        else 'IGNORED (!) — should stay tracked'))
        ok = ok and (not sig)
    return ok, details


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #
def _print_success(root, mapped, newly_ignored, mode, dry_run):
    data_dir = os.path.join(root, 'data')
    print('\n' + '=' * 70)
    print('  consolidate_data — local data mapping complete'
          + ('  (DRY RUN — nothing changed)' if dry_run else ''))
    print('=' * 70)
    if mapped:
        print(f'\n  Real experiments mapped into {data_dir} ({mode}):')
        for name, status in mapped:
            print(f'    - {name:35s} [{status}]')
    else:
        print('\n  No real experiment folders were found to map.')
    print('\n  .gitignore: '
          + (f'added {len(newly_ignored)} new ignore rule(s)' if newly_ignored
             else 'already up to date')
          + ' — your real data can never be committed.')
    print('\n  How to run the pipeline')
    print('  ----------------------')
    print('  • Out-of-the-box on synthetic demo data (no setup, safe to commit):')
    print('        leave config.json EXPERIMENT as-is  →  uses data/sample_experiment/')
    print('        python -m standard.run_analysis')
    print('  • On your own (private, git-ignored) data:')
    print('        python -m shared.consolidate_data "/path/to/your/data/snapshot"')
    print('        set  "EXPERIMENT": "experiment_a"  in config.json')
    print('        python -m standard.run_analysis')
    print('=' * 70 + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Map local real datasets into data/ and lock them out of Git.')
    parser.add_argument('source',
                        help='Path to your real data snapshot (a repo with data/, '
                             'or a data/ directory directly).')
    parser.add_argument('--copy', action='store_true',
                        help='Copy folders instead of symlinking (more disk).')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would happen without changing anything.')
    parser.add_argument('--no-link', action='store_true',
                        help='Only update .gitignore; do not create links/copies.')
    parser.add_argument('--no-gitignore', action='store_true',
                        help='Skip the .gitignore update (not recommended).')
    parser.add_argument('--repo', default=repo_root(),
                        help='Repository root (defaults to this checkout).')
    args = parser.parse_args(argv)

    root = os.path.abspath(os.path.expanduser(args.repo))
    mode = 'copy' if args.copy else 'symlink'

    try:
        src_root, experiments = discover_experiments(args.source)
    except FileNotFoundError as exc:
        print(f'  [error] {exc}', file=sys.stderr)
        return 2

    print(f'  Source experiments root : {src_root}')
    print(f'  Destination data dir    : {os.path.join(root, "data")}')
    print(f'  Mode                    : {mode}'
          + ('  (links skipped: --no-link)' if args.no_link else ''))
    if not experiments:
        print('  [warning] No real experiment folders (with .csv/.txt) found to map.')

    names = [name for name, _ in experiments]

    mapped = []
    if not args.no_link:
        for name, abspath in experiments:
            dst = os.path.join(root, 'data', name)
            status = map_experiment(abspath, dst, mode, args.dry_run)
            mapped.append((name, status))

    newly_ignored = []
    if not args.no_gitignore and names:
        newly_ignored, gi_path = update_gitignore(root, names, args.dry_run)
        if not args.dry_run:
            print(f'  Updated .gitignore      : {gi_path}')

    _print_success(root, mapped or [(n, 'gitignore-only') for n in names],
                   newly_ignored, mode, args.dry_run)

    if not args.dry_run and names:
        ok, details = verify_ignored(root, names)
        print('  Privacy verification (git check-ignore):')
        for rel, status in details:
            print(f'    {status:25s} {rel}')
        if not ok:
            print('  [error] Privacy boundary NOT fully locked down — review above.',
                  file=sys.stderr)
            return 1
        print('  ✓ All real datasets are git-ignored; the synthetic sample stays '
              'tracked.\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
