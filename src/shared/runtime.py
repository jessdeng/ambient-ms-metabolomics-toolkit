"""
runtime.py — out-of-the-box configuration safety + run provenance
=================================================================
Two small, dependency-light utilities used by the run_analysis entry points:

* :func:`resolve_experiment_dir` — pick the experiment folder to analyse. If the
  configured ``EXPERIMENT`` is a placeholder or points at a folder with no data,
  it falls back to a bundled sample experiment instead of crashing, so a fresh
  clone runs end-to-end with ``python -m standard.run_analysis``.

* :func:`write_run_manifest` — capture the software environment (package
  versions, Git commit, config parameters) and write ``run_manifest.json`` next
  to the results, so every figure is traceable to the exact code + settings that
  produced it.
"""

import os
import sys
import json
import datetime
import platform
import subprocess

# Default name of the bundled sample experiment shipped under data/.
SAMPLE_EXPERIMENT = 'sample_experiment'
# EXPERIMENT values treated as "not configured yet".
_PLACEHOLDERS = {'', 'your_experiment_folder', 'your_experiment_folder_A',
                 'your_experiment_folder_B'}


def _has_data_files(directory):
    """True if `directory` contains at least one .csv/.txt anywhere below it."""
    if not directory or not os.path.isdir(directory):
        return False
    for _dp, _dn, filenames in os.walk(directory):
        if any(f.lower().endswith(('.csv', '.txt')) and not f.startswith('.')
               for f in filenames):
            return True
    return False


def resolve_experiment_dir(base_dir, experiment, sample_name=SAMPLE_EXPERIMENT):
    """Resolve which experiment directory to analyse, with a safe fallback.

    Resolution order:
      1. ``data/<experiment>`` if ``experiment`` is set (not a placeholder) and the
         folder actually contains data — used as configured.
      2. otherwise the bundled ``data/<sample_name>`` if it has data — a loud
         warning is printed so the user knows results are from the demo set.
      3. otherwise a clear ``FileNotFoundError`` listing the available folders.

    Parameters
    ----------
    base_dir : str        repository root (the folder that contains ``data/``).
    experiment : str      the configured ``config.EXPERIMENT`` value.
    sample_name : str     bundled fallback experiment folder name.

    Returns
    -------
    (experiment_dir, experiment_name, used_fallback)
    """
    data_root = os.path.join(base_dir, 'data')
    configured = os.path.join(data_root, experiment) if experiment else None

    if (experiment and experiment not in _PLACEHOLDERS
            and _has_data_files(configured)):
        return configured, experiment, False

    sample_dir = os.path.join(data_root, sample_name)
    if _has_data_files(sample_dir):
        reason = ("is a placeholder" if (experiment or '') in _PLACEHOLDERS
                  else f"folder {configured!r} has no .csv/.txt data")
        print(f"  [warning] config.EXPERIMENT {reason}; falling back to the bundled "
              f"sample experiment '{sample_name}'. Set EXPERIMENT in config.json to "
              f"your own folder under data/ to analyse real data.")
        return sample_dir, sample_name, True

    available = []
    if os.path.isdir(data_root):
        available = sorted(d for d in os.listdir(data_root)
                           if os.path.isdir(os.path.join(data_root, d))
                           and not d.startswith('.'))
    raise FileNotFoundError(
        f"No usable experiment data found. config.EXPERIMENT={experiment!r} did not "
        f"resolve to a folder with .csv/.txt files, and no bundled "
        f"'{sample_name}' was found under {data_root!r}. Available folders: "
        f"{available or '(none)'}."
    )


# --------------------------------------------------------------------------- #
# Run provenance manifest                                                       #
# --------------------------------------------------------------------------- #
def _package_versions(packages=('numpy', 'pandas', 'scipy', 'scikit-learn',
                                'joblib', 'matplotlib')):
    """Map package -> installed version string (or 'not installed')."""
    try:
        from importlib import metadata as _md
    except Exception:                          # pragma: no cover (py<3.8)
        import importlib_metadata as _md        # type: ignore
    out = {}
    for pkg in packages:
        try:
            out[pkg] = _md.version(pkg)
        except Exception:
            out[pkg] = 'not installed'
    return out


def _git_info(repo_dir):
    """Best-effort Git provenance: commit SHA, branch, dirty flag. None if no Git."""
    def _run(args):
        return subprocess.run(['git', '-C', repo_dir, *args],
                              capture_output=True, text=True, timeout=5)
    try:
        head = _run(['rev-parse', 'HEAD'])
        if head.returncode != 0:
            return None
        commit = head.stdout.strip()
        branch = _run(['rev-parse', '--abbrev-ref', 'HEAD']).stdout.strip() or None
        dirty = bool(_run(['status', '--porcelain']).stdout.strip())
        return {'commit': commit, 'branch': branch, 'dirty': dirty}
    except Exception:
        return None


def _load_config_json(base_dir):
    """Return the raw dict of overrides from config.json, or {} if absent."""
    path = os.path.join(base_dir, 'config.json')
    if not os.path.exists(path):
        return {}, None
    try:
        with open(path) as f:
            return json.load(f), path
    except Exception:
        return {}, path


def _resolved_config(config_module):
    """Snapshot the effective UPPER_CASE config values (JSON-serialisable only)."""
    if config_module is None:
        return {}
    snap = {}
    for k in dir(config_module):
        if not k.isupper():
            continue
        v = getattr(config_module, k)
        if isinstance(v, (bool, int, float, str)) or v is None:
            snap[k] = v
    return snap


def write_run_manifest(out_dir, base_dir, config_module=None,
                       experiment_name=None, experiment_dir=None,
                       pipeline=None, extra=None, filename='run_manifest.json'):
    """Write ``run_manifest.json`` capturing the run's software/parameter provenance.

    Records the timestamp, Python/OS, key package versions (numpy, pandas, scipy,
    scikit-learn, joblib, matplotlib), the active Git commit/branch/dirty flag (if
    the tree is a Git repo), the raw ``config.json`` overrides, and the effective
    resolved config values — so any figure can be traced back to its exact
    environment and settings.

    Returns
    -------
    (manifest_dict, out_path)
    """
    os.makedirs(out_dir, exist_ok=True)
    config_json, config_json_path = _load_config_json(base_dir)
    manifest = {
        'generated_at': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
        'experiment': experiment_name,
        'experiment_dir': experiment_dir,
        'pipeline': pipeline,
        'python_version': sys.version.split()[0],
        'platform': platform.platform(),
        'packages': _package_versions(),
        'git': _git_info(base_dir),
        'config_json_path': config_json_path,
        'config_json': config_json,
        'resolved_config': _resolved_config(config_module),
    }
    if extra:
        manifest['extra'] = extra

    out_path = os.path.join(out_dir, filename)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"  Wrote run manifest -> {out_path}")
    return manifest, out_path
