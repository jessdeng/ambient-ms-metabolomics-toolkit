"""
Experiment-folder resolution and run provenance (run_manifest.json).
"""

import datetime
import json
import os
import platform
import subprocess
import sys


def resolve_experiment_dir(base_dir, experiment):
    """``data/<experiment>`` if it contains spectra, else a clear error."""
    data_root = os.path.join(base_dir, 'data')
    path = os.path.join(data_root, experiment or '')
    has_data = os.path.isdir(path) and any(
        f.lower().endswith(('.csv', '.txt')) and f != 'metadata_manifest.csv'
        for _, _, files in os.walk(path) for f in files)
    if experiment and has_data:
        return path
    available = sorted(d for d in os.listdir(data_root)
                       if os.path.isdir(os.path.join(data_root, d))) if os.path.isdir(data_root) else []
    raise FileNotFoundError(f"EXPERIMENT={experiment!r} has no .csv/.txt spectra under data/. "
                            f"Available folders: {available}")


def _package_versions(packages=('numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib')):
    from importlib import metadata
    out = {}
    for pkg in packages:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = 'not installed'
    return out


def _git_info(repo_dir):
    """Commit, branch and dirty flag, or None outside a Git checkout."""
    run = lambda *a: subprocess.run(['git', '-C', repo_dir, *a], capture_output=True,
                                    text=True, timeout=5)
    try:
        head = run('rev-parse', 'HEAD')
    except (OSError, subprocess.SubprocessError):
        return None
    if head.returncode != 0:
        return None
    return {'commit': head.stdout.strip(),
            'branch': run('rev-parse', '--abbrev-ref', 'HEAD').stdout.strip() or None,
            'dirty': bool(run('status', '--porcelain').stdout.strip())}


def write_run_manifest(out_dir, base_dir, config_module, experiment_dir, extra=None):
    """Write run_manifest.json: time, versions, Git state and every setting used."""
    settings = {k: getattr(config_module, k) for k in dir(config_module)
                if k.isupper() and isinstance(getattr(config_module, k), (bool, int, float, str, type(None)))}
    manifest = {
        'generated_at': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
        'experiment_dir': experiment_dir,
        'python_version': sys.version.split()[0],
        'platform': platform.platform(),
        'packages': _package_versions(),
        'git': _git_info(base_dir),
        'settings': settings,
    }
    if extra:
        manifest['extra'] = extra
    path = os.path.join(out_dir, 'run_manifest.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=str)
    return path
