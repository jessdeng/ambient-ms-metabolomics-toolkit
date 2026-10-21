"""Shared helpers for the figure scripts in src/scripts."""
import os
import sys


def experiment_and_folder(output_dir, config, prefix='feature_overlap'):
    """Return (experiment, safe_name, folder) for the figure scripts.

    The experiment is the first command-line argument if given, otherwise
    EXPERIMENT from config.json. ``folder`` is the results folder that holds
    ``<prefix>_<safe_name>.csv``: ``results/<pipeline>/<experiment>/`` (v2) or,
    for older runs, ``results/<pipeline>/``. It is None if neither exists.
    """
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    experiment = (args[0] if args else str(config.EXPERIMENT)).strip()
    safe_name = experiment.replace(' ', '_').replace(':', '')
    for folder in (os.path.join(output_dir, safe_name), output_dir):
        if os.path.isfile(os.path.join(folder, f'{prefix}_{safe_name}.csv')):
            return experiment, safe_name, folder
    return experiment, safe_name, None


def report_missing(output_dir, experiment, safe_name, script, prefix='feature_overlap'):
    """Explain which file was looked for and list experiments that have results."""
    print(f"No {prefix}_{safe_name}.csv found for experiment {experiment!r}.")
    print(f"  Looked in: {os.path.join(output_dir, safe_name)}")
    done = []
    if os.path.isdir(output_dir):
        done = sorted(d for d in os.listdir(output_dir)
                      if os.path.isfile(os.path.join(output_dir, d, f'{prefix}_{d}.csv')))
    if done:
        print(f"  Experiments with results: {', '.join(done)}")
        print(f"  Set EXPERIMENT in config.json to one of these, or run e.g.:")
        print(f"      python src/scripts/{script}.py {done[0]}")
    else:
        print("  Run src/standard/run_analysis.py for this experiment first.")
