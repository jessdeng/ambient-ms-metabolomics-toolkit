"""Shared helpers for the figure scripts in src/scripts."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, 'results')


def experiment_and_folder(config, prefix='candidates'):
    """(experiment, safe_name, folder) for a figure script.

    The experiment is the first command-line argument, else EXPERIMENT from the
    settings file. ``folder`` is results/<experiment>/ if it holds
    ``<prefix>_<safe_name>.csv``, else None.
    """
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    experiment = (args[0] if args else str(config.EXPERIMENT)).strip()
    safe = experiment.replace(' ', '_').replace(':', '')
    folder = os.path.join(RESULTS, safe)
    if os.path.isfile(os.path.join(folder, f'{prefix}_{safe}.csv')):
        return experiment, safe, folder
    print(f"No {prefix}_{safe}.csv in {folder}. Run src/standard/run_analysis.py "
          f"for {experiment!r} first.")
    return experiment, safe, None
