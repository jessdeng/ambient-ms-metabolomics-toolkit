"""
condition_spectrum.py — interactive (HTML) condition-mean spectra with the
candidates marked; candidates consistent across colonies are bold and labelled.

    python src/scripts/condition_spectrum.py [experiment]
"""
import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.dirname(_SRC), _SRC]

import pandas as pd

import config
from scripts._common import experiment_and_folder, ROOT
from standard.preprocessing import load_experiment, bin_spectra
from shared.grouping import resolve_design
from shared.visualization import plot_spectrum_with_features


def main():
    experiment, safe, folder = experiment_and_folder(config)
    if folder is None:
        return
    cand = pd.read_csv(os.path.join(folder, f'candidates_{safe}.csv'))
    if cand.empty:
        print("No candidates for this experiment.")
        return
    exp_dir = os.path.join(ROOT, 'data', experiment)
    X, names, mz = load_experiment(exp_dir)
    cond, _ = resolve_design(exp_dir, names, config.METADATA_MANIFEST)
    Xb, mzb = bin_spectra(X, mz, config.BIN_WIDTH, config.MZ_MIN, config.MZ_MAX, config.BIN_LABELS)
    out = os.path.join(folder, f'condition_spectrum_{safe}.html')
    plot_spectrum_with_features(Xb, mzb, cond, cand, experiment, out,
                                high_mask=cand['colonies_consistent'].to_numpy(), interactive=True)


if __name__ == '__main__':
    main()
