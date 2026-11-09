"""
Ambient MS feature prioritisation — one run per experiment.

    python src/standard/run_analysis.py --config config.json --experiment MY_EXPERIMENT

Question: which m/z features differ between the conditions, consistently across
colonies, and look like real ions? Results go to results/<experiment>/.
"""

import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(_SRC)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _SRC)

# The matrices here are tiny (tens of spectra); multi-threaded BLAS (Apple
# Accelerate in particular) spends far longer coordinating threads than computing:
# logistic regression went from 16 s to 0.08 s per fit with one thread. Must be
# set before numpy is imported. An explicit setting in the environment wins.
for _v in ('VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')


def _parse_cli():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument('--config', help='JSON settings file (default: config.json)')
    ap.add_argument('--experiment', help='folder under data/ (overrides EXPERIMENT)')
    a, _ = ap.parse_known_args()
    if a.config:
        os.environ['TOOLKIT_CONFIG'] = os.path.abspath(a.config)
    if a.experiment:
        os.environ['TOOLKIT_EXPERIMENT'] = a.experiment


if __name__ == '__main__':
    _parse_cli()

import time

import numpy as np
import pandas as pd

import config
from standard.preprocessing import load_experiment, build_matrices, scale, tic_normalise
from standard.pipeline import plot_vip
from shared.grouping import resolve_design
from shared.ranking import (colony_means, rank_features, colony_consistency,
                            variance_partition, technical_cv, concordance, VOTERS,
                            SENSITIVITY)
from shared.feature_groups import build_signal_groups
from shared.evidence import candidate_evidence
from shared.pca import run_pca
from shared.classifiers import compare_classifiers, plot_accuracy_comparison
from shared.visualization import plot_spectrum_with_features
from shared.runtime import resolve_experiment_dir, write_run_manifest


def main():
    t0 = time.time()
    experiment = os.environ.get('TOOLKIT_EXPERIMENT') or config.EXPERIMENT
    exp_dir = resolve_experiment_dir(_ROOT, experiment)
    safe = experiment.strip().replace(' ', '_').replace(':', '')
    out = os.path.join(_ROOT, 'results', safe)
    os.makedirs(out, exist_ok=True)
    print(f"Experiment: {experiment!r}\n  settings: {config.CONFIG_FILE}\n  output:   {out}")

    # -- spectra, design -------------------------------------------------------
    X_raw, names, raw_mz = load_experiment(exp_dir)
    cond, colony = resolve_design(exp_dir, names, config.METADATA_MANIFEST, config.EXPECTED_TECH_REPS)
    print(f"  {len(names)} spectra, {len(set(colony))} colonies, conditions: "
          + ', '.join(f"{c} ({len(set(colony[cond == c]))})" for c in np.unique(cond)))
    M = build_matrices(X_raw, raw_mz, config.BIN_WIDTH, config.BIN_LABELS, config.MZ_MIN,
                       config.MZ_MAX, config.VARIANCE_PERCENTILE, config.ABUNDANCE_PERCENTILE)
    mz = M['mz']
    print(f"  {M['binned'].shape[1]} bins in {config.MZ_MIN}-{config.MZ_MAX} Da; "
          f"{mz.size} after the variance/abundance filters")

    # -- unsupervised overview (spectra; labels only colour the plot) ----------
    pca_ev = run_pca(scale(M['log'], config.SCALING), cond, colony, mz, names, out, safe,
                     experiment, n_components=config.N_PCA_COMPONENTS)
    r = pca_ev.iloc[0]
    print(f"  PCA: PC1 {r.explained_variance_ratio:.0%} of variance; of its score variance "
          f"{r.frac_condition_plate:.0%} condition-plate, {r.frac_colony:.0%} colony, "
          f"{r.frac_technical:.0%} technical")

    # -- colony-level prioritisation -------------------------------------------
    Xc, colony_ids, cond_c = colony_means(M['log'], colony, cond)
    table, info = rank_features(scale(Xc, config.SCALING), Xc, cond_c, config.TOP_N_FEATURES,
                                config.MIN_RANKERS, config.RF_SEEDS, config.RANDOM_SEED)
    table.insert(0, 'mz', mz)
    table = pd.concat([table,
                       colony_consistency(Xc, cond_c, table['top_condition'].to_numpy(),
                                          table['bottom_condition'].to_numpy()),
                       variance_partition(M['log'], cond, colony)], axis=1)
    table['technical_cv_pct'] = technical_cv(M['linear'], colony)

    smap, spairs = build_signal_groups(M['binned'], M['mz_binned'], X_raw, raw_mz, colony,
                                       bin_width=config.BIN_WIDTH,
                                       mz0=float(np.floor(raw_mz.min() / config.BIN_WIDTH + 1e-9) * config.BIN_WIDTH),
                                       r_min=config.SIGNAL_R_MIN)
    sig = smap.set_index('mz_bin').reindex(mz)
    table['signal_group'] = sig['signal_group'].to_numpy()
    table['signal_mz'] = sig['signal_mz'].to_numpy()
    table['n_bins_in_signal'] = sig['n_bins_in_group'].to_numpy()

    table = table.sort_values(['n_rankers', 'mean_voter_rank'], ascending=[False, True])
    table.to_csv(os.path.join(out, f'feature_rankings_{safe}.csv'), index=False)
    smap.to_csv(os.path.join(out, f'signal_groups_{safe}.csv'), index=False)

    # -- candidates: add spectral evidence and colony values --------------------
    cand = table[table['candidate']].copy()
    if len(cand):
        cand = cand.join(candidate_evidence(cand, smap, M['binned'], X_raw, raw_mz, colony,
                                            config.BIN_WIDTH))
        col_vals = pd.DataFrame(Xc[:, cand.index].T, index=cand.index,
                                columns=[f'colony_log10_{c}' for c in colony_ids])
        cand = cand.join(col_vals)
    cand.to_csv(os.path.join(out, f'candidates_{safe}.csv'), index=False)

    conc = concordance(info['scores'], config.TOP_N_FEATURES)
    conc.to_csv(os.path.join(out, f'ranker_concordance_{safe}.csv'), index=False)

    # -- figures -----------------------------------------------------------------
    conds = np.unique(cond_c)
    plot_vip(info['scores']['vip'], mz, np.vstack([Xc[cond_c == c].mean(0) for c in conds]),
             list(conds), config.N_TOP_VIP, experiment, os.path.join(out, f'vip_scores_{safe}.png'))
    if len(cand):
        plot_spectrum_with_features(M['binned'], M['mz_binned'], cond, cand, experiment,
                                    os.path.join(out, f'spectrum_features_{safe}.png'),
                                    high_mask=cand['colonies_consistent'].to_numpy())

    # -- classifier comparison (condition-plate discriminability) ----------------
    acc = None
    if config.RUN_CLASSIFIERS:
        folds, acc = compare_classifiers(tic_normalise(M['binned']), cond, colony, config.CV_FOLDS,
                                         config.N_CV_REPEATS, config.RANDOM_SEED, config.SCALING,
                                         config.VARIANCE_PERCENTILE, config.ABUNDANCE_PERCENTILE)
        acc.to_csv(os.path.join(out, f'classifier_accuracy_{safe}.csv'), index=False)
        plot_accuracy_comparison(folds, experiment, os.path.join(out, f'classifier_comparison_{safe}.png'),
                                 chance=1.0 / len(np.unique(cond)))

    write_run_manifest(out, _ROOT, config, exp_dir,
                       extra={'rf_seed_agreement_topN': info['rf_seed_agreement']})

    # -- summary -----------------------------------------------------------------
    n = config.TOP_N_FEATURES
    print(f"\n  Rankers (top-{n}): " + ' + '.join(VOTERS) + " vote; " + ' + '.join(SENSITIVITY)
          + " are sensitivity columns")
    for row in conc.itertuples():
        print(f"    {row.ranker_a:>10} vs {row.ranker_b:<10} Jaccard {row.jaccard:.2f} "
              f"(chance {row.chance:.3f})")
    print(f"    random forest seed-to-seed agreement: {info['rf_seed_agreement']:.2f}")
    print(f"  Candidates (top-{n} in >= {config.MIN_RANKERS} of {len(VOTERS)} rankers): {len(cand)}; "
          f"consistent across all colonies: {int(cand['colonies_consistent'].sum()) if len(cand) else 0}")
    if acc is not None:
        print(f"  Classifier comparison ({acc.cv.iloc[0]}; chance {acc.chance_level.iloc[0]:.2f}):")
        for r in acc.itertuples():
            print(f"    {r.model:20} test {r.test_accuracy_mean:.2f} +/- {r.test_accuracy_sd:.2f}   "
                  f"train {r.train_accuracy_mean:.2f}")
    print(f"Done in {time.time() - t0:.0f} s -> {out}")


if __name__ == '__main__':
    main()
