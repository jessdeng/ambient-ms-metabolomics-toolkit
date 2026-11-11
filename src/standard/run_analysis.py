"""
Ambient MS feature prioritisation — everything in one run.

    python src/standard/run_analysis.py                           # EXPERIMENT from the settings
    python src/standard/run_analysis.py --experiment MY_EXPERIMENT
    python src/standard/run_analysis.py --all                     # every experiment under data/

Question: which m/z features differ between the conditions, consistently across
colonies, and look like real ions — i.e. which are worth chemical follow-up?

The run prints its progress stage by stage:

    design -> features -> structure (PCA) -> rankings -> method comparison ->
    follow-up list and spectral evidence -> classifier comparison ->
    permutation test -> report

Tables, figures and a plain-text summary go to results/<experiment>/. With --all,
tables and figures that span the experiments go to results/.
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

DEMO = 'sample_experiment'


def _parse_cli():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument('--config', help='JSON settings file (default: config.json, then config.local.json)')
    ap.add_argument('--experiment', help='folder under data/ (overrides EXPERIMENT)')
    ap.add_argument('--all', action='store_true',
                    help=f'run every experiment under data/ (the bundled {DEMO} is skipped '
                         'when you have your own)')
    a, _ = ap.parse_known_args()
    if a.config:
        os.environ['TOOLKIT_CONFIG'] = os.path.abspath(a.config)
    if a.experiment:
        os.environ['TOOLKIT_EXPERIMENT'] = a.experiment
    if a.all:
        os.environ['TOOLKIT_EXPERIMENT'] = 'all'


if __name__ == '__main__':
    _parse_cli()

import datetime
import time
import traceback

import numpy as np
import pandas as pd

import config
from standard.preprocessing import load_experiment, build_matrices, scale, tic_normalise
from standard.pipeline import plot_vip
from shared.grouping import resolve_design
from shared.ranking import (colony_means, rank_features, colony_consistency,
                            variance_partition, technical_cv, concordance, selected_by,
                            VOTERS, METHODS)
from shared.feature_groups import build_signal_groups
from shared.evidence import candidate_evidence
from shared.pca import run_pca
from shared.classifiers import compare_classifiers, plot_accuracy_comparison, MODELS
from shared.validation import RepeatedColonyKFold, colony_permutation_test
from shared.rank_aggregation import consensus_label
from shared.visualization import plot_spectrum_with_features
from shared.runtime import resolve_experiment_dir, write_run_manifest
from scripts import corroboration_figure, ranker_agreement, condition_abundance
from scripts import vip_vs_nmethods, pca_overview

RANKER_NAMES = {'univariate': 'Univariate F', 'vip': 'VIP (comp. 1)', 'ridge': 'Ridge',
                'lr': 'Logistic regression', 'svm': 'Linear SVM', 'rf': 'Random forest',
                'gb': 'Gradient boosting'}


class _Progress:
    """Prints '[k/n] title' when a stage starts and its duration when it ends."""

    def __init__(self, n):
        self.n, self.k, self.t0 = n, 0, None

    def stage(self, title):
        self.done()
        self.k += 1
        self.t0 = time.time()
        print(f"\n[{self.k}/{self.n}] {title}", flush=True)

    def done(self):
        if self.t0 is not None:
            print(f"      ({time.time() - self.t0:.0f} s)", flush=True)
            self.t0 = None


def _safe(experiment):
    return experiment.strip().replace(' ', '_').replace(':', '')


def _experiments():
    """The experiments to run: one name, or every folder with spectra under data/."""
    name = (os.environ.get('TOOLKIT_EXPERIMENT') or str(config.EXPERIMENT)).strip()
    if name.lower() != 'all':
        return [name]
    root = os.path.join(_ROOT, 'data')
    found = []
    for d in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        try:
            resolve_experiment_dir(_ROOT, d)
            found.append(d)
        except (FileNotFoundError, NotADirectoryError):
            pass
    own = [d for d in found if d != DEMO]
    if not found:
        raise FileNotFoundError("No experiment folders with spectra under data/.")
    return own or found


def _permutation_models():
    names = [n.strip() for n in str(config.PERMUTATION_MODELS).split(',') if n.strip()]
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        raise ValueError(f"PERMUTATION_MODELS: unknown model(s) {unknown}; choose from {list(MODELS)}")
    return names


def run_experiment(experiment):
    """All stages for one experiment. Returns a dict of headline results."""
    t0 = time.time()
    exp_dir = resolve_experiment_dir(_ROOT, experiment)
    safe = _safe(experiment)
    out = os.path.join(_ROOT, 'results', safe)
    os.makedirs(out, exist_ok=True)
    n_top = config.TOP_N_FEATURES
    seed = config.RANDOM_SEED
    progress = _Progress(7 + bool(config.RUN_CLASSIFIERS) + bool(config.RUN_PERMUTATION))
    print(f"\n{'=' * 72}\nExperiment: {experiment!r}\n  settings: {config.CONFIG_FILE}\n  output:   {out}")
    report = {'experiment': experiment, 'safe': safe, 'out': out, 'notes': []}

    # -- 1. spectra and design --------------------------------------------------
    progress.stage("Design: spectra, conditions and colonies")
    X_raw, names, raw_mz = load_experiment(exp_dir)
    cond, colony = resolve_design(exp_dir, names, config.METADATA_MANIFEST, config.EXPECTED_TECH_REPS)
    per_cond = {c: len(set(colony[cond == c])) for c in np.unique(cond)}
    print(f"  {len(names)} spectra, {len(set(colony))} colonies; colonies per condition: "
          + ', '.join(f"{c} {n}" for c, n in per_cond.items()))
    ids, counts = np.unique(colony, return_counts=True)
    usual = int(np.bincount(counts).argmax())
    odd = {i: int(n) for i, n in zip(ids, counts) if n != usual}
    if odd:
        note = (f"{len(odd)} colonies do not have the usual {usual} spectra: {odd}. If these are "
                "one colony named two ways, fix the file names or add a metadata manifest.")
        print(f"  WARNING: {note}")
        report['notes'].append(note)
    report.update(n_spectra=len(names), n_colonies=len(set(colony)), per_condition=per_cond)

    # -- 2. spectra -> features -------------------------------------------------
    progress.stage(f"Features: {config.BIN_WIDTH}-Da bins, TIC normalisation, filters, log10")
    M = build_matrices(X_raw, raw_mz, config.BIN_WIDTH, config.BIN_LABELS, config.MZ_MIN,
                       config.MZ_MAX, config.VARIANCE_PERCENTILE, config.ABUNDANCE_PERCENTILE,
                       config.LOG_OFFSET)
    mz = M['mz']
    print(f"  {M['binned'].shape[1]} bins in {config.MZ_MIN}-{config.MZ_MAX} Da; "
          f"{mz.size} after the variance/abundance filters")
    report.update(n_bins=int(M['binned'].shape[1]), n_features=int(mz.size))

    # -- 3. unsupervised overview (labels only colour the plot) -------------------
    progress.stage("Structure of the data: PCA and its variance components")
    pca_ev = run_pca(scale(M['log'], config.SCALING), cond, colony, mz, names, out, safe,
                     experiment, n_components=config.N_PCA_COMPONENTS)
    r = pca_ev.iloc[0]
    report['pca'] = (f"PC1 {r.explained_variance_ratio:.0%} of variance; of its score variance "
                     f"{r.frac_condition_plate:.0%} condition-plate, {r.frac_colony:.0%} colony, "
                     f"{r.frac_technical:.0%} technical")
    print(f"  {report['pca']}")

    # -- 4. rankings on colony means ---------------------------------------------
    progress.stage(f"Rankings: univariate F and six importance methods on {len(set(colony))} colony means")
    Xc, colony_ids, cond_c = colony_means(M['log'], colony, cond)
    table, info = rank_features(scale(Xc, config.SCALING), Xc, cond_c, n_top, config.MIN_RANKERS,
                                config.RF_SEEDS, seed, config.CONSENSUS_RULE,
                                config.CONSENSUS_MAX_WORST_FAMILY_RANK,
                                config.CONSENSUS_MIN_FAMILIES_TOPK,
                                gb_seeds=config.GB_SEEDS, n_jobs=config.N_JOBS)
    table.insert(0, 'mz', mz)
    table = pd.concat([table,
                       colony_consistency(Xc, cond_c, table['top_condition'].to_numpy(),
                                          table['bottom_condition'].to_numpy()),
                       variance_partition(M['log'], cond, colony)], axis=1)
    table['technical_cv_pct'] = technical_cv(M['linear'], colony)

    smap, spairs = build_signal_groups(M['binned'], M['mz_binned'], X_raw, raw_mz, colony,
                                       bin_width=config.BIN_WIDTH,
                                       mz0=float(config.MZ_MIN),
                                       r_min=config.SIGNAL_R_MIN)
    sig = smap.set_index('mz_bin').reindex(mz)
    table['signal_group'] = sig['signal_group'].to_numpy()
    table['signal_mz'] = sig['signal_mz'].to_numpy()
    table['n_bins_in_signal'] = sig['n_bins_in_group'].to_numpy()
    table['selected_by'] = selected_by(table)

    table = table.sort_values(['n_rankers', 'mean_voter_rank'], ascending=[False, True])
    table.to_csv(os.path.join(out, f'feature_rankings_{safe}.csv'), index=False)
    smap.to_csv(os.path.join(out, f'signal_groups_{safe}.csv'), index=False)
    print(f"  top-{n_top} of each ranking recorded for {mz.size} bins")
    print(f"  seed-to-seed agreement of the top-{n_top} (Jaccard, two halves of the seeds): "
          f"random forest {info['rf_seed_agreement']:.2f} ({config.RF_SEEDS} seeds), "
          f"gradient boosting {info['gb_seed_agreement']:.2f} ({config.GB_SEEDS} seeds)")
    report.update(rf_seed_agreement=info['rf_seed_agreement'],
                  gb_seed_agreement=info['gb_seed_agreement'])
    short = {RANKER_NAMES[m]: k for m, k in info['top_size'].items() if k < n_top}
    tied = {RANKER_NAMES[m]: k for m, k in info['ties_at_cutoff'].items() if k > 1}
    notes = []
    if short:
        notes.append(f"fewer than {n_top} bins have a non-zero score for {short}; those top "
                     "lists hold only the bins the method used")
    if tied:
        notes.append(f"the top-{n_top} cut falls among tied scores for {tied} (number of bins "
                     "tied); ties go to the lower m/z")
    for note in notes:
        print(f"  NOTE: {note}")
    report['notes'] += notes

    # -- 5. do the methods prioritise the same features? --------------------------
    progress.stage("Method comparison: do the rankings prioritise the same features?")
    conc = concordance(info['scores'], n_top)
    conc.to_csv(os.path.join(out, f'ranker_concordance_{safe}.csv'), index=False)
    overlap = table[(table['n_methods'] >= config.MIN_METHODS_OVERLAP) | table['consensus']]
    overlap = overlap.sort_values(['agg_worst_family_rank', 'agg_mean_family_rank', 'n_methods'],
                                  ascending=[True, True, False])
    overlap.to_csv(os.path.join(out, f'feature_overlap_{safe}.csv'), index=False)
    votes = table['n_methods'].value_counts().sort_index()
    rule = consensus_label(config.CONSENSUS_RULE, config.CONSENSUS_MAX_WORST_FAMILY_RANK,
                           config.CONSENSUS_MIN_FAMILIES_TOPK)
    print(f"  shared top-{n_top} features with VIP (comp. 1) (chance {n_top * n_top / mz.size:.1f}):")
    vip_rows = conc[(conc.ranker_a == 'vip') | (conc.ranker_b == 'vip')]
    for row in vip_rows.itertuples():
        other = row.ranker_b if row.ranker_a == 'vip' else row.ranker_a
        print(f"    {RANKER_NAMES[other]:<20} {row.shared:>3}")
    print("  bins by number of methods (of 6) that place them in the top-"
          f"{n_top}: " + ', '.join(f"{int(k)} methods: {int(v)}" for k, v in votes.items() if k > 0))
    print(f"  {rule}: {int(table['consensus'].sum())} bins")
    corroboration_figure.figure(experiment, safe, out)
    ranker_agreement.figure(experiment, safe, out)
    report.update(concordance=conc, votes=votes, consensus_rule=rule,
                  n_consensus=int(table['consensus'].sum()), n_overlap=len(overlap))

    # -- 6. the follow-up list, with spectral evidence ---------------------------
    progress.stage("Follow-up list: consistent across colonies? a real ion?")
    cand = table[table['candidate'] | table['consensus']].copy()
    if len(cand):
        cand = cand.join(candidate_evidence(cand, smap, M['binned'], X_raw, raw_mz, colony,
                                            config.BIN_WIDTH))
        col_vals = pd.DataFrame(Xc[:, cand.index].T, index=cand.index,
                                columns=[f'colony_log10_{c}' for c in colony_ids])
        cand = cand.join(col_vals)
    cand.to_csv(os.path.join(out, f'candidates_{safe}.csv'), index=False)
    n_voter = int(table['candidate'].sum())
    n_cons = int(cand['colonies_consistent'].sum()) if len(cand) else 0
    print(f"  {len(cand)} bins on the list: {n_voter} by the screen with VIP (top-{n_top} in >= "
          f"{config.MIN_RANKERS} of {len(VOTERS)}), {int(table['consensus'].sum())} by the method "
          f"consensus, {int((table['candidate'] & table['consensus']).sum())} by both")
    print(f"  consistent across all colonies: {n_cons}")
    conds = np.unique(cond_c)
    plot_vip(info['scores']['vip'], mz, np.vstack([Xc[cond_c == c].mean(0) for c in conds]),
             list(conds), config.N_TOP_VIP, experiment, os.path.join(out, f'vip_scores_{safe}.png'))
    if len(cand):
        high = cand['colonies_consistent'].to_numpy()
        plot_spectrum_with_features(M['binned'], M['mz_binned'], cond, cand, experiment,
                                    os.path.join(out, f'spectrum_features_{safe}.png'), high_mask=high)
        condition_abundance.figure(experiment, safe, out)
        try:
            plot_spectrum_with_features(M['binned'], M['mz_binned'], cond, cand, experiment,
                                        os.path.join(out, f'condition_spectrum_{safe}.html'),
                                        high_mask=high, interactive=True)
        except ImportError:
            print("  (interactive spectrum skipped: plotly is not installed)")
    report.update(candidates=cand, n_candidates=len(cand), n_voter=n_voter, n_consistent=n_cons)

    # -- 7. classifier comparison (condition-plate discriminability) --------------
    acc = None
    X_tic = tic_normalise(M['binned'])
    if config.RUN_CLASSIFIERS:
        progress.stage("Classifier comparison: can the conditions be told apart in unseen colonies?")
        folds, acc = compare_classifiers(
            X_tic, cond, colony, config.CV_FOLDS, config.N_CV_REPEATS, seed, config.SCALING,
            config.VARIANCE_PERCENTILE, config.ABUNDANCE_PERCENTILE,
            progress=lambda r, n: print(f"  repeat {r}/{n}", flush=True),
            log_offset=config.LOG_OFFSET)
        acc.to_csv(os.path.join(out, f'classifier_accuracy_{safe}.csv'), index=False)
        plot_accuracy_comparison(folds, experiment, os.path.join(out, f'classifier_comparison_{safe}.png'),
                                 chance=1.0 / len(np.unique(cond)))
        print(f"  {acc.cv.iloc[0]}; chance {acc.chance_level.iloc[0]:.2f}")
        for r in acc.itertuples():
            print(f"    {r.model:20} test {r.test_accuracy_mean:.2f} +/- {r.test_accuracy_sd:.2f}   "
                  f"train {r.train_accuracy_mean:.2f}")
    report['accuracy'] = acc

    # -- 8. permutation test -------------------------------------------------------
    perm = None
    if config.RUN_PERMUTATION:
        progress.stage(f"Permutation test: {config.N_PERMUTATIONS} shuffles of the condition labels "
                       "among colonies")
        min_col = min(per_cond.values())
        k = int(max(2, min(config.CV_FOLDS, min_col)))
        cv = RepeatedColonyKFold(n_splits=k, n_repeats=1, random_state=seed)
        rows = []
        for name in _permutation_models():
            obs, null, p = colony_permutation_test(
                MODELS[name](seed), X_tic, cond, colony, cv, n_permutations=config.N_PERMUTATIONS,
                seed=seed, n_jobs=config.N_JOBS, scaling=config.SCALING,
                variance_percentile=config.VARIANCE_PERCENTILE,
                abundance_percentile=config.ABUNDANCE_PERCENTILE,
                log_offset=config.LOG_OFFSET)
            rows.append(dict(model=name, observed_accuracy=obs, null_mean=null.mean(),
                             null_95th_percentile=np.percentile(null, 95), null_max=null.max(),
                             p_value=p, n_permutations=config.N_PERMUTATIONS,
                             cv=f'{k}-fold colony-grouped x 1'))
            print(f"    {name:20} observed {obs:.2f}   shuffled {null.mean():.2f} "
                  f"(max {null.max():.2f})   p = {p:.3f}", flush=True)
        perm = pd.DataFrame(rows)
        perm.to_csv(os.path.join(out, f'permutation_test_{safe}.csv'), index=False)
    report['permutation'] = perm

    # -- 9. report -------------------------------------------------------------------
    progress.stage("Report: summary and run manifest")
    write_run_manifest(out, _ROOT, config, exp_dir,
                       extra={'rf_seed_agreement_topN': info['rf_seed_agreement'],
                              'gb_seed_agreement_topN': info['gb_seed_agreement']})
    report['seconds'] = time.time() - t0
    path = write_summary(report)
    print(f"  Saved -> {path}")
    progress.done()
    print(f"\nDone in {time.time() - t0:.0f} s -> {out}")
    return report


def write_summary(rep):
    """Plain-text summary of one run: results/<experiment>/summary_<experiment>.txt."""
    n_top = config.TOP_N_FEATURES
    L = ["=" * 72, "AMBIENT MS FEATURE PRIORITISATION -- RUN SUMMARY",
         f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M}", "=" * 72, "",
         "Question: which m/z features differ between the conditions, consistently",
         "across colonies, and look like real ions -- i.e. which are worth chemical",
         "follow-up? Each condition was grown on one plate, so every difference below",
         "is a condition-plate association, not a proven effect of the treatment.", "",
         "-- Experiment " + "-" * 58,
         f"  Name      : {rep['experiment']}",
         f"  Spectra   : {rep['n_spectra']}   Colonies: {rep['n_colonies']}",
         "  Colonies per condition: " + ', '.join(f"{c} {n}" for c, n in rep['per_condition'].items())]
    L += [f"  NOTE: {n}" for n in rep['notes']]
    L += ["", "-- Settings " + "-" * 60,
          f"  Bins {config.BIN_WIDTH} Da over m/z {config.MZ_MIN}-{config.MZ_MAX}; filters: lowest "
          f"{config.VARIANCE_PERCENTILE}% by RSD, lowest {config.ABUNDANCE_PERCENTILE}% by mean",
          f"  TIC normalisation, log10 ({config.LOG_OFFSET} zero offset), {config.SCALING}; "
          f"top-N = {n_top}; seed {config.RANDOM_SEED}",
          f"  Features: {rep['n_bins']} bins, {rep['n_features']} after the filters",
          "", "-- Structure of the data " + "-" * 47, f"  PCA: {rep['pca']}",
          "", "-- Method comparison " + "-" * 51,
          f"  Shared top-{n_top} features between rankings (chance "
          f"{n_top * n_top / rep['n_features']:.1f}):"]
    conc = rep['concordance']
    for r in conc.itertuples():
        L.append(f"    {RANKER_NAMES[r.ranker_a]:<20} vs {RANKER_NAMES[r.ranker_b]:<20} {r.shared:>3}"
                 f"   (Jaccard {r.jaccard:.2f})")
    L.append(f"  Seed-to-seed agreement (top-{n_top} Jaccard between two halves of the seeds): "
             f"random forest {rep['rf_seed_agreement']:.2f} ({config.RF_SEEDS} seeds), "
             f"gradient boosting {rep['gb_seed_agreement']:.2f} ({config.GB_SEEDS} seeds)")
    L.append(f"  Bins in the top-{n_top} of k of the six methods ({', '.join(METHODS)}):")
    L.append("    " + ', '.join(f"k={int(k)}: {int(v)}" for k, v in rep['votes'].items()))
    L.append(f"  {rep['consensus_rule']}: {rep['n_consensus']} bins")

    L += ["", "-- Follow-up list " + "-" * 54,
          f"  {rep['n_candidates']} bins: {rep['n_voter']} by the colony-level screen with VIP, "
          f"{rep['n_consensus']} by the method consensus",
          f"  Consistent across all colonies: {rep['n_consistent']}", ""]
    cand = rep['candidates']
    if len(cand):
        L.append(f"  {'m/z':>8}  {'methods':>7}  {'F rank':>6}  {'VIP rank':>8}  {'apex':>7}  "
                 f"{'baseline':>8}  colonies / selected by")
        for r in cand.head(40).itertuples():
            pattern = r.colony_pattern if r.colonies_consistent else 'not consistent'
            L.append(f"  {r.mz:8.1f}  {int(r.n_methods):>5}/6  {int(r.univariate_rank):>6}  "
                     f"{int(r.vip_rank):>8}  {r.apex_mz:7.1f}  {r.baseline_share:8.2f}  "
                     f"{pattern}; {r.selected_by}")
        if len(cand) > 40:
            L.append(f"  ... and {len(cand) - 40} more in candidates_{rep['safe']}.csv")
        L.append("  baseline = local baseline as a share of the bin's signal (near 1 = no resolved peak)")

    acc = rep['accuracy']
    if acc is not None:
        L += ["", "-- Classifier comparison " + "-" * 47,
              f"  {acc.cv.iloc[0]}; chance {acc.chance_level.iloc[0]:.2f}"]
        for r in acc.itertuples():
            L.append(f"    {r.model:<20} test {r.test_accuracy_mean:.2f} +/- {r.test_accuracy_sd:.2f}   "
                     f"train {r.train_accuracy_mean:.2f}")
    perm = rep['permutation']
    if perm is not None:
        L += ["", "-- Permutation test " + "-" * 52,
              f"  {int(perm.n_permutations.iloc[0])} shuffles of the condition labels among colonies "
              f"({perm.cv.iloc[0]}); smallest attainable p = {1 / (perm.n_permutations.iloc[0] + 1):.3f}"]
        for r in perm.itertuples():
            L.append(f"    {r.model:<20} observed {r.observed_accuracy:.2f}   shuffled "
                     f"{r.null_mean:.2f} (max {r.null_max:.2f})   p = {r.p_value:.3f}")

    L += ["", "-- Output files " + "-" * 56]
    for f in sorted(set(os.listdir(rep['out'])) | {f"summary_{rep['safe']}.txt"}):
        L.append(f"  {f}")
    L += ["", f"Run time: {rep['seconds']:.0f} s", "=" * 72, ""]
    path = os.path.join(rep['out'], f"summary_{rep['safe']}.txt")
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L))
    return path


def across_experiments(reports):
    """Tables and figures that span the experiments -> results/."""
    out = os.path.join(_ROOT, 'results')
    print(f"\n{'=' * 72}\nAcross {len(reports)} experiments")
    accs = [r['accuracy'].assign(experiment=r['experiment']) for r in reports if r['accuracy'] is not None]
    if accs:
        long = pd.concat(accs, ignore_index=True)
        wide = long.pivot(index='model', columns='experiment', values='test_accuracy_mean')
        wide = wide.loc[list(dict.fromkeys(long.model)), [r['experiment'] for r in reports
                                                         if r['accuracy'] is not None]]
        path = os.path.join(out, 'classifier_accuracy_all_experiments.csv')
        wide.round(3).to_csv(path)
        print(f"  Saved -> {path}")
        print(wide.round(2).to_string())
    perms = [r['permutation'].assign(experiment=r['experiment']) for r in reports
             if r['permutation'] is not None]
    if perms:
        path = os.path.join(out, 'permutation_test_all_experiments.csv')
        pd.concat(perms, ignore_index=True).to_csv(path, index=False)
        print(f"  Saved -> {path}")
    cands = [r['candidates'].assign(experiment=r['experiment']) for r in reports if len(r['candidates'])]
    if cands:
        keep = ['experiment', 'mz', 'signal_mz', 'selected_by', 'n_methods', 'univariate_rank',
                'vip_rank', 'colonies_consistent', 'colony_pattern', 'apex_mz', 'apex_sd',
                'baseline_share', 'related_ions']
        path = os.path.join(out, 'candidates_all_experiments.csv')
        pd.concat(cands, ignore_index=True)[keep].to_csv(path, index=False)
        print(f"  Saved -> {path}")
    safes = [r['safe'] for r in reports]
    pca_overview.main(safes, out_dir=out)
    vip_vs_nmethods.figure(safes, output_dir=out)


def main():
    t0 = time.time()
    experiments = _experiments()
    if len(experiments) > 1:
        print(f"Running {len(experiments)} experiments: {', '.join(experiments)}")
    reports, failed = [], []
    for experiment in experiments:
        try:
            reports.append(run_experiment(experiment))
        except Exception:
            if len(experiments) == 1:
                raise
            failed.append(experiment)
            print(f"\nFAILED: {experiment!r}\n{traceback.format_exc()}")
    if len(reports) > 1:
        across_experiments(reports)
    if len(experiments) > 1:
        print(f"\nAll done in {time.time() - t0:.0f} s: {len(reports)} experiments finished"
              + (f", {len(failed)} failed ({', '.join(failed)})" if failed else ""))


if __name__ == '__main__':
    main()
