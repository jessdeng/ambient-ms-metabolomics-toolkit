# Ambient MS Metabolomics Toolkit

A Python pipeline for untargeted ambient mass spectrometry (LMJ-SSP on a unit-resolution triple quadrupole, MS1 profile spectra). It answers one question:

> **Which m/z features differ between the conditions, consistently across colonies, and look like real ions — i.e. which are worth chemical follow-up?**

Statistical importance is a reason to look at a feature, not proof of its chemical or biological significance. Identification needs high-resolution MS, MS/MS or standards.

## Design limits, stated once

- **Condition and plate are aliased.** Each condition was grown on a single plate/enclosure, so every difference is a condition-plate association. No statistic here can separate a light effect from a plate effect.
- **The colony is the experimental unit.** Technical replicates are repeated sprays of one colony. They are averaged before any ranking and used only to measure spray reproducibility.
- With three colonies per condition and thousands of bins, models that need many samples to fix their ranking (penalised regression, random forests) are compared side by side and never decide alone. See "Method comparison".

## Quick start

```bash
conda env create -f environment.yml        # once
conda activate ms-toolkit
python -m pytest -q                        # about 5 s
python src/standard/run_analysis.py --experiment sample_experiment
python src/standard/run_analysis.py --experiment MY_EXPERIMENT
python src/standard/run_analysis.py --all       # every experiment under data/, then the cross-experiment tables and figures
```

One run does everything: it prints its progress stage by stage (`[1/9]` … `[9/9]`) and writes every table, every figure and a plain-text summary. Results go to `results/<experiment>/`; with `--all`, tables and figures that span the experiments go to `results/`.

To run with one click (for example the Run button of your editor on `src/standard/run_analysis.py`), put your own settings in `config.local.json` next to `config.json`, e.g. `{"EXPERIMENT": "MY_EXPERIMENT"}` or `{"EXPERIMENT": "all"}`. It is read after `config.json` and is ignored by Git. A step-by-step guide for macOS is in [docs/MAC_GUIDE.md](docs/MAC_GUIDE.md).

## Data

One folder per experiment under `data/`, one subfolder per condition. Each file is one spectrum with an m/z column (`mz`, `Mass/Charge`) and an intensity column (`int`, `Intensity`).

```
data/MY_EXPERIMENT/
    metadata_manifest.csv     <- recommended: condition and colony of every file
    Control/  Control_A1T1.csv  Control_A1T2.csv  Control_A1T3.csv  Control_B1T1.csv ...
    Dark/     ...
```

Without a manifest the colony is read from the layout: `<condition>/<colony>/<file>`, or a file name ending in `T<n>` (`A1T3` → colony `A1`). Write a manifest to check it:

```bash
python src/shared/grouping.py data/MY_EXPERIMENT
```

Then fix `class` and `colony` for any file named inconsistently (e.g. `AmbB5T1` next to `AmberB5T2`). `EXPECTED_TECH_REPS` (e.g. `3`) stops the run if a colony has a different number of spectra.

## What the analysis does

1. **Spectra → bins.** All instrument points are summed into 0.5-Da bins on a fixed grid (absorbs unit-resolution m/z drift).
2. **TIC normalisation.** Each spectrum is divided by its own total ion current over all bins in range. This corrects spray-to-spray differences in sampled material.
3. **Filters.** Label-free MetaboAnalyst defaults: the lowest 25 % of bins by relative SD and the lowest 5 % by mean intensity are dropped.
4. **log10.** Spray noise is multiplicative.
5. **Overview.** PCA on the spectra. For each component, the share of score variance that is condition-plate, colony or technical.
6. **Colony means.** Technical replicates are averaged on the log scale. All rankings use one row per colony.
7. **Rankers that vote** (top-N each, default 50):

   | Ranker | Question it answers |
   |---|---|
   | Univariate F (one-way ANOVA on colony means) | Does this bin's level differ between conditions, relative to colony-to-colony variation? |
   | PLS-DA VIP, component 1 (autoscaled) | Does it load on the leading direction separating the conditions? |

8. **Method comparison.** Do different definitions of importance prioritise the same features? Six are computed on the colony means, each with its own native measure:

   | Method | Importance | Family |
   |---|---|---|
   | PLS-DA VIP, component 1 | VIP | PLS |
   | Ridge | mean \|coefficient\| | linear |
   | Logistic regression | mean \|coefficient\| | linear |
   | Linear SVM | mean \|weight\| over the class pairs | linear |
   | Random forest | mean impurity importance over `RF_SEEDS` seeds | tree |
   | Gradient boosting | split gain | tree |

   - **`n_methods`** counts how many of the six place a bin in their top-N. It is descriptive: the three linear models are closely related, so their votes are not independent.
   - **Cross-family consensus** gives each family one vote. By default a bin qualifies when it is top-N in at least two of the three families.
   - **Figures:** the method intersections and the top features by consensus (`corroboration_<exp>.png`), and the pairwise agreement of all seven rankings (`ranker_agreement_<exp>.png`).
   - **Reading it:** with 15 colonies and about 2,700 bins, Ridge's top-50 is set by its penalty and the forest's top-50 changes with the random seed (seed-to-seed agreement 0.23–0.47, printed in every run). Agreement between methods is a reason to look at a feature, not proof.
9. **Follow-up list.** A bin is on the list if it is top-N in both voting rankers, or if it meets the cross-family consensus. The column `selected_by` says which.
10. **Evidence for each bin:**
   - share of variance that is condition-plate / colony / technical;
   - technical %CV;
   - whether one condition's colonies all sit above (or below) every other colony (`colony_pattern`);
   - for the follow-up list, the raw-profile apex position, the local baseline share, and co-varying 13C-isotope / Na, K, NH4-adduct partners.
11. **Classifier comparison.** Six models with default settings: Random Forest, linear SVM, Gradient Boosting, Logistic Regression, LDA, Ridge.
   - **Evaluation:** folds hold out whole colonies and keep every condition in every test fold (3-fold × 10 repeats with three colonies per condition). Filters, log offset and scaling are refitted on the training colonies in every fold.
   - **Output:** test accuracy per fold, and mean train vs test accuracy.
   - **Reading it:** accuracy measures how well the condition-plates can be told apart, not a light effect. The fold-to-fold spread is as large as the differences between models, so which model "wins" is not a stable finding with three colonies per condition. On F11A the earlier version of this pipeline, which differs in preprocessing, ranked Ridge first; this one ranks Random Forest first, and with folds that were not balanced by condition it ranked Gradient Boosting first.
12. **Permutation test.** The condition labels are shuffled among colonies (a colony's spectra keep one label) and the whole cross-validation is repeated, `N_PERMUTATIONS` times per model. The smallest attainable p-value is 1 / (permutations + 1). Gradient Boosting is left out by default because it is about 20 times slower.
13. **Report.** `summary_<exp>.txt` gathers the design, the settings, the method comparison, the follow-up list with its evidence, the classifier accuracies and the permutation test.

## Settings

All settings and their defaults are in [config.py](config.py). Override them in `config.json`, in your own `config.local.json` (read after it, ignored by Git), or in a JSON file passed with `--config`. Unknown names are reported.

| Setting | Default | Meaning |
|---|---|---|
| `EXPERIMENT` | `sample_experiment` | folder under `data/` (or `--experiment`); `all` runs every experiment (or `--all`) |
| `METADATA_MANIFEST`, `EXPECTED_TECH_REPS` | `metadata_manifest.csv`, none | design file; replicate-count check |
| `MZ_MIN`, `MZ_MAX`, `BIN_WIDTH` | 100, 2000, 0.5 | m/z range and bin width (Da) |
| `BIN_LABELS` | `mean_mz` | bin label: mean m/z of its points, or `lower_edge` (MetaboAnalyst) |
| `VARIANCE_PERCENTILE`, `ABUNDANCE_PERCENTILE` | 25, 5 | label-free filters |
| `SCALING` | `autoscale` | for the multivariate rankers; or `pareto` |
| `TOP_N_FEATURES`, `MIN_RANKERS` | 50, 2 | top-N of every ranking; voters a bin must be top-N in |
| `RF_SEEDS`, `RANDOM_SEED` | 25, 42 | random-forest seeds averaged; seed for everything |
| `CONSENSUS_RULE`, `CONSENSUS_MIN_FAMILIES_TOPK`, `CONSENSUS_MAX_WORST_FAMILY_RANK` | `n_families_topk`, 2, 50 | cross-family consensus: top-N in at least this many families, or (`worst_family_rank`) ranked this well in every family |
| `MIN_METHODS_OVERLAP` | 2 | `feature_overlap` table: bins in the top-N of at least this many methods |
| `RUN_CLASSIFIERS`, `CV_FOLDS`, `N_CV_REPEATS` | true, 5, 10 | classifier comparison; folds reduced to the smallest colonies-per-condition |
| `RUN_PERMUTATION`, `N_PERMUTATIONS`, `PERMUTATION_MODELS`, `N_JOBS` | true, 200, all but Gradient Boosting, -1 | permutation test; models as a comma-separated list; parallel workers (-1 = all cores) |
| `N_PCA_COMPONENTS`, `N_TOP_VIP`, `SIGNAL_R_MIN` | 5, 30, 0.9 | descriptive outputs |

## Outputs (`results/<experiment>/`)

| File | Content |
|---|---|
| `summary_<exp>.txt` | the run in one page: design, settings, method comparison, follow-up list with evidence, accuracies, permutation test |
| `candidates_<exp>.csv` | the follow-up list with `selected_by`, `n_methods`, all evidence columns and each colony's mean |
| `feature_rankings_<exp>.csv` | every bin: score, rank and top-N membership for all seven rankings, `n_rankers`, `n_methods`, family ranks (`agg_*`), `consensus`, the colony-level screen (`screen_F`, `screen_eta2`, `screen_p`, `screen_q`, log2 fold change), variance partition, technical %CV, colony pattern, signal group |
| `feature_overlap_<exp>.csv` | bins in the top-N of at least two methods, or meeting the cross-family consensus; sorted by consensus rank |
| `ranker_concordance_<exp>.csv` | top-N Jaccard and shared count between every pair of rankings, with the chance level |
| `corroboration_<exp>.png` | method comparison: which methods select the same bins (intersections), and the top bins by cross-family consensus with their vote count and top condition |
| `ranker_agreement_<exp>.png` | pairwise top-N agreement of all rankings; univariate-F rank against VIP rank |
| `pca_*` | explained variance with its condition-plate / colony / technical split, scores, top loadings, figure |
| `signal_groups_<exp>.csv` | adjacent bins merged into one peak when the raw profile supports it |
| `classifier_comparison_<exp>.png`, `classifier_accuracy_<exp>.csv` | test accuracy per fold for the six classifiers, train vs test; mean ± SD per model with the chance level |
| `permutation_test_<exp>.csv` | observed and shuffled-label accuracy and the p-value for each model |
| `vip_scores_<exp>.png`, `spectrum_features_<exp>.png` | top-VIP plot; condition-mean spectra with the follow-up list marked |
| `condition_abundance_<exp>.png` | the follow-up list by condition and by colony |
| `condition_spectrum_<exp>.html` | interactive condition-mean spectra with the follow-up list marked |
| `run_manifest.json` | settings, package versions and Git commit of the run |

`screen_p` and `screen_q` compare colonies across condition-plates. They describe separation and are not tests of a light effect.

With `--all`, `results/` also gets:

| File | Content |
|---|---|
| `classifier_accuracy_all_experiments.csv` | mean test accuracy, one row per model and one column per experiment |
| `permutation_test_all_experiments.csv` | the permutation tests of all experiments |
| `candidates_all_experiments.csv` | the follow-up lists of all experiments with their main evidence columns |
| `pca_overview.png` | PCA of the experiments side by side |
| `vip_vs_nmethods.png` | vote count against component-1 VIP for every experiment, cross-family consensus ringed |

## Figures

Every figure is drawn by the main run. To redraw one without re-running the analysis:

```bash
python src/scripts/corroboration_figure.py MY_EXPERIMENT   # method intersections and consensus
python src/scripts/ranker_agreement.py MY_EXPERIMENT       # pairwise agreement of the rankings
python src/scripts/condition_abundance.py MY_EXPERIMENT    # follow-up list by condition and by colony
python src/scripts/condition_spectrum.py MY_EXPERIMENT     # interactive spectra, follow-up list marked (HTML)
python src/scripts/pca_overview.py EXP_A EXP_B EXP_C EXP_D # PCA of several experiments side by side
python src/scripts/vip_vs_nmethods.py EXP_A EXP_B          # vote count against VIP across experiments
```

## Validation helpers

`src/shared/validation.py` provides the folds and the permutation test the main run uses. Use it from Python, started in the repository root, to go beyond the classifier comparison (per-repeat and colony-level accuracy, a permutation test, PLS-DA):

- **`RepeatedColonyKFold`** holds out whole colonies and keeps every condition in every test fold. `StratifiedGroupKFold` with shuffling does not guarantee the second property: with three colonies per condition, a test fold can hold two colonies of one condition and none of another. The classifier comparison uses these folds too.
- **`grouped_cv`** returns, for each repeat, the accuracy and balanced accuracy over spectra and the accuracy over colonies (majority vote of a colony's spectra). Filters, log offset and scaling are refitted on the training colonies in every fold.
- **`colony_permutation_test`** shuffles the condition labels among colonies, so a colony's spectra keep one label, and repeats the whole procedure for every permutation. The null distribution is reproducible from the seed whatever `n_jobs` is.
- **`PLSDAClassifier`** scores the model behind VIP in the same way as the other classifiers.

```python
import sys
sys.path[:0] = ['src', '.']
from sklearn.linear_model import RidgeClassifier
from standard.preprocessing import load_experiment, build_matrices, tic_normalise
from shared.grouping import resolve_design
from shared.validation import RepeatedColonyKFold, grouped_cv, colony_permutation_test

exp_dir = 'data/sample_experiment'
X_raw, names, mz = load_experiment(exp_dir)
condition, colony = resolve_design(exp_dir, names)
X_tic = tic_normalise(build_matrices(X_raw, mz)['binned'])

cv = RepeatedColonyKFold(n_splits=3, n_repeats=10, random_state=42)
scores = grouped_cv(RidgeClassifier(), X_tic, condition, colony, cv)
observed, null, p = colony_permutation_test(RidgeClassifier(), X_tic, condition, colony, cv,
                                            n_permutations=1000, n_jobs=4)
```

The smallest attainable p-value is 1 / (permutations + 1). As everywhere in this toolkit, a small p-value shows that the condition-plates can be told apart; it is not a test of a light effect.

## Reproducing the manuscript's Table 1

The exact published Table 1 used the earlier version (legacy 5000-point resampling, preprocessing before the folds), frozen at tag `v2-table1-snapshot`:

```bash
git checkout v2-table1-snapshot
python src/standard/run_analysis.py --config configs/manuscript_original.json --experiment MY_EXPERIMENT
git checkout main
```
