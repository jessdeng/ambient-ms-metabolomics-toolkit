# Ambient MS Metabolomics Toolkit (v2)

A Python pipeline for untargeted ambient mass spectrometry metabolomics. It asks a practical question: **when thousands of m/z features are detected, which ones are worth investigating further?** It answers it with complementary, cross-validated analyses:

- unsupervised PCA, to see the dominant structure of the spectra before any labels are used;
- PLS-DA with VIP, the established reference, reported with explicitly labelled component definitions;
- six classifiers (Random Forest, linear SVM, Gradient Boosting, Logistic Regression, LDA, Ridge), evaluated with colony-grouped cross-validation;
- an ensemble feature-importance comparison, reporting where the methods and model families agree and where they disagree.

Statistical importance is a reason to look at a feature, not proof of its chemical or biological significance. See [Interpreting the results](#interpreting-the-results).

> **What changed in v2:** see [docs/CHANGES_v2.md](docs/CHANGES_v2.md). The original behaviour is still available as a documented legacy option (`configs/manuscript_original.json`) and in the Git history (`main`).

---

## Contents

1. [Repository structure](#repository-structure)
2. [Quick start](#quick-start)
3. [Your data and metadata](#your-data-and-metadata)
4. [Settings](#settings)
5. [What the standard analysis does](#what-the-standard-analysis-does)
6. [Output files](#output-files)
7. [Optional validation analyses](#optional-validation-analyses)
8. [Legacy reproduction of the original results](#legacy-reproduction-of-the-original-results)
9. [Interpreting the results](#interpreting-the-results)
10. [Runtime and platform notes](#runtime-and-platform-notes)
11. [Tests](#tests)

A beginner-friendly, step-by-step guide for macOS is in **[docs/MAC_GUIDE.md](docs/MAC_GUIDE.md)**.

---

## Repository structure

```
ambient-ms-metabolomics-toolkit/
├── README.md
├── environment.yml / requirements.txt   ← pinned, tested dependencies
├── config.json          ← example settings (runs the bundled demo)
├── config.py            ← default values for every setting (do not edit)
├── configs/
│   ├── standard.json            ← recommended settings for your own data
│   ├── manuscript_original.json ← legacy: reproduces the original Table 1
│   └── validation.json          ← settings for the optional validation package
├── data/                ← your experiment folders (git-ignored); sample_experiment/ is a demo
├── docs/                ← MAC_GUIDE.md, CHANGES_v2.md, NOTES.md, CHANGELOG*.md
├── results/             ← outputs (git-ignored), one folder per experiment
├── src/
│   ├── standard/        ← standard pipeline (run_analysis.py, extras.py)
│   ├── r_comparable/    ← MetaboAnalyst-offset bin labels for R comparison
│   ├── shared/          ← preprocessing, grouping, classifiers, PCA, VIP, statistics
│   ├── scripts/         ← figure scripts (e.g. vip_vs_nmethods.py)
│   └── validation/      ← OPTIONAL advanced analyses (stability, sensitivity, ...)
└── tests/               ← automated tests (pytest)
```

## Quick start

```bash
conda env create -f environment.yml        # once
conda activate ms-toolkit
python src/standard/run_analysis.py --config config.json            # demo data
python src/standard/run_analysis.py --config configs/standard.json --experiment MY_EXPERIMENT
```

Results appear in `results/standard/<experiment>/`.

## Your data and metadata

**Spectra.** Put one folder per experiment under `data/`, with one subfolder per condition (class):

```
data/MY_EXPERIMENT/
├── metadata_manifest.csv      ← recommended: biological replicate (colony) of every file
├── Control/   ctrl_A1T1.csv  ctrl_A1T2.csv  ctrl_A1T3.csv  ctrl_B1T1.csv ...
├── Dark/      ...
└── ...
```

Each file is one acquisition, with an m/z column and an intensity column (`.csv` comma-separated or `.txt` tab-separated; see [data/README.md](data/README.md)).

**Biological versus technical replicates.** Cross-validation must keep all technical replicates of one biological replicate (here, a colony) together. Otherwise accuracy is inflated by repeat measurements of the same colony. Colony identity is taken from `metadata_manifest.csv` when that file exists, and from the file names otherwise.

Create a template and check it:

```bash
python src/shared/grouping.py data/MY_EXPERIMENT     # writes data/MY_EXPERIMENT/metadata_manifest.csv
```

Open the CSV in Excel or Numbers and check, for every file:
- `class`: the condition;
- `biological_replicate_group`: the colony / biological replicate (you may name this column `colony`);
- `technical_replicate`: the repeat number.

Correct anything inferred wrongly from the file names. The pipeline stops with a clear message if a file is missing from the manifest or its class disagrees with its folder. Set `EXPECTED_TECH_REPS` (e.g. `3`) to also stop if any colony has a different number of replicates.

## Settings

Copy `configs/standard.json`, set `EXPERIMENT`, and pass it with `--config`. You never need to edit the Python code. Important settings:

| Setting | Default | Meaning |
|---|---|---|
| `EXPERIMENT` | — | data folder name under `data/` (or use `--experiment`) |
| `MZ_MIN`, `MZ_MAX`, `BIN_WIDTH` | 100, 2000, 0.5 | m/z range and bin width (Da) |
| `SPECTRAL_RESAMPLING` | `native` | `native`: every instrument data point is summed into its bin. `linspace5000`: original 5000-point resampling (legacy) |
| `NORMALIZATION`, `LOG_TRANSFORM`, `SCALING` | `tic`, `log10`, `autoscale` | preprocessing, refitted inside every CV training fold |
| `VARIANCE_PERCENTILE`, `ABUNDANCE_PERCENTILE`, `PREVALENCE_THRESHOLD` | 25, 5, 0.5 | feature filters |
| `CV_FOLDS`, `N_CV_REPEATS` | 5 (auto-reduced to the number of colonies per class), 10 | colony-grouped CV |
| `TOP_N_FEATURES` | 50 | top-N used for the vote count and the family-aware consensus |
| `CONSENSUS_RULE`, `CONSENSUS_MAX_WORST_FAMILY_RANK`, `CONSENSUS_MIN_FAMILIES_TOPK` | `n_families_topk`, 50, 2 | headline **cross-family consensus** rule used by the figure scripts (default `agg_n_families_topk >= 2`; stricter alternative `agg_worst_family_rank <= 50`) |
| `SMALL_SAMPLE_GUARD_KEY` | `per_class` | colony count (smallest class, or legacy `total`) that switches on small-sample regularisation for the classifiers and importance fits |
| `PLSDA_VIP_NUM_COMPONENTS` | 1 | component-1 VIP (MetaboAnalyst "Comp. 1" equivalent) |
| `RUN_MULTICOMPONENT_VIP`, `PLS_MAX_COMPONENTS`, `Q2_PARSIMONY_TOL` | true, 8, 0.05 | multi-component VIP with A* = smallest number of latent variables within `Q2_PARSIMONY_TOL` of the best grouped Q² |
| `RUN_PCA`, `RUN_SIGNAL_GROUPS`, `RUN_LOCO_CV` | true | PCA; signal-group annotation; leave-one-colony-out |
| `METADATA_MANIFEST`, `EXPECTED_TECH_REPS` | `metadata_manifest.csv`, none | replicate metadata |
| `RUN_FEATURE_STABILITY`, `RUN_OVERLAP_PERMUTATION`, `RUN_PERMUTATION_IMPORTANCE`, `RUN_PLSDA_Q2` | false | slow optional analyses (see below) |

Unknown setting names are reported, not silently ignored.

## What the standard analysis does

1. **Load** the spectra and colony metadata, and check the design (≥2 colonies per class).
2. **Bin** into 0.5-Da bins (all instrument points summed), then filter the m/z range.
3. **PCA** (unsupervised). Explained variance, score plots of spectra and colony means, and, for each PC, the share of score variance due to condition-plate, colony (biological) and technical-replicate differences.
4. **PLS-DA VIP**, two clearly labelled analyses:
   - (a) **component-1 VIP**, numerically equivalent to MetaboAnalyst's "Comp. 1" VIP;
   - (b) **multi-component VIP** (Wold PLS2 definition) with A* latent variables. A* is the smallest number of latent variables whose colony-grouped Q² is within `Q2_PARSIMONY_TOL` (0.05) of the best (all preprocessing is refitted inside every fold); the raw argmax is also printed, and a warning is logged when it sits on the search boundary.
5. **Classification**: six classifiers.
   - Principal: 3-fold colony-grouped CV (all preprocessing inside the folds), reporting accuracy, balanced accuracy, macro-F1 and confusion matrices.
   - Supplementary: leave-one-colony-out with pooled predictions.
6. **Feature importance and consensus.** RF impurity, GB gain, |coefficients| of SVM, LR and Ridge, and component-1 VIP are computed on all data. For each feature the outputs are:
   - `n_methods`: the original vote count of top-N memberships among these six (descriptive; the votes are not independent);
   - **cross-family consensus** (`agg_*` columns): methods are averaged within family (linear / tree / PLS) so each family casts one vote; `agg_worst_family_rank` is the worst of the three family ranks and `agg_n_families_topk` the number of families with the feature in their top-N. The figures mark features top-ranked in at least two families (`agg_n_families_topk >= 2`; configurable) as *Cross-family consensus*. `agg_rra_score` is a ranking score, not a p-value;
   - model-family support: linear (SVM/LR/Ridge), tree (RF/GB) and PLS (VIP), with `n_families`, because the three linear models give closely related solutions;
   - multi-component VIP rank;
   - a descriptive colony-level screen (`screen_F`, `screen_p`, `screen_q`). Light is aliased with plate, so these describe condition-plate separation, not a treatment effect, and no table is sorted by them.
7. **Signal groups.** Adjacent 0.5-Da bins whose colony-mean profiles co-vary and whose raw profile shows a single peak are annotated as one spectral signal (e.g. bins 159.5–160.0 and 160.0–160.5 → one signal near m/z 160.1). Features stay 0.5-Da bins. The grouping is an annotation for interpretation and never silently merges features.

## Output files

All in `results/standard/<experiment>/`:

| File | Content |
|---|---|
| `classifier_accuracy_<exp>.csv` | principal table: 3-fold grouped CV mean ± SD, LOCO accuracy / balanced accuracy / macro-F1, chance level, numbers of spectra and colonies |
| `classifier_comparison_<exp>.png` | accuracy per fold, train vs test |
| `feature_rankings_<exp>.csv` | **every** feature: importance and rank per method, top-N membership, `n_methods`, family support, family-aware consensus (`agg_*`), multi-component VIP, signal group; sorted by `agg_worst_family_rank` |
| `feature_overlap_<exp>.csv` | candidates with ≥2 votes **or** meeting the cross-family consensus rule, plus the colony-level screen (`screen_F/p/q`, `screen_test`), `agg_*` columns and condition means; sorted by `agg_worst_family_rank` |
| `vip_<exp>.csv`, `vip_scores_<exp>.png` | component-1 and multi-component VIP |
| `plsda_q2_profile_<exp>.csv` | grouped Q² for 1–8 latent variables (A* = parsimonious choice; see `Q2_PARSIMONY_TOL`) |
| `pca_<exp>.png`, `pca_explained_variance_<exp>.csv`, `pca_scores_<exp>.csv`, `pca_top_loadings_<exp>.csv` | PCA |
| `signal_groups_<exp>.csv` | bin → signal-group map (evidence in `supplementary/`) |
| `spectrum_features_<exp>.png` | condition-mean spectra with candidate features marked |
| `model_importance_correlation_<exp>.csv` | Spearman correlation between methods' importances |
| `supplementary/` | confusion matrices, classification metrics, QC (technical %CV, technical/total SD ratio) |
| `run_manifest.json` | settings, package versions, Git commit and input files of the run |

**Multi-experiment figures** (after the standard analysis has been run for each experiment):

```bash
python src/scripts/pca_overview.py experiment_a experiment_b experiment_c experiment_d      # PCA overview -> results/standard/pca_overview.png
python src/scripts/vip_vs_nmethods.py experiment_a experiment_b experiment_c experiment_d   # vote count vs component-1 VIP, cross-family consensus ringed -> results/standard/vip_vs_nmethods.png
```

**Single-experiment figures** (EXPERIMENT from `config.json`, or give the name; saved in `results/standard/<experiment>/`):

```bash
python src/scripts/corroboration_figure.py my_experiment   # which methods select which m/z
python src/scripts/condition_abundance.py my_experiment    # condition means of the cross-family-consensus features
python src/scripts/condition_spectrum.py my_experiment     # condition-mean spectra with candidate features marked (HTML)
```

## Optional validation analyses

These analyses are slow (minutes to hours) and are **never run by default**. They were used to validate the feature prioritisation reported in the manuscript.

```bash
python src/standard/extras.py --config configs/standard.json --experiment MY_EXPERIMENT   # exploratory permutation test, VIP-filtered classifiers (nested)
PYTHONPATH=src:. python -m validation.run_stability   --isolate MY_EXPERIMENT --design oa27    # colony-removal selection stability
PYTHONPATH=src:. python -m validation.run_technical   --isolate MY_EXPERIMENT                  # technical-replicate stability
PYTHONPATH=src:. python -m validation.run_sensitivity --isolates MY_EXPERIMENT                 # preprocessing / binning sensitivity
PYTHONPATH=src:. python -m validation.run_unsupervised --isolates MY_EXPERIMENT                # extended PCA diagnostics
```

**Prerequisites (v2 limitation).** `run_sensitivity`, `run_classification` and `run_benchmark` are self-contained. `run_stability`, `run_technical` and `run_agreement` (and the importance-dependent parts of `run_unsupervised`) read the per-isolate importance tables and the grouped-Q² A* table in `results/validation/step4_importance/`. The scripts that generate those tables have not yet been ported to this release (see docs/CHANGES_v2.md, "Remaining limitations").

Shuffled-label analyses report **exceedance proportions** (the share of shuffles reaching the observed value), never p-values. See below.

## Legacy reproduction of the original results

```bash
python src/standard/run_analysis.py --config configs/manuscript_original.json --experiment MY_EXPERIMENT
```

This uses the original 5000-point resampling and file-name grouping, with no v2 additions. It reproduced 16 of 18 published Table 1 entries exactly from the original spectra (requires scikit-learn ≥ 1.8; see docs/CHANGES_v2.md).

## Interpreting the results

- **Importance ≠ reproducibility ≠ chemical significance.** A high rank says a feature helps a model separate the conditions in this data set. Whether the signal is reproducible needs replicate, stability and spectral checks. Whether it means anything chemically needs identification (high-resolution MS, MS/MS, standards). At unit resolution, nominal m/z does not identify a compound.
- **Agreement between methods is not proof.** The linear models are closely related, so several votes can come from one underlying solution. Prefer the family-aware `agg_*` columns (cross-family consensus) to `n_methods`.
- **Design limits.** If each condition was grown on a single plate or enclosure, condition and plate effects cannot be separated, and condition differences are condition-plate associations. Shuffling labels between colonies then does not give a valid significance test, so shuffled-label results are exploratory exceedance proportions.
- **Effective sample size** is the number of biological replicates (colonies), not the number of spectra.

## Runtime and platform notes

Measured on macOS (8-core laptop, 16 GB RAM), single-threaded:

| Data set | Standard analysis | Peak memory |
|---|---|---|
| bundled demo (12 spectra) | 7 s | — |
| one isolate: 45 spectra, 19,001 points each, m/z 100–2000 | 89–105 s | 570–700 MiB |

On macOS, set `export VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1` before running several analyses at once. Otherwise Apple's Accelerate library starts extra threads in every process and the machine slows down. Windows and Linux should work but were not tested for v2.

## Tests

```bash
python -m pytest -q        # 299 tests, about 10 s
```
