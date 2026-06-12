# Ambient MS Metabolomics Toolkit

A Python pipeline for untargeted ambient mass spectrometry metabolomics. It performs multi-classifier machine learning, ensemble feature importance voting, and cross-validated PLS-DA. Two parallel pipelines are provided: one with bin labels compatible with R-based PLS-DA packages, and one with data-driven bin labels for accurate compound identification.

For detailed notes on parameter choices and output interpretation, see [NOTES.md](NOTES.md).

> **Instrument note:** Default parameters are optimised for fungal metabolomics data collected on a SCIEX 4500 triple quadrupole in MS1-only mode (LMJ-SSP). Adjust `config.py` for other instruments or sample types.

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Quick Start](#quick-start)
3. [Data Format](#data-format)
4. [Methodological Note](#methodological-note)
5. [Which Pipeline Should I Use?](#which-pipeline-should-i-use)
6. [Output Files](#output-files)
7. [Environment Setup](#environment-setup)
8. [Reproducibility](#reproducibility)
9. [Citation](#citation)

---

## Repository Structure

```
ambient-ms-metabolomics-toolkit/
├── config.py                  ← only file you need to edit
├── requirements.txt
├── environment.yml
├── setup.py
├── README.md
├── NOTES.md
│
├── data/                      ← your experiment data goes here (gitignored)
│   └── <experiment_folder>/
│       ├── Group1/
│       └── Group2/
│
├── results/                   ← pipeline outputs go here (gitignored)
│   ├── standard/
│   └── r_comparable/
│
├── notebooks/                 ← Jupyter notebooks (optional)
│
├── shared/                    ← modules shared between both pipelines
│   ├── classifier_comparison_standard.py
│   ├── classifier_comparison.py
│   └── visualization.py
│
├── standard/                  ← standard pipeline
│   ├── preprocessing.py
│   ├── pipeline.py
│   ├── run_analysis.py
│   └── extras.py
│
├── r_comparable/              ← R package-compatible pipeline
│   ├── preprocessing.py
│   ├── pipeline.py
│   ├── run_analysis.py
│   └── extras.py
│
└── scripts/                   ← post-hoc analysis scripts
    ├── condition_abundance.py
    ├── condition_spectrum.py
    └── vip_vs_nmethods.py
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/jessdeng/ambient-ms-metabolomics-toolkit.git
cd ambient-ms-metabolomics-toolkit

# 2. Create environment (conda) or install packages (pip)
conda env create -f environment.yml
conda activate ambient-ms
# OR
python setup.py

# 3. Place your data (see Data Format section)
#    data/<EXPERIMENT>/Group1/sample1.csv  ...

# 4. Set EXPERIMENT in config.py
#    EXPERIMENT = '<your_experiment_folder>'

# 5. Run
python -m standard.run_analysis

# Optional: additional plots and permutation test
python -m standard.extras
```

---

## Data Format

Place experiment data inside `data/<EXPERIMENT_NAME>/`. Each immediate subfolder is one biological group; each file inside a subfolder is one sample.

```
data/
└── my_experiment/
    ├── Control/
    │   ├── F11A_A6T1.csv
    │   ├── F11A_A6T2.csv
    │   └── F11A_A7T1.csv
    └── Treatment/
        ├── F11B_B3T1.csv
        └── F11B_B3T2.csv
```

**Supported formats:** `.csv` (comma-separated) or `.txt` (tab-separated).

**Required columns** (header names are case-insensitive):

| Column | Accepted names |
|--------|----------------|
| m/z    | `mz`, `m/z`, `Mass/Charge`, `mass` |
| Intensity | `int`, `intensity`, `Intensity` |

**Filename convention for biological replicates (GroupKFold):** For the leave-one-replicate-out cross-validation to work correctly, filenames should follow the pattern `<prefix><well>T<replicate>.<ext>`, e.g., `F11A_A6T1.csv`. The pipeline extracts the colony well ID (e.g., `A6`) and builds group labels `condition::well` (e.g., `Control::A6`). Files not matching this pattern fall back to full filename as the group identifier.

---

## Methodological Note

> **This section documents two methodological decisions that prevent common biases in small-n metabolomics studies. Both are required for defensible peer-review-grade analysis.**

### 1. Leave-One-Biological-Replicate-Out Cross-Validation (GroupKFold)

**Problem — pseudoreplication:** When technical replicates (multiple measurements of the same biological sample) are randomly split across training and test folds, the same colony can appear in both. This inflates reported accuracy because the classifier learns the specific chemical fingerprint of each individual colony rather than generalising across biological replicates.

**Solution:** `StratifiedGroupKFold` is used so that all technical replicates from one biological replicate are always on the same side of the train/test boundary. Groups are constructed as `condition::well` (e.g., `Control::A6`), parsed from the filename using the convention above. With three biological replicates per condition, this yields three-fold cross-validation — each fold tests the classifier on one unseen colony.

This is implemented in `shared/classifier_comparison_standard.py` (`make_groups`, `auto_n_splits`, `_run_grouped_cv`) and applied identically in both pipelines.

### 2. Preprocessing Inside Cross-Validation (sklearn Pipeline)

**Problem — preprocessing leakage:** When preprocessing steps (variance filtering, normalization, log-transformation, scaling) are fit on the whole dataset before cross-validation, information from the test fold contaminates the training fold. This makes the classifier appear to generalise better than it does.

**Solution:** All preprocessing is wrapped in an `sklearn.Pipeline` and fit exclusively on the training fold inside each CV iteration. The test fold is transformed using parameters estimated from training data only.

The custom sklearn-compatible transformers `VarianceFilter`, `AbundanceFilter`, `Normalizer`, `LogTransform`, and `Scaler` are defined in `shared/classifier_comparison_standard.py` and assembled via `make_preprocessor()`.

**Note on feature importance:** `feature_importance_analysis()` fits descriptive models on the full preprocessed dataset. This is intentional — it is a descriptive summary of which features drive the class separation, not a generalisation estimate. CV results from `run_analysis.py` provide the generalisation estimate.

---

## Which Pipeline Should I Use?

**If you are not comparing against R-based PLS-DA output, use the standard pipeline.**

| | Standard | R-comparable |
|---|---|---|
| **Run with** | `python -m standard.run_analysis` | `python -m r_comparable.run_analysis` |
| **Extras** | `python -m standard.extras` | `python -m r_comparable.extras` |
| **Outputs to** | `results/standard/` | `results/r_comparable/` |
| **Bin labels** | Mean of actual m/z values in bin | Lower edge of bin interval (MetaboAnalyst convention) |
| **Use when** | General use — accurate m/z for database lookup | Matching R/MetaboAnalyst PLS-DA results |

Classifier accuracies, VIP scores, and cross-validation results are numerically identical between the two versions. Only the reported m/z for each feature differs.

---

## Output Files

Outputs are saved to `results/standard/` or `results/r_comparable/`.

| File | Description |
|------|-------------|
| `plsda_scores_3d_<exp>.html` | Interactive 3D PLS-DA scores plot (open in browser) |
| `vip_scores_<exp>.png` | Top VIP features with per-group intensity heatmap |
| `classifier_comparison_<exp>.png` | Per-fold accuracy and train/test comparison for all 6 classifiers |
| `spectrum_features_<exp>.png` | Mean spectrum per group with ensemble features marked |
| `feature_overlap_<exp>.csv` | Ensemble feature candidates with per-group mean intensity, signed Ridge coefficients, VIP score, and a "top condition" call |
| `classifier_results_<exp>.npz` | Saved CV accuracy arrays — auto-loaded by extras.py |
| `summary_<exp>.txt` | Plain-text run summary (extras.py) |
| `variation_<exp>.png` | Within-group feature RSD violin plot (extras.py) |
| `correlation_<exp>.png` | Pairwise correlation of top VIP features (extras.py) |
| `permutation_test_<exp>.png` | Null accuracy distribution vs observed (extras.py) |
| `permutation_null_<exp>.npy` | Raw permutation accuracy array (extras.py) |

---

## Environment Setup

### Option A — conda (recommended)

```bash
conda env create -f environment.yml
conda activate ambient-ms
```

### Option B — pip

```bash
python setup.py
# or
pip install -r requirements.txt
```

**Python version:** 3.11 is tested and recommended. Python 3.9+ should work.

---

## Reproducibility

All stochastic classifiers use `RANDOM_SEED = 42` (set in `config.py`). To verify that results are robust across seeds, change this value (e.g., to `0` or `99`) and re-run — key findings should not change substantially.

```python
# config.py
RANDOM_SEED = 42   # change to verify robustness
```

To reproduce results from the paper exactly, keep all `config.py` values at their defaults.

---

## Citation

If you use this toolkit in your research, please cite:

> [Citation to be added upon publication]

For the scaling method comparison that informed default parameter selection:

> van den Berg, R.A., Hoefsloot, H.C., Westerhuis, J.A., Smilde, A.K., & van der Werf, M.J. (2006). Centering, scaling, and transformations: improving the biological information content of metabolomics data. *BMC Genomics*, 7, 142. https://doi.org/10.1186/1471-2164-7-142
