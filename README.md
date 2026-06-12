# Ambient MS Metabolomics Toolkit

A Python pipeline for untargeted ambient mass spectrometry metabolomics. It performs multi-classifier machine learning, ensemble feature importance voting, and cross-validated PLS-DA. Two parallel pipelines are provided: one with bin labels compatible with R-based PLS-DA packages, and one with data-driven bin labels for accurate compound identification.

For detailed notes on parameter choices and output interpretation, see [NOTES.md](NOTES.md).

> **Instrument note:** Default parameters (0.5 Da bins, 100–1000 Da m/z range, TIC normalization) are appropriate for low-resolution ambient MS and direct infusion workflows. Adjust `config.json` or `config.py` for your instrument and sample type.

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Quick Start](#quick-start)
3. [Configuration (config.json)](#configuration-configjson)
4. [Data Format](#data-format)
5. [Methodological Note](#methodological-note)
6. [Which Pipeline Should I Use?](#which-pipeline-should-i-use)
7. [Output Files](#output-files)
8. [Environment Setup](#environment-setup)
9. [Windows Compatibility](#windows-compatibility)
10. [Reproducibility](#reproducibility)
11. [Citation](#citation)

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
git clone https://github.com/<your-username>/ambient-ms-metabolomics-toolkit.git
cd ambient-ms-metabolomics-toolkit

# 2. Create environment (conda) or install packages (pip)
conda env create -f environment.yml
conda activate ambient-ms
# OR
python setup.py

# 3. Place your data (see Data Format section)
#    data/<EXPERIMENT>/Group1/sample1.csv  ...

# 4. Set EXPERIMENT in config.json (no Python editing needed)
#    { "EXPERIMENT": "<your_experiment_folder>" }

# 5. Run
python -m standard.run_analysis

# Optional: additional plots and permutation test
python -m standard.extras
```

---

## Configuration (config.json)

`config.json` in the repository root is the recommended way to set your experiment parameters. Edit it without touching any Python files:

```json
{
  "EXPERIMENT": "my_experiment_folder",
  "MZ_MIN": 100,
  "MZ_MAX": 1000,
  "BIN_WIDTH": 0.5,
  "NORMALIZATION": "tic",
  "LOG_TRANSFORM": "log10",
  "SCALING": "autoscale",
  "RANDOM_SEED": 42,
  "RUN_PERMUTATION_TEST": true
}
```

Only include the keys you want to change — any key absent from `config.json` falls back to the default in `config.py`. See `config.py` for the full list of available parameters and their documentation.

The only required key is `EXPERIMENT`, which must match the name of your experiment folder inside `data/`.

---

## Data Format

Place experiment data inside `data/<EXPERIMENT_NAME>/`. Each immediate subfolder is one biological group; each file inside a subfolder is one sample.

```
data/
└── my_experiment/
    ├── Control/
    │   ├── ConditionA_W1T1.csv
    │   ├── ConditionA_W1T2.csv
    │   └── ConditionA_W2T1.csv
    └── Treatment/
        ├── ConditionB_W3T1.csv
        └── ConditionB_W3T2.csv
```

**Supported formats:** `.csv` (comma-separated) or `.txt` (tab-separated).

**Required columns** (header names are case-insensitive):

| Column | Accepted names |
|--------|----------------|
| m/z    | `mz`, `m/z`, `Mass/Charge`, `mass` |
| Intensity | `int`, `intensity`, `Intensity` |

**Filename convention for biological replicates (GroupKFold):** For the leave-one-replicate-out cross-validation to work correctly, filenames should follow the pattern `<prefix><well>T<replicate>.<ext>`, e.g., `ConditionA_W1T1.csv`. The pipeline extracts the well/replicate ID (e.g., `W1`) and builds group labels `condition::well` (e.g., `Control::W1`). Files not matching this pattern fall back to full filename as the group identifier.

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

## Windows Compatibility

This repository is fully Windows-compatible. All file paths use `os.path.join()` — no hardcoded forward slashes. All dependencies in `environment.yml` are cross-platform.

**Running on Windows (CMD or PowerShell):**

```cmd
:: Step 1 — Create and activate environment (conda)
conda env create -f environment.yml
conda activate ambient-ms

:: Step 2 — Or install with pip
python setup.py

:: Step 3 — Run from the repository root
python -m standard.run_analysis
python -m r_comparable.run_analysis
python -m standard.extras
```

**Important:** Always run `python -m` commands from the **repository root directory** (the folder containing `config.py`). On Windows this means opening CMD or PowerShell, navigating to the repo folder with `cd`, and then running the commands above.

```cmd
cd C:\path\to\ambient-ms-metabolomics-toolkit
python -m standard.run_analysis
```

**Character encoding:** All terminal output uses ASCII characters and is safe on Windows with the default code page. No `PYTHONIOENCODING` changes are required.

---

## Reproducibility

All stochastic classifiers use `RANDOM_SEED = 42` (set in `config.json` or `config.py`). To verify that results are robust across seeds, change this value and re-run — key findings should not change substantially.

```json
{ "RANDOM_SEED": 99 }
```

To reproduce results from the paper exactly, keep all `config.json` values at their defaults.

---

## Citation

If you use this toolkit in your research, please cite:

> [Citation to be added upon publication]

For the scaling method comparison that informed default parameter selection:

> van den Berg, R.A., Hoefsloot, H.C., Westerhuis, J.A., Smilde, A.K., & van der Werf, M.J. (2006). Centering, scaling, and transformations: improving the biological information content of metabolomics data. *BMC Genomics*, 7, 142. https://doi.org/10.1186/1471-2164-7-142
