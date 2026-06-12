# Ambient MS Metabolomics Toolkit

A Python pipeline for untargeted ambient mass spectrometry metabolomics. Performs multi-classifier machine learning, ensemble feature importance voting, and cross-validated PLS-DA — with two parallel pipelines: one with data-driven bin labels for accurate compound identification, and one with MetaboAnalyst-offset labels for direct R comparison.

> **Instrument note:** Default parameters (0.5 Da bins, 100–1000 Da m/z range, TIC normalization) are appropriate for low-resolution ambient MS and direct infusion workflows. Adjust `config.json` for your instrument.

For detailed notes on parameter choices and output interpretation, see [docs/NOTES.md](docs/NOTES.md).

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Quick Start](#quick-start)
3. [Configuration (config.json)](#configuration-configjson)
4. [Data Format](#data-format)
5. [Methodological Note](#methodological-note)
6. [Which Pipeline?](#which-pipeline)
7. [Output Files](#output-files)
8. [Environment Setup](#environment-setup)
9. [Windows Compatibility](#windows-compatibility)
10. [Reproducibility](#reproducibility)
11. [Citation](#citation)

---

## Repository Structure

```
ambient-ms-metabolomics-toolkit/
├── README.md
├── LICENSE
├── environment.yml
├── config.json            ← edit this to configure your experiment
├── config.py              ← default parameter values (do not edit)
├── requirements.txt
├── setup.py
│
├── data/                  ← your experiment data goes here (gitignored)
│   └── <experiment_folder>/
│       ├── Group1/
│       └── Group2/
│
├── docs/                  ← supplementary documentation
│   ├── NOTES.md           ← parameter rationale and output interpretation
│   └── CHANGELOG.md
│
├── results/               ← pipeline outputs (gitignored, auto-created on run)
│   ├── standard/
│   └── r_comparable/
│
└── src/                   ← all source code
    ├── standard/          ← standard pipeline (data-driven bin labels)
    │   ├── preprocessing.py
    │   ├── pipeline.py
    │   ├── run_analysis.py
    │   └── extras.py
    ├── r_comparable/      ← R/MetaboAnalyst-compatible pipeline
    │   ├── preprocessing.py
    │   ├── pipeline.py
    │   ├── run_analysis.py
    │   └── extras.py
    ├── shared/            ← modules shared by both pipelines
    │   ├── classifier_comparison_standard.py
    │   ├── classifier_comparison.py
    │   └── visualization.py
    └── scripts/           ← post-hoc analysis scripts
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

# 2. Create environment
conda env create -f environment.yml
conda activate ambient-ms

# 3. Place your data
#    data/<EXPERIMENT>/Group1/sample_W1T1.csv
#    data/<EXPERIMENT>/Group2/sample_W2T1.csv  ...

# 4. Set your experiment name in config.json
#    { "EXPERIMENT": "<your_experiment_folder>" }

# 5. Run from the repository root (the folder containing config.py)
python -m src.standard.run_analysis

# Optional: additional plots and permutation test
python -m src.standard.extras
```

---

## Configuration (config.json)

Edit `config.json` in the repository root — no Python editing required:

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

Only include the keys you want to change — missing keys fall back to defaults in `config.py`. See `config.py` for the full parameter list and documentation.

**The only required key is `EXPERIMENT`**, which must match the name of your experiment folder inside `data/`.

---

## Data Format

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

**Required columns** (case-insensitive):

| Column | Accepted names |
|--------|----------------|
| m/z | `mz`, `m/z`, `Mass/Charge`, `mass` |
| Intensity | `int`, `intensity`, `Intensity` |

**Filename convention for GroupKFold CV:** Use `<prefix><well>T<replicate>.<ext>`, e.g. `ConditionA_W1T1.csv`. The pipeline extracts the well ID (`W1`) and groups all technical replicates of that biological replicate together so they never straddle the train/test boundary. Files not matching this pattern fall back to per-file grouping (no error, but CV is not pseudoreplication-corrected).

---

## Methodological Note

> **Two methodological decisions prevent the most common biases in small-n metabolomics studies. Both are required for defensible peer-review-grade analysis.**

### 1. Leave-One-Biological-Replicate-Out Cross-Validation (GroupKFold)

**Problem — pseudoreplication:** When technical replicates of the same biological sample are randomly split across folds, the classifier learns the chemical fingerprint of each individual colony rather than generalising across replicates — inflating reported accuracy.

**Solution:** `StratifiedGroupKFold` ensures all technical replicates from one biological replicate are always on the same side of every fold boundary. Groups are `condition::well` (e.g. `Control::W1`), parsed from the filename convention above.

Implemented in `src/shared/classifier_comparison_standard.py` via `make_groups`, `auto_n_splits`, and `_run_grouped_cv`.

### 2. Preprocessing Inside Cross-Validation (sklearn Pipeline)

**Problem — preprocessing leakage:** Fitting variance filters, normalisation, and scaling on the full dataset before CV leaks test-fold information into training, making accuracy appear higher than it is.

**Solution:** All preprocessing is wrapped in an `sklearn.Pipeline` fitted exclusively on the training fold inside each CV iteration. The test fold is transformed using parameters from training data only.

Custom transformers (`VarianceFilter`, `AbundanceFilter`, `Normalizer`, `LogTransform`, `Scaler`) are defined in `src/shared/classifier_comparison_standard.py` and assembled via `make_preprocessor()`.

**Note:** `feature_importance_analysis()` fits on the full preprocessed dataset — this is intentional. It is a descriptive summary of feature drivers, not a generalisation estimate. CV results from `run_analysis.py` provide the generalisation estimate.

---

## Which Pipeline?

**If you are not comparing against R-based PLS-DA output, use the standard pipeline.**

| | Standard | R-comparable |
|---|---|---|
| **Run** | `python -m src.standard.run_analysis` | `python -m src.r_comparable.run_analysis` |
| **Extras** | `python -m src.standard.extras` | `python -m src.r_comparable.extras` |
| **Output** | `results/standard/` | `results/r_comparable/` |
| **Bin labels** | Mean of actual m/z values in bin | Lower bin edge (MetaboAnalyst convention) |
| **Use when** | General use, database lookup | Matching R/MetaboAnalyst PLS-DA results |

Classifier accuracies and VIP scores are numerically identical between both versions; only the reported m/z per feature differs.

---

## Output Files

Outputs are saved to `results/standard/` or `results/r_comparable/`.

| File | Description |
|------|-------------|
| `plsda_scores_3d_<exp>.html` | Interactive 3D PLS-DA scores plot (open in browser) |
| `vip_scores_<exp>.png` | Top VIP features with per-group intensity heatmap |
| `classifier_comparison_<exp>.png` | Per-fold accuracy for all 6 classifiers |
| `spectrum_features_<exp>.png` | Mean spectrum per group with ensemble features marked |
| `feature_overlap_<exp>.csv` | Ensemble candidates with per-group means, Ridge coefficients, VIP scores |
| `classifier_results_<exp>.npz` | Saved CV accuracy arrays (auto-loaded by extras.py) |
| `summary_<exp>.txt` | Plain-text run summary (extras.py) |
| `variation_<exp>.png` | Within-group feature RSD violin plot (extras.py) |
| `correlation_<exp>.png` | Pairwise correlation of top VIP features (extras.py) |
| `permutation_test_<exp>.png` | Null distribution vs observed accuracy (extras.py) |
| `permutation_null_<exp>.npy` | Raw permutation accuracy array (extras.py) |

### Post-hoc scripts

Run after `run_analysis.py`. Set `PIPELINE = 'standard'` or `'r_comparable'` at the top of each script.

```bash
python -m src.scripts.condition_abundance
python -m src.scripts.vip_vs_nmethods
python -m src.scripts.condition_spectrum
```

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
# or directly:
pip install -r requirements.txt
```

**Python version:** 3.11 tested and recommended. Python 3.9+ should work.

---

## Windows Compatibility

All file paths use `os.path.join()` — no hardcoded forward slashes. All dependencies are cross-platform.

**Running on Windows (CMD or PowerShell):**

```cmd
:: Navigate to the repo root
cd C:\path\to\ambient-ms-metabolomics-toolkit

:: Create and activate environment
conda env create -f environment.yml
conda activate ambient-ms

:: Run the pipeline (always from repo root)
python -m src.standard.run_analysis
python -m src.standard.extras
```

**Important:** Always run `python -m` commands from the **repository root directory** (the folder containing `config.py`). All terminal output uses ASCII characters only — no `PYTHONIOENCODING` changes required.

---

## Reproducibility

All stochastic classifiers use `RANDOM_SEED = 42` (set in `config.json` or `config.py`). To verify robustness, change this value and re-run — key findings should not change substantially.

```json
{ "RANDOM_SEED": 99 }
```

To reproduce results from the paper exactly, keep all `config.json` values at their defaults.

---

## Citation

If you use this toolkit in your research, please cite:

> [Citation to be added upon publication]

For the scaling method comparison informing default parameter selection:

> van den Berg, R.A., Hoefsloot, H.C., Westerhuis, J.A., Smilde, A.K., & van der Werf, M.J. (2006). Centering, scaling, and transformations: improving the biological information content of metabolomics data. *BMC Genomics*, 7, 142. https://doi.org/10.1186/1471-2164-7-142
