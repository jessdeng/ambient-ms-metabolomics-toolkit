# Running the toolkit on your Mac — step by step

This guide assumes you know mass spectrometry and statistics but have not managed Python projects before. Every command is typed into the **Terminal** app (Applications → Utilities → Terminal). Lines starting with `#` are comments; do not type them.

**Time needed:** about 20 minutes the first time, then about 2 minutes per analysis.

---

## 1. Install Miniforge (one time)

Miniforge provides `conda`, which creates an isolated Python installation with exactly the tested package versions. It does not touch the Python that came with your Mac.

1. Open https://github.com/conda-forge/miniforge#install and download the macOS installer for your chip:
   - **Apple Silicon** (M1/M2/M3/M4): `Miniforge3-MacOSX-arm64.sh`
   - **Intel**: `Miniforge3-MacOSX-x86_64.sh`

   To check your chip: Apple menu → About This Mac → "Chip" or "Processor".
2. In Terminal:
   ```bash
   cd ~/Downloads
   bash Miniforge3-MacOSX-arm64.sh        # or the x86_64 file name
   ```
   Press Enter to page through the licence, type `yes` to accept, accept the default location, and answer `yes` when asked to initialise conda.
3. **Close Terminal and open a new window.** Your prompt should now start with `(base)`.

## 2. Get the toolkit (one time)

If you already have the repository folder (for example a GitHub clone or a downloaded zip), skip to step 3. Otherwise:

```bash
cd ~/Documents
git clone https://github.com/jessdeng/ambient-ms-metabolomics-toolkit.git
cd ambient-ms-metabolomics-toolkit
```

If macOS asks to install the "command line developer tools" for `git`, click Install, then repeat the command.

## 3. Create the Python environment (one time)

From inside the toolkit folder:

```bash
conda env create -f environment.yml
```

This takes a few minutes. Then, **every time you open a new Terminal window**, activate it:

```bash
cd ~/Documents/ambient-ms-metabolomics-toolkit
conda activate ms-toolkit
```

The prompt now starts with `(ms-toolkit)`.

## 4. Check that everything works

```bash
python -m pytest -q
python src/standard/run_analysis.py --experiment sample_experiment
```

The first command should end with `34 passed`. The second analyses the small bundled demo, finishes in a few seconds, and ends with `Done in … s -> …/results/sample_experiment`. Open that folder in Finder:

```bash
open results/sample_experiment
```

## 5. Add your own data

1. Copy your experiment folder into `data/`, with one subfolder per condition:
   ```
   data/my_experiment/
       Amber/   Amber_A6T1.csv  Amber_A6T2.csv  ...
       Control/ ...
       Dark/    ...
       Green/   ...
       White/   ...
   ```
   Each file is one exported spectrum with an m/z column and an intensity column (the Analyst text export works as is).
2. Create the replicate metadata template:
   ```bash
   python src/shared/grouping.py data/my_experiment
   ```
   This writes `data/my_experiment/metadata_manifest.csv`.
3. Open `metadata_manifest.csv` in Excel or Numbers. For every spectrum, check that:
   - `class` is its condition;
   - `colony` is its **colony** (biological replicate). All technical replicates of one colony must share the same value;
   - `technical_replicate` is the repeat number (1, 2, 3).

   Fix any value that was guessed wrongly from the file name (for example a typo such as `AmbB5T1` next to `AmberB5T2`). Save it as CSV with the same name.

   *Why this matters:* the colony is the experimental unit. Its technical replicates are averaged before any feature is ranked, so a misnamed file would create a fake extra colony.

Your data stay on your computer: everything under `data/` is excluded from Git.

## 6. Configure and run the standard analysis

1. Make your own copy of the recommended settings:
   ```bash
   cp configs/standard.json configs/my_settings.json
   open -e configs/my_settings.json        # opens it in TextEdit
   ```
   Set `EXPERIMENT`, and check `MZ_MIN`/`MZ_MAX` (100–2000 for the LMJ-SSP data) and `EXPECTED_TECH_REPS` (3 = stop if a colony does not have three spectra). You do not need to change anything else. Save the file. In TextEdit, use Format → Make Plain Text if asked.
2. Run:
   ```bash
   python src/standard/run_analysis.py --config configs/my_settings.json --experiment my_experiment
   ```
   One isolate (45 spectra) takes about 35 minutes on an 8-core laptop, almost all of it the permutation test for Gradient Boosting. For a run of about 2 minutes, remove `Gradient Boosting` from `PERMUTATION_MODELS` in your settings file; set `"RUN_PERMUTATION": false` and `"RUN_CLASSIFIERS": false` for a 15-second run without either.
3. Repeat with `--experiment experiment_b`, `--experiment experiment_c`, and so on.

## 7. Find and read the results

```bash
open results/my_experiment
```

Start with these files:

| File | What to look at |
|---|---|
| `candidates_my_experiment.csv` | the prioritised m/z bins: how they rank, which condition stands apart (`colony_pattern`), how reproducible the sprays are (`technical_cv_pct`), whether the raw profile shows a real peak (`apex_mz`, `baseline_share`) and any isotope/adduct partners |
| `pca_my_experiment.png` + `pca_explained_variance_my_experiment.csv` | dominant structure; for each PC, how much of its variance follows condition-plate, colony or technical replicate |
| `feature_rankings_my_experiment.csv` | every m/z bin, with the same columns |
| `ranker_concordance_my_experiment.csv` | how far the rankers agree, against two unrelated lists and, for univariate F against VIP, against shuffled labels |
| `classifier_comparison_my_experiment.png` + `classifier_accuracy_my_experiment.csv` | test accuracy per fold for six classifiers, train vs test (how well the condition-plates can be told apart) |
| `spectrum_features_my_experiment.png` | where the candidates sit in the spectra |

Then make the figures:

```bash
python src/scripts/condition_abundance.py my_experiment
python src/scripts/corroboration_figure.py my_experiment
```

Before following up a feature, look at its raw spectral region: is it a resolved peak, or the edge of a neighbouring peak? `baseline_share` near 1 means no resolved peak.

## 8. Reproduce the manuscript's Table 1

The exact published numbers used the earlier version, kept at a Git tag:

```bash
git checkout v2-table1-snapshot
python src/standard/run_analysis.py --config configs/manuscript_original.json --experiment my_experiment
git checkout main
```

## Troubleshooting

| Message | What to do |
|---|---|
| `conda: command not found` | Open a new Terminal window after installing Miniforge, or rerun its installer and answer `yes` to "initialise". |
| `EXPERIMENT ... contains no .csv/.txt spectra` | The name after `--experiment` must match a folder in `data/` exactly (capitals, spaces). The message lists the available folders. |
| `file(s) missing from metadata_manifest.csv` | A spectrum has no row in the manifest. Add it and rerun. |
| `Colonies without 3 spectra` | A colony has a different number of files, often a misnamed file. Fix its `colony` in the manifest. |
| `ModuleNotFoundError` | The environment is not active: `conda activate ms-toolkit`. |

## What has and has not been tested

- **Tested (macOS, development environment):** all tests, the demo analysis, and the analysis plus figure scripts on four real isolates.
- **Not yet tested:** a clean environment built from `environment.yml`, and a fresh clone on another Mac.

Please report any error message exactly as printed.
