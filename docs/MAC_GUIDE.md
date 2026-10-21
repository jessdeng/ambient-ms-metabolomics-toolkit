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
git checkout release/toolkit-v2      # the updated version (until it is merged into main)
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
python src/standard/run_analysis.py --config config.json
```

The first command should end with `299 passed`. The second analyses the small bundled demo, should finish in seconds, and ends with `Done in … s. Results: …/results/standard/sample_experiment`. Open that folder in Finder:

```bash
open results/standard/sample_experiment
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
   - `biological_replicate_group` is its **colony** (biological replicate). All technical replicates of one colony must share the same value;
   - `technical_replicate` is the repeat number (1, 2, 3).

   Fix any value that was guessed wrongly from the file name (for example a typo such as `AmbB5T1` next to `AmberB5T2`). Save it as CSV with the same name.

   *Why this matters:* cross-validation holds out whole colonies. If two replicates of one colony end up in different groups, accuracy is inflated.

Your data stay on your computer: everything under `data/` is excluded from Git.

## 6. Configure and run the standard analysis

1. Make your own copy of the recommended settings:
   ```bash
   cp configs/standard.json configs/my_settings.json
   open -e configs/my_settings.json        # opens it in TextEdit
   ```
   Check `MZ_MIN`/`MZ_MAX` (100–2000 for the LMJ-SSP data) and the preprocessing choices. You do not need to change anything else. Save the file. In TextEdit, use Format → Make Plain Text if asked.
2. Run:
   ```bash
   python src/standard/run_analysis.py --config configs/my_settings.json --experiment my_experiment
   ```
   One isolate (45 spectra) took about 90–105 s on a laptop. Progress messages appear as the steps run.
3. Repeat with `--experiment experiment_b`, `--experiment experiment_c`, and so on.

## 7. Find and read the results

```bash
open results/standard/my_experiment
```

Start with these files:

| File | What to look at |
|---|---|
| `classifier_accuracy_my_experiment.csv` | test accuracy (3-fold colony-grouped CV, mean ± SD) and leave-one-colony-out accuracy for the six classifiers, with the chance level |
| `pca_my_experiment.png` + `pca_explained_variance_my_experiment.csv` | dominant structure; for each PC, how much of its variance follows condition-plate, colony or technical replicate |
| `feature_rankings_my_experiment.csv` | every m/z bin: rank in each method, `n_methods` (original ensemble count), `n_families`, multi-component VIP rank, `signal_mz` |
| `vip_my_experiment.csv`, `plsda_q2_profile_my_experiment.csv` | component-1 and multi-component VIP; Q² by number of latent variables |
| `spectrum_features_my_experiment.png` | where the candidate features sit in the spectra |

Tip: in Excel, sort `feature_rankings` by `n_methods` and then by `n_families`. Before following up a feature, look at its raw spectral region: is it a resolved peak, or the edge of a neighbouring peak?

## 8. Optional analyses

These are slow and only needed for validation work; see the README section "Optional validation analyses". Before running several of them, type:

```bash
export VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
```

and run them one at a time.

## 9. Reproduce the original (published) Table 1

```bash
python src/standard/run_analysis.py --config configs/manuscript_original.json --experiment my_experiment
```

This uses the original processing (5000-point resampling, file-name grouping).

## Troubleshooting

| Message | What to do |
|---|---|
| `conda: command not found` | Open a new Terminal window after installing Miniforge, or rerun its installer and answer `yes` to "initialise". |
| `EXPERIMENT ... contains no .csv/.txt spectra` | The name after `--experiment` must match a folder in `data/` exactly (capitals, spaces). The message lists the available folders. |
| `Manifest/grouping inconsistency` | A file is missing from `metadata_manifest.csv`, or its `class` differs from its folder. Edit the CSV and rerun. |
| `ModuleNotFoundError` | The environment is not active: `conda activate ms-toolkit`. |
| The Mac becomes very slow | Run one analysis at a time and set `export VECLIB_MAXIMUM_THREADS=1` first. |

## What has and has not been tested

- **Tested by the developer (macOS, 8-core laptop):**
  - a clean environment built from the pinned versions;
  - a fresh copy of the repository;
  - all tests;
  - the demo analysis;
  - the standard analysis on four real isolates;
  - identical results between the fresh copy and the development copy.
- **Still to be tested by you on your own Mac:**
  - installing Miniforge;
  - `conda env create -f environment.yml` on your chip type;
  - running the commands above on your copy of the data.

Please report any error message exactly as printed.
