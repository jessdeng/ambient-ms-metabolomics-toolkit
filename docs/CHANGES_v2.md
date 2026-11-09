# Changes in toolkit v2 (branch `release/toolkit-v2`, based on `main` @ b64dfc7)

v2 corrects defects in the original toolkit and adds analyses that separate four questions: whether a feature **discriminates** the conditions, whether its measurement is **analytically reproducible**, whether its **selection is stable**, and whether it has **chemical relevance**. v2 answers the first three; the fourth needs identification (high-resolution MS, MS/MS, standards). The original behaviour remains available through `configs/manuscript_original.json`.

## 1. Defects corrected

| Area | Original behaviour | v2 |
|---|---|---|
| VIP-filtered classifiers (`extras.py`) | Crashed under the current CV API; when it ran, VIP features were chosen on all labelled data before CV (selection bias). | VIP selection is refitted inside each colony-grouped training fold. |
| Cross-experiment comparison | Unpacked 4 of 7 return values and crashed. | Fixed. |
| Seed reproducibility report | Omitted Gradient Boosting; reported deterministic models as "100% stable"; reduced Ridge to one feature in binary designs. | GB included; deterministic models labelled `seed-independent`; binary designs handled. |
| Figure 3 script | Dropped all points with 6 votes. | Fixed; reads per-experiment output folders. |
| Other figure scripts (`corroboration_figure`, `condition_abundance`, `condition_spectrum`) | Looked for results only in the flat `results/standard/` folder; two failed with `No module named 'src'` when run by file path (e.g. VS Code Play); the spectrum script ignored `SPECTRAL_RESAMPLING`. | Read `results/standard/<experiment>/` (flat folder still accepted), take the experiment from the command line or `config.json`, save next to the results, list experiments with results when none is found; spectrum script uses the analysis grid. |
| Colony grouping | Inferred only from file names; inconsistent names split colonies (in one data set, 17 apparent colonies instead of 15). | `metadata_manifest.csv` (file, class, colony) with consistency checks; file names are used only when no manifest exists. |
| Overlap reference distribution (`feature_stats.py`) | Shuffled labels per spectrum (technical replicates are not exchangeable). | Shuffles whole colonies; reports an exceedance proportion. |
| Permutation classifier test | Classifier chosen after inspecting results; reported as a p-value. | Classifier pre-specified (`PERMUTATION_CLASSIFIER`); reported as an exploratory exceedance proportion, never a p-value. |
| Wrong experiment name | Silently analysed the bundled sample data. | Stops with an error listing the available folders (`ALLOW_SAMPLE_FALLBACK` = false). |
| Summary report | Without stored results it re-ran classifiers with ungrouped CV on preprocessed data. | Reports "not computed"; grouped results only. |
| "Broadhurst D-ratio" | Computed without pooled QCs. | Renamed "technical/total SD ratio". |
| Shipped settings and dependencies | `config.json` differed from the manuscript settings; pinned versions differed from those used. | `configs/standard.json`, `configs/manuscript_original.json`, `configs/validation.json`; tested pins in `environment.yml` / `requirements.txt`. |

## 2. Methods added (standard analysis)

- **Native spectral grid**: every instrument point (0.1 Da) is summed into 0.5-Da bins (`SPECTRAL_RESAMPLING = "native"`). The original 5000-point resampling used about a quarter of the instrument points.
- **PCA** (unsupervised; labels used only for colouring and for a nested variance partition of the scores).
- **Two labelled VIP definitions**: component-1 VIP (as in the original toolkit and the MetaboAnalystR/pls code path) and multi-component VIP (Wold PLS2) at A* latent variables, with A* chosen by colony-grouped Q² and all preprocessing refitted inside the folds. Cross-checked against ropls 1.42.0 and mixOmics 6.34.0.
- **Family support** (linear / tree / PLS) reported next to the ensemble count `n_methods`.
- **Leave-one-colony-out** accuracy as a supplementary estimate next to the principal 3-fold colony-grouped CV.
- **Signal groups**: an annotation that links adjacent bins belonging to one peak; features remain 0.5-Da bins.
- **Run manifest** (`run_manifest.json`): settings, package versions, Git commit and input files.

Optional validation package (`src/validation`, never run by default): seed, technical-replicate and colony-removal selection stability; preprocessing and binning sensitivity; method agreement; spectral evidence for selected features.

## 3. Breaking changes

- Outputs are written to `results/standard/<experiment>/` (previously one flat folder).
- `run_vip_filtered_classifiers(X_cv, y_labels, groups, prep_steps, safe_name, out_dir, ...)` requires colony groups and the preprocessing steps.
- The seed reproducibility CSV labels deterministic models `seed-independent`.
- Slow analyses (bootstrap stability, overlap reference distribution, permutation importance, legacy Q² permutations) are off by default.
- No fallback to the sample data.
- Figure 3: `python src/scripts/vip_vs_nmethods.py EXP1 EXP2 ...`; Figure 2: `python src/scripts/pca_overview.py EXP1 EXP2 ...`.
- `CV_FOLDS` is an upper limit: the number of folds used is min(`CV_FOLDS`, smallest number of colonies in any class), which is 3 for the manuscript data.

## 4. Effect on previously reported results

- With `configs/manuscript_original.json` the original code path is unchanged; under scikit-learn >= 1.8 it reproduced the published accuracy table from the original spectra (16 of 18 values exactly).
- The corrected standard analysis (native grid, manifest grouping, 10 CV repeats) gives different accuracies and feature lists; these are reported in the revised manuscript.
- Label-permutation p-values are no longer produced: when each condition occupies one plate, a label shuffle cannot test a condition effect.
- VIP-filtered accuracies from the original code are withdrawn (selection bias).
- Adjacent 0.5-Da bins can be two halves of one peak, and a bin on the edge of a peak can rank highly without carrying an independent signal; check `signal_mz` and the spectra before interpreting a feature.
- High ensemble support combined with component-1 VIP < 1 is largely a property of component-1 VIP, which follows one dominant class contrast.

## 5. Remaining limitations

- Condition and plate/enclosure are aliased; all condition differences are condition-plate associations.
- Files whose colony identity cannot be resolved from the sample records should be kept as their own unit and their exclusion reported as a sensitivity check.
- Unregularised LDA is kept for continuity with the original Table 1; its training accuracy is below 1, and its test accuracy was the lowest of the six classifiers in two of the four manuscript data sets.
- Media blanks are not used.
- The validation runners `run_stability`, `run_technical` and `run_agreement` need the importance tables in `results/validation/step4_importance/`; the scripts that create them exist on the research branch but are not yet part of this release.
- The plotting scripts for the spectral-evidence, stability and sensitivity figures are not packaged; those figures are drawn from the CSV outputs of the validation package.
