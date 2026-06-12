# Data Directory

Place your experiment folders here. Each folder becomes one experiment in the pipeline.

## Expected Structure

```
data/
└── your_experiment_name/
    ├── Group1/
    │   ├── sample_A1T1.csv
    │   ├── sample_A1T2.csv
    │   └── sample_A1T3.csv
    ├── Group2/
    │   ├── sample_B2T1.csv
    │   └── ...
    └── Group3/
        └── ...
```

- Each **subfolder** = one biological group (condition/isolate/treatment)
- Each **file** = one sample (one MS acquisition)
- Supported formats: `.csv` (comma-separated) or `.txt` (tab-separated)

## Required File Columns

Each sample file must contain at least these two columns (names are case-insensitive):

| Column | Accepted names | Description |
|--------|---------------|-------------|
| m/z | `mz`, `Mass/Charge`, `m/z` | Feature mass-to-charge ratio |
| Intensity | `int`, `Intensity`, `intensity` | Feature intensity |

## Filename Convention for GroupKFold CV

For the pseudoreplication-corrected cross-validation to work correctly, filenames should encode the **biological replicate** (colony/well) and **technical replicate** number using the pattern:

```
<prefix><well><T><replicate>.<ext>
```

Examples: `ConditionA_W1T1.csv`, `ConditionB_W2T2.txt`, `ConditionC_W3T1.csv`

The pipeline extracts the well/replicate ID (e.g. `W1`) from the filename and groups all technical replicates of that biological replicate together so they never straddle the train/test boundary in cross-validation. If your filenames do not follow this pattern, grouping falls back to per-file (each file is its own group), which disables grouping but does not cause an error.

## Configuration

After placing your data here, set `EXPERIMENT` in **config.json** (recommended) or `config.py`:

```json
{ "EXPERIMENT": "your_experiment_name" }
```

The pipeline resolves the full path as `data/<EXPERIMENT>`.
