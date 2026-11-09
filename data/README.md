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

- Each **subfolder** = one condition (treatment)
- Each **file** = one sample (one MS acquisition)
- Supported formats: `.csv` (comma-separated) or `.txt` (tab-separated)

## Required File Columns

Each sample file must contain at least these two columns (names are case-insensitive):

| Column | Accepted names | Description |
|--------|---------------|-------------|
| m/z | `mz`, `Mass/Charge`, `m/z` | Feature mass-to-charge ratio |
| Intensity | `int`, `Intensity`, `intensity` | Feature intensity |

## Colonies and technical replicates

The colony (biological replicate) is the experimental unit; its technical replicates are averaged before any ranking. The colony of each file is read from `metadata_manifest.csv` in the experiment folder if present (columns `file_path`, `class`, `colony`), otherwise from the layout:

- `<condition>/<colony>/<file>`: the file's folder is the colony;
- `<condition>/<name>T<n>.csv`: the name without the trailing `T<n>` is the colony (`ConditionA_W1T2.csv` -> `ConditionA_W1`).

If every file ends up as its own colony the run warns; a condition with fewer than two colonies stops it. `python src/shared/grouping.py data/<experiment>` writes a manifest to check.

## Configuration

After placing your data here, set `EXPERIMENT` in **config.json** (or pass `--experiment`):

```json
{ "EXPERIMENT": "your_experiment_name" }
```

The pipeline resolves the full path as `data/<EXPERIMENT>`.
