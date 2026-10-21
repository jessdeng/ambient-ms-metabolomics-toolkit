"""Data access for the validated workflow.

Reproduces the original descriptive preprocessing path of standard/run_analysis.py
exactly (same functions, same order), and exposes the raw binned matrix plus the
leak-free per-fold preprocessor for resampling analyses.
"""
import os
import re
import sys
import json

import numpy as np

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(_SRC)
for _p in (_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from standard.preprocessing import (load_experiment, bin_features, filter_mass_range,
                                    filter_snr_floor, filter_prevalence,
                                    filter_low_variance, filter_low_abundance, preprocess)
from shared.grouping import resolve_groups, load_manifest_groups
from shared.classifier_comparison_standard import make_preprocessor


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    return {k: v for k, v in cfg.items() if not k.startswith('_')}


def load_isolate(data_dir, experiment, cfg, grouping='manifest', resample=None,
                 feature_level='bin', return_raw=False):
    """Load one experiment and return a dict with raw binned data, groups and the
    descriptive (full-data) matrix exactly as built by standard/run_analysis.py.

    grouping : 'manifest' (metadata_manifest.csv; provisional corrections),
               'filename' (original topology rule), or 'manifest_exclude'
               (manifest, dropping files whose assignment_status is UNRESOLVED and
               whose colony differs from the file-name colony).
    resample : None -> cfg['SPECTRAL_RESAMPLING'] (default 'linspace5000', the
               original behaviour) or 'native'.
    feature_level : 'bin' (0.5-Da bins) or 'signal' (bins merged into evidence-based
               signal groups by shared.feature_groups; raw intensities summed within a
               group before the usual filters). Grouping is label-free.
    """
    exp_dir = os.path.join(data_dir, experiment)
    resample = resample or cfg.get('SPECTRAL_RESAMPLING', 'linspace5000')
    X_raw, y, names, mz = load_experiment(exp_dir, validate=False, resample=resample)
    y = np.asarray(y)
    manifest = os.path.join(exp_dir, cfg.get('METADATA_MANIFEST', 'metadata_manifest.csv'))
    keep = np.ones(len(names), dtype=bool)
    if grouping == 'filename' or not os.path.exists(manifest):
        groups = resolve_groups(y, names, None, None)
    else:
        groups = load_manifest_groups(manifest, names, y_labels=y)
        if grouping == 'manifest_exclude':
            import pandas as pd
            man = pd.read_csv(manifest, dtype=str).fillna('').set_index('file_path')
            for i, n in enumerate(names):
                st = man.loc[n].get('assignment_status', '')
                well = re.search(r'([A-Z]\d+)T\d+\.\w+$', n)
                if st.startswith('UNRESOLVED') and well and well.group(1) != man.loc[n, 'colony']:
                    keep[i] = False
    X_raw, y, groups = X_raw[keep], y[keep], np.asarray(groups)[keep]
    names = [n for n, k in zip(names, keep) if k]

    if feature_level == 'nominal':
        from standard.preprocessing import bin_features_nominal
        Xb, mzb = bin_features_nominal(X_raw, mz, offset=cfg.get('NOMINAL_BIN_OFFSET', 0.6))
    else:
        Xb, mzb = bin_features(X_raw, mz, bin_width=cfg['BIN_WIDTH'])
    Xb, mzb = filter_mass_range(Xb, mzb, mz_min=cfg['MZ_MIN'], mz_max=cfg['MZ_MAX'])
    group_map = pair_table = None
    if feature_level == 'signal':
        from shared.feature_groups import build_signal_groups, aggregate_to_groups
        group_map, pair_table = build_signal_groups(
            Xb, mzb, X_raw, mz, groups, bin_width=cfg['BIN_WIDTH'], mz0=float(mz.min()),
            r_min=cfg.get('SIGNAL_R_MIN', 0.9))
        Xb, _gids, mzb = aggregate_to_groups(Xb, group_map)

    # Descriptive copy: identical order to standard/run_analysis.py steps 2c-4.
    Xd, mzd = Xb.copy(), mzb.copy()
    if cfg.get('SNR_FLOOR_ENABLED', False):
        Xd, mzd = filter_snr_floor(Xd, mzd, y, snr_threshold=cfg['SNR_THRESHOLD'],
                                   noise_quantile=cfg['NOISE_QUANTILE'],
                                   min_fraction=cfg['MIN_FRACTION_IN_GROUP'])
    if cfg.get('PREVALENCE_THRESHOLD', 0) > 0:
        Xd, mzd = filter_prevalence(Xd, mzd, y, threshold=cfg['PREVALENCE_THRESHOLD'])
    if cfg['VARIANCE_PERCENTILE'] > 0:
        Xd, mzd = filter_low_variance(Xd, mzd, percentile=cfg['VARIANCE_PERCENTILE'])
    if cfg['ABUNDANCE_PERCENTILE'] > 0:
        Xd, mzd = filter_low_abundance(Xd, mzd, percentile=cfg['ABUNDANCE_PERCENTILE'])
    X = preprocess(Xd, normalization=cfg['NORMALIZATION'],
                   log_transform=cfg['LOG_TRANSFORM'], scaling=cfg['SCALING'])
    prep_steps = make_preprocessor(
        normalization=cfg['NORMALIZATION'], log_transform=cfg['LOG_TRANSFORM'],
        scaling=cfg['SCALING'], variance_percentile=cfg['VARIANCE_PERCENTILE'],
        abundance_percentile=cfg['ABUNDANCE_PERCENTILE'],
        prevalence_threshold=cfg.get('PREVALENCE_THRESHOLD', 0.5),
        snr_floor_enabled=False)
    return dict(experiment=experiment, y=y, groups=groups, names=names,
                X_binned=Xb, mz_binned=mzb, X_filt_raw=Xd, mz=mzd, X=X,
                prep_steps=prep_steps, n_dropped=int((~keep).sum()),
                resample=resample, feature_level=feature_level,
                group_map=group_map, pair_table=pair_table,
                **({'X_raw': X_raw, 'raw_mz': mz} if return_raw else {}))
