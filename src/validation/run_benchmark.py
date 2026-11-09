"""Exploratory shuffled-label benchmark for consensus and stability (plan step 8).

    PYTHONPATH=src:. python -m validation.run_benchmark --isolate MY_EXPERIMENT --n-shuffles 50

Shuffle 0 = observed labels; shuffles 1..B = colony-level label shuffles. Every
label set goes through the identical pipeline: full-data rankings plus the
9-subset colony-removal design (validation.consensus.DESIGN9), with preprocessing
and PLS-DA A* refitted in every fit and the same seeds (RF 25, GB 5). One file per
label set (resumable). Results are exceedance proportions, never p-values.
"""
import argparse, contextlib, io, json, os, sys, time, warnings
import numpy as np
warnings.simplefilter('ignore')
from validation.data import load_config, load_isolate
from validation.stability import rank_subset, subset_indices
from validation.consensus import (DESIGN9, full_data_statistics, resampled_statistics, shuffle_colony_labels)
from shared.importance import top_n

METHODS = ('vip_ref', 'vip_multi', 'ridge', 'lr', 'svm', 'lda', 'rf', 'gb')


def run_label_set(d, y, seed=42):
    g = np.asarray(d['groups']); Xb = d['X_binned']
    full, info = rank_subset(Xb, y, g, d['prep_steps'], methods=METHODS, seed=seed)
    subs, a_sub = [], []
    for drop in DESIGN9:
        idx, _ = subset_indices(y, g, drop)
        imps, inf = rank_subset(Xb[idx], y[idx], g[idx], d['prep_steps'], methods=METHODS, seed=seed)
        subs.append(imps); a_sub.append(inf['a_star'])
    stats = dict(full_data_statistics(full)); r, _ = resampled_statistics(subs)
    stats.update(r); stats.update(a_star_full=info['a_star'], a_star_subsets=a_sub)
    tops = {f'full_{m}': np.array(top_n(np.nan_to_num(full[m], nan=-np.inf), 100)) for m in METHODS}
    for s, imps in enumerate(subs):
        for m in METHODS:
            tops[f's{s}_{m}'] = np.array(top_n(np.nan_to_num(imps[m], nan=-np.inf), 100))
    return stats, tops


STOP_FILE = 'results/validation/STOP'   # cooperative stop: checked before every label set


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--isolate', required=True); ap.add_argument('--n-shuffles', type=int, default=50)
    ap.add_argument('--seed', type=int, default=2026); ap.add_argument('--level', default='signal')
    ap.add_argument('--out', default='results/validation/step8_benchmark')
    a = ap.parse_args(argv)
    cfg = load_config('configs/validation.json')
    with contextlib.redirect_stdout(io.StringIO()):
        d = load_isolate('data', a.isolate, cfg, resample='native', feature_level=a.level)
    out = os.path.join(a.out, f'{a.isolate}_{a.level}'); os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(a.seed + sum(map(ord, a.isolate)))
    labels = [np.asarray(d['y'])] + [shuffle_colony_labels(d['y'], d['groups'], rng) for _ in range(a.n_shuffles)]
    t0 = time.time()
    for b, y in enumerate(labels):
        f = os.path.join(out, f'labels_{b:03d}.npz')
        if os.path.exists(f):
            continue
        if os.path.exists(STOP_FILE):
            print(f'{a.isolate} stopped by {STOP_FILE}', file=sys.__stdout__, flush=True); return
        with contextlib.redirect_stdout(io.StringIO()):
            stats, tops = run_label_set(d, y)
        tmp = f[:-4] + '.tmp.npz'
        np.savez_compressed(tmp, y=y, stats=json.dumps(stats), **tops); os.replace(tmp, f)
        print(f'{a.isolate} label set {b} done [{time.time() - t0:.0f}s]', file=sys.__stdout__, flush=True)
    print(f'{a.isolate} benchmark complete', flush=True)


if __name__ == '__main__':
    main()
