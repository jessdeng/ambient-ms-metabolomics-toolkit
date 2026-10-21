"""Classification validation with a colony-level shuffled-label benchmark (step 11).

    PYTHONPATH=src:. python -m validation.run_classification --datasets EXP_A EXP_B:filename

Observed labels: leave-one-colony-out (pooled predictions) and 3-fold x 5
grouped CV for all six original classifiers. Benchmark: 20 colony-level label
shuffles (pre-specified), leave-one-colony-out accuracy for each classifier.
Resumable: one file per dataset x label set.
"""
import argparse, contextlib, io, json, os, sys, time, warnings
import numpy as np, pandas as pd
warnings.simplefilter('ignore')
from validation.data import load_config, load_isolate
from validation.classification import evaluate_all, loco_evaluate, original_estimators
from validation.consensus import shuffle_colony_labels


STOP_FILE = 'results/validation/STOP'   # cooperative stop: checked before every unit


def _atomic_json(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', required=True, help='EXPERIMENT or EXPERIMENT:grouping (manifest|filename|manifest_exclude)')
    ap.add_argument('--n-shuffles', type=int, default=20); ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--out', default='results/validation/step11_classification')
    a = ap.parse_args(argv)
    cfg = load_config('configs/validation.json'); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    for ds in a.datasets:
        iso, grouping = (ds.split(':') + ['manifest'])[:2]
        tag = f'{iso}__{grouping}'
        with contextlib.redirect_stdout(io.StringIO()):
            d = load_isolate('data', iso, cfg, grouping=grouping, resample='native', feature_level='signal')
        y, g = np.asarray(d['y']), np.asarray(d['groups'])
        f0 = os.path.join(a.out, f'{tag}__observed.json')
        if os.path.exists(STOP_FILE):
            print(f'stopped by {STOP_FILE}', file=sys.__stdout__, flush=True); return
        if not os.path.exists(f0):
            with contextlib.redirect_stdout(io.StringIO()):
                tab, conf = evaluate_all(d['X_binned'], y, g, d['prep_steps'])
            _atomic_json(dict(table=tab.to_dict('records'), n_spectra=len(y), n_colonies=len(set(g)),
                              confusion={m: c.to_dict('split') for m, c in conf.items()}), f0)
            print(f'{tag} observed done [{time.time() - t0:.0f}s]', file=sys.__stdout__, flush=True)
        rng = np.random.default_rng(a.seed + sum(map(ord, tag)))
        shuffles = [shuffle_colony_labels(y, g, rng) for _ in range(a.n_shuffles)]
        est = original_estimators()
        for b, ys in enumerate(shuffles, start=1):
            fb = os.path.join(a.out, f'{tag}__shuffle_{b:03d}.json')
            if os.path.exists(fb):
                continue
            if os.path.exists(STOP_FILE):
                print(f'stopped by {STOP_FILE}', file=sys.__stdout__, flush=True); return
            with contextlib.redirect_stdout(io.StringIO()):
                acc = {m: loco_evaluate(e, d['X_binned'], ys, g, d['prep_steps'])['accuracy'] for m, e in est.items()}
            _atomic_json(acc, fb)
        print(f'{tag} benchmark done [{time.time() - t0:.0f}s]', file=sys.__stdout__, flush=True)


if __name__ == '__main__':
    main()
