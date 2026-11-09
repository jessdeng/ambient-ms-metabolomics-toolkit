"""Colony-removal stability driver (plan step 7). Resumable: one file per subset.

    PYTHONPATH=src:. python -m validation.run_stability --isolate MY_EXPERIMENT --design oa27
    PYTHONPATH=src:. python -m validation.run_stability --isolate MY_EXPERIMENT --design full --methods no_gb
    PYTHONPATH=src:. python -m validation.run_stability --isolate MY_EXPERIMENT --design full --methods gb
"""
import argparse, contextlib, io, os, sys, time, warnings
import pandas as pd
warnings.simplefilter('ignore')
from validation.data import load_config, load_isolate
from validation.stability import subset_design, run_subsets, METHODS

SETS = {'all': METHODS, 'no_gb': tuple(m for m in METHODS if m != 'gb'), 'gb': ('gb',)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--isolate', required=True); ap.add_argument('--design', default='oa27')
    ap.add_argument('--methods', default='all', choices=list(SETS)); ap.add_argument('--level', default='signal')
    ap.add_argument('--gb-seeds', type=int, default=5); ap.add_argument('--rf-seeds', type=int, default=25)
    ap.add_argument('--out', default='results/validation/step7_stability')
    a = ap.parse_args(argv)
    cfg = load_config('configs/validation.json')
    AS = pd.read_csv('results/validation/step4_importance/q2_profile_validated_features.csv').set_index(['level', 'isolate']).A_star
    with contextlib.redirect_stdout(io.StringIO()):
        d = load_isolate('data', a.isolate, cfg, resample='native', feature_level=a.level)
    out = os.path.join(a.out, f'{a.isolate}_{a.level}'); tag = f'{a.design}_{a.methods}'
    t0 = time.time(); n = [0]

    def log(s):
        n[0] += 1
        if n[0] % 9 == 0:
            print(f'{s}  [{n[0]} new, {time.time() - t0:.0f}s]', file=sys.__stdout__, flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        meta = run_subsets(d, d['prep_steps'], subset_design(design=a.design), out, tag, log=log,
                           methods=SETS[a.methods], gb_seeds=a.gb_seeds, rf_seeds=a.rf_seeds,
                           fixed_A=int(AS[(a.level, a.isolate)]))
    meta.to_csv(os.path.join(out, f'meta_{tag}.csv'), index=False)
    print(f'{tag} complete: {len(meta)} subsets, {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
