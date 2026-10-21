"""Technical-replicate stability (plan step 7). Resumable.

    PYTHONPATH=src:. python -m validation.run_technical --isolate MY_EXPERIMENT

Designs (colonies fixed throughout):
  tech30 : drop one technical replicate per colony (n = 30; same n as the
           colony-removal subsets, which drop one colony per condition).
  d15single : one technical replicate per colony (n = 15).
  d15mean   : colony means of the three technical replicates (n = 15), averaged
              on the raw scale before normalisation and log transform.
"""
import argparse, contextlib, io, os, sys, time, warnings
import numpy as np, pandas as pd
warnings.simplefilter('ignore')
from validation.data import load_config, load_isolate
from validation.stability import technical_draws, colony_mean_data, run_index_sets, rank_subset, METHODS


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--isolate', required=True); ap.add_argument('--level', default='signal')
    ap.add_argument('--b-tech30', type=int, default=100); ap.add_argument('--b-d15single', type=int, default=25)
    ap.add_argument('--out', default='results/validation/step7_stability')
    a = ap.parse_args(argv)
    cfg = load_config('configs/validation.json')
    AS = pd.read_csv('results/validation/step4_importance/q2_profile_validated_features.csv').set_index(['level', 'isolate']).A_star
    fixed_A = int(AS[(a.level, a.isolate)])
    with contextlib.redirect_stdout(io.StringIO()):
        d = load_isolate('data', a.isolate, cfg, resample='native', feature_level=a.level)
    out = os.path.join(a.out, f'{a.isolate}_{a.level}'); t0 = time.time()

    def log(s):
        if s.split()[1] in ('24', '49', '74', '99'):
            print(f'{s}  [{time.time() - t0:.0f}s]', file=sys.__stdout__, flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        m30 = run_index_sets(d, d['prep_steps'], technical_draws(d['groups'], 2, a.b_tech30, seed=30), out, 'tech30',
                             log=log, fixed_A=fixed_A)
        m15 = run_index_sets(d, d['prep_steps'], technical_draws(d['groups'], 1, a.b_d15single, seed=15), out, 'd15single',
                             log=log, fixed_A=fixed_A)
        Xm, ym, cm = colony_mean_data(d['X_binned'], d['y'], d['groups'])
        mm = run_index_sets(d, d['prep_steps'], [np.arange(len(ym))], out, 'd15mean', log=log,
                            X_binned=Xm, y=ym, groups=cm, fixed_A=fixed_A)
    for tag, m in (('tech30', m30), ('d15single', m15), ('d15mean', mm)):
        m.to_csv(os.path.join(out, f'meta_{tag}.csv'), index=False)
    print(f'{a.isolate} technical designs complete, {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
