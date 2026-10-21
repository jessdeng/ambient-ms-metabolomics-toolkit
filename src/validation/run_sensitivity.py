"""Run the step-9 sensitivity settings. Resumable (one file per isolate x setting).

    PYTHONPATH=src:. python -m validation.run_sensitivity --isolates EXP_A EXP_B
"""
import argparse, contextlib, io, os, sys, time, warnings
import numpy as np
warnings.simplefilter('ignore')
from validation.data import load_config, load_isolate
from validation.stability import rank_subset
from validation.sensitivity import SETTINGS, GROUPING_VARIANTS

METHODS = ('vip_ref', 'vip_multi', 'ridge', 'lr', 'svm', 'lda', 'rf', 'gb')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--isolates', nargs='+', required=True, help='experiment folder names under data/')
    ap.add_argument('--grouping-variants', action='store_true', help='also run filename and exclusion grouping variants')
    ap.add_argument('--out', default='results/validation/step9_sensitivity')
    a = ap.parse_args(argv)
    base = load_config('configs/validation.json'); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    for iso in a.isolates:
        for name, desc, kw, over in SETTINGS + (GROUPING_VARIANTS if a.grouping_variants else []):
            f = os.path.join(a.out, f'{iso}__{name}.npz')
            if os.path.exists(f):
                continue
            cfg = dict(base, **over)
            kw = dict({'resample': 'native', 'feature_level': 'signal'}, **kw)
            with contextlib.redirect_stdout(io.StringIO()):
                d = load_isolate('data', iso, cfg, **kw)
                imps, info = rank_subset(d['X_binned'], np.asarray(d['y']), np.asarray(d['groups']), d['prep_steps'],
                                         methods=METHODS)
            np.savez_compressed(f, mz_binned=d['mz_binned'], eligible=~np.isnan(imps['ridge']),
                                n_spectra=len(d['y']), n_colonies=len(set(d['groups'])),
                                a_star=info['a_star'], q2_profile=info['q2_profile'], description=desc,
                                **{m: imps[m] for m in METHODS})
            print(f'{iso} {name}: {int((~np.isnan(imps["ridge"])).sum())} eligible features, A*={info["a_star"]} '
                  f'[{time.time() - t0:.0f}s]', file=sys.__stdout__, flush=True)


if __name__ == '__main__':
    main()
