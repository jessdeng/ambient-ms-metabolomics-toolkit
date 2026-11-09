"""
Spectral evidence for candidate features — is the bin a real ion?

At unit resolution a 0.5-Da bin can hold the tail of a neighbouring peak or a
raised baseline. For each candidate this reads the raw (instrument-grid)
profiles inside the bin's signal group and reports

  apex_mz / apex_sd   where the profile peaks in each spectrum, and how much that
                      position moves (a real ion has a stable apex)
  baseline_share      local baseline (median of flank points either side) as a
                      share of the summed intensity; near 1 = no resolved peak
  related_ions        candidate 13C-isotopologue (+1.003) and Na/K/NH4 adduct
                      partners whose colony means co-vary (flags, not identities)
"""

import numpy as np
import pandas as pd

from shared.feature_groups import aggregate_to_groups, flag_relations


def peak_evidence(raw_X, raw_mz, lo, hi, gap=0.2, flank=0.7):
    """Apex position and baseline share of the raw profile in [lo, hi)."""
    win = (raw_mz >= lo) & (raw_mz < hi)
    if not win.any():
        return dict(apex_mz=np.nan, apex_sd=np.nan, baseline_share=np.nan)
    apex = raw_mz[win][np.argmax(raw_X[:, win], axis=1)]
    fl = (((raw_mz >= lo - gap - flank) & (raw_mz < lo - gap))
          | ((raw_mz >= hi + gap) & (raw_mz < hi + gap + flank)))
    gross = raw_X[:, win].sum(1)
    base = (np.median(raw_X[:, fl], axis=1) * win.sum()) if fl.any() else np.zeros(len(gross))
    share = np.divide(base, gross, out=np.full_like(gross, np.nan), where=gross > 0)
    return dict(apex_mz=float(np.median(apex)), apex_sd=float(np.std(apex, ddof=1)),
                baseline_share=float(np.median(share)))


def candidate_evidence(candidates, signal_map, X_binned, raw_X, raw_mz, colony_id, bin_width):
    """One row of evidence per candidate (aligned with ``candidates``' index)."""
    Xg, gids, smz = aggregate_to_groups(X_binned, signal_map)
    rel = flag_relations(smz, Xg, colony_id)
    pos = {g: i for i, g in enumerate(gids)}
    sig = signal_map.set_index('mz_bin')
    rows = []
    for idx, mz in candidates['mz'].items():
        gid = int(sig.loc[mz, 'signal_group'])
        bins = signal_map[signal_map.signal_group == gid]
        lo, hi = bins.bin_lo.min(), bins.bin_lo.max() + bin_width
        k = pos[gid]
        notes = [f"{r.relation} -> m/z {r.mz_j:.1f} (r={r.r:.2f})" for r in rel[rel.group_i == k].itertuples()]
        notes += [f"partner of m/z {r.mz_i:.1f}: {r.relation} (r={r.r:.2f})" for r in rel[rel.group_j == k].itertuples()]
        rows.append(dict(index=idx, **peak_evidence(raw_X, raw_mz, lo, hi),
                         related_ions='; '.join(notes)))
    return pd.DataFrame(rows).set_index('index')
