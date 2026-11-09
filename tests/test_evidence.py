"""Spectral evidence: apex position and baseline share of a raw profile."""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.evidence import peak_evidence

MZ = np.round(np.arange(150.0, 160.0001, 0.1), 1)


def test_resolved_peak_has_stable_apex_and_low_baseline():
    rng = np.random.default_rng(0)
    X = np.vstack([1e5 * np.exp(-0.5 * ((MZ - 155.2) / 0.3) ** 2) + rng.normal(100, 5, MZ.size)
                   for _ in range(6)])
    e = peak_evidence(X, MZ, 155.0, 155.5)
    assert e['apex_mz'] == pytest.approx(155.2) and e['apex_sd'] < 0.05
    assert e['baseline_share'] < 0.05


def test_flat_baseline_has_share_near_one():
    X = np.full((4, MZ.size), 200.0)
    assert peak_evidence(X, MZ, 155.0, 155.5)['baseline_share'] == pytest.approx(1.0)
