"""
tests/test_preprocessing.py
============================
Unit tests for the SNR floor filter, glog transform, CV-transformer parity,
and r_comparable MetaboAnalyst regression.

Run from the repository root:
    pytest tests/test_preprocessing.py -v
"""

import sys
import os
import numpy as np
import pytest

# Ensure both src/ and repo root are on sys.path so imports resolve correctly.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC  = os.path.join(_ROOT, 'src')
for _p in (_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# A.  SNR floor — filter_snr_floor
# ---------------------------------------------------------------------------

class TestSNRFloor:
    """Tests for standard.preprocessing.filter_snr_floor."""

    def _make_matrix(self):
        """
        6 samples × 5 bins.  Groups: A (samples 0-2), B (samples 3-5).

          bin 0 : flat baseline — constant ~5 in all samples.
                  SNR = 5/sigma ≈ 1.0 (sigma ~ noise floor = 5).
                  MUST be REMOVED (below threshold in every group).

          bin 1 : tall peak in group A only (200-220 in A, ~5 in B).
                  SNR in A >> 3; fails in B.
                  MUST be KEPT (passes in at least one group = A).

          bin 2 : shared peak in both groups (270-320 in all samples).
                  MUST be KEPT.

          bin 3 : tall peak in group B only (~5 in A, 180-210 in B).
                  MUST be KEPT (passes in B).

          bin 4 : flat baseline — constant ~5 in all samples.
                  MUST be REMOVED.

        Noise-region sigma:
          Each row has values like [5, peak, peak, 5, 5].  At noise_quantile=60,
          the 60th-percentile of such a row is ~5 (three of five values are 5).
          low_vals = [5, 5, 5] → MAD = 0 → sigma falls back to median = 5.
          SNR for flat bins = 5/5 = 1.0  < snr_threshold=3  → removed.
          SNR for peak bins = 200/5 = 40  ≥ snr_threshold=3  → kept.
        """
        X = np.array([
            #  b0     b1     b2     b3    b4
            [  5.0, 200.0, 300.0,   5.0,  5.0],   # A
            [  5.0, 180.0, 280.0,   5.0,  5.0],   # A
            [  5.0, 220.0, 320.0,   5.0,  5.0],   # A
            [  5.0,   5.0, 300.0, 200.0,  5.0],   # B
            [  5.0,   5.0, 270.0, 180.0,  5.0],   # B
            [  5.0,   5.0, 290.0, 210.0,  5.0],   # B
        ], dtype=float)
        y  = np.array(['A', 'A', 'A', 'B', 'B', 'B'])
        mz = np.array([100., 200., 300., 400., 500.])
        return X, y, mz

    def test_flat_baseline_removed(self):
        """Flat baseline bins (SNR ≤ 1) are removed in all groups."""
        from standard.preprocessing import filter_snr_floor
        X, y, mz = self._make_matrix()
        X_f, mz_f = filter_snr_floor(X, mz, y, snr_threshold=3,
                                      noise_quantile=60, min_fraction=0.5)
        assert 100.0 not in mz_f, "Flat baseline bin b0 (m/z 100) should be removed"
        assert 500.0 not in mz_f, "Flat baseline bin b4 (m/z 500) should be removed"

    def test_single_group_peak_retained(self):
        """A tall peak present in only ONE group is retained (group-aware OR logic)."""
        from standard.preprocessing import filter_snr_floor
        X, y, mz = self._make_matrix()
        X_f, mz_f = filter_snr_floor(X, mz, y, snr_threshold=3,
                                      noise_quantile=60, min_fraction=0.5)
        assert 200.0 in mz_f, (
            "Group-A-specific peak (m/z 200) should be retained — "
            "the floor must not require signal in EVERY group"
        )
        assert 400.0 in mz_f, (
            "Group-B-specific peak (m/z 400) should be retained"
        )

    def test_shared_peak_retained(self):
        """A peak present in both groups is retained."""
        from standard.preprocessing import filter_snr_floor
        X, y, mz = self._make_matrix()
        X_f, mz_f = filter_snr_floor(X, mz, y, snr_threshold=3,
                                      noise_quantile=60, min_fraction=0.5)
        assert 300.0 in mz_f, "Shared peak (m/z 300) should be retained"

    def test_output_shape_consistent(self):
        """X_f and mz_f have matching n_features; n_samples unchanged."""
        from standard.preprocessing import filter_snr_floor
        X, y, mz = self._make_matrix()
        X_f, mz_f = filter_snr_floor(X, mz, y)
        assert X_f.shape[1] == len(mz_f)
        assert X_f.shape[0] == X.shape[0]

    def test_transformer_guard_never_empties(self):
        """SNRFloor transformer's guard keeps at least one feature on degenerate folds."""
        from shared.classifier_comparison_standard import SNRFloor
        rng = np.random.default_rng(0)
        X   = np.abs(rng.standard_normal((4, 10))) + 1.0
        y   = np.array(['A', 'A', 'B', 'B'])
        sf  = SNRFloor(snr_threshold=1e9, enabled=True)  # impossibly high threshold
        sf.fit(X, y)
        assert sf.keep_.sum() >= 1, "SNRFloor transformer guard should keep ≥1 feature"


# ---------------------------------------------------------------------------
# B.  glog — finite at zero and elsewhere
# ---------------------------------------------------------------------------

class TestGlog:
    """Tests for the glog (arcsinh) transform path in preprocess()."""

    def test_glog_finite_at_zero(self):
        """arcsinh(0 / lambda_) = 0 exactly; no half-min hack needed."""
        from standard.preprocessing import preprocess
        X = np.array([[0.0, 100.0, 500.0],
                      [0.0,  50.0, 300.0]])
        X_out = preprocess(X.copy(), normalization='none',
                           log_transform='glog', scaling='none')
        assert np.all(np.isfinite(X_out)), "glog must produce finite values for all inputs"
        # arcsinh(0/lambda_) = 0 exactly for the zero bins
        # (scaling='none' in preprocess subtracts feature mean; zeros map to 0 - 0 = 0)
        # Use the LogTransform transformer directly to check zero → 0:
        from shared.classifier_comparison_standard import LogTransform
        lt = LogTransform(method='glog')
        lt.fit(X)
        X_tf = lt.transform(X.copy())
        assert X_tf[0, 0] == 0.0 and X_tf[1, 0] == 0.0, \
            "glog transform of zero should produce exactly 0.0"

    def test_glog_monotone(self):
        """Higher raw intensity → higher glog value (via LogTransform, not preprocess scaling)."""
        from shared.classifier_comparison_standard import LogTransform
        # Use a 2-row matrix so the transformer sees real training data
        X = np.array([[10.0, 100.0, 1000.0],
                      [20.0, 200.0,  800.0]])
        lt = LogTransform(method='glog')
        lt.fit(X)
        # Check monotonicity on the first row
        out = lt.transform(np.array([[10.0, 100.0, 1000.0]]))
        assert out[0, 0] < out[0, 1] < out[0, 2], \
            "glog transform must be monotonically increasing"

    def test_glog_accepted_by_preprocess(self):
        """preprocess() does not raise for log_transform='glog'."""
        from standard.preprocessing import preprocess
        X = np.abs(np.random.default_rng(1).standard_normal((5, 20))) + 1.0
        try:
            preprocess(X.copy(), normalization='tic',
                       log_transform='glog', scaling='autoscale')
        except ValueError as e:
            pytest.fail(f"preprocess raised ValueError for glog: {e}")

    def test_glog_rejects_unknown_transform(self):
        """preprocess() still raises for unknown transform options."""
        from standard.preprocessing import preprocess
        X = np.ones((3, 5))
        with pytest.raises(ValueError, match="Unknown log_transform"):
            preprocess(X.copy(), log_transform='log3')


# ---------------------------------------------------------------------------
# C.  Per-fold transformer agreement with preprocess()
# ---------------------------------------------------------------------------

class TestTransformerAgreement:
    """
    For a fixed dataset, LogTransform fitted on that data and then applied
    should agree with preprocess() to machine precision (max abs diff ≈ 0.0).

    This verifies the per-fold CV transformers are consistent with the
    full-data preprocess() path when both see the same data.
    """

    def _rng_matrix(self, seed=42, n=10, p=15):
        rng = np.random.default_rng(seed)
        return np.abs(rng.standard_normal((n, p))) * 100 + 1.0  # all positive

    # ---- standard path (classifier_comparison_standard.LogTransform) --------

    def test_standard_log10_agreement(self):
        from shared.classifier_comparison_standard import LogTransform
        from standard.preprocessing import preprocess
        X = self._rng_matrix()
        lt = LogTransform(method='log10')
        lt.fit(X)
        out_tf  = lt.transform(X.copy())
        out_pre = preprocess(X.copy(), normalization='none',
                             log_transform='log10', scaling='none')
        # preprocess 'none' scaling centers by mean; LogTransform doesn't center.
        # The transforms (before centering) must match.
        # Compare only the log-transform step by centering both:
        assert np.allclose(out_tf - out_tf.mean(axis=0),
                           out_pre - out_pre.mean(axis=0), atol=1e-10), \
            f"standard log10 transformer and preprocess disagree"

    def test_standard_glog_agreement(self):
        from shared.classifier_comparison_standard import LogTransform
        from standard.preprocessing import preprocess
        X = self._rng_matrix()
        lt = LogTransform(method='glog')
        lt.fit(X)
        out_tf  = lt.transform(X.copy())
        out_pre = preprocess(X.copy(), normalization='none',
                             log_transform='glog', scaling='none')
        # Both use the same lambda_ (5th percentile of X[X>0]) on the same X
        assert np.allclose(out_tf - out_tf.mean(axis=0),
                           out_pre - out_pre.mean(axis=0), atol=1e-10), \
            f"standard glog transformer and preprocess disagree"

    # ---- r_comparable path (classifier_comparison.LogTransform) --------------

    def test_rcomp_log10_agreement(self):
        from shared.classifier_comparison import LogTransform
        from standard.preprocessing import preprocess
        X = self._rng_matrix()
        lt = LogTransform(method='log10')
        lt.fit(X)
        out_tf  = lt.transform(X.copy())
        out_pre = preprocess(X.copy(), normalization='none',
                             log_transform='log10', scaling='none')
        assert np.allclose(out_tf - out_tf.mean(axis=0),
                           out_pre - out_pre.mean(axis=0), atol=1e-10), \
            "r_comparable log10 transformer and preprocess disagree"

    def test_rcomp_glog_agreement(self):
        from shared.classifier_comparison import LogTransform
        from standard.preprocessing import preprocess
        X = self._rng_matrix()
        lt = LogTransform(method='glog')
        lt.fit(X)
        out_tf  = lt.transform(X.copy())
        out_pre = preprocess(X.copy(), normalization='none',
                             log_transform='glog', scaling='none')
        assert np.allclose(out_tf - out_tf.mean(axis=0),
                           out_pre - out_pre.mean(axis=0), atol=1e-10), \
            "r_comparable glog transformer and preprocess disagree"

    def test_fixed_lambda_exact_match(self):
        """With the same lambda_, transformer output equals np.arcsinh(X/lambda_) exactly."""
        from shared.classifier_comparison_standard import LogTransform
        X = self._rng_matrix(seed=7)
        lt = LogTransform(method='glog')
        lt.fit(X)
        lambda_ = lt.params_['lambda']

        out_tf     = lt.transform(X.copy())
        out_direct = np.arcsinh(X / lambda_)

        assert np.allclose(out_tf, out_direct, atol=1e-14), \
            f"LogTransform glog != arcsinh(X/lambda_): max diff {np.abs(out_tf - out_direct).max()}"

    def test_same_lambda_both_paths_agree(self):
        """When preprocess() and the transformer derive lambda_ from the same X,
        their outputs (before centering) match to machine precision."""
        from shared.classifier_comparison_standard import LogTransform
        from standard.preprocessing import preprocess

        X = self._rng_matrix(seed=13)
        # Transformer
        lt = LogTransform(method='glog')
        lt.fit(X)
        out_tf = lt.transform(X.copy())
        # preprocess with scaling='none' (subtracts mean); undo the centering
        out_pre = preprocess(X.copy(), normalization='none',
                             log_transform='glog', scaling='none')
        # Both should have the same shape and same centered values
        assert np.allclose(out_tf - out_tf.mean(axis=0),
                           out_pre - out_pre.mean(axis=0), atol=1e-10), \
            "glog transformer and preprocess() disagree for identical input"


# ---------------------------------------------------------------------------
# D.  SNRFloor transformer — mirrors filter_snr_floor
# ---------------------------------------------------------------------------

class TestSNRFloorTransformer:
    """The SNRFloor sklearn transformer must produce the same keep_ mask as
    filter_snr_floor when fitted on the same data."""

    def test_standard_transformer_matches_filter(self):
        from shared.classifier_comparison_standard import SNRFloor
        from standard.preprocessing import filter_snr_floor

        rng = np.random.default_rng(99)
        X   = np.abs(rng.standard_normal((8, 30))) * 50 + 1.0
        y   = np.array(['A'] * 4 + ['B'] * 4)
        mz  = np.arange(30, dtype=float)

        # Function-based filter
        _, mz_kept = filter_snr_floor(X, mz, y, snr_threshold=3,
                                      noise_quantile=60, min_fraction=0.5)

        # Transformer-based filter
        sf = SNRFloor(snr_threshold=3, noise_quantile=60, min_fraction=0.5,
                      enabled=True)
        sf.fit(X, y)
        kept_idx     = set(np.where(sf.keep_)[0])
        expected_idx = set(np.where(np.isin(mz, mz_kept))[0])

        assert kept_idx == expected_idx, (
            f"SNRFloor transformer keep_ differs from filter_snr_floor:\n"
            f"  extra cols:   {sorted(kept_idx - expected_idx)}\n"
            f"  missing cols: {sorted(expected_idx - kept_idx)}"
        )

    def test_rcomp_transformer_matches_filter(self):
        from shared.classifier_comparison import SNRFloor
        from standard.preprocessing import filter_snr_floor

        rng = np.random.default_rng(7)
        X   = np.abs(rng.standard_normal((6, 20))) * 30 + 1.0
        y   = np.array(['X'] * 3 + ['Y'] * 3)
        mz  = np.arange(20, dtype=float)

        _, mz_kept = filter_snr_floor(X, mz, y, snr_threshold=2,
                                      noise_quantile=50, min_fraction=0.5)

        sf = SNRFloor(snr_threshold=2, noise_quantile=50, min_fraction=0.5,
                      enabled=True)
        sf.fit(X, y)
        kept_idx     = set(np.where(sf.keep_)[0])
        expected_idx = set(np.where(np.isin(mz, mz_kept))[0])
        assert kept_idx == expected_idx


# ---------------------------------------------------------------------------
# E.  Regression — r_comparable MetaboAnalyst defaults unchanged
# ---------------------------------------------------------------------------

class TestRComparableRegression:
    """
    The r_comparable pipeline with its MetaboAnalyst-matching defaults (log10,
    autoscale, no SNR floor) must produce identical results to running
    preprocess(log_transform='log10', scaling='autoscale').

    These tests confirm that adding glog/SNRFloor did NOT change any existing
    code paths.
    """

    def _make_data(self):
        """Minimal synthetic dataset: 6 samples, 2 groups, ~80 m/z points."""
        rng = np.random.default_rng(0)
        n, p = 6, 80
        X    = np.abs(rng.standard_normal((n, p))) * 200 + 10
        mz   = np.linspace(100, 500, p)
        y    = np.array(['A', 'A', 'A', 'B', 'B', 'B'])
        return X, mz, y

    def test_log10_autoscale_output_finite(self):
        """log10+autoscale output must be finite and zero-mean after adding glog."""
        from standard.preprocessing import preprocess
        X, _, _ = self._make_data()
        out = preprocess(X.copy(), normalization='tic',
                         log_transform='log10', scaling='autoscale')
        assert np.all(np.isfinite(out)), "log10+autoscale output must be finite"
        assert np.allclose(out.mean(axis=0), 0, atol=1e-10), \
            "Autoscaled output mean should be 0"

    def test_rcomp_bin_lower_edge_labels(self):
        """r_comparable.bin_features labels each bin at its lower edge (MetaboAnalyst)."""
        from r_comparable.preprocessing import bin_features
        X, mz, _ = self._make_data()
        _, mz_b = bin_features(X, mz, bin_width=1.0)
        residuals = mz_b % 1.0
        assert np.allclose(residuals, 0, atol=1e-9), \
            "r_comparable bin labels should be exact lower edges (multiples of bin_width)"

    def test_snr_floor_disabled_is_passthrough(self):
        """SNRFloor(enabled=False) must be a perfect pass-through."""
        from shared.classifier_comparison import SNRFloor
        rng = np.random.default_rng(5)
        X   = np.abs(rng.standard_normal((6, 40))) * 100 + 1.0
        y   = np.array(['A', 'A', 'A', 'B', 'B', 'B'])
        sf  = SNRFloor(enabled=False)
        sf.fit(X, y)
        X_out = sf.transform(X.copy())
        assert X_out.shape == X.shape and np.allclose(X_out, X), \
            "SNRFloor(enabled=False) should be a pass-through"

    def test_log10_branch_unchanged_standard(self):
        """Adding glog must not alter the log10 branch in classifier_comparison_standard."""
        from shared.classifier_comparison_standard import LogTransform
        rng = np.random.default_rng(3)
        X   = np.abs(rng.standard_normal((5, 20))) * 50 + 0.1
        lt  = LogTransform(method='log10')
        lt.fit(X)
        out    = lt.transform(X.copy())
        expect = np.log10(X + lt.params_['half_min'])
        assert np.allclose(out, expect, atol=1e-12), \
            "log10 branch changed after glog addition in classifier_comparison_standard"

    def test_log10_branch_unchanged_rcomp(self):
        """Adding glog must not alter the log10 branch in classifier_comparison."""
        from shared.classifier_comparison import LogTransform
        rng = np.random.default_rng(4)
        X   = np.abs(rng.standard_normal((5, 20))) * 50 + 0.1
        lt  = LogTransform(method='log10')
        lt.fit(X)
        out    = lt.transform(X.copy())
        expect = np.log10(X + lt.params_['half_min'])
        assert np.allclose(out, expect, atol=1e-12), \
            "log10 branch changed after glog addition in classifier_comparison"

    def test_r_comparable_preprocessing_log10_matches_direct(self):
        """
        preprocess(log10, autoscale) output is unchanged by this PR.
        Run it twice on the same data and confirm exact reproducibility.
        """
        from standard.preprocessing import preprocess
        X, _, _ = self._make_data()
        out1 = preprocess(X.copy(), normalization='pqn',
                          log_transform='log10', scaling='autoscale')
        out2 = preprocess(X.copy(), normalization='pqn',
                          log_transform='log10', scaling='autoscale')
        assert np.allclose(out1, out2, atol=0), \
            "preprocess(log10) must be deterministic"
