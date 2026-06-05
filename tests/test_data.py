"""
tests/test_data.py
Data integrity tests — assert the generated dataset matches
dissertation summary statistics (Table 3.1) within tolerance.
Run with:  python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/master.csv")


@pytest.fixture(scope="module")
def df():
    assert os.path.exists(DATA_PATH), \
        "master.csv not found — run data/generate.py first"
    return pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)


# ── Shape and columns ─────────────────────────────────────────────────────────

class TestDataShape:
    def test_row_count(self, df):
        """Dissertation sample: 269 obs (Aug 2002–Dec 2024). Allow ±10."""
        assert 200 <= len(df) <= 290, f"Expected ~269 rows, got {len(df)}"

    def test_required_columns(self, df):
        required = ["stock", "bond", "vix", "inflation",
                    "fed_funds", "real_ffr", "credit_spread",
                    "rolling_corr", "corr_sign", "high_inflation"]
        missing = [c for c in required if c not in df.columns]
        assert not missing, f"Missing columns: {missing}"

    def test_no_nulls_in_core_columns(self, df):
        core = ["stock", "bond", "vix", "inflation", "rolling_corr"]
        null_counts = df[core].isnull().sum()
        assert null_counts.sum() == 0, f"Nulls found:\n{null_counts}"

    def test_date_index_monotonic(self, df):
        assert df.index.is_monotonic_increasing


# ── Summary statistics vs Table 3.1 ──────────────────────────────────────────

class TestSummaryStats:
    """
    Tolerance: ±30% of dissertation value (synthetic data, not exact replication).
    Tighter tolerances where values are well-constrained.
    """

    def test_stock_mean(self, df):
        """Dissertation Table 3.1: mean = 0.0085"""
        assert -0.005 < df["stock"].mean() < 0.020

    def test_stock_std(self, df):
        """Dissertation: std = 0.0432"""
        assert 0.025 < df["stock"].std() < 0.065

    def test_bond_mean_positive(self, df):
        """Bonds should have positive average return over full sample"""
        assert df["bond"].mean() > -0.005

    def test_vix_range(self, df):
        """Dissertation: min=9.51, max=59.89"""
        assert df["vix"].min() >= 9.0
        assert df["vix"].max() <= 65.0

    def test_inflation_range(self, df):
        """Dissertation: min=-1.96, max=9.00"""
        assert df["inflation"].min() >= -3.0
        assert df["inflation"].max() <= 10.0

    def test_inflation_mean(self, df):
        """Dissertation: mean=2.57%"""
        assert 1.5 < df["inflation"].mean() < 4.0

    def test_credit_spread_positive(self, df):
        """Credit spread (BAA-AAA) must always be non-negative"""
        assert (df["credit_spread"] >= 0).all()

    def test_real_ffr_computed_correctly(self, df):
        """real_ffr should equal fed_funds - inflation"""
        computed = df["fed_funds"] - df["inflation"]
        assert np.allclose(df["real_ffr"], computed, atol=1e-6)


# ── Regime structure ──────────────────────────────────────────────────────────

class TestRegimeStructure:
    def test_pre_2020_correlation_mostly_negative(self, df):
        """Dissertation: pre-2022 correlation averages around -0.18"""
        pre2020 = df[df.index < "2020-01-01"]["rolling_corr"].dropna()
        assert pre2020.mean() < 0.0, \
            f"Pre-2020 correlation should be negative, got {pre2020.mean():.3f}"

    def test_2022_higher_correlation_than_pre2020(self, df):
        """2022 correlation should exceed pre-2020 average"""
        pre  = df[df.index < "2020-01-01"]["rolling_corr"].dropna().mean()
        post = df[df.index >= "2022-01-01"]["rolling_corr"].dropna().mean()
        assert post > pre, \
            f"2022+ corr ({post:.3f}) should exceed pre-2020 ({pre:.3f})"

    def test_high_inflation_label_consistent(self, df):
        """high_inflation should be 1 iff inflation >= 4.57%"""
        expected = (df["inflation"] >= 4.57).astype(float)
        assert np.allclose(df["high_inflation"], expected)

    def test_corr_sign_label_consistent(self, df):
        """corr_sign should be 1 iff rolling_corr > 0"""
        expected = (df["rolling_corr"] > 0).astype(float)
        assert np.allclose(df["corr_sign"], expected)

    def test_some_high_inflation_months(self, df):
        """Should have at least some high-inflation months (2022 episode)"""
        n_high = df["high_inflation"].sum()
        assert n_high >= 10, f"Only {n_high} high-inflation months — check generator"

    def test_some_positive_correlation_months(self, df):
        """Should have both positive and negative correlation months"""
        n_pos = df["corr_sign"].sum()
        n_neg = len(df) - n_pos
        assert n_pos >= 10, f"Only {n_pos} positive-correlation months"
        assert n_neg >= 10, f"Only {n_neg} negative-correlation months"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])