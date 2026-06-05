"""
tests/test_metrics.py
Unit tests for dissertation formulas implemented in utils/metrics.py.
Run with:  python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from utils.metrics import (
    portfolio_variance,
    hedge_effectiveness,
    min_variance_bond_weight,
    breakeven_correlation,
    inflation_threshold,
    marginal_effect,
    sharpe_ratio,
    annualised_vol,
)


# ── Dissertation Table 4.1 / Eq 4.1 ──────────────────────────────────────────

class TestInflationThreshold:
    """Dissertation Eq 4.1: Inflation* = −β̂₁/β̂₂ = −(−0.378)/0.0828 = 4.57%"""

    def test_ols_threshold(self):
        threshold = inflation_threshold(beta1=-0.3780, beta2=0.0828)
        assert abs(threshold - 4.57) < 0.01, \
            f"Expected ~4.57%, got {threshold:.4f}%"

    def test_positive_beta2_gives_positive_threshold(self):
        t = inflation_threshold(beta1=-0.5, beta2=0.1)
        assert t > 0

    def test_zero_beta2_returns_nan(self):
        t = inflation_threshold(beta1=-0.378, beta2=0.0)
        assert np.isnan(t)

    def test_uk_threshold(self):
        """UK replication: β̂₂=0.0636 → threshold 2.61% (dissertation §4.3.4)"""
        t = inflation_threshold(beta1=-0.1657, beta2=0.0636)
        assert abs(t - 2.61) < 0.05, f"Expected ~2.61%, got {t:.4f}%"


class TestMarginalEffect:
    """Dissertation Eq 4.2: ∂r_bond/∂r_stock = β̂₁ + β̂₂ · Inflation"""

    def test_at_mean_inflation(self):
        """At 2.57% CPI (sample mean), ME ≈ −0.166 (§4.1.1)"""
        me = marginal_effect(-0.3780, 0.0828, inflation=2.57)
        assert abs(me - (-0.166)) < 0.01, f"Expected ~-0.166, got {me:.4f}"

    def test_at_peak_inflation(self):
        """At 9% CPI (2022 peak), ME ≈ +0.367 (§4.1.1)"""
        me = marginal_effect(-0.3780, 0.0828, inflation=9.0)
        assert abs(me - 0.367) < 0.01, f"Expected ~+0.367, got {me:.4f}"

    def test_sign_change_at_threshold(self):
        """ME should be ~0 at the threshold (4.57%)"""
        me = marginal_effect(-0.3780, 0.0828, inflation=4.57)
        assert abs(me) < 0.01


# ── Dissertation Eq 3.7 / §3.2.5 ─────────────────────────────────────────────

class TestPortfolioVariance:
    def test_zero_correlation(self):
        """With rho=0: σ²p = ws²σ²s + wb²σ²b"""
        var = portfolio_variance(0.6, 0.4, 0.14, 0.05, rho=0)
        expected = 0.6**2 * 0.14**2 + 0.4**2 * 0.05**2
        assert abs(var - expected) < 1e-10

    def test_positive_correlation_increases_variance(self):
        var_neg = portfolio_variance(0.6, 0.4, 0.14, 0.05, rho=-0.3)
        var_pos = portfolio_variance(0.6, 0.4, 0.14, 0.05, rho= 0.3)
        assert var_pos > var_neg

    def test_perfect_correlation_equals_weighted_vol(self):
        """rho=1: σp = ws·σs + wb·σb"""
        var = portfolio_variance(0.6, 0.4, 0.14, 0.05, rho=1.0)
        expected = (0.6*0.14 + 0.4*0.05)**2
        assert abs(var - expected) < 1e-10


# ── Dissertation Eq 4.4 ───────────────────────────────────────────────────────

class TestMinVarianceBondWeight:
    def test_2022_vols_low_correlation(self):
        """At 2022 vols, low/negative rho: w*_bond should be moderate-high"""
        w = min_variance_bond_weight(sigma_s=0.168, sigma_b=0.063, rho=-0.18)
        assert 0.4 < w < 0.9, f"Expected 0.4–0.9, got {w:.4f}"

    def test_2022_vols_positive_correlation(self):
        """
        At 2022 vols (σs=16.8%, σb=6.3%), the min-var weight is high (>0.5)
        because bond vol is much lower than stock vol — even at positive
        correlation you want more bonds, not fewer. This is precisely the
        dissertation's key insight (§4.2.5, §5.4).
        """
        w = min_variance_bond_weight(sigma_s=0.168, sigma_b=0.063, rho=0.15)
        assert w > 0.5, f"Expected w* > 0.5 at 2022 vols, got {w:.4f}"
        assert w <= 1.0

    def test_weight_between_0_and_1(self):
        for rho in [-0.8, -0.3, 0.0, 0.3, 0.8]:
            w = min_variance_bond_weight(0.15, 0.05, rho)
            assert 0.0 <= w <= 1.0, f"Weight out of bounds at rho={rho}"


# ── Dissertation Eq 4.5 ───────────────────────────────────────────────────────

class TestBreakevenCorrelation:
    def test_2022_rho_star_exceeds_1(self):
        """
        Key result: at 2022 vols, ρ* = 4.39 > 1.
        No feasible correlation at which cutting bonds helps (§4.2.5, §5.4).
        """
        rho_star = breakeven_correlation(
            ws_a=0.60, wb_a=0.40,   # 60/40
            ws_b=0.80, wb_b=0.20,   # 80/20 (reduced bonds)
            sigma_s=0.168,
            sigma_b=0.063,
        )
        assert rho_star > 1.0, \
            f"Expected ρ* > 1 at 2022 vols, got {rho_star:.4f}"
        assert abs(rho_star - 4.39) < 0.05, \
            f"Expected ~4.39 (dissertation Eq 4.5), got {rho_star:.4f}"

    def test_equal_portfolios_return_zero(self):
        """If A == B, variance is always equal → rho* undefined (near 0 denominator)"""
        rho_star = breakeven_correlation(0.6, 0.4, 0.6, 0.4, 0.15, 0.05)
        assert np.isnan(rho_star) or abs(rho_star) > 1e6


# ── Dissertation Eq 3.8 ───────────────────────────────────────────────────────

class TestHedgeEffectiveness:
    def test_he_low_inflation(self):
        """HE ~ 57.6% in low-inflation periods (Table 4.4). Allow ±10%."""
        var_port   = portfolio_variance(0.6, 0.4, 0.142, 0.051, rho=-0.18)
        var_equity = 0.142**2
        he = hedge_effectiveness(var_port, var_equity)
        assert 0.45 < he < 0.75, f"Expected 45–75% HE in low-inflation, got {he:.4f}"

    def test_he_positive_means_variance_reduction(self):
        var_port   = portfolio_variance(0.6, 0.4, 0.15, 0.05, rho=-0.3)
        var_equity = 0.15**2
        he = hedge_effectiveness(var_port, var_equity)
        assert he > 0

    def test_he_can_go_negative(self):
        """Positive correlation + high bond vol can make HE negative"""
        var_port   = portfolio_variance(0.6, 0.4, 0.10, 0.15, rho=0.95)
        var_equity = 0.10**2
        he = hedge_effectiveness(var_port, var_equity)
        assert he < 0


# ── Basic return metrics ──────────────────────────────────────────────────────

class TestReturnMetrics:
    def setup_method(self):
        np.random.seed(0)
        self.ret = np.random.normal(0.008, 0.04, 120)   # 10 years monthly

    def test_sharpe_positive_for_positive_mean(self):
        assert sharpe_ratio(self.ret) > 0

    def test_annualised_vol_reasonable(self):
        vol = annualised_vol(self.ret)
        assert 0.05 < vol < 0.30   # 5–30% is sensible for equities

    def test_zero_returns_zero_sharpe(self):
        assert sharpe_ratio(np.zeros(100)) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])