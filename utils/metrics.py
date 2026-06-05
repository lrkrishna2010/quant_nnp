"""
utils/metrics.py
Reusable portfolio and correlation metrics.
All formulas reference the dissertation section where they appear.
"""

import numpy as np
import pandas as pd


# ── Portfolio metrics ─────────────────────────────────────────────────────────

def sharpe_ratio(returns: np.ndarray, ann: int = 12) -> float:
    """Annualised Sharpe ratio (assumes risk-free = 0)."""
    mu    = returns.mean() * ann
    sigma = returns.std() * np.sqrt(ann)
    return float(mu / sigma) if sigma > 0 else 0.0


def annualised_vol(returns: np.ndarray, ann: int = 12) -> float:
    """Annualised volatility."""
    return float(returns.std() * np.sqrt(ann))


def annualised_return(returns: np.ndarray, ann: int = 12) -> float:
    return float(returns.mean() * ann)


def portfolio_variance(ws: float, wb: float,
                       sigma_s: float, sigma_b: float,
                       rho: float) -> float:
    """
    Two-asset portfolio variance.
    Dissertation Eq 3.7: σ²_port = ws²σ²s + wb²σ²b + 2·ws·wb·ρ·σs·σb
    """
    return ws**2 * sigma_s**2 + wb**2 * sigma_b**2 + 2*ws*wb*rho*sigma_s*sigma_b


def hedge_effectiveness(var_portfolio: float, var_equity: float) -> float:
    """
    Percentage variance reduction from holding bonds alongside equities.
    Dissertation Eq 3.8 / §4.2.4:  HE = 1 - Var(60/40) / Var(100% equity)
    Returns value in [−∞, 1]; negative means bonds increase portfolio risk.
    """
    return 1.0 - var_portfolio / var_equity


def min_variance_bond_weight(sigma_s: float, sigma_b: float,
                              rho: float) -> float:
    """
    Minimum-variance bond weight for a two-asset stock/bond portfolio.
    Dissertation Eq 4.4:
      w*_bond = (σ²s − ρ·σs·σb) / (σ²s + σ²b − 2ρ·σs·σb)
    """
    num = sigma_s**2 - rho * sigma_s * sigma_b
    den = sigma_s**2 + sigma_b**2 - 2*rho*sigma_s*sigma_b
    if abs(den) < 1e-12:
        return 0.5
    return float(np.clip(num / den, 0.0, 1.0))


def breakeven_correlation(ws_a: float, wb_a: float,
                           ws_b: float, wb_b: float,
                           sigma_s: float, sigma_b: float) -> float:
    """
    Correlation ρ* at which two portfolios A and B have equal variance.
    Dissertation Eq 4.5 — used to show bond reduction was variance-increasing
    at 2022 volatility levels (result: ρ* = 4.39 > 1, so no feasible value).

    Portfolio A: (ws_a, wb_a) e.g. 60/40
    Portfolio B: (ws_b, wb_b) e.g. 80/20
    """
    num = (ws_a**2 - ws_b**2) * sigma_s**2 + (wb_a**2 - wb_b**2) * sigma_b**2
    den = 2 * (ws_b*wb_b - ws_a*wb_a) * sigma_s * sigma_b
    if abs(den) < 1e-12:
        return np.nan
    return float(num / den)


# ── Correlation metrics ───────────────────────────────────────────────────────

def rolling_correlation(s: pd.Series, b: pd.Series, window: int = 24) -> pd.Series:
    """24-month rolling Pearson correlation. Dissertation §3.2.1."""
    return s.rolling(window).corr(b)


def inflation_threshold(beta1: float, beta2: float) -> float:
    """
    CPI level at which stock–bond beta turns positive.
    Dissertation Eq 4.1:  Inflation* = −β̂₁ / β̂₂
    Default coefficients from Table 4.1: β̂₁=−0.378, β̂₂=0.0828 → 4.57%
    """
    if abs(beta2) < 1e-12:
        return np.nan
    return float(-beta1 / beta2)


def marginal_effect(beta1: float, beta2: float, inflation: float) -> float:
    """
    Marginal effect of stock returns on bond returns at a given inflation level.
    Dissertation Eq 4.2:  ∂r_bond/∂r_stock = β̂₁ + β̂₂ · Inflation
    """
    return beta1 + beta2 * inflation


# ── Summary table ─────────────────────────────────────────────────────────────

def portfolio_summary(returns_dict: dict, ann: int = 12) -> pd.DataFrame:
    """
    Given a dict of {label: return_series}, return a comparison DataFrame
    with annualised return, volatility, and Sharpe ratio.
    """
    rows = []
    for label, ret in returns_dict.items():
        r = np.asarray(ret)
        rows.append({
            "Strategy":          label,
            "Ann. Return (%)":   round(annualised_return(r, ann) * 100, 2),
            "Ann. Volatility (%)": round(annualised_vol(r, ann) * 100, 2),
            "Sharpe Ratio":      round(sharpe_ratio(r, ann), 3),
        })
    return pd.DataFrame(rows).set_index("Strategy")