"""
utils/portfolio.py
Portfolio construction, rebalancing, and backtesting utilities.
"""

import numpy as np
import pandas as pd
from utils.metrics import (portfolio_variance, hedge_effectiveness,
                            min_variance_bond_weight, sharpe_ratio,
                            annualised_vol, annualised_return, portfolio_summary)


# ── Static portfolios ─────────────────────────────────────────────────────────

def static_portfolio(df: pd.DataFrame,
                     w_stock: float = 0.60,
                     w_bond: float  = 0.40,
                     label: str     = "60/40") -> pd.Series:
    """Fixed-weight portfolio return series."""
    ret = w_stock * df["stock"] + w_bond * df["bond"]
    ret.name = label
    return ret


def naive_cpi_rule(df: pd.DataFrame,
                   cpi_threshold: float = 3.0,
                   w_bond_normal: float = 0.40,
                   w_bond_high:   float = 0.20) -> pd.Series:
    """
    Cut bonds from 40% to 20% whenever CPI >= threshold.
    Dissertation §4.2.5 Table 4.5: raises volatility in every regime.
    """
    wb  = np.where(df["inflation"] >= cpi_threshold, w_bond_high, w_bond_normal)
    ws  = 1 - wb
    ret = ws * df["stock"].values + wb * df["bond"].values
    return pd.Series(ret, index=df.index, name=f"Naive CPI>{cpi_threshold}%")


# ── Dynamic / NN-driven portfolio ─────────────────────────────────────────────

def nn_regime_portfolio(df: pd.DataFrame,
                        regime_probs: np.ndarray,
                        w_bond_normal: float = 0.40,
                        w_bond_high:   float = 0.511) -> pd.Series:
    """
    Soft regime-switching: bond weight linearly interpolates between
    w_bond_normal (low inflation) and w_bond_high (high inflation /
    min-variance at 2022 vol levels) based on NN regime probability.

    w_bond_high = 0.511 from dissertation Eq 4.4 at 2022 vol levels.
    """
    wb  = (1 - regime_probs) * w_bond_normal + regime_probs * w_bond_high
    ws  = 1 - wb
    ret = ws * df["stock"].values + wb * df["bond"].values
    return pd.Series(ret, index=df.index, name="NN Regime-Switch")


# ── Rolling variance and HE ───────────────────────────────────────────────────

def rolling_portfolio_stats(df: pd.DataFrame,
                             w_stock: float = 0.60,
                             w_bond:  float = 0.40,
                             window:  int   = 24) -> pd.DataFrame:
    """
    Rolling annualised portfolio volatility and hedge effectiveness.
    Dissertation §3.2.5, §4.2.1.
    """
    port_ret  = w_stock * df["stock"] + w_bond * df["bond"]
    port_vol  = port_ret.rolling(window).std() * np.sqrt(12)
    equity_vol= df["stock"].rolling(window).std() * np.sqrt(12)

    he = 1 - (port_vol ** 2) / (equity_vol ** 2)

    return pd.DataFrame({
        "portfolio_vol":  port_vol,
        "equity_vol":     equity_vol,
        "hedge_effectiveness": he,
    })


# ── Full backtest comparison ──────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame,
                 regime_probs: np.ndarray | None = None,
                 cpi_threshold: float = 3.0) -> dict:
    """
    Compare static 60/40, naive CPI rule, and (optionally) NN portfolio.
    Returns dict of {label: Series} for use with portfolio_summary().
    """
    portfolios = {
        "Static 60/40":  static_portfolio(df),
        f"Naive CPI>{cpi_threshold}%": naive_cpi_rule(df, cpi_threshold),
    }
    if regime_probs is not None:
        portfolios["NN Regime-Switch"] = nn_regime_portfolio(df, regime_probs)

    return portfolios


def regime_breakdown(df: pd.DataFrame,
                     portfolios: dict,
                     inflation_col: str = "inflation") -> pd.DataFrame:
    """
    Break down portfolio stats by inflation regime:
    low (<3%), high (>=3%), and 2022 shock (>=6%).
    Mirrors dissertation Table 4.3 / 4.5 structure.
    """
    regimes = {
        "Low inflation (<3%)":   df[inflation_col] < 3.0,
        "High inflation (>=3%)": df[inflation_col] >= 3.0,
        "2022 shock (>=6%)":     df[inflation_col] >= 6.0,
    }

    rows = []
    for regime_label, mask in regimes.items():
        if mask.sum() == 0:
            continue
        for strat_label, ret_series in portfolios.items():
            r = ret_series[mask].values
            rows.append({
                "Regime":   regime_label,
                "Strategy": strat_label,
                "N months": int(mask.sum()),
                "Ann. Return (%)":     round(annualised_return(r) * 100, 2),
                "Ann. Volatility (%)": round(annualised_vol(r) * 100, 2),
                "Sharpe Ratio":        round(sharpe_ratio(r), 3),
            })

    return pd.DataFrame(rows)