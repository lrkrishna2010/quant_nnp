"""
utils/plotting.py
Dissertation-style charts: rolling correlations, regime overlays,
hedge effectiveness, threshold curves.
Saves to results/ as .png files.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

STYLE = {
    "figure.figsize": (12, 5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
}
plt.rcParams.update(STYLE)

# Dissertation shading colours
SHADING = {
    "GFC":     ("2008-09-01", "2009-06-30", "#d3d3d3", "GFC 2008-09"),
    "COVID":   ("2020-02-01", "2020-09-30", "#ffcc99", "COVID 2020"),
    "INFLATE": ("2022-01-01", "2022-12-31", "#ffaaaa", "2022 Inflation"),
}


def _add_regime_shading(ax, df_index):
    """Only shade regions that overlap with the actual data range."""
    patches = []
    data_start, data_end = df_index.min(), df_index.max()
    for key, (start, end, color, label) in SHADING.items():
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        if e < data_start or s > data_end:
            continue   # region entirely outside data — skip
        ax.axvspan(max(s, data_start), min(e, data_end),
                   alpha=0.35, color=color, label=label)
        patches.append(mpatches.Patch(color=color, alpha=0.5, label=label))
    return patches


def plot_rolling_correlation(df: pd.DataFrame,
                              save_path: str = "results/rolling_corr.png"):
    """Replicates dissertation Figure 4.1."""
    fig, ax = plt.subplots()
    ax.plot(df.index, df["rolling_corr"], color="#2c6fad", lw=1.5,
            label="24-month rolling correlation")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    patches = _add_regime_shading(ax, df.index)
    ax.set_title("24-Month Rolling Stock–Bond Correlation")
    ax.set_ylabel("Pearson Correlation")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_hedge_effectiveness(df: pd.DataFrame,
                              he_series: pd.Series,
                              save_path: str = "results/hedge_effectiveness.png"):
    """Replicates dissertation Figure 4.3 lower panel."""
    fig, ax = plt.subplots()
    ax.plot(he_series.index, he_series * 100, color="#2ca02c", lw=1.5,
            label="Hedge Effectiveness (%)")
    ax.axhline(0, color="red", lw=0.8, ls="--", label="Zero HE")
    ax.axhline(57.6, color="gray", lw=0.8, ls=":", label="Mean low-inflation HE")
    _add_regime_shading(ax, df.index)
    ax.set_title("Hedge Effectiveness of 60/40 Portfolio")
    ax.set_ylabel("HE (%)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_inflation_threshold(threshold_df: pd.DataFrame,
                              ols_threshold: float = 4.57,
                              nn_threshold:  float | None = None,
                              save_path: str = "results/threshold_curve.png"):
    """
    Plots the marginal effect of stock returns on bond returns
    as a function of CPI — OLS line vs NN learned curve.
    Dissertation §4.1.1, Eq 4.2.
    """
    fig, ax = plt.subplots()
    ax.plot(threshold_df["cpi_grid"],
            threshold_df["marginal_effect_ols"],
            color="#2c6fad", lw=2, label="OLS (Eq 3.5)")

    if "marginal_effect_nn" in threshold_df.columns:
        ax.plot(threshold_df["cpi_grid"],
                threshold_df["marginal_effect_nn"],
                color="#d62728", lw=2, ls="--", label="ThresholdNet (NN)")

    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(ols_threshold, color="#2c6fad", lw=1, ls=":",
               label=f"OLS threshold: {ols_threshold}%")
    if nn_threshold is not None and not np.isnan(nn_threshold):
        ax.axvline(nn_threshold, color="#d62728", lw=1, ls=":",
                   label=f"NN threshold: {nn_threshold:.2f}%")

    ax.set_xlabel("CPI Inflation (%)")
    ax.set_ylabel("Marginal Effect ∂r_bond/∂r_stock")
    ax.set_title("Inflation Threshold: OLS vs Non-Parametric NN")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_portfolio_comparison(bt: pd.DataFrame,
                               save_path: str = "results/portfolio_comparison.png"):
    """Cumulative return comparison across strategies."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Data x-limits — clip shading to actual data range
    x_min, x_max = bt.index[0], bt.index[-1]

    # Cumulative returns
    for col, label, color in [
        ("r_nn",   "NN Regime-Switch", "#d62728"),
        ("r_6040", "Static 60/40",     "#2c6fad"),
        ("r_naive","Naive CPI Rule",   "#ff7f0e"),
    ]:
        if col in bt.columns:
            cum = (1 + bt[col]).cumprod()
            axes[0].plot(cum.index, cum, label=label, color=color, lw=1.5)

    axes[0].set_title("Cumulative Portfolio Returns (Test Period)")
    axes[0].set_ylabel("Growth of $1")
    axes[0].set_xlim(x_min, x_max)
    axes[0].legend(fontsize=9)
    _add_regime_shading(axes[0], bt.index)

    # Rolling volatility
    for col, label, color in [
        ("r_nn",   "NN",    "#d62728"),
        ("r_6040", "60/40", "#2c6fad"),
    ]:
        if col in bt.columns:
            vol = bt[col].rolling(12).std() * np.sqrt(12) * 100
            axes[1].plot(vol.index, vol, label=label, color=color, lw=1.5)

    axes[1].set_title("12-Month Rolling Volatility (%)")
    axes[1].set_ylabel("Ann. Volatility (%)")
    axes[1].set_xlim(x_min, x_max)
    axes[1].legend(fontsize=9)
    _add_regime_shading(axes[1], bt.index)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_lstm_predictions(results: pd.DataFrame,
                           save_path: str = "results/lstm_predictions_plot.png"):
    """LSTM predicted probability vs true correlation sign."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    axes[0].plot(results.index, results["prob_positive"],
                 color="#d62728", lw=1.5, label="P(positive correlation)")
    axes[0].axhline(0.5, color="black", lw=0.8, ls="--")
    axes[0].set_ylabel("Predicted Probability")
    axes[0].set_title("LSTM: Predicted P(Stock–Bond Correlation > 0)")
    axes[0].legend(fontsize=9)

    axes[1].bar(results.index, results["true_sign"],
                color=["#d62728" if v else "#2c6fad"
                       for v in results["true_sign"]],
                width=20, label="True sign (1=positive, 0=negative)")
    axes[1].set_ylabel("True Sign")
    axes[1].set_title("Realised Correlation Sign")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")