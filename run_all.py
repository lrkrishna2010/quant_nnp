"""
run_all.py  –  Train all three models, generate all charts
"""

import sys, os
import random
import numpy as np
import pandas as pd
import torch

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
torch.use_deterministic_algorithms(True, warn_only=True)

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data"))
sys.path.insert(0, os.path.join(ROOT, "models"))

# ── Create output directories ──────────────────────────────────────────────────
for d in ["results", "models"]:
    os.makedirs(os.path.join(ROOT, d), exist_ok=True)

RESULTS = os.path.join(ROOT, "results")

print("=" * 65)
print("QUANT-NN PROJECT: Extending the Dissertation with PyTorch")
print("=" * 65)

# ── Data ───────────────────────────────────────────────────────────────────────
master_csv = os.path.join(ROOT, "data", "master.csv")
if os.path.exists(master_csv):
    df = pd.read_csv(master_csv, index_col=0, parse_dates=True)
    print(f"\n[1/4] Loaded real master.csv  ({len(df)} rows)")
else:
    print("\n[1/4] master.csv not found — generating synthetic dataset...")
    from data.generate import build_dataset
    df = build_dataset(save=True)

print(f"      {len(df)} monthly observations | "
      f"{df['high_inflation'].sum():.0f} high-inflation months")

# ── Model 1 ───────────────────────────────────────────────────────────────────
print("\n[2/4] Training Model 1: LSTM Correlation-Sign Classifier...")
from models.model1_lstm_classifier import run as run1
model1, hist1, results1 = run1()

# ── Model 2 ───────────────────────────────────────────────────────────────────
print("\n[3/4] Training Model 2: Regime Detector → Portfolio...")
from models.model2_regime_detector import run as run2
model2, bt2, hist2 = run2()

# ── Model 3 ───────────────────────────────────────────────────────────────────
print("\n[4/4] Training Model 3: Non-Parametric Threshold Learner...")
from models.model3_nonparametric_threshold import run as run3
model3, thresh_df, nn_thresh, hist3 = run3()

# ── Charts ─────────────────────────────────────────────────────────────────────
print("\n[5/5] Generating charts...")
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils.plotting import (
    plot_rolling_correlation,
    plot_hedge_effectiveness,
    plot_inflation_threshold,
    plot_portfolio_comparison,
    plot_lstm_predictions,
)
from utils.portfolio import rolling_portfolio_stats
from utils.metrics import hedge_effectiveness, portfolio_variance

# Chart 1 — Rolling correlation (dissertation Figure 4.1)
plot_rolling_correlation(
    df,
    save_path=os.path.join(RESULTS, "rolling_corr.png"),
)

# Chart 2 — Inflation vs rolling correlation (core dissertation finding)
fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()
ax1.plot(df.index, df["rolling_corr"], color="#2c6fad", lw=1.5,
         label="Rolling correlation (left)")
ax1.axhline(0, color="#2c6fad", lw=0.6, ls="--", alpha=0.5)
ax2.plot(df.index, df["inflation"], color="#d62728", lw=1.5,
         label="CPI inflation % (right)")
ax2.axhline(4.57, color="#d62728", lw=0.8, ls=":", alpha=0.8,
            label="OLS threshold 4.57%")
ax2.axhline(nn_thresh if nn_thresh == nn_thresh else 4.57,
            color="#ff7f0e", lw=0.8, ls=":",
            label=f"NN threshold {nn_thresh:.2f}%"
                  if nn_thresh == nn_thresh else "")
for start, end, color, label in [
    ("2008-09-01", "2009-06-30", "#d3d3d3", "GFC"),
    ("2020-02-01", "2020-09-30", "#ffcc99", "COVID"),
    ("2022-01-01", "2022-12-31", "#ffaaaa", "2022 Inflation"),
]:
    ax1.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                alpha=0.25, color=color, label=label)
ax1.set_ylabel("Correlation", color="#2c6fad")
ax2.set_ylabel("CPI Inflation (%)", color="#d62728")
ax1.set_title("Rolling Stock–Bond Correlation vs CPI Inflation")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "corr_vs_inflation.png"), dpi=150)
plt.close()
print(f"Saved: results/corr_vs_inflation.png")

# Chart 3 — Hedge effectiveness
stats = rolling_portfolio_stats(df)
he_series = stats["hedge_effectiveness"].dropna()
plot_hedge_effectiveness(
    df,
    he_series,
    save_path=os.path.join(RESULTS, "hedge_effectiveness.png"),
)

# Chart 4 — Inflation threshold: OLS vs NN (Model 3)
plot_inflation_threshold(
    thresh_df,
    ols_threshold=4.57,
    nn_threshold=nn_thresh if nn_thresh == nn_thresh else None,
    save_path=os.path.join(RESULTS, "threshold_curve.png"),
)

# Chart 5 — Portfolio comparison (Model 2)
plot_portfolio_comparison(
    bt2,
    save_path=os.path.join(RESULTS, "portfolio_comparison.png"),
)

# Chart 6 — LSTM predictions (Model 1)
plot_lstm_predictions(
    results1,
    save_path=os.path.join(RESULTS, "lstm_predictions_plot.png"),
)

# Chart 7 — Training loss curves for all three models
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Model 1: train + val
axes[0].plot(hist1["train"], label="Train", color="#2c6fad", lw=1.5)
axes[0].plot(hist1["val"],   label="Val",   color="#d62728", lw=1.5)
axes[0].set_title("Model 1: LSTM (BCEWithLogitsLoss)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend(fontsize=9)

# Model 2: val only
axes[1].plot(hist2["val"], color="#d62728", lw=1.5, label="Val")
axes[1].set_title("Model 2: Regime Detector (BCE)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend(fontsize=9)

# Model 3: val only
axes[2].plot(hist3["val"], color="#d62728", lw=1.5, label="Val")
axes[2].set_title("Model 3: Threshold Net (MSE)")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Loss")
axes[2].legend(fontsize=9)

plt.suptitle("Training Loss Curves", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "training_curves.png"), dpi=150)
plt.close()
print(f"Saved: results/training_curves.png")

# ── Summary ────────────────────────────────────────────────────────────────────
thresh_str = f"{nn_thresh:.2f}%" if nn_thresh == nn_thresh else "not identified"
print("\n" + "=" * 65)
print("PROJECT SUMMARY")
print("=" * 65)
print(f"""
  Model 1 – LSTM Classifier
    Predicts next-month sign of rolling stock-bond correlation.

  Model 2 – Regime Detector → Portfolio
    NN soft-switches bond weight; lower vol than 60/40 in
    high-inflation periods.

  Model 3 – Non-Parametric Threshold
    Dissertation OLS threshold: 4.57% CPI
    NN learned threshold:       {thresh_str}
""")

print("Results saved to results/:")
for f in sorted(os.listdir(RESULTS)):
    print(f"  ✓  {f}")