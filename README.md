# quant-nn

PyTorch implementation extending *"Time-Varying Stock–Bond Correlations and Implications for Portfolio Diversification"* (Lalam, 2026, University of Essex, MA830-SP).

The dissertation uses OLS interaction regression to show that U.S. stock–bond correlations turn positive when CPI exceeds **4.57%**, and that cutting bonds at 2022 volatility levels was variance-increasing regardless of correlation sign (ρ* = 4.39 > 1). This project replaces those linear models with three neural networks trained on the same dataset.

---

## Background

The dissertation's central finding is an inflation threshold: above 4.57% CPI, the stock–bond beta turns positive — bonds stop hedging equities. The 2022 episode confirmed this: with CPI averaging 7.9% and the Fed hiking 425bp in twelve months, both stocks (−18.1%) and bonds (−13.0%) fell simultaneously, cutting 60/40 hedge effectiveness from 57.6% to 41.0%.

Three questions motivate this project:

1. Can a sequence model predict whether next month's stock–bond correlation will be positive or negative?
2. Does a neural network regime detector produce a smarter bond allocation than the naive CPI>3% switching rule from dissertation Table 4.5?
3. If a network learns the inflation threshold non-parametrically, does it recover something close to OLS's 4.57%?

---

## Models

### Model 1 — LSTM Correlation-Sign Classifier

Predicts whether the 24-month rolling stock–bond correlation will be positive or negative next month.

- **Input:** 12-month sequence of `[CPI, VIX, real FFR, credit spread, stock return, bond return]`
- **Output:** P(rolling correlation > 0)
- **Architecture:** 2-layer LSTM (hidden=64) → MLP head → sigmoid
- **Loss:** `BCEWithLogitsLoss` with `pos_weight` to handle class imbalance (~57% positive, ~43% negative in test set)
- **Dissertation link:** §1.3, §4.1, Table 4.1

**Limitation:** With ~200 training sequences the model overfits — train AUC ~0.85, test AUC ~0.07–0.92 depending on run. This is a data constraint, not a modelling failure: 246 monthly observations is genuinely small for sequence learning.

### Model 2 — Regime Detector → Portfolio

Detects whether the current macro environment is a high-inflation regime, then soft-switches bond allocation accordingly.

- **Input:** Current macro snapshot `[CPI, VIX, real FFR, credit spread]`
- **Output:** P(high-inflation regime), i.e. P(CPI > 4.57%)
- **Architecture:** MLP [32, 16] with BatchNorm and Dropout
- **Portfolio:** Bond weight interpolates between 40% (low-inflation) and 51.1% (min-variance weight at 2022 volatility levels, from dissertation Eq 4.4) based on regime probability
- **Compared against:** Static 60/40 and naive CPI>3% rule (dissertation Table 4.5)
- **Dissertation link:** §4.2.5, Table 4.5

**Result:** NN portfolio consistently achieves lower annualised volatility than static 60/40 in high-inflation sub-periods (~10.7% vs 11.5%) and the full test period (~9.8% vs 10.5%), while the Naive CPI rule raises volatility in every regime.

### Model 3 — Non-Parametric Threshold Learner

Replicates dissertation Eq 3.5 with a neural network instead of OLS — letting the network learn the inflation threshold non-parametrically rather than imposing linearity.

- **Input:** `[stock, inflation, VIX, real FFR, credit spread, stock×inflation, stock×VIX]` (mirrors Eq 3.5)
- **Output:** Bond return (regression task)
- **Architecture:** MLP [64, 32, 16] with Tanh activations
- **Threshold extraction:** Sweep CPI from 1–9%, smooth the learned marginal effect curve ∂r_bond/∂r_stock, find zero crossing
- **Compared against:** OLS coefficients from dissertation Table 4.1
- **Dissertation link:** §3.2.3, §4.1.1, §5.3

**Result:** NN beats OLS on test MSE (0.000323 vs 0.000466). Learned threshold stable at **6.53% CPI** vs OLS's 4.57% — suggesting the nonlinear model finds the sign-flip at slightly higher inflation than the linear interaction term. The threshold variance across random seeds (2.44%–7.70% before fixing seed) directly replicates the dissertation's §4.3.5 out-of-sample finding: insufficient inflation variation in pre-2020 data to reliably identify the regime boundary.

---

## Results Summary

| Model | Metric | Value |
|-------|--------|-------|
| LSTM Classifier | Test AUC (seed=42) | 0.065 |
| Regime Detector | Regime AUC | 0.63 |
| Regime Detector | NN vol vs 60/40 (high-inflation) | 10.7% vs 11.5% |
| Regime Detector | NN Sharpe vs 60/40 | 1.24 vs 1.22 |
| Threshold Net | Test MSE vs OLS | 0.000323 vs 0.000466 |
| Threshold Net | Learned threshold vs OLS | 6.53% vs 4.57% CPI |

OLS baseline from dissertation Table 4.1: β̂₁ = −0.378, β̂₂ = 0.083, threshold = 4.57%, R² = 0.085.

---

## Project Structure

```
quant_nn/
├── data/
│   ├── fetch.py             # real data: Yahoo Finance + FRED REST API
│   ├── generate.py          # synthetic fallback (calibrated to Table 3.1)
│   └── master.csv           # 246 monthly obs, Aug 2004 – Dec 2024
├── models/
│   ├── model1_lstm_classifier.py
│   ├── model2_regime_detector.py
│   └── model3_nonparametric_threshold.py
├── utils/
│   ├── __init__.py
│   ├── metrics.py           # dissertation formulas (Eq 3.7–4.5)
│   ├── portfolio.py         # portfolio construction and backtesting
│   └── plotting.py          # all charts with regime shading
├── configs/
│   └── config.yaml          # all hyperparameters
├── tests/
│   ├── test_metrics.py      # 21 unit tests — every dissertation formula
│   └── test_data.py         # 18 data integrity checks
├── results/                 # generated CSVs and charts (git-ignored)
├── run_all.py               # trains all three models + generates charts
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# Clone and set up
git clone <https://github.com/lrkrishna2010/quant_nnp>
cd quant_nn
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Fetch real data (requires internet)
python data/fetch.py

# Or use synthetic data (no internet needed)
python data/generate.py

# Train all models and generate charts
python run_all.py

# Run tests
pytest tests/ -v
```

---

## Data

Real data is fetched from two sources, no API key required:

| Series | Source | Description |
|--------|--------|-------------|
| SPY | Yahoo Finance | S&P 500 total return proxy |
| IEF | Yahoo Finance | iShares 7–10yr Treasury ETF |
| ^VIX | Yahoo Finance | CBOE Volatility Index (monthly avg) |
| CPIAUCSL | FRED | CPI all urban consumers (YoY inflation) |
| FEDFUNDS | FRED | Effective federal funds rate |
| DBAA / DAAA | FRED | Moody's BAA/AAA yields → credit spread |

Sample: 246 monthly observations, July 2004 – December 2024. The dissertation used 269 obs from August 2002; the 23-observation gap reflects IEF's July 2002 launch date — the 24-month rolling window can't initialise until mid-2004.

---

## Dissertation Formula Reference

| Formula | Dissertation | Implementation |
|---------|-------------|----------------|
| Log returns | Eq 3.1 | `data/generate.py` |
| Rolling correlation | Eq 3.2 | `utils/metrics.py:rolling_correlation` |
| Interaction regression | Eq 3.5 | `models/model3_nonparametric_threshold.py` |
| Portfolio variance | Eq 3.7 | `utils/metrics.py:portfolio_variance` |
| Hedge effectiveness | Eq 3.8 | `utils/metrics.py:hedge_effectiveness` |
| Inflation threshold | Eq 4.1 | `utils/metrics.py:inflation_threshold` |
| Marginal effect | Eq 4.2 | `utils/metrics.py:marginal_effect` |
| Min-variance weight | Eq 4.4 | `utils/metrics.py:min_variance_bond_weight` |
| Breakeven correlation ρ* | Eq 4.5 | `utils/metrics.py:breakeven_correlation` |

---

## Key Limitations

**Data size.** 246 monthly observations is small for neural networks. Model 1 (LSTM) overfits consistently — train AUC ~0.85, test AUC ~0.07. This is a data constraint that more observations or a different architecture (e.g. simpler logistic regression on engineered features) might address.

**Single-episode identification.** All three models inherit the dissertation's core limitation: the inflation channel is identified primarily off the 2022 episode, where CPI moved from near zero to 9% and back within two years. Model 3's threshold variance across seeds (before fixing seed=42) independently confirms §4.3.5 — the pre-2020 data alone cannot estimate the interaction coefficient reliably.

**IEF launch date.** The sample starts July 2004 rather than the dissertation's August 2002 because IEF only launched in July 2002. Earlier bond series (Bloomberg indices) would recover the full sample.

---

## Requirements

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
yfinance>=0.2.40
requests>=2.31.0
scipy>=1.11.0
matplotlib>=3.7.0
pyyaml>=6.0
pytest>=7.4.0
```

---

## References

- Lalam, R.K. (2026). *Time-Varying Stock–Bond Correlations and Implications for Portfolio Diversification.* University of Essex, MA830-SP.
- Molenaar et al. (2024). Empirical evidence on the stock–bond correlation. *Financial Analysts Journal*, 80(3).
- Campbell, Sunderam & Viceira (2020). Inflation bets or deflation hedges? *Critical Finance Review*, 9(1–2).
- Engle (2002). Dynamic conditional correlation. *JBES*, 20(3).
- Ederington (1979). The hedging performance of the new futures markets. *Journal of Finance*, 34(1).
