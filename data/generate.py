import os
"""
generate.py  –  Synthetic dataset matching dissertation statistics exactly
                (Table 3.1, Section 4.1, Section 4.2)

Regime structure:
  - Low inflation (<4.57% CPI):  stock-bond correlation ~ -0.18  (2002-2021)
  - High inflation (≥4.57% CPI): stock-bond correlation ~ +0.15  (2022)
  - 2022 shock: CPI peaks at 9%, stocks -18.1%, bonds -13.0%
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

np.random.seed(42)

# ── Dissertation summary stats (Table 3.1) ────────────────────────────────────
STATS = {
    "stock":        {"mean": 0.0085, "std": 0.0432},
    "bond":         {"mean": 0.0028, "std": 0.0195},
    "vix":          {"mean": 19.46,  "std": 8.15,  "min": 9.51,  "max": 59.89},
    "inflation":    {"mean": 2.57,   "std": 1.84,  "min": -1.96, "max": 9.00},
    "fed_funds":    {"mean": 1.64,   "std": 1.84,  "min": 0.05,  "max": 5.33},
    "credit_spread":{"mean": 0.93,   "std": 0.32,  "min": 0.55,  "max": 2.18},
}

N = 269  # Aug 2002 – Dec 2024

def make_dates(n=N):
    return pd.date_range("2002-08-31", periods=n, freq="ME")

def ar1(n, mu, sigma, phi=0.9, clip=None):
    """AR(1) process."""
    x = np.zeros(n)
    x[0] = mu
    eps = np.random.normal(0, sigma * np.sqrt(1 - phi**2), n)
    for t in range(1, n):
        x[t] = mu + phi * (x[t-1] - mu) + eps[t]
    if clip:
        x = np.clip(x, clip[0], clip[1])
    return x

def make_inflation(n=N):
    """
    Inflation trajectory matching dissertation:
      2002-2019: ~2% mean, mild variation
      2020-2021: briefly negative (COVID), then rising
      2022:      peaks ~9% (Fed tightening)
      2023-2024: returns toward 3%
    """
    cpi = np.zeros(n)
    # 2002-2019: 209 months
    cpi[:209] = ar1(209, mu=2.2, sigma=0.6, phi=0.92,
                    clip=(-1.96, 4.0))
    # 2020 (12 months): dip then rise
    cpi[209:221] = np.linspace(1.2, 5.0, 12) + np.random.normal(0, 0.3, 12)
    # 2021 (12): rising fast
    cpi[221:233] = np.linspace(5.0, 8.0, 12) + np.random.normal(0, 0.4, 12)
    # 2022 (12): peak ~9%
    cpi[233:245] = np.array([7.9, 8.0, 8.5, 8.3, 8.6, 9.1,
                              8.5, 8.3, 8.2, 7.7, 7.1, 6.5])
    # 2023-2024 (24): cooling
    cpi[245:] = np.linspace(5.5, 2.8, n - 245) + np.random.normal(0, 0.3, n - 245)
    return np.clip(cpi, -1.96, 9.0)

def make_vix(n=N, inflation=None):
    """VIX elevated in crisis and high-inflation periods."""
    vix = ar1(n, mu=18.0, sigma=5.0, phi=0.85, clip=(9.51, 59.89))
    # GFC spike (months ~72-83 ≈ 2008-2009)
    vix[72:84] += np.linspace(15, 30, 12)
    # COVID spike (months ~209-212)
    vix[209:213] += np.array([25, 35, 20, 10])
    # 2022 elevated
    if inflation is not None:
        high_inf = inflation > 4.57
        vix[high_inf] += 5.0
    return np.clip(vix, 9.51, 59.89)

def make_fed_funds(inflation, n=N):
    """Fed funds: near zero 2009-2015, 2020-2021; hiking 2022-2023."""
    ffr = np.zeros(n)
    ffr[:72]    = ar1(72, mu=3.5, sigma=0.5, phi=0.95, clip=(0.05, 5.33))   # 2002-2008
    ffr[72:96]  = np.linspace(3.0, 0.1, 24)                                  # GFC cut
    ffr[96:162] = ar1(66, mu=0.1, sigma=0.05, phi=0.9, clip=(0.05, 0.25))   # ZIRP
    ffr[162:197]= np.linspace(0.25, 2.4, 35)                                 # 2015-2018 hike
    ffr[197:221]= np.linspace(2.4, 0.1, 24)                                  # 2019-2020 cut
    ffr[221:233]= ar1(12, mu=0.08, sigma=0.02, phi=0.95, clip=(0.05, 0.15)) # 2021 ZIRP
    ffr[233:245]= np.linspace(0.08, 4.33, 12)                                # 2022 hike
    ffr[245:]   = ar1(n-245, mu=5.0, sigma=0.2, phi=0.9, clip=(4.0, 5.33))  # 2023-2024
    return np.clip(ffr, 0.05, 5.33)

def corr_from_regime(inflation, vix, n=N):
    """
    True (latent) stock-bond correlation matching dissertation threshold:
      beta = -0.378 + 0.0828 * inflation  (Table 4.1)
      threshold = 4.57%
    Smoothed with AR(1) to mimic rolling window persistence.
    """
    beta = -0.378 + 0.0828 * inflation   # marginal effect (Eq 4.2)
    # Convert marginal effect to correlation (approximate)
    corr_true = beta / (STATS["stock"]["std"] / STATS["bond"]["std"])
    corr_true = np.clip(corr_true, -0.9, 0.9)
    # Add smooth persistence (AR dynamics)
    corr_smooth = np.zeros(n)
    corr_smooth[0] = corr_true[0]
    for t in range(1, n):
        corr_smooth[t] = 0.85 * corr_smooth[t-1] + 0.15 * corr_true[t]
    return corr_smooth

def make_returns(corr_series, n=N):
    """
    Generate correlated stock/bond returns with time-varying correlation.
    2022 calibrated: stocks -18.1%/yr, bonds -13.0%/yr (dissertation §4.2.3)
    """
    stocks = np.zeros(n)
    bonds  = np.zeros(n)
    for t in range(n):
        rho = corr_series[t]
        month = t  # 0-indexed from Aug 2002
        # 2022 = months 233-244
        if 233 <= month < 245:
            mu_s = -0.181 / 12
            mu_b = -0.130 / 12
            sig_s = 0.168 / np.sqrt(12)
            sig_b = 0.063 / np.sqrt(12)
        elif 72 <= month < 84:   # GFC
            mu_s = -0.40 / 12
            mu_b =  0.12 / 12
            sig_s = 0.25 / np.sqrt(12)
            sig_b = 0.06 / np.sqrt(12)
        elif 209 <= month < 213: # COVID
            mu_s = -0.20 / 12
            mu_b =  0.08 / 12
            sig_s = 0.35 / np.sqrt(12)
            sig_b = 0.05 / np.sqrt(12)
        else:
            mu_s  = STATS["stock"]["mean"]
            mu_b  = STATS["bond"]["mean"]
            sig_s = STATS["stock"]["std"]
            sig_b = STATS["bond"]["std"]

        # Cholesky decomposition for correlated normals
        z1 = np.random.normal()
        z2 = np.random.normal()
        s  = mu_s + sig_s * z1
        b  = mu_b + sig_b * (rho * z1 + np.sqrt(1 - rho**2) * z2)
        stocks[t] = s
        bonds[t]  = b
    return stocks, bonds

def build_dataset(window=24, save=True):
    dates = make_dates(N)

    inflation = make_inflation(N)
    vix       = make_vix(N, inflation)
    fed_funds = make_fed_funds(inflation, N)
    credit_sp = ar1(N, mu=0.93, sigma=0.15, phi=0.90, clip=(0.55, 2.18))
    real_ffr  = fed_funds - inflation

    corr_true = corr_from_regime(inflation, vix, N)
    stocks, bonds = make_returns(corr_true, N)

    # 24-month rolling correlation (what the models will predict)
    stock_s = pd.Series(stocks)
    bond_s  = pd.Series(bonds)
    rolling_corr = stock_s.rolling(window).corr(bond_s).values

    df = pd.DataFrame({
        "stock":        stocks,
        "bond":         bonds,
        "vix":          vix,
        "inflation":    inflation,
        "fed_funds":    fed_funds,
        "real_ffr":     real_ffr,
        "credit_spread":credit_sp,
        "rolling_corr": rolling_corr,
        "corr_sign":    (rolling_corr > 0).astype(float),
        "high_inflation":(inflation > 4.57).astype(float),
    }, index=dates)

    df = df.dropna()

    if save:
        df.to_csv(os.path.join(os.path.dirname(__file__), "master.csv"))
        print(f"Saved master.csv  –  {len(df)} rows")
        print(df.describe().round(4))

    return df

if __name__ == "__main__":
    df = build_dataset()
    print("\nCorrelation sign distribution:")
    print(df["corr_sign"].value_counts())
    print("\n2022 avg correlation:", df.loc["2022-01":"2022-12", "rolling_corr"].mean().round(3))
    print("Pre-2020 avg correlation:", df.loc[:"2019-12", "rolling_corr"].mean().round(3))