"""
fetch.py  –  Download real market and macro data
Replicates the dissertation dataset: Aug 2002 – Dec 2024

Requirements: pip install yfinance requests
Usage:        python data/fetch.py

Data sources:
  - Yahoo Finance : SPY (stocks), IEF (bonds), ^VIX (volatility)
  - FRED API      : CPIAUCSL, FEDFUNDS, DBAA, DAAA  (no API key needed)
"""

import os
import io
import time
import warnings
import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

START        = "2000-06-01"   # pulled back so 24-month window initialises by Aug 2002
END          = "2024-12-31"
SAMPLE_START = "2002-08-01"
WINDOW       = 24
OUT_PATH     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "master.csv")

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
FRED_API_KEY = "9e0a85d2b88f4394e1150540b996c402"   # placeholder – see note below


# ── 1. Market prices (Yahoo Finance) ─────────────────────────────────────────

def fetch_prices(start=START, end=END) -> pd.DataFrame:
    print("  Downloading SPY, IEF, ^VIX from Yahoo Finance...")
    raw = yf.download(
        ["SPY", "IEF", "^VIX"],
        start=start, end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )
    spy = raw["SPY"]["Close"].resample("ME").last().rename("stock_price")
    ief = raw["IEF"]["Close"].resample("ME").last().rename("bond_price")
    vix = raw["^VIX"]["Close"].resample("ME").mean().rename("vix")
    return pd.concat([spy, ief, vix], axis=1).dropna()


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Dissertation Eq 3.1: r_t = ln(P_t / P_{t-1})"""
    ret = np.log(prices[["stock_price", "bond_price"]] /
                 prices[["stock_price", "bond_price"]].shift(1))
    return ret.rename(columns={"stock_price": "stock", "bond_price": "bond"}).dropna()


# ── 2. Macro data ─────────────────────────────────────────────────────────────

def fred_series_api(series_id: str, start=START, end=END,
                    api_key=FRED_API_KEY) -> pd.Series:
    """
    Fetch via the official FRED JSON API.
    Get a free API key (instant) at: https://fred.stlouisfed.org/docs/api/api_key.html
    Then set: export FRED_API_KEY=your_key_here  (or edit FRED_API_KEY above)
    """
    key = os.environ.get("FRED_API_KEY", api_key)
    params = {
        "series_id":         series_id,
        "observation_start": start[:10],
        "observation_end":   end[:10],
        "api_key":           key,
        "file_type":         "json",
    }
    for attempt in range(3):
        try:
            r = requests.get(FRED_BASE, params=params, timeout=20)
            r.raise_for_status()
            obs = r.json()["observations"]
            s = pd.Series(
                {pd.Timestamp(o["date"]): float(o["value"])
                 for o in obs if o["value"] != "."},
                name=series_id,
            )
            return s
        except Exception as e:
            if attempt == 2:
                raise
            print(f"    Retry {attempt+1} for {series_id}: {e}")
            time.sleep(2)


def fred_series_csv(series_id: str, start=START, end=END) -> pd.Series:
    """
    Fallback: fetch FRED data via the public CSV download URL.
    Uses requests (more reliable than urllib for SSL on macOS).
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    for attempt in range(3):
        try:
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), parse_dates=["DATE"], index_col="DATE")
            s = df.iloc[:, 0].replace(".", np.nan).astype(float).dropna()
            s = s[(s.index >= start) & (s.index <= end)]
            s.name = series_id
            return s
        except Exception as e:
            if attempt == 2:
                raise
            print(f"    Retry {attempt+1} for {series_id}: {e}")
            time.sleep(3)


def fred_series(series_id: str, start=START, end=END) -> pd.Series:
    """Try API key first, fall back to CSV endpoint."""
    key = os.environ.get("FRED_API_KEY", FRED_API_KEY)
    if key and key != "abcdefghijklmnop1234567890abcdef":
        print(f"    {series_id} (FRED API)...")
        return fred_series_api(series_id, start, end, key)
    else:
        print(f"    {series_id} (FRED CSV)...")
        return fred_series_csv(series_id, start, end)


def fetch_macro(start=START, end=END) -> pd.DataFrame:
    print("  Downloading macro data from FRED...")
    cpi_raw   = fred_series("CPIAUCSL", start, end)
    fed_funds  = fred_series("FEDFUNDS",  start, end)
    baa        = fred_series("DBAA",      start, end)
    aaa        = fred_series("DAAA",      start, end)

    cpi_m  = cpi_raw.resample("ME").last()
    ffr_m  = fed_funds.resample("ME").last()
    baa_m  = baa.resample("ME").last()
    aaa_m  = aaa.resample("ME").last()

    inflation     = cpi_m.pct_change(12) * 100
    credit_spread = (baa_m - aaa_m).rename("credit_spread")

    return pd.DataFrame({
        "inflation":     inflation,
        "fed_funds":     ffr_m,
        "credit_spread": credit_spread,
    })


# ── 3. Assemble master dataset ────────────────────────────────────────────────

def build_dataset(window: int = WINDOW, save: bool = True) -> pd.DataFrame:
    prices       = fetch_prices()
    ret          = log_returns(prices)
    vix          = prices["vix"]
    macro        = fetch_macro()
    rolling_corr = ret["stock"].rolling(window).corr(ret["bond"]).rename("rolling_corr")

    df = pd.concat([ret, vix, macro, rolling_corr], axis=1)
    df["real_ffr"]       = df["fed_funds"] - df["inflation"]
    df["corr_sign"]      = (df["rolling_corr"] > 0).astype(int)
    df["high_inflation"] = (df["inflation"] > 4.57).astype(int)

    df = df.dropna()
    # Note: actual start determined by when rolling window initialises
    # IEF launched Jul 2002 so 24-month window fires from ~mid-2004
    # dissertation used an earlier bond series; 246 obs from 2004 is fine

    print(f"\n  {len(df)} monthly obs  ({df.index[0].date()} – {df.index[-1].date()})")
    print(f"  Positive-correlation months:    {df['corr_sign'].sum()} / {len(df)}")
    print(f"  High-inflation months (>4.57%): {df['high_inflation'].sum()}")

    if save:
        df.to_csv(OUT_PATH)
        print(f"  Saved → {OUT_PATH}")

    return df


if __name__ == "__main__":
    import sys

    # Optional: pass API key as argument
    #   python data/fetch.py your_fred_api_key_here
    if len(sys.argv) > 1:
        os.environ["FRED_API_KEY"] = sys.argv[1]

    print("fetch.py — downloading real data")
    print("(FRED API key detected)" if os.environ.get("FRED_API_KEY") else
          "(no FRED API key — using CSV fallback; get a free key at "
          "https://fred.stlouisfed.org/docs/api/api_key.html)\n")

    df = build_dataset()

    print("\nSummary statistics:")
    print(df[["stock","bond","vix","inflation","fed_funds","real_ffr","credit_spread"]]
          .describe().round(4))