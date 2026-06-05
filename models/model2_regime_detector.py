import os
"""
model2_regime_detector.py
────────────────────────────────────────────────────────────────────────────────
MODEL 2: MLP Regime Detector → Regime-Switching Portfolio
  Input:  current macro snapshot [CPI, VIX, real_ffr, credit_spread]
  Output: regime probability (high-inflation / positive-correlation regime)
  Portfolio: soft-switches bond weight based on regime probability
  Compared against: static 60/40 and naive CPI>3% rule (dissertation §4.2.5,
                    Table 4.5)
────────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURES = ["inflation", "vix", "real_ffr", "credit_spread"]
TARGET   = "high_inflation"    # 1 when CPI > 4.57% (dissertation threshold)
HIDDEN   = [32, 16]
DROPOUT  = 0.2
LR       = 5e-4
EPOCHS   = 200
BATCH    = 32
TEST_SPLIT = 0.2

# Portfolio weights in each regime
W_BOND_NORMAL  = 0.40    # standard 60/40
W_BOND_HIGH_INF = 0.511  # minimum-variance weight from dissertation §4.2.5


# ── Model ─────────────────────────────────────────────────────────────────────
class RegimeDetector(nn.Module):
    def __init__(self, n_in, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        layers = []
        in_d = n_in
        for h in hidden:
            layers += [nn.Linear(in_d, h), nn.BatchNorm1d(h), nn.ReLU(),
                       nn.Dropout(dropout)]
            in_d = h
        layers += [nn.Linear(in_d, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Portfolio simulation ───────────────────────────────────────────────────────
def backtest(df, regime_probs, test_idx):
    """
    Soft regime-switching: bond weight interpolates between W_BOND_NORMAL
    and W_BOND_HIGH_INF based on regime probability.
    """
    results = []
    for i, idx in enumerate(test_idx):
        row    = df.iloc[idx]
        p_high = regime_probs[i]

        w_bond  = (1 - p_high) * W_BOND_NORMAL + p_high * W_BOND_HIGH_INF
        w_stock = 1 - w_bond

        # Simple returns (dissertation §3.2.5 note)
        r_port  = w_stock * row["stock"] + w_bond * row["bond"]
        r_6040  = 0.60   * row["stock"] + 0.40   * row["bond"]
        r_naive = (0.80 if row["inflation"] > 3.0 else 0.60) * row["stock"] + \
                  (0.20 if row["inflation"] > 3.0 else 0.40) * row["bond"]

        results.append({
            "date":      df.index[idx],
            "r_nn":      r_port,
            "r_6040":    r_6040,
            "r_naive":   r_naive,
            "w_bond_nn": w_bond,
            "regime_p":  p_high,
            "inflation": row["inflation"],
        })

    return pd.DataFrame(results).set_index("date")


def portfolio_stats(returns, ann=12):
    """Annualised mean, vol, Sharpe."""
    mu    = returns.mean() * ann
    sigma = returns.std()  * np.sqrt(ann)
    sharpe= mu / sigma if sigma > 0 else 0
    return {"Ann. Return (%)": round(mu * 100, 2),
            "Ann. Volatility (%)": round(sigma * 100, 2),
            "Sharpe Ratio": round(sharpe, 3)}


# ── Training ──────────────────────────────────────────────────────────────────
def run():
    # ── Reproducibility ──────────────────────────────────────────────────────
    torch.manual_seed(42)
    np.random.seed(42)
    df = pd.read_csv(os.path.join(ROOT, "data", "master.csv"),
                     index_col=0, parse_dates=True)

    X = df[FEATURES].values
    y = df[TARGET].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    split  = int(len(X) * (1 - TEST_SPLIT))
    X_tr, X_te = X_sc[:split], X_sc[split:]
    y_tr, y_te = y[:split],    y[split:]
    test_idx   = list(range(split, len(df)))

    tr_ds  = torch.utils.data.TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32))
    te_ds  = torch.utils.data.TensorDataset(
        torch.tensor(X_te, dtype=torch.float32),
        torch.tensor(y_te, dtype=torch.float32))
    tr_ld  = torch.utils.data.DataLoader(tr_ds, batch_size=BATCH, shuffle=True)
    te_ld  = torch.utils.data.DataLoader(te_ds, batch_size=BATCH)

    model     = RegimeDetector(len(FEATURES)).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val, best_state = 1e9, None
    history = {'train': [], 'val': []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for X_b, y_b in tr_ld:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(X_b), y_b).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds, val_true = [], []
            val_loss = 0
            for X_b, y_b in te_ld:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                p = model(X_b)
                val_loss += criterion(p, y_b).item() * len(y_b)
                val_preds.extend(p.cpu().numpy())
                val_true.extend(y_b.cpu().numpy())
            val_loss /= len(te_ds)

        history['val'].append(val_loss)
        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0:
            auc = roc_auc_score(val_true, val_preds)
            print(f"Epoch {epoch:3d} | val_loss {val_loss:.4f} | AUC {auc:.3f}")

    # ── Portfolio backtest ────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        regime_probs = model(
            torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
        ).cpu().numpy()

    bt = backtest(df, regime_probs, test_idx)

    print("\n" + "="*60)
    print("MODEL 2: Regime Detector → Portfolio Backtest")
    print("="*60)
    print(f"\nTest AUC (regime detection): "
          f"{roc_auc_score(val_true, val_preds):.4f}")

    for strat, col in [("NN Regime-Switch", "r_nn"),
                        ("Static 60/40",    "r_6040"),
                        ("Naive CPI Rule",  "r_naive")]:
        stats = portfolio_stats(bt[col])
        print(f"\n{strat}:")
        for k, v in stats.items(): print(f"  {k}: {v}")

    # 2022 sub-period
    bt_2022 = bt[bt["inflation"] > 6.0]
    if len(bt_2022) > 0:
        print(f"\n--- High-inflation sub-period (CPI>6%) ---")
        for strat, col in [("NN", "r_nn"), ("60/40", "r_6040"), ("Naive", "r_naive")]:
            vol = bt_2022[col].std() * np.sqrt(12) * 100
            print(f"  {strat} Ann.Vol: {vol:.2f}%")

    # Save
    torch.save({"model_state": best_state, "scaler": scaler},
               os.path.join(ROOT, "models", "regime_detector.pt"))
    bt.to_csv(os.path.join(ROOT, "results", "regime_backtest.csv"))
    print(f"\nSaved model → models/regime_detector.pt")
    print(f"Saved backtest → results/regime_backtest.csv")

    return model, bt, history


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    model, bt = run()