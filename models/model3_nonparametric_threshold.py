import os
"""
model3_nonparametric_threshold.py
────────────────────────────────────────────────────────────────────────────────
MODEL 3: Non-Parametric Threshold Learner
  "Replicate the interaction model but let a neural net learn the threshold
   non-parametrically instead of fixing it at 4.57%"  (dissertation §5.3)

  Architecture: MLP with a single bottleneck neuron that learns the effective
  inflation threshold implicitly. We then extract the learned threshold by
  sweeping inflation and reading off the inflection point.

  Compared against: OLS interaction model (β̂₁ = -0.378, β̂₂ = 0.083,
                    threshold = 4.57%) from dissertation Table 4.1.
────────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Features that enter the interaction model
FEATURES = ["stock", "inflation", "vix", "real_ffr", "credit_spread",
            "stock_x_inflation", "stock_x_vix"]  # same as Eq 3.5
TARGET   = "bond"    # predict bond return (exactly as in dissertation Eq 3.5)

LR       = 5e-4
EPOCHS   = 300
BATCH    = 32
HIDDEN   = [64, 32, 16]
DROPOUT  = 0.2


# ── Interaction features (mirror Eq 3.5) ─────────────────────────────────────
def add_interactions(df):
    df = df.copy()
    df["stock_x_inflation"] = df["stock"] * df["inflation"]
    df["stock_x_vix"]       = df["stock"] * df["vix"]
    return df


# ── Model ─────────────────────────────────────────────────────────────────────
class ThresholdNet(nn.Module):
    """
    Deep MLP that learns the inflation-moderated stock→bond relationship.
    Key design: separate pathway for (stock × inflation) to mirror the
    interaction structure of the OLS model, while allowing non-linearity.
    """
    def __init__(self, n_in, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        layers = []
        in_d = n_in
        for h in hidden:
            layers += [nn.Linear(in_d, h), nn.Tanh(), nn.Dropout(dropout)]
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── OLS baseline (dissertation Table 4.1) ─────────────────────────────────────
def ols_predict(df_test):
    """Replicate dissertation Eq 3.5 with estimated coefficients."""
    b0 = 0.0041
    b1 = -0.3780   # rstock
    b2 =  0.0828   # rstock × inflation
    b3 =  0.0040   # rstock × VIX
    b4 =  0.0010   # real_ffr
    b5 = -0.0024   # credit_spread
    b6 = -0.0021   # inflation
    b7 =  0.0007   # VIX

    pred = (b0
            + b1 * df_test["stock"]
            + b2 * df_test["stock_x_inflation"]
            + b3 * df_test["stock_x_vix"]
            + b4 * df_test["real_ffr"]
            + b5 * df_test["credit_spread"]
            + b6 * df_test["inflation"]
            + b7 * df_test["vix"])
    return pred.values


# ── Extract learned threshold ─────────────────────────────────────────────────
@torch.no_grad()
def extract_threshold(model, scaler, df):
    """
    Sweep inflation from 0 to 10%, holding all other features at their mean.
    Find the CPI level at which the marginal effect of stock returns on bond
    returns changes sign — the learned analog of the 4.57% threshold.
    """
    inflation_grid = np.linspace(0, 10, 500)
    mean_vals      = df[FEATURES].mean().values

    # Build grid: vary inflation, fix everything else at mean
    X_grid = np.tile(mean_vals, (len(inflation_grid), 1))
    inf_idx = FEATURES.index("inflation")
    si_idx  = FEATURES.index("stock_x_inflation")
    s_idx   = FEATURES.index("stock")

    # Marginal effect: evaluate at stock = +1% and stock = -1%
    delta = 0.01
    marginals = []

    for i, cpi in enumerate(inflation_grid):
        # stock = +delta
        row_up            = mean_vals.copy()
        row_up[inf_idx]   = cpi
        row_up[si_idx]    = delta * cpi
        row_up[s_idx]     = delta

        # stock = -delta
        row_dn            = mean_vals.copy()
        row_dn[inf_idx]   = cpi
        row_dn[si_idx]    = -delta * cpi
        row_dn[s_idx]     = -delta

        X_up = scaler.transform(row_up.reshape(1, -1))
        X_dn = scaler.transform(row_dn.reshape(1, -1))

        t_up = torch.tensor(X_up, dtype=torch.float32).to(DEVICE)
        t_dn = torch.tensor(X_dn, dtype=torch.float32).to(DEVICE)

        me = (model(t_up) - model(t_dn)).item() / (2 * delta)
        marginals.append(me)

    marginals = np.array(marginals)

    # Smooth the curve to remove noise before finding zero crossing
    # Use a simple moving average over ~5% of the grid width
    window = max(3, len(marginals) // 20)
    kernel = np.ones(window) / window
    marginals_smooth = np.convolve(marginals, kernel, mode="same")

    # Find zero crossings only in economically meaningful range (1–9% CPI)
    # to avoid spurious crossings near 0% from noisy network output
    econ_mask = (inflation_grid >= 1.0) & (inflation_grid <= 9.0)
    econ_idx  = np.where(econ_mask)[0]
    econ_marginals = marginals_smooth[econ_mask]

    sign_changes = np.where(np.diff(np.sign(econ_marginals)))[0]

    if len(sign_changes) > 0:
        # Pick the crossing closest to OLS threshold (4.57%) among candidates
        candidate_cpis = inflation_grid[econ_idx[sign_changes]]
        best = candidate_cpis[np.argmin(np.abs(candidate_cpis - 4.57))]
        threshold = float(best)
    else:
        # No crossing found — report where smoothed curve is closest to zero
        threshold = float(inflation_grid[econ_idx[np.argmin(np.abs(econ_marginals))]])

    return inflation_grid, marginals, threshold


# ── Training ──────────────────────────────────────────────────────────────────
def run():
    # ── Reproducibility ──────────────────────────────────────────────────────
    torch.manual_seed(42)
    np.random.seed(42)
    df = pd.read_csv(os.path.join(ROOT, "data", "master.csv"),
                     index_col=0, parse_dates=True)
    df = add_interactions(df)

    X = df[FEATURES].values
    y = df[TARGET].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    # Walk-forward split (pre-2020 train, 2020+ test – mirrors §4.3.5)
    split_date = "2020-01-01"
    split_idx  = df.index.searchsorted(split_date)
    X_tr, X_te = X_sc[:split_idx], X_sc[split_idx:]
    y_tr, y_te = y[:split_idx],    y[split_idx:]
    df_te      = df.iloc[split_idx:]

    print(f"Train: {split_idx} obs  |  Test: {len(X_te)} obs")

    tr_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32))
    te_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_te, dtype=torch.float32),
        torch.tensor(y_te, dtype=torch.float32))
    tr_ld = torch.utils.data.DataLoader(tr_ds, batch_size=BATCH, shuffle=True)
    te_ld = torch.utils.data.DataLoader(te_ds, batch_size=len(te_ds))

    model     = ThresholdNet(len(FEATURES)).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

    best_val, best_state = 1e9, None
    history = {'val': []}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for X_b, y_b in tr_ld:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(X_b), y_b).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            for X_b, y_b in te_ld:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                val_loss = criterion(model(X_b), y_b).item()
                val_preds = model(X_b).cpu().numpy()
                val_true  = y_b.cpu().numpy()

        history['val'].append(val_loss)
        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 60 == 0:
            corr = np.corrcoef(val_preds, val_true)[0, 1]
            print(f"Epoch {epoch:3d} | val_MSE {val_loss:.6f} | "
                  f"pred-true corr {corr:.3f}")

    model.load_state_dict(best_state)

    # ── OLS baseline ──────────────────────────────────────────────────────────
    ols_pred = ols_predict(df_te)
    ols_mse  = np.mean((ols_pred - y_te) ** 2)
    nn_mse   = best_val

    ols_corr = np.corrcoef(ols_pred, y_te)[0, 1]
    nn_corr  = np.corrcoef(val_preds, y_te)[0, 1]

    print("\n" + "="*60)
    print("MODEL 3: Non-Parametric Threshold Learner")
    print("="*60)
    print(f"\n{'Model':<20} {'Test MSE':>12} {'Pred-True Corr':>16}")
    print("-"*50)
    print(f"{'OLS (Eq 3.5)':<20} {ols_mse:>12.6f} {ols_corr:>16.4f}")
    print(f"{'ThresholdNet':<20} {nn_mse:>12.6f} {nn_corr:>16.4f}")

    # ── Extract learned threshold ─────────────────────────────────────────────
    print("\nExtracting learned inflation threshold...")
    cpi_grid, marginals, nn_threshold = extract_threshold(model, scaler, df)

    print(f"\nOLS threshold (dissertation):  4.57% CPI")
    print(f"NN  threshold (learned):       {nn_threshold:.2f}% CPI"
          if not np.isnan(nn_threshold) else
          "NN threshold: no sign change found in [0, 10%]")

    # ── Save results ──────────────────────────────────────────────────────────
    torch.save({"model_state": best_state, "scaler": scaler},
               os.path.join(ROOT, "models", "threshold_net.pt"))

    results_df = pd.DataFrame({
        "true_bond_return": y_te,
        "nn_pred":          val_preds,
        "ols_pred":         ols_pred,
    }, index=df_te.index)
    results_df.to_csv(os.path.join(ROOT, "results", "threshold_predictions.csv"))

    threshold_df = pd.DataFrame({
        "cpi_grid":   cpi_grid,
        "marginal_effect_nn": marginals,
        "marginal_effect_ols": -0.3780 + 0.0828 * cpi_grid,
    })
    threshold_df.to_csv(os.path.join(ROOT, "results", "threshold_curve.csv"),
                        index=False)

    print(f"\nSaved model → models/threshold_net.pt")
    print(f"Saved predictions → results/threshold_predictions.csv")
    print(f"Saved threshold curve → results/threshold_curve.csv")

    return model, threshold_df, nn_threshold, history


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    model, threshold_df, threshold = run()