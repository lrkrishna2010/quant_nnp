import os
"""
model1_lstm_classifier.py
────────────────────────────────────────────────────────────────────────────────
MODEL 1: LSTM Correlation-Sign Classifier
  Input:  sequence of [CPI, VIX, real_ffr, credit_spread] (look-back window)
  Output: P(rolling_corr > 0) next month
  Based on:  dissertation §1.3, §4.1  (inflation threshold predicts sign)
────────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN   = 12        # 12-month look-back
FEATURES  = ["inflation", "vix", "real_ffr", "credit_spread", "stock", "bond"]
TARGET    = "corr_sign"
HIDDEN    = 64
LAYERS    = 2
DROPOUT   = 0.3
LR        = 1e-3
EPOCHS    = 150
BATCH     = 32
TEST_FRAC = 0.2       # walk-forward: last 20% = 2020-2024


# ── Dataset ───────────────────────────────────────────────────────────────────
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def prepare_data(df, seq_len=SEQ_LEN):
    scaler = StandardScaler()
    feat   = scaler.fit_transform(df[FEATURES].values)
    target = df[TARGET].values

    X, y = [], []
    for i in range(seq_len, len(feat)):
        X.append(feat[i - seq_len:i])   # shape (seq_len, n_features)
        y.append(target[i])

    X, y = np.array(X), np.array(y)

    split = int(len(X) * (1 - TEST_FRAC))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    return X_tr, X_te, y_tr, y_te, scaler


# ── Model ─────────────────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden=HIDDEN, n_layers=LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, n_layers,
                            batch_first=True, dropout=dropout)
        self.head  = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)  # raw logit — sigmoid applied by BCEWithLogitsLoss
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)   # raw logit


# ── Training loop ─────────────────────────────────────────────────────────────
def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        optimizer.zero_grad()
        pred = model(X_b)
        loss = criterion(pred, y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_b)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, preds, trues = 0, [], []
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        pred = model(X_b)
        total_loss += criterion(pred, y_b).item() * len(y_b)
        preds.extend(torch.sigmoid(pred).cpu().numpy())
        trues.extend(y_b.cpu().numpy())
    return total_loss / len(loader.dataset), np.array(preds), np.array(trues)


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    # ── Reproducibility ──────────────────────────────────────────────────────
    torch.manual_seed(42)
    np.random.seed(42)
    df = pd.read_csv(os.path.join(ROOT, "data", "master.csv"),
                     index_col=0, parse_dates=True)

    X_tr, X_te, y_tr, y_te, scaler = prepare_data(df)

    tr_ds  = SequenceDataset(X_tr, y_tr)
    te_ds  = SequenceDataset(X_te, y_te)
    tr_ld  = torch.utils.data.DataLoader(tr_ds, batch_size=BATCH, shuffle=True)
    te_ld  = torch.utils.data.DataLoader(te_ds, batch_size=BATCH)

    model     = LSTMClassifier(len(FEATURES)).to(DEVICE)
    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)]).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

    best_val, patience, best_state = 1e9, 20, None
    history = {"train": [], "val": []}

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train(model, tr_ld, criterion, optimizer)
        va_loss, preds, trues = evaluate(model, te_ld, criterion)
        scheduler.step()

        history["train"].append(tr_loss)
        history["val"].append(va_loss)

        if va_loss < best_val:
            best_val   = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pat_count  = 0
        else:
            pat_count += 1

        if epoch % 25 == 0:
            auc = roc_auc_score(trues, preds)
            acc = ((preds > 0.5) == trues).mean()
            print(f"Epoch {epoch:3d} | train {tr_loss:.4f} | val {va_loss:.4f} "
                  f"| AUC {auc:.3f} | Acc {acc:.3f}")

        if pat_count >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # ── Final evaluation ──────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    _, preds, trues = evaluate(model, te_ld, criterion)
    # Lower threshold accounts for class imbalance (~16% positive months)
    pred_labels = (preds > 0.35).astype(int)

    auc = roc_auc_score(trues, preds)
    print("\n" + "="*60)
    print("MODEL 1: LSTM Correlation-Sign Classifier")
    print("="*60)
    print(f"Test AUC:      {auc:.4f}")
    print(f"Test Accuracy: {(pred_labels == trues).mean():.4f}")
    print("\nClassification Report:")
    print(classification_report(trues, pred_labels,
                                target_names=["Negative Corr", "Positive Corr"]))

    # Save model and predictions
    torch.save({"model_state": best_state,
                "scaler": scaler,
                "history": history},
               os.path.join(ROOT, "models", "lstm_classifier.pt"))

    results = pd.DataFrame({"prob_positive": preds, "true_sign": trues},
                           index=df.index[-len(trues):])
    results.to_csv(os.path.join(ROOT, "results", "lstm_predictions.csv"))
    print(f"\nSaved model → models/lstm_classifier.pt")
    print(f"Saved predictions → results/lstm_predictions.csv")

    return model, history, results


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    model, history, results = run()