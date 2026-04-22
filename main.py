"""
====================================================
 Flash Flood Prediction — Model Training (v2)
 Models: XGBoost + LSTM
 Fixes: leakage-safe split, no SMOTE, focal loss,
        threshold tuning, probability calibration
====================================================
Run AFTER feature_engineering.py.
Run: python train.py
Outputs: outputs/models/
"""
import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Config ────────────────────────────────────────
FEATURE_PATH = "outputs/features/features.parquet"
MODEL_DIR    = "outputs/models"
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET_CLS   = "is_flood"
SEQ_LEN      = 24
BATCH_SIZE   = 512
LSTM_EPOCHS  = 40
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE = 42

# Leakage-safe: features that encode future or direct flood state
# are excluded. The feature_engineering step already uses .shift(1)
# for all rolling windows, so we just audit at load time.
LEAKAGE_FEATURES = {
    # direct discharge at t=0 is used as a feature only via lagged versions
    # any column not lagged is potential leakage — we drop them here
    "cfs", "log_discharge",
}

print(f"Device: {DEVICE}")


# ── 1. Load & Audit Features ──────────────────────
def load_features():
    df = pd.read_parquet(FEATURE_PATH)
    with open("outputs/features/feature_cols.txt") as f:
        feature_cols = [line.strip() for line in f if line.strip()]

    # Audit: remove any leakage columns that slipped through
    safe_cols = [c for c in feature_cols if c not in LEAKAGE_FEATURES]
    dropped = set(feature_cols) - set(safe_cols)
    if dropped:
        print(f"  [Leakage audit] Dropped: {dropped}")
    feature_cols = safe_cols

    # Sanity-check: non-lagged discharge columns should not appear
    suspicious = [c for c in feature_cols
                  if "discharge" in c and not any(
                      k in c for k in ["lag", "diff", "roll", "ratio"])]
    if suspicious:
        print(f"  [Leakage WARNING] Possibly leaky features: {suspicious}")
        feature_cols = [c for c in feature_cols if c not in suspicious]

    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_CLS].values.astype(int)
    print(f"X: {X.shape} | features: {len(feature_cols)} | flood rate: {y.mean()*100:.2f}%")
    return X, y, feature_cols, df


# ── 2. Event-Aware Temporal Split ────────────────
def _label_flood_events(y, gap=3):
    """
    Assign a unique integer ID to each contiguous flood episode.
    Gaps of <= `gap` rows between flood rows are bridged (they are
    almost certainly the same physical event separated by a missing
    hour or a brief dip below threshold).
    Returns array of length n: 0 = non-flood, ≥1 = event ID.
    """
    event_id  = np.zeros(len(y), dtype=int)
    current   = 0
    in_flood  = False
    last_flood_idx = -999

    for i, v in enumerate(y):
        if v == 1:
            if not in_flood or (i - last_flood_idx) > gap:
                current += 1      # new event
            event_id[i] = current
            last_flood_idx = i
            in_flood = True
        else:
            in_flood = False

    return event_id


def temporal_split(X, y, df):
    """
    Event-aware chronological 70/15/15 split.

    Problem with naive chronological split on flood data:
        All flood events cluster early → Val/Test have zero positives.

    Problem with block-stratified split (our previous fix):
        Multi-day flood episodes get sliced across Train/Val/Test.
        The LSTM trains on the start of a storm and is "tested" on
        its tail — trivial memorisation, not generalisation.
        This explains Best iteration: 0 and AUC-PR ~1.0.

    Correct approach — treat each flood *episode* as an atomic unit:
        1. Label every contiguous flood episode with a unique ID.
        2. Assign complete episodes to splits chronologically (70/15/15
           by episode count, not by row count).
        3. Non-flood rows are assigned to whichever split contains
           their nearest neighbouring flood rows, preserving the
           contiguous time-block structure of the original series.
        4. Hard boundary: enforce a `buffer` row gap between splits
           so that rolling-window features from Train cannot bleed
           into Val/Test rows.

    If there are fewer than 3 distinct events the function falls back
    to a plain chronological cut and warns loudly.
    """
    BUFFER = SEQ_LEN  # rows dropped at each split boundary

    event_id = _label_flood_events(y, gap=3)
    n_events  = event_id.max()

    print(f"  Flood episodes detected: {n_events}")

    if n_events < 3:
        print("  ⚠ Fewer than 3 flood episodes — using plain chronological split.")
        print("    Val and/or Test may have no positive examples.")
        n = len(X)
        train_end = int(n * 0.70)
        val_end   = int(n * 0.85)
        X_train, y_train = X[:train_end],        y[:train_end]
        X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
        X_test,  y_test  = X[val_end:],          y[val_end:]
    else:
        # Chronological episode boundaries
        e_train_end = max(1, int(n_events * 0.70))
        e_val_end   = max(e_train_end + 1, int(n_events * 0.85))

        # Row index of the *last row* belonging to each episode boundary
        def _last_row_of_episode(eid):
            rows = np.where(event_id == eid)[0]
            return rows[-1] if len(rows) else 0

        train_row_end = _last_row_of_episode(e_train_end)
        val_row_end   = _last_row_of_episode(e_val_end)

        # Apply buffer gaps to prevent feature bleed across boundaries
        train_slice = slice(None, train_row_end + 1)
        val_slice   = slice(train_row_end + 1 + BUFFER, val_row_end + 1)
        test_slice  = slice(val_row_end   + 1 + BUFFER, None)

        X_train, y_train = X[train_slice], y[train_slice]
        X_val,   y_val   = X[val_slice],   y[val_slice]
        X_test,  y_test  = X[test_slice],  y[test_slice]

        print(f"  Episode boundary rows — Train ends: {train_row_end} | "
              f"Val ends: {val_row_end} | Buffer: {BUFFER} rows each gap")

    for split_name, ys in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        floods = ys.sum()
        rate   = floods / len(ys) * 100 if len(ys) > 0 else 0
        print(f"  {split_name}: {len(ys):,} rows | {floods} floods ({rate:.1f}%)")
        if floods == 0:
            print(f"  ⚠ WARNING: No flood events in {split_name} split! "
                  f"Metrics will be meaningless for this set.")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ── 3. Leakage Probe ──────────────────────────────
def leakage_probe(X_train, y_train, X_test, y_test, feature_cols):
    """
    Two probes:

    A) Shuffled-label probe — detects hard feature leakage.
       If a model trained on random labels still scores well,
       some feature is directly encoding the target.

    B) Near-duplicate row probe — detects temporal autocorrelation
       bleed across the split boundary.  Computes the fraction of
       test rows whose nearest training neighbour (L∞ distance) is
       within a tight tolerance.  >5 % overlap is a red flag.
    """
    print("\n[Leakage Probe] Training with SHUFFLED labels...")
    y_shuffled = y_train.copy()
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(y_shuffled)

    probe = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    probe.fit(X_train, y_shuffled)

    if len(np.unique(y_test)) > 1:
        proba = probe.predict_proba(X_test)[:, 1]
        auc   = roc_auc_score(y_test, proba)
        print(f"  Shuffled-label AUC on test: {auc:.4f}")
        if auc > 0.70:
            print("  ⛔ LEAKAGE CONFIRMED — AUC high even with random labels!")
        else:
            print("  ✅ No hard leakage detected (AUC near chance).")
    else:
        print("  [Skip] Single class in test — cannot compute AUC.")

    # ── Near-duplicate probe ──────────────────────
    print("\n[Near-Duplicate Probe] Checking train/test row similarity...")
    # Sample at most 2000 rows from each set for speed
    rng = np.random.default_rng(RANDOM_STATE)
    n_sample = 2000
    tr_idx = rng.choice(len(X_train), min(n_sample, len(X_train)), replace=False)
    te_idx = rng.choice(len(X_test),  min(n_sample, len(X_test)),  replace=False)
    X_tr_s = X_train[tr_idx]
    X_te_s = X_test[te_idx]

    # L∞ (Chebyshev) distance: max absolute feature difference per pair
    # Using broadcasting on sample — O(n_sample²) but bounded by 2000²
    dists = np.abs(X_te_s[:, None, :] - X_tr_s[None, :, :]).max(axis=-1)  # (te, tr)
    min_dists = dists.min(axis=1)  # closest training row for each test row

    # Threshold: rows where every feature agrees within 1 % of its range
    feature_ranges = X_train.max(axis=0) - X_train.min(axis=0)
    feature_ranges = np.where(feature_ranges == 0, 1.0, feature_ranges)
    tol = 0.01 * feature_ranges.max()   # 1 % of the widest feature range

    near_dup_frac = (min_dists < tol).mean()
    print(f"  Near-duplicate fraction (tol={tol:.4f}): {near_dup_frac*100:.1f}% of test rows")
    if near_dup_frac > 0.05:
        print("  ⛔ HIGH near-duplicate overlap — split likely contaminated by "
              "same-event rows appearing in both Train and Test.")
    else:
        print("  ✅ Low near-duplicate overlap — split looks clean.")

# ── Helpers ───────────────────────────────────────
def _csi(y_true, y_pred):
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    return tp / (tp + fp + fn + 1e-9)


def _best_threshold(y_true, proba, metric="f1"):
    """Find threshold that maximises F1 or CSI on the given set."""
    thresholds = np.linspace(0.01, 0.99, 200)
    scores = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        if metric == "f1":
            scores.append(f1_score(y_true, pred, zero_division=0))
        else:
            scores.append(_csi(y_true, pred))
    best_t = thresholds[np.argmax(scores)]
    best_s = max(scores)
    print(f"  Best threshold ({metric}): {best_t:.3f} → {metric.upper()}={best_s:.4f}")
    return best_t


def _save_metrics(metrics, name):
    path = f"{MODEL_DIR}/{name.lower().replace(' ', '_')}_metrics.json"
    with open(path, "w") as fh:
        json.dump(metrics, fh, indent=2)


def _report(name, y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    has_both = len(np.unique(y_true)) > 1

    m = {
        "threshold": float(threshold),
        "auc_roc": float(roc_auc_score(y_true, proba)) if has_both else 0.0,
        "auc_pr":  float(average_precision_score(y_true, proba)) if has_both else 0.0,
        "f1":      float(f1_score(y_true, pred, zero_division=0)),
        "csi":     float(_csi(y_true, pred)),
        "brier":   float(brier_score_loss(y_true, proba)),
    }
    print(f"\n{'─'*40}\n  {name} — Test Metrics (threshold={threshold:.3f})\n{'─'*40}")
    for k, v in m.items():
        print(f"  {k.upper():10s}: {v:.4f}")
    print(f"\n{classification_report(y_true, pred, target_names=['No Flood','Flood'], zero_division=0)}")
    _save_metrics(m, name)
    return m


# ══════════════════════════════════════════════════
# MODEL A: XGBoost  (class weights, no SMOTE)
# ══════════════════════════════════════════════════
def train_xgboost(X_train, y_train, X_val, y_val, feature_cols):
    print("\n" + "=" * 50)
    print("  Training XGBoost  (scale_pos_weight, no SMOTE)")
    print("=" * 50)

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw   = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0
    print(f"  scale_pos_weight = {spw:.1f}  (neg={n_neg} pos={n_pos})")

    model = xgb.XGBClassifier(
        n_estimators=2000,
        max_depth=5,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=10,   # higher = less overfit on rare events
        gamma=0.5,
        reg_alpha=0.5,
        reg_lambda=2.0,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        early_stopping_rounds=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=200)
    print(f"\nBest iteration: {model.best_iteration}")
    joblib.dump(model, f"{MODEL_DIR}/xgboost_model.pkl")

    # Calibrate probabilities with Platt scaling (on val set)
    print("  Calibrating probabilities (Platt scaling on val set)...")
    raw_val_proba = model.predict_proba(X_val)[:, 1].reshape(-1, 1)
    calibrator    = LogisticRegression()
    calibrator.fit(raw_val_proba, y_val)
    joblib.dump(calibrator, f"{MODEL_DIR}/xgb_calibrator.pkl")

    return model, calibrator


def evaluate_xgboost(model, calibrator, X_val, y_val, X_test, y_test):
    # Calibrated probabilities
    raw_proba  = model.predict_proba(X_test)[:, 1].reshape(-1, 1)
    cal_proba  = calibrator.predict_proba(raw_proba)[:, 1]

    # Tune threshold on val set
    raw_val  = model.predict_proba(X_val)[:, 1].reshape(-1, 1)
    cal_val  = calibrator.predict_proba(raw_val)[:, 1]
    threshold = _best_threshold(y_val, cal_val, metric="csi")

    return cal_proba, _report("XGBoost", y_test, cal_proba, threshold)


# ══════════════════════════════════════════════════
# MODEL B: LSTM  (focal loss, tuned threshold)
# ══════════════════════════════════════════════════
class FocalLoss(nn.Module):
    """
    Focal loss: down-weights easy negatives so the model
    focuses on hard flood events.  gamma=2 is standard.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce  = nn.functional.binary_cross_entropy_with_logits(
                   logits, targets, reduction="none")
        pt   = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()


class FloodDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X       = torch.tensor(X, dtype=torch.float32)
        self.y       = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, idx):
        return self.X[idx: idx + self.seq_len], self.y[idx + self.seq_len]


class FloodLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.attention  = nn.Linear(hidden_size, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _  = self.lstm(x)
        attn_w  = torch.softmax(self.attention(out), dim=1)
        context = (out * attn_w).sum(dim=1)
        return self.classifier(context).squeeze(-1)


def train_lstm(X_train, y_train, X_val, y_val):
    print("\n" + "=" * 50)
    print("  Training LSTM  (focal loss, no SMOTE)")
    print("=" * 50)

    input_size = X_train.shape[1]
    seq_len    = min(SEQ_LEN, max(1, len(X_train) // 20))
    if seq_len != SEQ_LEN:
        print(f"  SEQ_LEN adjusted: {SEQ_LEN} → {seq_len}")

    train_ds = FloodDataset(X_train, y_train, seq_len)
    val_ds   = FloodDataset(X_val,   y_val,   seq_len)

    if len(train_ds) == 0:
        print("  [Skip] Training set too small.")
        return FloodLSTM(input_size).to(DEVICE), seq_len

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = FloodLSTM(input_size).to(DEVICE)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=LSTM_EPOCHS, eta_min=1e-6)

    history      = {"train_loss": [], "val_loss": [], "val_auc": []}
    best_auc     = 0.0
    patience_ctr = 0
    PATIENCE     = 10

    for epoch in range(1, LSTM_EPOCHS + 1):
        model.train()
        t_losses = []
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_losses.append(loss.item())
        scheduler.step()

        model.eval()
        v_losses, probas, labels = [], [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                v_losses.append(criterion(logits, yb).item())
                probas.extend(torch.sigmoid(logits).cpu().numpy())
                labels.extend(yb.cpu().numpy())

        val_auc = roc_auc_score(labels, probas) if len(np.unique(labels)) > 1 else 0.0
        t_loss  = np.mean(t_losses)
        v_loss  = np.mean(v_losses)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_auc"].append(val_auc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{LSTM_EPOCHS}  "
                  f"train={t_loss:.4f}  val={v_loss:.4f}  val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), f"{MODEL_DIR}/lstm_best.pt")
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    if not os.path.exists(f"{MODEL_DIR}/lstm_best.pt"):
        torch.save(model.state_dict(), f"{MODEL_DIR}/lstm_best.pt")

    model.load_state_dict(torch.load(f"{MODEL_DIR}/lstm_best.pt", map_location=DEVICE))
    print(f"\nBest val AUC: {best_auc:.4f}")
    _plot_lstm_history(history)

    cfg = {"input_size": input_size, "hidden_size": 128, "num_layers": 2,
           "seq_len": seq_len, "device": DEVICE}
    with open(f"{MODEL_DIR}/lstm_config.json", "w") as fh:
        json.dump(cfg, fh, indent=2)
    return model, seq_len


def _plot_lstm_history(history):
    if not history["train_loss"]:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("Focal Loss"); axes[0].legend()
    axes[1].plot(history["val_auc"], color="green")
    axes[1].set_title("Validation AUC-ROC")
    plt.tight_layout()
    plt.savefig(f"{MODEL_DIR}/lstm_training_curves.png")
    plt.close()


def evaluate_lstm(model, seq_len, X_val, y_val, X_test, y_test):
    def _infer(X, y):
        ds = FloodDataset(X, y, seq_len)
        if len(ds) == 0:
            return np.array([]), np.array([])
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        model.eval()
        ps, ls = [], []
        with torch.no_grad():
            for xb, yb in dl:
                logits = model(xb.to(DEVICE))
                ps.extend(torch.sigmoid(logits).cpu().numpy())
                ls.extend(yb.numpy())
        return np.array(ps), np.array(ls)

    val_proba,  val_labels  = _infer(X_val,  y_val)
    test_proba, test_labels = _infer(X_test, y_test)

    if len(val_proba) == 0 or len(test_proba) == 0:
        m = {"auc_roc": 0, "auc_pr": 0, "f1": 0, "csi": 0, "brier": 1, "threshold": 0.5}
        _save_metrics(m, "LSTM")
        return test_proba, test_labels, m

    # Tune threshold on val set
    threshold = _best_threshold(val_labels, val_proba, metric="csi")
    return test_proba, test_labels, _report("LSTM", test_labels, test_proba, threshold)


# ── Dashboard ─────────────────────────────────────
def plot_evaluation(xgb_proba, xgb_labels,
                    lstm_proba, lstm_labels,
                    feature_cols, xgb_model):
    fig = plt.figure(figsize=(20, 13))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("Flash Flood Model Evaluation (Leakage-Audited)", fontsize=15, fontweight="bold")
    colors = {"XGBoost": "#F44336", "LSTM": "#2196F3"}

    pairs = {}
    if len(xgb_proba) > 0: pairs["XGBoost"] = (xgb_proba, xgb_labels)
    if len(lstm_proba) > 0: pairs["LSTM"]    = (lstm_proba, lstm_labels)

    # ROC
    ax1 = fig.add_subplot(gs[0, 0])
    for nm, (p, l) in pairs.items():
        if len(np.unique(l)) > 1:
            fpr, tpr, _ = roc_curve(l, p)
            ax1.plot(fpr, tpr, label=f"{nm} (AUC={roc_auc_score(l,p):.3f})",
                     color=colors[nm])
    ax1.plot([0,1],[0,1],"k--",lw=0.8)
    ax1.set_title("ROC Curve"); ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR"); ax1.legend()

    # PR
    ax2 = fig.add_subplot(gs[0, 1])
    for nm, (p, l) in pairs.items():
        if len(np.unique(l)) > 1:
            prec, rec, _ = precision_recall_curve(l, p)
            ax2.plot(rec, prec,
                     label=f"{nm} (AP={average_precision_score(l,p):.3f})",
                     color=colors[nm])
    ax2.set_title("Precision-Recall Curve"); ax2.set_xlabel("Recall"); ax2.legend()

    # Confusion (XGBoost)
    ax3 = fig.add_subplot(gs[0, 2])
    if len(xgb_proba) > 0:
        with open(f"{MODEL_DIR}/xgboost_metrics.json") as fh:
            thresh = json.load(fh).get("threshold", 0.5)
        cm = confusion_matrix(xgb_labels, (xgb_proba >= thresh).astype(int), labels=[0,1])
        ax3.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax3.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                         fontsize=13, color="white" if cm[i,j]>cm.max()/2 else "black")
        ax3.set_xticks([0,1]); ax3.set_yticks([0,1])
        ax3.set_xticklabels(["No Flood","Flood"])
        ax3.set_yticklabels(["No Flood","Flood"])
        ax3.set_title(f"XGBoost Confusion (t={thresh:.2f})")

    # Feature importance
    ax4 = fig.add_subplot(gs[1, 0:2])
    try:
        imp = pd.Series(xgb_model.feature_importances_, index=feature_cols)
        imp.nlargest(20).sort_values().plot.barh(ax=ax4, color="#42A5F5")
        ax4.set_title("Top-20 Feature Importances (XGBoost — gain)")
    except Exception:
        ax4.set_title("Feature importance unavailable")

    # Threshold sweep (both models)
    ax5 = fig.add_subplot(gs[1, 2])
    for nm, (p, l) in pairs.items():
        if len(np.unique(l)) > 1:
            ts   = np.linspace(0.01, 0.99, 150)
            csis = [_csi(l, (p >= t).astype(int)) for t in ts]
            ax5.plot(ts, csis, label=nm, color=colors[nm])
    ax5.set_title("CSI vs Threshold"); ax5.set_xlabel("Threshold")
    ax5.set_ylabel("CSI"); ax5.legend()

    plt.savefig(f"{MODEL_DIR}/evaluation_dashboard.png", bbox_inches="tight")
    plt.close()
    print(f"Dashboard saved → {MODEL_DIR}/evaluation_dashboard.png")


# ── Main ──────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  Flash Flood Model Training  (v2 — leakage-safe)")
    print("=" * 50)

    X, y, feature_cols, df = load_features()
    X_train, X_val, X_test, y_train, y_val, y_test = temporal_split(X, y, df)

    # ── Leakage probe (non-blocking) ──
    leakage_probe(X_train, y_train, X_test, y_test, feature_cols)

    # ── XGBoost ──
    xgb_model, xgb_cal = train_xgboost(X_train, y_train, X_val, y_val, feature_cols)
    xgb_proba, xgb_m   = evaluate_xgboost(xgb_model, xgb_cal, X_val, y_val, X_test, y_test)

    # ── LSTM ──
    lstm_model, seq_len                  = train_lstm(X_train, y_train, X_val, y_val)
    lstm_proba, lstm_labels, lstm_m      = evaluate_lstm(
        lstm_model, seq_len, X_val, y_val, X_test, y_test)

    # ── Dashboard ──
    try:
        plot_evaluation(xgb_proba, y_test,
                        lstm_proba, lstm_labels,
                        feature_cols, xgb_model)
    except Exception as e:
        print(f"  [Dashboard skipped]: {e}")

    # ── Summary ──
    print("\n" + "=" * 50)
    print("  Final Summary")
    print("=" * 50)
    for nm, m in [("XGBoost", xgb_m), ("LSTM", lstm_m)]:
        print(f"  {nm:10s}  "
              f"AUC-ROC={m['auc_roc']:.4f}  "
              f"AUC-PR={m['auc_pr']:.4f}  "
              f"F1={m['f1']:.4f}  "
              f"CSI={m['csi']:.4f}  "
              f"Brier={m['brier']:.4f}  "
              f"threshold={m['threshold']:.3f}")
    print(f"\nAll outputs → '{MODEL_DIR}/'")