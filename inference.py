"""
====================================================
 Flash Flood Prediction — Inference Engine
====================================================
Loads trained XGBoost + LSTM models from outputs/models/
and exposes a unified predict() interface used by server.py.

Usage (standalone):
    from inference import FloodPredictor
    predictor = FloodPredictor()
    result = predictor.predict(feature_dict)
"""
import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
import torch
import torch.nn as nn

MODEL_DIR = "outputs/models"
FEATURE_COLS_PATH = "outputs/features/feature_cols.txt"

LEAKAGE_FEATURES = {"cfs", "log_discharge"}


# ── LSTM architecture (must match main.py exactly) ──────────────────
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


# ── Risk label helper ────────────────────────────────────────────────
def _risk_label(prob: float) -> tuple[str, str]:
    """Return (label, hex_colour) for a probability."""
    if prob < 0.20:
        return "Low",      "#22c55e"   # green
    elif prob < 0.45:
        return "Moderate", "#f59e0b"   # amber
    elif prob < 0.70:
        return "High",     "#f97316"   # orange
    else:
        return "Critical", "#ef4444"   # red


class FloodPredictor:
    """
    Loads all model artefacts once at construction time.
    Call predict(features_dict) at inference time.

    features_dict — {feature_name: value, ...} for a single timestep.
    For XGBoost: uses the single feature vector directly.
    For LSTM:    expects a 'history' key containing a list of
                 SEQ_LEN feature dicts (oldest → newest).
                 Falls back to repeating the single row if absent.
    """

    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"

        # ── Feature columns ─────────────────────────────────────────
        with open(FEATURE_COLS_PATH) as f:
            all_cols = [l.strip() for l in f if l.strip()]
        suspicious = [c for c in all_cols
                      if "discharge" in c and not any(
                          k in c for k in ["lag", "diff", "roll", "ratio"])]
        self.feature_cols = [c for c in all_cols
                             if c not in LEAKAGE_FEATURES and c not in suspicious]

        # ── XGBoost ─────────────────────────────────────────────────
        xgb_path = os.path.join(model_dir, "xgboost_model.pkl")
        cal_path  = os.path.join(model_dir, "xgb_calibrator.pkl")
        self.xgb_model    = joblib.load(xgb_path)
        self.xgb_cal      = joblib.load(cal_path)

        with open(os.path.join(model_dir, "xgboost_metrics.json")) as fh:
            self.xgb_threshold = json.load(fh).get("threshold", 0.5)

        # ── LSTM ─────────────────────────────────────────────────────
        lstm_cfg_path = os.path.join(model_dir, "lstm_config.json")
        lstm_wt_path  = os.path.join(model_dir, "lstm_best.pt")
        with open(lstm_cfg_path) as fh:
            cfg = json.load(fh)

        self.seq_len   = cfg["seq_len"]
        self.lstm_model = FloodLSTM(
            input_size  = cfg["input_size"],
            hidden_size = cfg["hidden_size"],
            num_layers  = cfg["num_layers"],
        ).to(self.device)
        self.lstm_model.load_state_dict(
            torch.load(lstm_wt_path, map_location=self.device, weights_only=True)
        )
        self.lstm_model.eval()

        with open(os.path.join(model_dir, "lstm_metrics.json")) as fh:
            self.lstm_threshold = json.load(fh).get("threshold", 0.5)

        print(f"[FloodPredictor] Loaded — device={self.device} | "
              f"features={len(self.feature_cols)} | "
              f"xgb_t={self.xgb_threshold:.3f} | lstm_t={self.lstm_threshold:.3f}")

    # ── Internal helpers ─────────────────────────────────────────────
    def _row_to_vec(self, feature_dict: dict) -> np.ndarray:
        """Convert a feature dict to a 1-D float32 array aligned to feature_cols."""
        return np.array(
            [float(feature_dict.get(c, 0.0)) for c in self.feature_cols],
            dtype=np.float32,
        )

    def _xgb_predict(self, x_vec: np.ndarray) -> float:
        raw   = self.xgb_model.predict_proba(x_vec[None, :])[: , 1].reshape(-1, 1)
        cal   = self.xgb_cal.predict_proba(raw)[0, 1]
        return float(cal)

    def _lstm_predict(self, history: list[dict]) -> float:
        """history: list of feature dicts, oldest first, length >= seq_len."""
        # Pad / truncate to exactly seq_len rows
        if len(history) < self.seq_len:
            pad  = [history[0]] * (self.seq_len - len(history))
            history = pad + history
        history = history[-self.seq_len:]

        seq = np.stack([self._row_to_vec(h) for h in history], axis=0)  # (T, F)
        tensor = torch.tensor(seq[None, :, :], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logit = self.lstm_model(tensor)
            prob  = torch.sigmoid(logit).item()
        return float(prob)

    # ── Public API ───────────────────────────────────────────────────
    def predict(self, feature_dict: dict, history: list[dict] | None = None) -> dict:
        """
        Parameters
        ----------
        feature_dict : dict
            Current-timestep features {name: value}.
        history : list[dict] | None
            Optional list of SEQ_LEN previous feature dicts (oldest first).
            If omitted, the current row is repeated to fill the LSTM window
            (degrades LSTM quality but keeps the API simple for quick testing).

        Returns
        -------
        dict with keys:
            xgb_prob       float   calibrated XGBoost probability
            xgb_flag       bool    XGBoost flood prediction at trained threshold
            lstm_prob      float   LSTM sigmoid probability
            lstm_flag      bool    LSTM flood prediction at trained threshold
            ensemble_prob  float   simple average of both probabilities
            ensemble_flag  bool    ensemble prediction (majority of flags)
            risk_label     str     Low / Moderate / High / Critical
            risk_color     str     hex colour for the risk label
            xgb_threshold  float
            lstm_threshold float
            features_used  int
        """
        x_vec = self._row_to_vec(feature_dict)

        # XGBoost
        xgb_prob = self._xgb_predict(x_vec)
        xgb_flag = xgb_prob >= self.xgb_threshold

        # LSTM
        if history is None:
            history = [feature_dict] * self.seq_len
        lstm_prob = self._lstm_predict(history)
        lstm_flag = lstm_prob >= self.lstm_threshold

        # Ensemble
        ensemble_prob = (xgb_prob + lstm_prob) / 2.0
        ensemble_flag = bool(xgb_flag) or bool(lstm_flag)   # OR = high-recall bias

        risk_label, risk_color = _risk_label(ensemble_prob)

        return {
            "xgb_prob":        round(xgb_prob,       4),
            "xgb_flag":        bool(xgb_flag),
            "lstm_prob":       round(lstm_prob,       4),
            "lstm_flag":       bool(lstm_flag),
            "ensemble_prob":   round(ensemble_prob,   4),
            "ensemble_flag":   ensemble_flag,
            "risk_label":      risk_label,
            "risk_color":      risk_color,
            "xgb_threshold":   round(self.xgb_threshold,  4),
            "lstm_threshold":  round(self.lstm_threshold,  4),
            "features_used":   len(self.feature_cols),
        }

    def feature_names(self) -> list[str]:
        """Return the ordered list of expected feature names."""
        return list(self.feature_cols)
