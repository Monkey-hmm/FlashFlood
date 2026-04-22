"""
====================================================
 Flash Flood Prediction — Feature Engineering
 Dataset: FlowDB Sample (Kaggle: isaacmg/flowdb-sample)
====================================================
Run AFTER eda.py.
Run: python feature_engineering.py
Output: outputs/features/features.parquet
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# ── Config ────────────────────────────────────────
INPUT_PATH  = "outputs/eda/flowdb_raw_with_labels.parquet"
OUTPUT_DIR  = "outputs/features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COL     = "cfs"
PRECIP_COL     = "precip"
FLOOD_QUANTILE = 0.95


# ── 1. Load ───────────────────────────────────────
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(INPUT_PATH)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    df.sort_values([c for c in ["catchment_id", "date"] if c in df.columns], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Loaded: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df


def clean_source_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Drop rows with no discharge
    - Fill missing precip with 0
    - Drop columns >50% missing
    - Drop ALL string/object columns except date & catchment_id (prevents coercion nuke later)
    """
    print("  Cleaning source data...")
    n_before = len(df)

    if TARGET_COL in df.columns:
        df.dropna(subset=[TARGET_COL], inplace=True)

    if PRECIP_COL in df.columns:
        df[PRECIP_COL] = df[PRECIP_COL].fillna(0.0)
    else:
        print(f"  [Warn] '{PRECIP_COL}' not found; creating zero column.")
        df[PRECIP_COL] = 0.0

    # Drop sparse columns (>50% NaN)
    sparse = [
        c for c in df.columns
        if c not in {"date", "catchment_id", TARGET_COL, PRECIP_COL}
        and df[c].isna().mean() > 0.5
    ]
    if sparse:
        print(f"    Dropping sparse columns (>50% NaN): {sparse}")
        df.drop(columns=sparse, inplace=True)

    # Drop known junk string/index columns
    always_drop = ["valid", "hour_updated", "skyc1", "station_id",
                   "sensing_time", "base_url", "datetime", "index"]
    df.drop(columns=[c for c in always_drop if c in df.columns], inplace=True)

    # Drop ALL remaining object columns except date & catchment_id
    # (USGS _cd qualifier codes, free-text fields, etc.)
    obj_cols = [c for c in df.columns
                if df[c].dtype == object and c not in {"date", "catchment_id"}]
    if obj_cols:
        print(f"    Dropping remaining string columns: {obj_cols}")
        df.drop(columns=obj_cols, inplace=True)

    print(f"    Rows: {n_before:,} → {len(df):,}")
    print(f"    Remaining columns ({len(df.columns)}): {df.columns.tolist()}")
    return df


# ── 2. Rolling Rainfall Features ──────────────────
def add_rainfall_features(df: pd.DataFrame) -> pd.DataFrame:
    """Antecedent precipitation indices at multiple lag windows."""
    print("  Adding rainfall rolling features...")
    windows = [1, 3, 6, 12, 24, 48, 72]
    for w in windows:
        df[f"precip_sum_{w}h"] = (
            df.groupby("catchment_id")[PRECIP_COL]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).sum())
        )
        df[f"precip_max_{w}h"] = (
            df.groupby("catchment_id")[PRECIP_COL]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).max())
        )
        df[f"precip_mean_{w}h"] = (
            df.groupby("catchment_id")[PRECIP_COL]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )

    df["precip_diff_1h"] = df.groupby("catchment_id")[PRECIP_COL].diff(1)
    df["precip_diff_3h"] = df.groupby("catchment_id")[PRECIP_COL].diff(3)

    def api(series, k=0.9):
        result = np.zeros(len(series))
        for i in range(1, len(series)):
            result[i] = k * result[i - 1] + series.iloc[i]
        return pd.Series(result, index=series.index)

    df["api_0.9"] = df.groupby("catchment_id")[PRECIP_COL].transform(api)
    return df


# ── 3. Streamflow Lag Features ────────────────────
def add_discharge_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    print("  Adding discharge lag features...")
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"discharge_lag_{lag}"] = (
            df.groupby("catchment_id")[TARGET_COL]
            .transform(lambda x: x.shift(lag))
        )

    df["discharge_diff_1"] = df.groupby("catchment_id")[TARGET_COL].diff(1)
    df["discharge_diff_3"] = df.groupby("catchment_id")[TARGET_COL].diff(3)

    for w in [3, 6, 24]:
        df[f"discharge_roll_mean_{w}"] = (
            df.groupby("catchment_id")[TARGET_COL]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )
        df[f"discharge_roll_std_{w}"] = (
            df.groupby("catchment_id")[TARGET_COL]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).std())
        )
    return df


# ── 4. Temporal Features ──────────────────────────
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        print("  [Skip] Temporal features — no 'date' column.")
        return df
    print("  Adding temporal features...")
    df["hour"]       = df["date"].dt.hour
    df["dayofweek"]  = df["date"].dt.dayofweek
    df["month"]      = df["date"].dt.month
    df["dayofyear"]  = df["date"].dt.dayofyear
    df["quarter"]    = df["date"].dt.quarter
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    df["hour_sin"]      = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]      = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"]     = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]     = np.cos(2 * np.pi * df["month"] / 12)
    df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)
    return df


# ── 5. Soil Moisture Proxy ────────────────────────
def add_soil_moisture_proxy(df: pd.DataFrame) -> pd.DataFrame:
    print("  Adding soil moisture proxy...")

    def bucket_model(series, decay=0.95, max_val=200):
        sm = np.zeros(len(series))
        for i in range(1, len(series)):
            sm[i] = min(max_val, max(0, sm[i - 1] * decay + series.iloc[i]))
        return pd.Series(sm, index=series.index)

    df["soil_moisture_proxy"] = (
        df.groupby("catchment_id")[PRECIP_COL].transform(bucket_model)
    )
    return df


# ── 6. Interaction Features ───────────────────────
def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    print("  Adding interaction features...")
    if "precip_sum_6h" in df.columns and "soil_moisture_proxy" in df.columns:
        df["rain_x_soil"] = df["precip_sum_6h"] * df["soil_moisture_proxy"]
    if "discharge_roll_mean_24" in df.columns:
        df["discharge_ratio_24h"] = df[TARGET_COL] / (df["discharge_roll_mean_24"] + 1e-9)
    return df


# ── 7. Catchment Encoding ─────────────────────────
def encode_catchment(df: pd.DataFrame) -> pd.DataFrame:
    print("  Encoding catchment IDs...")
    le = LabelEncoder()
    df["catchment_enc"] = le.fit_transform(df["catchment_id"].astype(str))
    joblib.dump(le, f"{OUTPUT_DIR}/catchment_encoder.pkl")
    return df


# ── 8. Label / Target ─────────────────────────────
def add_target(df: pd.DataFrame) -> pd.DataFrame:
    if "is_flood" not in df.columns:
        thresh = df[TARGET_COL].quantile(FLOOD_QUANTILE)
        df["is_flood"] = (df[TARGET_COL] >= thresh).astype(int)
    df["log_discharge"] = np.log1p(df[TARGET_COL])
    return df


# ── 9. Clean Up & Save ────────────────────────────
def finalise(df: pd.DataFrame):
    non_features = {
        "date", "catchment_id", TARGET_COL, "is_flood", "log_discharge",
        "height", "tz_cd", "site_no", "precip"
    }

    # Only select numeric columns not in non_features and not _cd qualifiers
    feature_cols = [
        c for c in df.columns
        if c not in non_features
        and not c.endswith("_cd")
        and df[c].dtype != object
    ]

    print(f"  Feature columns selected ({len(feature_cols)}): {feature_cols[:10]} ...")

    print("  Dropping NaNs from rolling windows...")
    n_before = len(df)
    df.dropna(subset=feature_cols + [TARGET_COL, "is_flood"], inplace=True)
    print(f"  Rows: {n_before:,} → {len(df):,} (dropped {n_before - len(df):,})")

    if len(df) == 0:
        raise RuntimeError(
            "DataFrame is empty after dropping NaNs. "
            "Inspect which feature columns have NaN with: "
            "df[feature_cols].isna().sum().sort_values(ascending=False)"
        )

    # Scale numeric features
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.pkl")
    print(f"  Scaler saved → {OUTPUT_DIR}/scaler.pkl")

    with open(f"{OUTPUT_DIR}/feature_cols.txt", "w") as f:
        f.write("\n".join(feature_cols))

    return df, feature_cols


# ── Main ──────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  Flash Flood Feature Engineering")
    print("=" * 50)

    df = load_data()
    df = clean_source_data(df)
    df = add_rainfall_features(df)
    df = add_discharge_lag_features(df)
    df = add_temporal_features(df)
    df = add_soil_moisture_proxy(df)
    df = add_interaction_features(df)
    df = encode_catchment(df)
    df = add_target(df)
    df, feature_cols = finalise(df)

    out_path = f"{OUTPUT_DIR}/features.parquet"
    df.to_parquet(out_path, index=False)

    print(f"\nDone!")
    print(f"  Feature matrix : {df.shape}")
    print(f"  Feature count  : {len(feature_cols)}")
    print(f"  Flood rate     : {df['is_flood'].mean()*100:.2f}%")
    print(f"  Saved to       : {out_path}")
    print(f"\nFeature list:")
    for c in feature_cols:
        print(f"  {c}")