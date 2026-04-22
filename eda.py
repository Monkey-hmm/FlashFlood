"""
====================================================
 Flash Flood Prediction — EDA
 Dataset: FlowDB Sample (Kaggle: isaacmg/flowdb-sample)
====================================================
Run: python eda.py
Outputs saved to: outputs/eda/
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

import kagglehub
from kagglehub import KaggleDatasetAdapter

# ── Config ────────────────────────────────────────
OUTPUT_DIR = "outputs/eda"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FLOOD_QUANTILE = 0.95   # define "flood" as top 5% streamflow
TARGET_COL  = "cfs"    # discharge in cubic feet per second
PRECIP_COL  = "p_mean" # mean precipitation column in flowdb-sample

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})


# ── 1. Load Data via KaggleHub ────────────────────
def load_flowdb() -> pd.DataFrame:
    """
    Load FlowDB Sample dataset directly from Kaggle using kagglehub.
    The dataset may be a single DataFrame or need per-file loading.
    """
    print("Downloading dataset from Kaggle: isaacmg/flowdb-sample ...")

    # Try loading the full dataset (all files merged) first
    try:
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "isaacmg/flowdb-sample",
            "",  # empty string loads default / first file
        )
    except Exception as e:
        print(f"  Default load failed ({e}), attempting path-based load...")
        path = kagglehub.dataset_download("isaacmg/flowdb-sample")
        frames = []
        for root, _, files in os.walk(path):
            for fname in sorted(files):
                if not fname.endswith(".csv"):
                    continue
                catchment_id = fname.replace(".csv", "")
                fp = os.path.join(root, fname)
                try:
                    tmp = pd.read_csv(fp)
                    tmp["catchment_id"] = catchment_id
                    frames.append(tmp)
                except Exception as read_err:
                    print(f"    Skipping {fname}: {read_err}")
        if not frames:
            raise FileNotFoundError("No CSV files found in downloaded dataset.")
        df = pd.concat(frames, ignore_index=True)

    print(f"Raw shape: {df.shape}")
    print(f"Columns  : {df.columns.tolist()}")

    # ── Normalise column names ────────────────────
    df.columns = [c.strip().lower() for c in df.columns]

    # Map common date column names
    for date_alias in ["date", "datetime", "time", "timestamp"]:
        if date_alias in df.columns:
            df.rename(columns={date_alias: "date"}, inplace=True)
            break

    # Map discharge column (try several aliases)
    for q_alias in ["cfs", "q", "streamflow", "discharge", "flow"]:
        if q_alias in df.columns and q_alias != "cfs":
            df.rename(columns={q_alias: "cfs"}, inplace=True)
            break

    # Map precipitation column
    for p_alias in ["p_mean", "precip", "p01m", "precipitation", "rain"]:
        if p_alias in df.columns:
            df.rename(columns={p_alias: "precip"}, inplace=True)
            PRECIP_COL_ACTUAL = "precip"
            break
    else:
        PRECIP_COL_ACTUAL = None

    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None)
        df.dropna(subset=["date"], inplace=True)

    # Add catchment_id if missing
    if "catchment_id" not in df.columns:
        df["catchment_id"] = "default"

    # Drop unnamed index columns
    df.drop(columns=[c for c in df.columns if c.startswith("unnamed")],
            inplace=True, errors="ignore")

    # ── Deduplicate column names (keep first occurrence) ─────
    seen, deduped = {}, []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            deduped.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            deduped.append(col)
    df.columns = deduped
    # Drop the renamed duplicate columns (suffixed _1, _2, …)
    import re
    dup_cols = [c for c in df.columns if re.search(r'_\d+$', c)
                and c.rsplit('_', 1)[0] in seen and seen[c.rsplit('_', 1)[0]] > 0]
    df.drop(columns=dup_cols, inplace=True, errors="ignore")

    # Sort
    sort_cols = [c for c in ["catchment_id", "date"] if c in df.columns]
    df.sort_values(sort_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"Loaded {len(df):,} rows | {df['catchment_id'].nunique()} catchment(s)")
    return df


# ── 2. Basic Summary ──────────────────────────────
def basic_summary(df: pd.DataFrame):
    print("\n=== Shape ===")
    print(df.shape)
    print("\n=== Dtypes ===")
    print(df.dtypes)
    print("\n=== Missing Values (%) ===")
    miss = df.isnull().mean().mul(100).round(2)
    print(miss[miss > 0].to_string() if miss.any() else "  None")
    print("\n=== Numeric Summary ===")
    print(df.describe().T.round(3))
    return miss


# ── 3. Target Distribution ────────────────────────
def plot_target_distribution(df: pd.DataFrame, col: str):
    if col not in df.columns:
        print(f"  [Skip] '{col}' column not found. Available: {df.columns.tolist()}")
        return df

    flood_thresh = df[col].quantile(FLOOD_QUANTILE)
    df["is_flood"] = (df[col] >= flood_thresh).astype(int)
    flood_pct = df["is_flood"].mean() * 100

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Target Variable Distribution", fontsize=14, fontweight="bold")

    axes[0].hist(df[col].dropna(), bins=80, color="#2196F3", edgecolor="none", alpha=0.8)
    axes[0].axvline(flood_thresh, color="red", linestyle="--",
                    label=f"Flood threshold (p{int(FLOOD_QUANTILE*100)})")
    axes[0].set_title("Discharge Volume — Raw")
    axes[0].set_xlabel(col)
    axes[0].legend()

    log_vals = np.log1p(df[col].dropna())
    axes[1].hist(log_vals, bins=80, color="#4CAF50", edgecolor="none", alpha=0.8)
    axes[1].set_title("Discharge Volume — log1p")
    axes[1].set_xlabel(f"log1p({col})")

    counts = df["is_flood"].value_counts().sort_index()
    axes[2].bar(["No Flood", "Flood"], counts.values, color=["#607D8B", "#F44336"], alpha=0.8)
    axes[2].set_title(f"Class Balance (flood = {flood_pct:.1f}%)")
    for i, v in enumerate(counts.values):
        axes[2].text(i, v + max(counts) * 0.01, f"{v:,}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_target_distribution.png")
    plt.close()
    print(f"Flood threshold: {flood_thresh:.3f} | Flood events: {flood_pct:.1f}%")
    return df


# ── 4. Time-Series Plot ───────────────────────────
def plot_timeseries(df: pd.DataFrame, n_catchments: int = 4):
    if "date" not in df.columns or TARGET_COL not in df.columns:
        print("  [Skip] Missing date or cfs column for time-series plot.")
        return
    catchments = df["catchment_id"].unique()[:n_catchments]
    fig, axes = plt.subplots(len(catchments), 1,
                             figsize=(16, 3 * len(catchments)), sharex=False)
    if len(catchments) == 1:
        axes = [axes]
    for ax, cid in zip(axes, catchments):
        sub = df[df["catchment_id"] == cid].set_index("date")
        ax.plot(sub.index, sub[TARGET_COL], linewidth=0.7, color="#1565C0")
        if "is_flood" in sub.columns:
            floods = sub[sub["is_flood"] == 1]
            ax.scatter(floods.index, floods[TARGET_COL], color="red", s=4, zorder=3, label="Flood")
        ax.set_title(f"Catchment: {cid}", fontsize=10)
        ax.set_ylabel("Discharge (cfs)")
    axes[0].legend(loc="upper right")
    plt.suptitle("Streamflow Time Series", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_timeseries.png", bbox_inches="tight")
    plt.close()
    print("  Time-series plot saved.")


# ── 5. Rainfall vs Discharge ──────────────────────
def plot_rainfall_discharge(df: pd.DataFrame, rain_col: str = "precip"):
    if rain_col not in df.columns:
        print(f"  [Skip] Rainfall plot — column '{rain_col}' not found.")
        return
    if TARGET_COL not in df.columns:
        print(f"  [Skip] Rainfall plot — column '{TARGET_COL}' not found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Rainfall vs Discharge", fontsize=13, fontweight="bold")

    sample = df.sample(min(5000, len(df)), random_state=42)
    color = ["#F44336" if f else "#90CAF9" for f in sample.get("is_flood", [0] * len(sample))]
    axes[0].scatter(sample[rain_col], sample[TARGET_COL], c=color, s=5, alpha=0.5)
    axes[0].set_xlabel(rain_col)
    axes[0].set_ylabel(TARGET_COL)
    axes[0].set_title("Scatter (red = flood)")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    num_cols = [c for c in num_cols if c not in ["is_flood"]][:10]
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, ax=axes[1], annot=True, fmt=".2f",
                cmap="coolwarm", linewidths=0.4, cbar_kws={"shrink": 0.8})
    axes[1].set_title("Feature Correlation Matrix")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_rainfall_discharge.png")
    plt.close()
    print("  Rainfall-discharge plot saved.")


# ── 6. Seasonal Analysis ──────────────────────────
def plot_seasonal(df: pd.DataFrame):
    if "date" not in df.columns or TARGET_COL not in df.columns:
        print("  [Skip] Seasonal plot — missing date or cfs column.")
        return
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn"
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Seasonal Patterns", fontsize=13, fontweight="bold")

    monthly_mean = df.groupby("month")[TARGET_COL].mean()
    axes[0].bar(monthly_mean.index, monthly_mean.values,
                color=sns.color_palette("Blues_d", 12))
    axes[0].set_xticks(range(1, 13))
    axes[0].set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                              "Jul","Aug","Sep","Oct","Nov","Dec"])
    axes[0].set_title("Mean Discharge by Month")
    axes[0].set_ylabel("Mean Discharge")

    season_order = ["Winter", "Spring", "Summer", "Autumn"]
    sns.boxplot(data=df, x="season", y=TARGET_COL, order=season_order,
                palette="Set2", ax=axes[1], showfliers=False)
    axes[1].set_title("Discharge Distribution by Season")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_seasonal.png")
    plt.close()
    print("  Seasonal plot saved.")


# ── 7. Missing Value Heatmap ──────────────────────
def plot_missing(df: pd.DataFrame, miss: pd.Series):
    cols_with_miss = miss[miss > 0].index.tolist()
    if not cols_with_miss:
        print("  No missing values — skipping missing value heatmap.")
        return
    sample = df[cols_with_miss].sample(min(500, len(df)), random_state=42)
    plt.figure(figsize=(max(8, len(cols_with_miss)), 5))
    sns.heatmap(sample.isnull(), cbar=False, yticklabels=False, cmap="viridis")
    plt.title("Missing Value Pattern (sample of 500 rows)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_missing_values.png")
    plt.close()
    print("  Missing value heatmap saved.")


# ── Main ──────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  Flash Flood EDA  (FlowDB Sample via Kaggle)")
    print("=" * 50)

    df = load_flowdb()
    miss = basic_summary(df)

    print("\n[1/5] Plotting target distribution...")
    df = plot_target_distribution(df, TARGET_COL)
    print("[2/5] Plotting time series...")
    plot_timeseries(df)
    print("[3/5] Plotting rainfall vs discharge...")
    plot_rainfall_discharge(df, rain_col="precip")
    print("[4/5] Plotting seasonal patterns...")
    plot_seasonal(df)
    print("[5/5] Plotting missing values...")
    plot_missing(df, miss)

    # Coerce any object columns that contain mixed numeric/string data (e.g. 'M' sentinels)
    for col in df.select_dtypes(include="object").columns:
        if col in ("date", "catchment_id", "valid", "hour_updated", "skyc1", "station_id"):
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        # Only replace if the conversion kept ≥50% of non-null values
        if converted.notna().sum() >= df[col].notna().sum() * 0.5:
            df[col] = converted

    out_parquet = f"{OUTPUT_DIR}/flowdb_raw_with_labels.parquet"
    df.to_parquet(out_parquet, index=False)
    print(f"\nDone! All plots saved to '{OUTPUT_DIR}/'")
    print(f"Labeled data saved to '{out_parquet}'")