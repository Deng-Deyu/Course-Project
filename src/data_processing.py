"""
data_processing.py
------------------
Complete data loading, cleaning, alignment, and feature engineering pipeline
for the Kaggle Solar Power Generation Dataset (Plant 1).

Pipeline steps:
  1. Load raw generation and weather sensor CSVs.
  2. Parse and align DATE_TIME timestamps.
  3. Aggregate generation data to plant-level per timestamp
     (the raw file is inverter-level with multiple SOURCE_KEY per timestamp).
  4. Merge generation and weather on aligned timestamp.
  5. Distinguish normal nocturnal zero-generation from anomalous daytime zeros.
  6. Impute or drop remaining missing values.
  7. Engineer temporal, environmental, and lag features.
  8. Split into train / validation / test sets (chronologically).
"""

import os
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
GEN_PATH = os.path.join(DATA_DIR, "Plant_1_Generation_Data.csv")
WEATHER_PATH = os.path.join(DATA_DIR, "Plant_1_Weather_Sensor_Data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


# ---------------------------------------------------------------------------
# Step 1: Load raw data
# ---------------------------------------------------------------------------
def load_raw(gen_path: str = GEN_PATH, weather_path: str = WEATHER_PATH):
    logger.info("Loading raw generation data from %s", gen_path)
    df_gen = pd.read_csv(gen_path)
    logger.info("  Shape: %s", df_gen.shape)

    logger.info("Loading raw weather data from %s", weather_path)
    df_weather = pd.read_csv(weather_path)
    logger.info("  Shape: %s", df_weather.shape)

    return df_gen, df_weather


# ---------------------------------------------------------------------------
# Step 2: Parse timestamps
# ---------------------------------------------------------------------------
def _parse_timestamps(df: pd.DataFrame, col: str = "DATE_TIME") -> pd.DataFrame:
    """
    Try multiple format strings to handle the mixed date formats present
    in the Kaggle dataset ('15-05-2020 00:00' vs '2020-05-15 00:00').
    """
    df = df.copy()
    parsed = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    n_failed = parsed.isna().sum()
    if n_failed > 0:
        logger.warning("%d timestamps could not be parsed and will be dropped.", n_failed)
        df = df[~parsed.isna()].copy()
        parsed = parsed[~parsed.isna()]
    df[col] = parsed
    return df


# ---------------------------------------------------------------------------
# Step 3: Aggregate inverter-level generation to plant-level
# ---------------------------------------------------------------------------
def aggregate_generation(df_gen: pd.DataFrame) -> pd.DataFrame:
    """
    The generation CSV has one row per inverter (SOURCE_KEY) per timestamp.
    Aggregate AC_POWER and DC_POWER to plant level by summing over inverters.
    DAILY_YIELD and TOTAL_YIELD are averaged across inverters (they track
    per-unit yield, so the mean is the representative plant-level value).
    """
    logger.info("Aggregating inverter-level generation to plant-level per timestamp.")
    agg = df_gen.groupby("DATE_TIME").agg(
        AC_POWER=("AC_POWER", "sum"),
        DC_POWER=("DC_POWER", "sum"),
        DAILY_YIELD=("DAILY_YIELD", "mean"),
        TOTAL_YIELD=("TOTAL_YIELD", "mean"),
    ).reset_index()
    logger.info("  Aggregated shape: %s", agg.shape)
    return agg


# ---------------------------------------------------------------------------
# Step 4: Merge generation and weather data
# ---------------------------------------------------------------------------
def merge_data(df_gen_agg: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    """
    Both datasets use 15-minute intervals with the same DATE_TIME field.
    Inner join to keep only timestamps present in both sources.
    Weather data has one unique sensor per plant, so no aggregation needed.
    """
    logger.info("Merging generation and weather data on DATE_TIME.")
    # Drop PLANT_ID and SOURCE_KEY from weather (redundant after merge)
    cols_drop = [c for c in ["PLANT_ID", "SOURCE_KEY"] if c in df_weather.columns]
    df_weather_clean = df_weather.drop(columns=cols_drop).drop_duplicates(subset="DATE_TIME")

    merged = pd.merge(df_gen_agg, df_weather_clean, on="DATE_TIME", how="inner")
    merged = merged.sort_values("DATE_TIME").reset_index(drop=True)
    logger.info("  Merged shape: %s", merged.shape)
    return merged


# ---------------------------------------------------------------------------
# Step 5: Anomaly detection — distinguish nocturnal zeros from equipment faults
# ---------------------------------------------------------------------------
def flag_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nocturnal zeros: AC_POWER == 0 AND IRRADIATION == 0 AND hour is outside
    the typical solar generation window (< 06:00 or > 19:00). These are NORMAL.

    Anomalous zeros: AC_POWER == 0 BUT IRRADIATION > threshold during daylight.
    This suggests inverter failure or data dropout — flag for removal.
    """
    df = df.copy()
    hour = df["DATE_TIME"].dt.hour

    # Daytime is approximately 06:00 – 18:30 for the Indian dataset region
    is_daytime = (hour >= 6) & (hour <= 18)

    irr_threshold = 0.005  # W/m² — essentially zero irradiation

    # Normal night zeros: not daytime and irradiation near zero
    normal_night = (~is_daytime) & (df["IRRADIATION"] <= irr_threshold)

    # Anomalous: daytime with zero AC_POWER despite non-trivial irradiation
    anomalous = is_daytime & (df["AC_POWER"] == 0) & (df["IRRADIATION"] > irr_threshold)

    df["is_night"] = (~is_daytime).astype(int)
    df["is_anomalous"] = anomalous.astype(int)

    n_anomalous = anomalous.sum()
    n_night = normal_night.sum()
    logger.info("  Normal night-zero records: %d", n_night)
    logger.info("  Anomalous daytime-zero records flagged for removal: %d", n_anomalous)

    return df


# ---------------------------------------------------------------------------
# Step 6: Clean missing values and duplicates
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning data: removing duplicates, anomalies, and missing values.")
    n_before = len(df)

    # Remove exact duplicate timestamps
    df = df.drop_duplicates(subset="DATE_TIME").sort_values("DATE_TIME").reset_index(drop=True)

    # Remove flagged anomalous records (equipment fault zeros during daylight)
    df = df[df["is_anomalous"] == 0].reset_index(drop=True)

    # Forward-fill minor gaps (up to 2 consecutive timestamps = 30 min)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].ffill(limit=2)

    # Drop remaining rows with missing critical features
    critical = ["AC_POWER", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
    df = df.dropna(subset=critical).reset_index(drop=True)

    logger.info("  Records before cleaning: %d | after: %d", n_before, len(df))
    return df


# ---------------------------------------------------------------------------
# Step 7: Feature engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame, lag_steps: int = 4) -> pd.DataFrame:
    """
    Temporal features:
      - hour, minute, day_of_year, month, weekday (integer)
      - hour_sin / hour_cos  : cyclic encoding of hour of day
      - is_daytime            : binary flag

    Environmental features (pass-through):
      - AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, IRRADIATION

    Lag and rolling features (over AC_POWER):
      - ac_lag_{k}  for k in 1..lag_steps  (k * 15-min history)
      - ac_roll_mean_4   : 1-hour rolling mean
      - ac_roll_mean_8   : 2-hour rolling mean
      - ac_roll_std_4    : 1-hour rolling std (volatility proxy)

    Interaction features:
      - irr_x_module_temp : IRRADIATION * MODULE_TEMPERATURE
    """
    df = df.copy()
    dt = df["DATE_TIME"]

    # --- Temporal ---
    df["hour"] = dt.dt.hour
    df["minute"] = dt.dt.minute
    df["day_of_year"] = dt.dt.dayofyear
    df["month"] = dt.dt.month
    df["weekday"] = dt.dt.weekday

    # Cyclic encoding prevents the model from treating hour 23 as far from hour 0
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["is_daytime"] = df["is_night"].apply(lambda x: 1 - x)

    # --- Lag features ---
    for k in range(1, lag_steps + 1):
        df[f"ac_lag_{k}"] = df["AC_POWER"].shift(k)

    # --- Rolling statistics ---
    df["ac_roll_mean_4"] = df["AC_POWER"].shift(1).rolling(4).mean()
    df["ac_roll_mean_8"] = df["AC_POWER"].shift(1).rolling(8).mean()
    df["ac_roll_std_4"] = df["AC_POWER"].shift(1).rolling(4).std().fillna(0)

    # --- Interaction ---
    df["irr_x_module_temp"] = df["IRRADIATION"] * df["MODULE_TEMPERATURE"]

    # Drop rows that have NaN from lag / rolling (initial window)
    df = df.dropna().reset_index(drop=True)

    logger.info("Feature engineering complete. Shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Step 8: Train / validation / test split (chronological, no shuffle)
# ---------------------------------------------------------------------------
def split_data(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15):
    """
    Chronological split: train | val | test
    Ratios default to 70% / 15% / 15%.
    Returns: df_train, df_val, df_test
    """
    n = len(df)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test

    df_train = df.iloc[:n_train].copy()
    df_val = df.iloc[n_train: n_train + n_val].copy()
    df_test = df.iloc[n_train + n_val:].copy()

    logger.info(
        "Train/Val/Test split: %d / %d / %d records (%.0f%% / %.0f%% / %.0f%%)",
        len(df_train), len(df_val), len(df_test),
        100 * len(df_train) / n, 100 * len(df_val) / n, 100 * len(df_test) / n,
    )
    return df_train, df_val, df_test


# ---------------------------------------------------------------------------
# Feature column definitions (shared by training and app)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "hour", "minute", "day_of_year", "month", "weekday",
    "hour_sin", "hour_cos", "is_daytime",
    "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION",
    "ac_lag_1", "ac_lag_2", "ac_lag_3", "ac_lag_4",
    "ac_roll_mean_4", "ac_roll_mean_8", "ac_roll_std_4",
    "irr_x_module_temp",
]

TARGET_COL = "AC_POWER"


# ---------------------------------------------------------------------------
# Master pipeline function
# ---------------------------------------------------------------------------
def build_dataset():
    """
    Run the full pipeline and return (df_train, df_val, df_test, df_full, scaler).
    The scaler is fitted on training features only.
    """
    df_gen_raw, df_weather_raw = load_raw()

    df_gen_raw = _parse_timestamps(df_gen_raw)
    df_weather_raw = _parse_timestamps(df_weather_raw)

    df_gen_agg = aggregate_generation(df_gen_raw)
    df_merged = merge_data(df_gen_agg, df_weather_raw)
    df_flagged = flag_anomalies(df_merged)
    df_clean = clean_data(df_flagged)
    df_features = engineer_features(df_clean)

    df_train, df_val, df_test = split_data(df_features)

    # Fit StandardScaler on training features only
    scaler = StandardScaler()
    scaler.fit(df_train[FEATURE_COLS])

    logger.info("Dataset build complete.")
    return df_train, df_val, df_test, df_features, scaler


if __name__ == "__main__":
    df_train, df_val, df_test, df_full, scaler = build_dataset()
    print("Train shape:", df_train.shape)
    print("Val shape  :", df_val.shape)
    print("Test shape :", df_test.shape)
    print("Features   :", FEATURE_COLS)
