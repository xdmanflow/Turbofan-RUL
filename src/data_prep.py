"""Data preparation for the C-MAPSS FD001 dataset.

- Loading the raw text files
- RUL label construction with piecewise-linear capping
- Feature selection (near-constant sensors removed)
- Engine-level train/validation split
- Min-max normalization fitted on training data only
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

SETTINGS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSORS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle"] + SETTINGS + SENSORS

# Near-constant under FD001's single operating condition
DROP_SENSORS = ["s1", "s5", "s10", "s16", "s18", "s19"]
KEPT_SENSORS = [s for s in SENSORS if s not in DROP_SENSORS]  # 15 sensors
FEATURES = SETTINGS + KEPT_SENSORS                            # 18 features

RUL_CAP = 125


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, : len(COLUMNS)]
    df.columns = COLUMNS
    return df


def load_fd001(data_dir="data"):
    """Return (train, test, rul_true) for FD001.

    rul_true is the true RUL at the last cycle of each test engine.
    """
    data_dir = Path(data_dir)
    train = _read(data_dir / "train_FD001.txt")
    test = _read(data_dir / "test_FD001.txt")
    rul_true = pd.read_csv(data_dir / "RUL_FD001.txt", sep=r"\s+", header=None).iloc[:, 0].to_numpy()
    return train, test, rul_true


def add_rul(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """Add the RUL target: cycles remaining until failure, capped at `cap`."""
    df = df.copy()
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = (max_cycle - df["cycle"]).clip(upper=cap)
    return df


def engine_split(df: pd.DataFrame, val_frac: float = 0.2, seed: int = 42):
    """Split by engine so no trajectory appears in both sets (prevents leakage)."""
    units = df["unit"].unique()
    rng = np.random.default_rng(seed)
    val_units = rng.choice(units, size=int(len(units) * val_frac), replace=False)
    val_mask = df["unit"].isin(val_units)
    return df[~val_mask].copy(), df[val_mask].copy()


def fit_scaler(train_df: pd.DataFrame, features=FEATURES) -> MinMaxScaler:
    """Fit min-max scaling on training data only."""
    return MinMaxScaler().fit(train_df[features])


def scale(df: pd.DataFrame, scaler: MinMaxScaler, features=FEATURES) -> pd.DataFrame:
    df = df.copy()
    df[features] = scaler.transform(df[features])
    return df


def last_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the last recorded cycle of each engine (used for test evaluation)."""
    return df.sort_values(["unit", "cycle"]).groupby("unit").tail(1)
