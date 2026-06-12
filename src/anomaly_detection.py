"""
Anomaly Detection Module for NIFTY-50 Investment Intelligence Platform.

Detects unusual market behaviour through multiple complementary methods:
    * Isolation Forest (multi-feature statistical anomalies).
    * Flash-crash detection (extreme negative returns).
    * Volume spike detection (z-score based).
    * Volatility regime classification (rolling-vol regimes).

Author: NIFTY-50 Intelligence Team
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ===================================================================
# 1. Isolation Forest Anomalies
# ===================================================================

def detect_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.02,
) -> pd.DataFrame:
    """Detect statistical anomalies using an Isolation Forest.

    Engineered features used for detection:
        - ``daily_return``: percentage daily return.
        - ``volume_change``: percentage change in volume.
        - ``high_low_spread``: ``(High - Low) / Close``.
        - ``open_close_spread``: ``(Close - Open) / Open``.

    Args:
        df: DataFrame with columns ``Close``, ``Open``, ``High``,
            ``Low``, ``Volume``.
        contamination: Expected fraction of anomalies (default 2 %).

    Returns:
        DataFrame with two new columns:
        - ``Anomaly``: -1 for anomaly, 1 for normal.
        - ``Anomaly_Score``: Isolation Forest decision-function score
          (lower ⇒ more anomalous).
    """
    df = df.copy()

    try:
        # --- Engineer features -------------------------------------------
        df["_daily_return"] = df["Close"].pct_change()
        df["_volume_change"] = df["Volume"].pct_change()
        df["_high_low_spread"] = (df["High"] - df["Low"]) / df["Close"]
        df["_open_close_spread"] = (df["Close"] - df["Open"]) / df["Open"].replace(0, np.nan)

        feature_cols = [
            "_daily_return",
            "_volume_change",
            "_high_low_spread",
            "_open_close_spread",
        ]

        # Drop rows with NaN features before fitting
        valid_mask = df[feature_cols].notna().all(axis=1)
        X = df.loc[valid_mask, feature_cols].values

        if len(X) < 10:
            logger.warning("Too few valid rows (%d) for anomaly detection", len(X))
            df["Anomaly"] = 1
            df["Anomaly_Score"] = 0.0
            return df

        # --- Fit Isolation Forest ----------------------------------------
        iso = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=200,
            n_jobs=-1,
        )
        iso.fit(X)

        df.loc[valid_mask, "Anomaly"] = iso.predict(X)
        df.loc[valid_mask, "Anomaly_Score"] = iso.decision_function(X)

        # Fill non-valid rows as normal
        df["Anomaly"] = df["Anomaly"].fillna(1).astype(int)
        df["Anomaly_Score"] = df["Anomaly_Score"].fillna(0.0)

        n_anom = int((df["Anomaly"] == -1).sum())
        logger.info("Isolation Forest: %d anomalies detected (%.1f%%)", n_anom, n_anom / len(df) * 100)

    except Exception as exc:  # noqa: BLE001
        logger.error("Anomaly detection failed: %s", exc)
        df["Anomaly"] = 1
        df["Anomaly_Score"] = 0.0

    finally:
        # Clean up temporary columns
        for c in ["_daily_return", "_volume_change", "_high_low_spread", "_open_close_spread"]:
            df.drop(columns=c, errors="ignore", inplace=True)

    return df


# ===================================================================
# 2. Flash Crash Detection
# ===================================================================

def detect_flash_crashes(
    df: pd.DataFrame,
    threshold: float = -0.05,
) -> pd.DataFrame:
    """Flag trading days with extreme negative returns.

    Args:
        df: DataFrame with a ``Close`` column (used to compute returns
            if ``Daily_Return`` is not already present).
        threshold: Return threshold below which a day is considered a
            flash crash (default -5 %).

    Returns:
        DataFrame with a new boolean column ``Flash_Crash``.
    """
    df = df.copy()

    if "Daily_Return" not in df.columns:
        df["Daily_Return"] = df["Close"].pct_change()

    df["Flash_Crash"] = df["Daily_Return"] < threshold

    n_crashes = int(df["Flash_Crash"].sum())
    logger.info("Flash crashes detected (threshold %.1f%%): %d days", threshold * 100, n_crashes)
    return df


# ===================================================================
# 3. Volume Spike Detection
# ===================================================================

def detect_volume_spikes(
    df: pd.DataFrame,
    z_threshold: float = 3.0,
    window: int = 60,
) -> pd.DataFrame:
    """Flag days with abnormally high or low volume.

    Uses a rolling z-score to identify volume that deviates
    significantly from the recent average.

    Args:
        df: DataFrame with a ``Volume`` column.
        z_threshold: z-score threshold for flagging (default 3.0).
        window: Rolling window for mean/std (default 60 trading days).

    Returns:
        DataFrame with new columns ``Volume_Spike`` (bool) and
        ``Volume_ZScore`` (float).
    """
    df = df.copy()

    vol = df["Volume"].astype(float)
    rolling_mean = vol.rolling(window=window, min_periods=10).mean()
    rolling_std = vol.rolling(window=window, min_periods=10).std()

    # Avoid division by zero
    z_scores = (vol - rolling_mean) / rolling_std.replace(0, np.nan)

    df["Volume_ZScore"] = z_scores
    df["Volume_Spike"] = z_scores.abs() > z_threshold

    n_spikes = int(df["Volume_Spike"].sum())
    logger.info("Volume spikes detected (|z| > %.1f): %d days", z_threshold, n_spikes)
    return df


# ===================================================================
# 4. Regime Change Detection
# ===================================================================

def detect_regime_changes(
    df: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """Classify market regimes based on rolling volatility.

    Regime labels:
        - ``low_vol``: rolling vol < (mean − 1 std).
        - ``high_vol``: rolling vol > (mean + 1 std).
        - ``normal``: everything in between.

    Args:
        df: DataFrame with ``Close`` column.
        window: Look-back window for rolling volatility (default 60).

    Returns:
        DataFrame with new columns ``Rolling_Volatility`` and ``Regime``.
    """
    df = df.copy()

    if "Daily_Return" not in df.columns:
        df["Daily_Return"] = df["Close"].pct_change()

    # Annualised rolling volatility
    rolling_vol = df["Daily_Return"].rolling(window=window, min_periods=10).std() * np.sqrt(252)
    df["Rolling_Volatility"] = rolling_vol

    vol_mean = rolling_vol.mean()
    vol_std = rolling_vol.std()

    conditions = [
        rolling_vol < (vol_mean - vol_std),
        rolling_vol > (vol_mean + vol_std),
    ]
    choices = ["low_vol", "high_vol"]
    df["Regime"] = np.select(conditions, choices, default="normal")

    # Log regime distribution
    regime_counts = df["Regime"].value_counts().to_dict()
    logger.info("Regime distribution: %s", regime_counts)
    return df


# ===================================================================
# 5. Anomaly Summary
# ===================================================================

def get_anomaly_summary(df: pd.DataFrame) -> Dict:
    """Aggregate summary statistics for all detected anomalies.

    Gracefully handles missing columns — if a particular anomaly type
    was not computed, its summary fields are set to ``None``.

    Args:
        df: DataFrame that has been processed by one or more of the
            detection functions above.

    Returns:
        Dict with keys: ``total_anomalies``, ``anomaly_pct``,
        ``total_flash_crashes``, ``total_volume_spikes``,
        ``regime_distribution``, ``most_recent_anomaly_date``,
        ``anomaly_dates``.
    """
    summary: Dict = {}
    total = len(df)

    # --- Isolation Forest anomalies ------------------------------------
    if "Anomaly" in df.columns:
        anomalies = df[df["Anomaly"] == -1]
        summary["total_anomalies"] = len(anomalies)
        summary["anomaly_pct"] = len(anomalies) / total * 100 if total else 0.0

        if "Date" in df.columns and not anomalies.empty:
            anomaly_dates = anomalies["Date"].tolist()
            summary["anomaly_dates"] = [str(d) for d in anomaly_dates]
            summary["most_recent_anomaly_date"] = str(anomaly_dates[-1])
        else:
            summary["anomaly_dates"] = []
            summary["most_recent_anomaly_date"] = None
    else:
        summary["total_anomalies"] = None
        summary["anomaly_pct"] = None
        summary["anomaly_dates"] = []
        summary["most_recent_anomaly_date"] = None

    # --- Flash crashes --------------------------------------------------
    if "Flash_Crash" in df.columns:
        summary["total_flash_crashes"] = int(df["Flash_Crash"].sum())
    else:
        summary["total_flash_crashes"] = None

    # --- Volume spikes --------------------------------------------------
    if "Volume_Spike" in df.columns:
        summary["total_volume_spikes"] = int(df["Volume_Spike"].sum())
    else:
        summary["total_volume_spikes"] = None

    # --- Regime distribution --------------------------------------------
    if "Regime" in df.columns:
        summary["regime_distribution"] = df["Regime"].value_counts().to_dict()
    else:
        summary["regime_distribution"] = None

    return summary
