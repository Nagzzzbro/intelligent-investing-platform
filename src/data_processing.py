"""
Data Processing Module for NIFTY-50 Investment Intelligence Platform.

This module handles loading, cleaning, and processing stock market data
from the Kaggle NIFTY-50 datasets. It provides functions for technical
indicator calculation, return computation, ML feature engineering, and
synthetic data generation for demo/testing purposes.

Supported dataset formats:
    - rohanrao/nifty50-stock-market-data:
        Columns: Date, Symbol, Prev Close, Open, High, Low, Last, Close,
                 VWAP, Volume, Turnover, Trades, Deliverable Volume, %Deliverble
    - stoicstatic/india-stock-data-nse-1990-2020:
        Columns: Date, Symbol, Open, High, Low, Close, Volume, etc.

Author: NIFTY-50 Intelligence Team
"""

import os
import glob
import logging
from typing import Tuple, Optional, List, Dict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Primary dataset columns (rohanrao)
PRIMARY_COLS = [
    "Date", "Symbol", "Prev Close", "Open", "High", "Low",
    "Last", "Close", "VWAP", "Volume", "Turnover", "Trades",
    "Deliverable Volume", "%Deliverble",
]

# Minimal required columns after normalisation
REQUIRED_COLS = ["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]

# Fictional stock symbols for demo data
SAMPLE_SYMBOLS = ["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON"]


# ===================================================================
# 1. Data Loading
# ===================================================================

def load_all_stocks(data_dir: str = "data/") -> pd.DataFrame:
    """Load all CSV files from *data_dir* and concatenate into one DataFrame.

    The function auto-detects the dataset format by inspecting column names
    and normalises every file to a common schema with at least the columns
    in ``REQUIRED_COLS``.

    Args:
        data_dir: Path to directory containing per-stock CSV files.

    Returns:
        pd.DataFrame: Concatenated DataFrame with a ``Symbol`` column
        identifying each stock.

    Raises:
        FileNotFoundError: If *data_dir* does not exist.
        ValueError: If no valid CSV files are found.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")

    frames: List[pd.DataFrame] = []
    for fpath in csv_files:
        try:
            df = pd.read_csv(fpath)
            df = _normalise_columns(df, fpath)
            frames.append(df)
            logger.info("Loaded %s — %d rows", os.path.basename(fpath), len(df))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s: %s", fpath, exc)

    if not frames:
        raise ValueError("No valid stock data could be loaded")

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "Total loaded: %d rows across %d stocks",
        len(combined),
        combined["Symbol"].nunique(),
    )
    return combined


def _normalise_columns(df: pd.DataFrame, filepath: str) -> pd.DataFrame:
    """Normalise column names across different dataset formats.

    Detects the dataset format from column names and renames / selects
    the relevant columns so the output always contains ``REQUIRED_COLS``.

    Args:
        df: Raw DataFrame loaded from a CSV.
        filepath: Path to the source file (used to infer Symbol if missing).

    Returns:
        pd.DataFrame with normalised column names.
    """
    # Standardise: strip whitespace and title-case
    df.columns = df.columns.str.strip()

    # ---- Check for primary dataset (rohanrao) ----
    if "Prev Close" in df.columns or "VWAP" in df.columns:
        logger.debug("Detected primary (rohanrao) format for %s", filepath)
        # These files typically already have Symbol
        if "Symbol" not in df.columns:
            # Infer from filename
            symbol = os.path.splitext(os.path.basename(filepath))[0].upper()
            df["Symbol"] = symbol

    # ---- Check for secondary dataset (stoicstatic) ----
    elif {"Date", "Open", "High", "Low", "Close"}.issubset(df.columns):
        logger.debug("Detected secondary (stoicstatic) format for %s", filepath)
        if "Symbol" not in df.columns:
            symbol = os.path.splitext(os.path.basename(filepath))[0].upper()
            df["Symbol"] = symbol

    else:
        # Attempt a case-insensitive fallback mapping
        col_map = {c: c.title() for c in df.columns}
        df.rename(columns=col_map, inplace=True)
        if "Symbol" not in df.columns:
            symbol = os.path.splitext(os.path.basename(filepath))[0].upper()
            df["Symbol"] = symbol

    # Ensure Volume column exists (may be named differently)
    for alias in ["Volume", "Volumes", "Total Traded Quantity"]:
        if alias in df.columns and "Volume" not in df.columns:
            df.rename(columns={alias: "Volume"}, inplace=True)
            break

    # Verify minimum required columns
    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns {missing} in {filepath}. "
            f"Available: {list(df.columns)}"
        )

    return df


# ===================================================================
# 2. Data Cleaning
# ===================================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, sort, forward-fill missing values, and drop critical NaN rows.

    Args:
        df: Raw DataFrame with at least ``REQUIRED_COLS``.

    Returns:
        pd.DataFrame: Cleaned DataFrame sorted by (Symbol, Date).
    """
    df = df.copy()

    # Parse dates -------------------------------------------------------
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=False)
    pre_len = len(df)
    df.dropna(subset=["Date"], inplace=True)
    dropped = pre_len - len(df)
    if dropped:
        logger.warning("Dropped %d rows with unparseable dates", dropped)

    # Sort ---------------------------------------------------------------
    df.sort_values(["Symbol", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Forward-fill within each stock ------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df.groupby("Symbol")[numeric_cols].ffill()

    # Drop rows missing critical OHLC values ----------------------------
    critical = ["Open", "High", "Low", "Close"]
    pre_len = len(df)
    df.dropna(subset=critical, inplace=True)
    dropped = pre_len - len(df)
    if dropped:
        logger.warning("Dropped %d rows with NaN in OHLC columns", dropped)

    logger.info("Cleaned data: %d rows, %d stocks", len(df), df["Symbol"].nunique())
    return df


# ===================================================================
# 3. Technical Indicators
# ===================================================================

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate and append technical indicators for each stock.

    Indicators added:
        - SMA_20, SMA_50, SMA_200: Simple Moving Averages
        - EMA_12, EMA_26: Exponential Moving Averages
        - RSI_14: 14-period Relative Strength Index
        - MACD, MACD_Signal, MACD_Histogram
        - BB_Upper, BB_Middle, BB_Lower: Bollinger Bands (20-day, 2σ)
        - ATR_14: 14-period Average True Range
        - OBV: On-Balance Volume

    Args:
        df: DataFrame with columns Date, Symbol, Open, High, Low, Close, Volume.
            Must be sorted by (Symbol, Date).

    Returns:
        pd.DataFrame: Input DataFrame augmented with indicator columns.
    """
    df = df.copy()

    # Process each stock individually.
    # NOTE: pandas groupby drops the key column from each group slice,
    # so we must re-insert 'Symbol' before calling the helper.
    processed_frames = []
    for symbol, group in df.groupby("Symbol", sort=False):
        group = group.copy()
        group["Symbol"] = symbol          # re-insert dropped key
        group = _compute_indicators_for_stock(group)
        processed_frames.append(group)

    if processed_frames:
        df = pd.concat(processed_frames, ignore_index=True)
    else:
        logger.warning("No stock groups found to process")
        return df

    symbol_count = df["Symbol"].nunique() if "Symbol" in df.columns else 0
    logger.info("Added technical indicators for %d stocks", symbol_count)
    return df


def _compute_indicators_for_stock(g: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators for a single stock group.

    This is an internal helper called by :func:`add_technical_indicators`.
    """
    close = g["Close"].astype(float)
    high = g["High"].astype(float)
    low = g["Low"].astype(float)
    volume = g["Volume"].astype(float)

    # ---- Simple Moving Averages ----------------------------------------
    g["SMA_20"] = close.rolling(window=20, min_periods=1).mean()
    g["SMA_50"] = close.rolling(window=50, min_periods=1).mean()
    g["SMA_200"] = close.rolling(window=200, min_periods=1).mean()

    # ---- Exponential Moving Averages -----------------------------------
    g["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    g["EMA_26"] = close.ewm(span=26, adjust=False).mean()

    # ---- RSI (14-period) -----------------------------------------------
    g["RSI_14"] = _compute_rsi(close, period=14)

    # ---- MACD ----------------------------------------------------------
    g["MACD"] = g["EMA_12"] - g["EMA_26"]
    g["MACD_Signal"] = g["MACD"].ewm(span=9, adjust=False).mean()
    g["MACD_Histogram"] = g["MACD"] - g["MACD_Signal"]

    # ---- Bollinger Bands (20-day, 2σ) ----------------------------------
    bb_mid = close.rolling(window=20, min_periods=1).mean()
    bb_std = close.rolling(window=20, min_periods=1).std()
    g["BB_Middle"] = bb_mid
    g["BB_Upper"] = bb_mid + 2 * bb_std
    g["BB_Lower"] = bb_mid - 2 * bb_std

    # ---- Average True Range (14-period) --------------------------------
    g["ATR_14"] = _compute_atr(high, low, close, period=14)

    # ---- On-Balance Volume ---------------------------------------------
    g["OBV"] = _compute_obv(close, volume)

    return g


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI).

    Uses the standard Wilder smoothing (exponential moving average with
    ``alpha = 1 / period``).

    Args:
        close: Series of closing prices.
        period: Look-back period (default 14).

    Returns:
        pd.Series of RSI values (0–100).
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Compute 14-period Average True Range (ATR).

    True Range = max(H-L, |H-prevC|, |L-prevC|).

    Args:
        high, low, close: Price series.
        period: Smoothing period.

    Returns:
        pd.Series of ATR values.
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period, min_periods=1).mean()
    return atr


def _compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute On-Balance Volume (OBV).

    OBV accumulates volume on up-days and subtracts on down-days.

    Args:
        close: Closing prices.
        volume: Trading volume.

    Returns:
        pd.Series of OBV values.
    """
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    return obv


# ===================================================================
# 4. Returns Calculation
# ===================================================================

def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add return columns grouped by stock symbol.

    Columns added:
        - ``Daily_Return``: Simple daily percentage return.
        - ``Log_Return``: Logarithmic daily return.
        - ``Weekly_Return``: 5-day rolling return.
        - ``Monthly_Return``: 21-day rolling return.

    Args:
        df: DataFrame with at least ``Close`` and ``Symbol`` columns.

    Returns:
        pd.DataFrame with return columns appended.
    """
    df = df.copy()

    def _returns(g: pd.DataFrame) -> pd.DataFrame:
        close = g["Close"].astype(float)
        g["Daily_Return"] = close.pct_change()
        g["Log_Return"] = np.log(close / close.shift(1))
        g["Weekly_Return"] = close.pct_change(periods=5)
        g["Monthly_Return"] = close.pct_change(periods=21)
        return g

    processed = []
    for symbol, g in df.groupby("Symbol", sort=False):
        g = g.copy()
        g["Symbol"] = symbol              # re-insert dropped key
        processed.append(_returns(g))
    if processed:
        df = pd.concat(processed, ignore_index=True)
    symbol_count = df["Symbol"].nunique() if "Symbol" in df.columns else 0
    logger.info("Calculated returns for %d stocks", symbol_count)
    return df


# ===================================================================
# 5. ML Feature Engineering
# ===================================================================

def prepare_ml_features(
    df: pd.DataFrame,
    target_col: str = "Direction",
    lookback: int = 5,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build a feature matrix and binary classification target.

    Features include:
        * Lagged daily returns (1 … *lookback* days).
        * Lagged volume percentage changes (1 … *lookback* days).
        * All technical indicator columns present in *df*.

    The target is 1 if next-day return > 0 (up), else 0 (down/flat).

    Args:
        df: DataFrame **already** enriched with technical indicators and
            returns (call ``add_technical_indicators`` and
            ``calculate_returns`` first).
        target_col: Name for the target column (default ``'Direction'``).
        lookback: Number of lagged periods to include (default 5).

    Returns:
        Tuple of ``(X, y)`` where *X* is a DataFrame of features and
        *y* is a binary Series.

    Raises:
        ValueError: If required columns are missing.
    """
    df = df.copy()

    if "Daily_Return" not in df.columns:
        raise ValueError("Daily_Return column missing — call calculate_returns first")

    # ---- Indicator columns (auto-detected) -----------------------------
    indicator_cols = [
        c for c in df.columns
        if c.startswith(("SMA_", "EMA_", "RSI_", "MACD", "BB_", "ATR_", "OBV"))
    ]

    # ---- Lagged features -----------------------------------------------
    for lag in range(1, lookback + 1):
        df[f"Return_Lag_{lag}"] = df.groupby("Symbol")["Daily_Return"].shift(lag)
        if "Volume" in df.columns:
            vol_pct = df.groupby("Symbol")["Volume"].pct_change()
            df[f"VolChg_Lag_{lag}"] = vol_pct.groupby(df["Symbol"]).shift(lag)

    lag_cols = [c for c in df.columns if "Lag_" in c]

    # ---- Target --------------------------------------------------------
    df[target_col] = (
        df.groupby("Symbol")["Daily_Return"]
        .shift(-1)  # next-day return
        .apply(lambda x: 1 if x > 0 else 0)
    )

    # ---- Assemble feature matrix ---------------------------------------
    feature_cols = indicator_cols + lag_cols + ["Daily_Return", "Log_Return"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Drop NaN rows introduced by lagging / rolling indicators
    subset = df[feature_cols + [target_col]].dropna()

    X = subset[feature_cols].astype(float)
    y = subset[target_col].astype(int)

    logger.info("Feature matrix: %d samples × %d features", X.shape[0], X.shape[1])
    return X, y


# ===================================================================
# 6. Synthetic / Sample Data Generator
# ===================================================================

def get_sample_data(
    n_days: int = 500,
    symbols: Optional[List[str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data for demo and testing.

    Creates realistic-looking stock data with:
        * Geometric Brownian Motion base price paths.
        * Trend and mean-reversion regimes.
        * Volatility clustering.
        * Volume patterns correlated with price moves.

    Each stock has a distinct character (growth, value, volatile, etc.)
    so that portfolio/risk analytics produce meaningful results.

    Args:
        n_days: Number of trading days to generate (default 500).
        symbols: List of ticker symbols (default ``SAMPLE_SYMBOLS``).
        seed: Random seed for reproducibility (default 42).

    Returns:
        pd.DataFrame with columns: Date, Symbol, Open, High, Low, Close, Volume.
    """
    rng = np.random.default_rng(seed)
    symbols = symbols or SAMPLE_SYMBOLS

    # Stock personality profiles: (annual_drift, annual_vol, base_volume)
    profiles: Dict[str, Tuple[float, float, int]] = {
        "ALPHA":   (0.18,  0.22, 5_000_000),   # Steady large-cap growth
        "BETA":    (0.12,  0.30, 3_000_000),    # Moderate growth, higher vol
        "GAMMA":   (0.25,  0.40, 2_000_000),    # Aggressive growth, high vol
        "DELTA":   (-0.05, 0.28, 4_000_000),    # Declining / value trap
        "EPSILON": (0.08,  0.18, 8_000_000),    # Stable blue-chip
    }

    # Date range: business days ending today
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)

    all_frames: List[pd.DataFrame] = []
    for symbol in symbols:
        drift, vol, base_vol = profiles.get(
            symbol, (0.10, 0.25, 3_000_000)
        )

        # Daily parameters
        mu_daily = drift / 252
        sigma_daily = vol / np.sqrt(252)

        # --- Price path via GBM with mean-reversion overlay ----
        log_returns = rng.normal(mu_daily, sigma_daily, n_days)

        # Add volatility clustering via GARCH-like effect
        vol_cluster = np.ones(n_days)
        for t in range(1, n_days):
            vol_cluster[t] = 0.9 * vol_cluster[t - 1] + 0.1 * abs(log_returns[t - 1]) / sigma_daily
        log_returns *= vol_cluster

        # Add mild mean-reversion (Ornstein-Uhlenbeck nudge)
        cum_returns = np.cumsum(log_returns)
        mean_level = np.convolve(cum_returns, np.ones(60) / 60, mode="same")
        reversion = -0.02 * (cum_returns - mean_level)
        log_returns += reversion

        # Build close prices
        start_price = rng.uniform(200, 3000)
        close_prices = start_price * np.exp(np.cumsum(log_returns))

        # OHLC from close
        daily_range = close_prices * rng.uniform(0.005, 0.035, n_days)
        open_prices = close_prices + rng.normal(0, 0.3, n_days) * daily_range
        high_prices = np.maximum(open_prices, close_prices) + rng.uniform(0, 1, n_days) * daily_range
        low_prices = np.minimum(open_prices, close_prices) - rng.uniform(0, 1, n_days) * daily_range

        # Ensure OHLC consistency
        high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
        low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
        low_prices = np.maximum(low_prices, 1.0)  # no negative prices

        # Volume: base + correlation with absolute return + noise
        abs_ret = np.abs(log_returns)
        volume = base_vol * (1.0 + 5.0 * abs_ret) * rng.uniform(0.6, 1.4, n_days)
        volume = volume.astype(int)

        # Inject a few volume spikes (earnings / events)
        spike_days = rng.choice(n_days, size=max(1, n_days // 100), replace=False)
        volume[spike_days] *= rng.integers(3, 8, size=len(spike_days))

        stock_df = pd.DataFrame(
            {
                "Date": dates[:n_days],
                "Symbol": symbol,
                "Open": np.round(open_prices, 2),
                "High": np.round(high_prices, 2),
                "Low": np.round(low_prices, 2),
                "Close": np.round(close_prices, 2),
                "Volume": volume,
            }
        )
        all_frames.append(stock_df)

    result = pd.concat(all_frames, ignore_index=True)
    logger.info(
        "Generated sample data: %d rows, %d stocks, %d days",
        len(result),
        len(symbols),
        n_days,
    )
    return result
