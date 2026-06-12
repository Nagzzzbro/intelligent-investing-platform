"""
Risk Assessment Module for NIFTY-50 Investment Intelligence Platform.

Provides a comprehensive suite of risk metrics commonly used in
quantitative finance, including volatility, drawdown analysis,
Value-at-Risk, CAPM beta, and ratio-based performance measures.

Key functions:
    comprehensive_risk_report — all metrics in one dict.
    compare_stocks_risk — tabular comparison across multiple stocks.

Author: NIFTY-50 Intelligence Team
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module-level logger & constants
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

TRADING_DAYS: int = 252


# ===================================================================
# 1. Annualised Volatility
# ===================================================================

def annualized_volatility(returns: pd.Series) -> float:
    """Annualised standard deviation of daily returns.

    Args:
        returns: Daily return series.

    Returns:
        Annualised volatility as a float.  Returns 0.0 for empty input.
    """
    if returns.empty or len(returns) < 2:
        return 0.0
    return float(returns.std() * np.sqrt(TRADING_DAYS))


# ===================================================================
# 2. Sharpe Ratio
# ===================================================================

def sharpe_ratio(returns: pd.Series, rf_rate: float = 0.06) -> float:
    """Annualised Sharpe ratio.

    Args:
        returns: Daily return series.
        rf_rate: Annual risk-free rate (default 6 % for India).

    Returns:
        Sharpe ratio.  Returns 0.0 when volatility is zero.
    """
    ann_vol = annualized_volatility(returns)
    if ann_vol == 0:
        return 0.0
    ann_ret = float(returns.mean()) * TRADING_DAYS
    return (ann_ret - rf_rate) / ann_vol


# ===================================================================
# 3. Sortino Ratio
# ===================================================================

def sortino_ratio(returns: pd.Series, rf_rate: float = 0.06) -> float:
    """Sortino ratio — penalises downside risk only.

    Args:
        returns: Daily return series.
        rf_rate: Annual risk-free rate.

    Returns:
        Sortino ratio.  Returns ``np.inf`` when there is no downside
        deviation (all returns non-negative).
    """
    downside = returns[returns < 0]
    if downside.empty:
        return np.inf  # type: ignore[return-value]
    downside_std = float(downside.std()) * np.sqrt(TRADING_DAYS)
    if downside_std == 0:
        return np.inf  # type: ignore[return-value]
    ann_ret = float(returns.mean()) * TRADING_DAYS
    return (ann_ret - rf_rate) / downside_std


# ===================================================================
# 4. Maximum Drawdown
# ===================================================================

def max_drawdown(prices: pd.Series) -> Tuple[float, Optional[object], Optional[object]]:
    """Maximum peak-to-trough decline in price.

    Args:
        prices: Daily price series (not returns).

    Returns:
        Tuple of ``(max_dd_value, peak_date, trough_date)`` where
        *max_dd_value* is a negative float representing the percentage
        decline.  Dates are ``None`` if the series is empty.
    """
    if prices.empty or len(prices) < 2:
        return 0.0, None, None

    running_max = prices.cummax()
    dd = (prices - running_max) / running_max

    max_dd_val = float(dd.min())
    trough_idx = dd.idxmin()
    # Peak is the running max value at the trough
    peak_idx = prices.loc[:trough_idx].idxmax()

    return max_dd_val, peak_idx, trough_idx


# ===================================================================
# 5. Drawdown Series
# ===================================================================

def drawdown_series(prices: pd.Series) -> pd.Series:
    """Full drawdown time series.

    Args:
        prices: Daily price series.

    Returns:
        pd.Series of drawdown values (negative or zero).
    """
    if prices.empty:
        return pd.Series(dtype=float)
    running_max = prices.cummax()
    return (prices - running_max) / running_max


# ===================================================================
# 6. Value at Risk (Historical)
# ===================================================================

def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical Value-at-Risk at the given confidence level.

    Args:
        returns: Daily return series.
        confidence: Confidence level (e.g. 0.95 for 95 %).

    Returns:
        VaR as a negative float (representing loss).
    """
    if returns.empty:
        return 0.0
    return float(returns.quantile(1 - confidence))


# ===================================================================
# 7. Conditional VaR (Expected Shortfall)
# ===================================================================

def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Conditional VaR (CVaR / Expected Shortfall).

    Average of losses beyond the VaR threshold.

    Args:
        returns: Daily return series.
        confidence: Confidence level.

    Returns:
        CVaR as a negative float.
    """
    if returns.empty:
        return 0.0
    var = value_at_risk(returns, confidence)
    tail = returns[returns <= var]
    if tail.empty:
        return var
    return float(tail.mean())


# ===================================================================
# 8. CAPM Beta
# ===================================================================

def beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    """CAPM beta — sensitivity of *stock_returns* to *market_returns*.

    Args:
        stock_returns: Daily returns for the stock.
        market_returns: Daily returns for the market index.

    Returns:
        Beta coefficient.  Returns 0.0 for degenerate inputs.
    """
    # Align on common index
    aligned = pd.concat(
        [stock_returns.rename("stock"), market_returns.rename("market")],
        axis=1,
    ).dropna()

    if len(aligned) < 2:
        return 0.0

    market_var = aligned["market"].var()
    if market_var == 0:
        return 0.0
    cov = aligned["stock"].cov(aligned["market"])
    return float(cov / market_var)


# ===================================================================
# 9. Calmar Ratio
# ===================================================================

def calmar_ratio(returns: pd.Series, prices: pd.Series) -> float:
    """Calmar ratio — annualised return / absolute max drawdown.

    Args:
        returns: Daily return series.
        prices: Daily price series.

    Returns:
        Calmar ratio.  Returns 0.0 if max drawdown is zero.
    """
    ann_ret = float(returns.mean()) * TRADING_DAYS
    mdd, _, _ = max_drawdown(prices)
    if mdd == 0:
        return 0.0
    return ann_ret / abs(mdd)


# ===================================================================
# 10. Comprehensive Risk Report
# ===================================================================

def comprehensive_risk_report(
    prices: pd.Series,
    returns: pd.Series,
    market_returns: Optional[pd.Series] = None,
) -> Dict:
    """Compute all risk metrics and return a single summary dict.

    Each metric is computed inside its own try/except so that a failure
    in one calculation does not prevent the others from being reported.

    Args:
        prices: Daily price series for one stock.
        returns: Corresponding daily return series.
        market_returns: Optional market (index) daily returns for beta.

    Returns:
        Dict with all risk metrics.
    """
    report: Dict = {}

    # --- Annualised return ---
    try:
        report["annualized_return"] = float(returns.mean()) * TRADING_DAYS
    except Exception:  # noqa: BLE001
        report["annualized_return"] = None

    # --- Volatility ---
    try:
        report["annualized_volatility"] = annualized_volatility(returns)
    except Exception:  # noqa: BLE001
        report["annualized_volatility"] = None

    # --- Sharpe ---
    try:
        report["sharpe_ratio"] = sharpe_ratio(returns)
    except Exception:  # noqa: BLE001
        report["sharpe_ratio"] = None

    # --- Sortino ---
    try:
        report["sortino_ratio"] = sortino_ratio(returns)
    except Exception:  # noqa: BLE001
        report["sortino_ratio"] = None

    # --- Max Drawdown ---
    try:
        mdd_val, peak_dt, trough_dt = max_drawdown(prices)
        report["max_drawdown"] = mdd_val
        report["max_drawdown_peak_date"] = str(peak_dt) if peak_dt is not None else None
        report["max_drawdown_trough_date"] = str(trough_dt) if trough_dt is not None else None
    except Exception:  # noqa: BLE001
        report["max_drawdown"] = None

    # --- VaR & CVaR ---
    try:
        report["var_95"] = value_at_risk(returns, 0.95)
        report["cvar_95"] = conditional_var(returns, 0.95)
        report["var_99"] = value_at_risk(returns, 0.99)
        report["cvar_99"] = conditional_var(returns, 0.99)
    except Exception:  # noqa: BLE001
        report["var_95"] = report["cvar_95"] = None
        report["var_99"] = report["cvar_99"] = None

    # --- Beta ---
    if market_returns is not None:
        try:
            report["beta"] = beta(returns, market_returns)
        except Exception:  # noqa: BLE001
            report["beta"] = None
    else:
        report["beta"] = None

    # --- Calmar ---
    try:
        report["calmar_ratio"] = calmar_ratio(returns, prices)
    except Exception:  # noqa: BLE001
        report["calmar_ratio"] = None

    # --- Distribution stats ---
    try:
        report["skewness"] = float(returns.skew())
        report["kurtosis"] = float(returns.kurtosis())
    except Exception:  # noqa: BLE001
        report["skewness"] = report["kurtosis"] = None

    # --- Day distribution ---
    try:
        total = len(returns.dropna())
        if total > 0:
            report["positive_days_pct"] = float((returns > 0).sum()) / total
            report["negative_days_pct"] = float((returns < 0).sum()) / total
        else:
            report["positive_days_pct"] = report["negative_days_pct"] = None
    except Exception:  # noqa: BLE001
        report["positive_days_pct"] = report["negative_days_pct"] = None

    return report


# ===================================================================
# 11. Cross-Stock Risk Comparison
# ===================================================================

def compare_stocks_risk(prices_dict: Dict[str, pd.Series]) -> pd.DataFrame:
    """Compare key risk metrics across multiple stocks.

    Args:
        prices_dict: Mapping ``{symbol: prices_series}`` for each stock.

    Returns:
        pd.DataFrame with stocks as rows and metrics as columns:
        annualized_return, annualized_vol, sharpe, sortino,
        max_drawdown, var_95, cvar_95, calmar.
    """
    rows = []
    for symbol, prices in prices_dict.items():
        try:
            rets = prices.pct_change().dropna()
            row = {
                "symbol": symbol,
                "annualized_return": float(rets.mean()) * TRADING_DAYS,
                "annualized_vol": annualized_volatility(rets),
                "sharpe": sharpe_ratio(rets),
                "sortino": sortino_ratio(rets),
                "max_drawdown": max_drawdown(prices)[0],
                "var_95": value_at_risk(rets, 0.95),
                "cvar_95": conditional_var(rets, 0.95),
                "calmar": calmar_ratio(rets, prices),
            }
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not compute risk for %s: %s", symbol, exc)

    result = pd.DataFrame(rows)
    if not result.empty:
        result.set_index("symbol", inplace=True)
    logger.info("Risk comparison: %d stocks evaluated", len(result))
    return result
