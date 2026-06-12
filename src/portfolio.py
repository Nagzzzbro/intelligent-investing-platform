"""
Portfolio Construction Module for NIFTY-50 Investment Intelligence Platform.

Implements Modern Portfolio Theory (MPT) to build optimal portfolios across
three risk profiles — conservative, balanced, and aggressive.  Uses
``scipy.optimize`` for mean-variance optimisation and provides an efficient
frontier generator.

Key functions:
    calculate_expected_returns — annualised mean or EWM returns.
    calculate_covariance_matrix — annualised return covariance.
    optimize_portfolio — profile-based optimisation (min-vol / max-Sharpe / max-return).
    get_efficient_frontier — trace the efficient frontier curve.
    construct_portfolios — build all three profiles at once.
    get_portfolio_summary — human-readable allocation summary.

Author: NIFTY-50 Intelligence Team
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Module-level logger & constants
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

RF_RATE: float = 0.06  # Indian risk-free rate ≈ 6 % p.a.
TRADING_DAYS: int = 252


# ===================================================================
# 1. Expected Returns
# ===================================================================

def calculate_expected_returns(
    prices_df: pd.DataFrame,
    method: str = "mean",
) -> pd.Series:
    """Compute annualised expected returns from a daily price DataFrame.

    Args:
        prices_df: DataFrame with stocks as columns and a DatetimeIndex.
            Each column holds daily closing prices for one stock.
        method: ``'mean'`` for arithmetic mean of daily returns (default),
            or ``'ewm'`` for exponentially weighted mean (span = 252).

    Returns:
        pd.Series indexed by stock symbol with annualised returns.

    Raises:
        ValueError: If *prices_df* is empty or has fewer than 2 rows.
    """
    if prices_df.empty or len(prices_df) < 2:
        raise ValueError("Need at least 2 price observations to compute returns")

    daily_returns = prices_df.pct_change().dropna()

    if method == "ewm":
        mean_daily = daily_returns.ewm(span=TRADING_DAYS).mean().iloc[-1]
    else:  # default: arithmetic mean
        mean_daily = daily_returns.mean()

    annualised = mean_daily * TRADING_DAYS
    logger.info("Expected returns computed (%s) for %d stocks", method, len(annualised))
    return annualised


# ===================================================================
# 2. Covariance Matrix
# ===================================================================

def calculate_covariance_matrix(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the annualised covariance matrix of daily returns.

    Args:
        prices_df: Daily price DataFrame (stocks as columns).

    Returns:
        pd.DataFrame: Annualised covariance matrix (N × N).
    """
    daily_returns = prices_df.pct_change().dropna()
    cov = daily_returns.cov() * TRADING_DAYS
    logger.info("Covariance matrix computed: %d × %d", cov.shape[0], cov.shape[1])
    return cov


# ===================================================================
# 3. Portfolio Performance
# ===================================================================

def portfolio_performance(
    weights: np.ndarray,
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
) -> Tuple[float, float, float]:
    """Evaluate a portfolio defined by *weights*.

    Args:
        weights: Array of asset weights summing to 1.
        expected_returns: Annualised expected returns per stock.
        cov_matrix: Annualised covariance matrix.

    Returns:
        Tuple of ``(portfolio_return, portfolio_volatility, sharpe_ratio)``.
    """
    w = np.asarray(weights, dtype=float)
    port_return = float(w @ expected_returns.values)
    port_vol = float(np.sqrt(w @ cov_matrix.values @ w))
    sharpe = (port_return - RF_RATE) / port_vol if port_vol > 0 else 0.0
    return port_return, port_vol, sharpe


# ===================================================================
# 4. Portfolio Optimisation
# ===================================================================

def optimize_portfolio(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    profile: str = "balanced",
) -> Dict:
    """Find optimal portfolio weights for the given risk profile.

    Profiles:
        * **conservative** — minimise volatility, max weight ≤ 5 %.
        * **balanced** — maximise Sharpe ratio, max weight ≤ 10 %.
        * **aggressive** — maximise return (with vol < 35 %), max weight ≤ 15 %.

    Args:
        expected_returns: Annualised expected returns (pd.Series).
        cov_matrix: Annualised covariance matrix (pd.DataFrame).
        profile: One of ``'conservative'``, ``'balanced'``, ``'aggressive'``.

    Returns:
        Dict with keys: ``weights``, ``expected_return``, ``volatility``,
        ``sharpe_ratio``, ``profile``, ``stock_names``.

    Raises:
        ValueError: If *profile* is not recognised.
    """
    n = len(expected_returns)
    stock_names = list(expected_returns.index)

    # Initial guess: equal weights
    w0 = np.ones(n) / n

    # Common constraint: weights sum to 1
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Profile-specific settings ----------------------------------------
    if profile == "conservative":
        max_weight = 0.05
        # Objective: minimise volatility
        objective = lambda w: np.sqrt(w @ cov_matrix.values @ w)

    elif profile == "balanced":
        max_weight = 0.10
        # Objective: maximise Sharpe → minimise negative Sharpe
        def objective(w):
            ret = w @ expected_returns.values
            vol = np.sqrt(w @ cov_matrix.values @ w)
            return -(ret - RF_RATE) / vol if vol > 1e-10 else 0.0

    elif profile == "aggressive":
        max_weight = 0.15
        # Objective: maximise return → minimise negative return
        objective = lambda w: -(w @ expected_returns.values)
        # Additional constraint: volatility < 35 %
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: 0.35 - np.sqrt(w @ cov_matrix.values @ w),
            }
        )
    else:
        raise ValueError(
            f"Unknown profile '{profile}'. Choose from: conservative, balanced, aggressive"
        )

    bounds = tuple((0.0, max_weight) for _ in range(n))

    try:
        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        if not result.success:
            logger.warning("Optimiser did not converge: %s", result.message)

        weights = result.x
        # Clean up tiny negative weights from numerical noise
        weights = np.maximum(weights, 0.0)
        weights /= weights.sum()  # re-normalise

    except Exception as exc:  # noqa: BLE001
        logger.error("Optimisation failed for '%s' profile: %s", profile, exc)
        # Fallback to equal-weight portfolio
        weights = np.ones(n) / n

    ret, vol, sharpe = portfolio_performance(weights, expected_returns, cov_matrix)

    logger.info(
        "%s portfolio — return: %.2f%%, vol: %.2f%%, Sharpe: %.2f",
        profile.title(),
        ret * 100,
        vol * 100,
        sharpe,
    )

    return {
        "weights": weights,
        "expected_return": ret,
        "volatility": vol,
        "sharpe_ratio": sharpe,
        "profile": profile,
        "stock_names": stock_names,
    }


# ===================================================================
# 5. Efficient Frontier
# ===================================================================

def get_efficient_frontier(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_points: int = 50,
) -> List[Tuple[float, float]]:
    """Generate the mean-variance efficient frontier.

    For each target return level between the minimum and maximum
    individual stock return, solve for the minimum-variance portfolio.

    Args:
        expected_returns: Annualised expected returns.
        cov_matrix: Annualised covariance matrix.
        n_points: Number of points on the frontier (default 50).

    Returns:
        List of ``(volatility, return)`` tuples tracing the frontier.
    """
    n = len(expected_returns)
    min_ret = expected_returns.min()
    max_ret = expected_returns.max()
    target_returns = np.linspace(min_ret, max_ret, n_points)

    frontier: List[Tuple[float, float]] = []

    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: w @ expected_returns.values - t},
        ]
        bounds = tuple((0.0, 1.0) for _ in range(n))
        w0 = np.ones(n) / n

        try:
            result = minimize(
                lambda w: np.sqrt(w @ cov_matrix.values @ w),
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-10},
            )
            if result.success:
                vol = float(np.sqrt(result.x @ cov_matrix.values @ result.x))
                frontier.append((vol, float(target)))
        except Exception:  # noqa: BLE001
            continue

    logger.info("Efficient frontier: %d points generated", len(frontier))
    return frontier


# ===================================================================
# 6. Construct All Three Profiles
# ===================================================================

def construct_portfolios(prices_df: pd.DataFrame) -> Dict:
    """Build conservative, balanced, and aggressive portfolios.

    Args:
        prices_df: Daily closing prices with stocks as columns.

    Returns:
        Dict keyed by profile name.  Each value is a sub-dict with:
        ``weights``, ``stock_names``, ``expected_return``, ``volatility``,
        ``sharpe_ratio``.
    """
    exp_ret = calculate_expected_returns(prices_df)
    cov_mat = calculate_covariance_matrix(prices_df)

    portfolios: Dict = {}
    for profile in ("conservative", "balanced", "aggressive"):
        try:
            result = optimize_portfolio(exp_ret, cov_mat, profile=profile)
            portfolios[profile] = result
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to build '%s' portfolio: %s", profile, exc)
            portfolios[profile] = {"error": str(exc)}

    return portfolios


# ===================================================================
# 7. Human-Readable Portfolio Summary
# ===================================================================

def get_portfolio_summary(
    weights: np.ndarray,
    stock_names: List[str],
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
) -> Dict:
    """Create a human-readable summary of a portfolio allocation.

    Args:
        weights: Array of portfolio weights.
        stock_names: Corresponding stock ticker symbols.
        expected_returns: Annualised expected returns per stock.
        cov_matrix: Annualised covariance matrix.

    Returns:
        Dict with keys: ``portfolio_return``, ``portfolio_volatility``,
        ``sharpe_ratio``, ``allocations`` (list of dicts sorted by weight),
        ``top_holdings`` (top 10), ``diversification_score``.
    """
    w = np.asarray(weights, dtype=float)
    port_ret, port_vol, sharpe = portfolio_performance(w, expected_returns, cov_matrix)

    # Build allocation list ------------------------------------------------
    allocations = sorted(
        [
            {"stock": name, "weight": float(wi), "weight_pct": f"{wi * 100:.2f}%"}
            for name, wi in zip(stock_names, w)
        ],
        key=lambda d: d["weight"],
        reverse=True,
    )

    # Diversification via Herfindahl–Hirschman Index -----------------------
    hhi = float(np.sum(w ** 2))
    diversification_score = 1.0 - hhi  # 0 = concentrated, ~1 = diversified

    return {
        "portfolio_return": port_ret,
        "portfolio_volatility": port_vol,
        "sharpe_ratio": sharpe,
        "allocations": allocations,
        "top_holdings": allocations[:10],
        "diversification_score": diversification_score,
    }
