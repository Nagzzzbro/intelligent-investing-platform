"""
Model Explainability Module for NIFTY-50 Investment Intelligence Platform.

Provides SHAP (SHapley Additive exPlanations) based model interpretation
utilities for tree-based classifiers (XGBoost, Random Forest).  Generates
human-readable explanations of individual predictions and global feature
importance rankings.

Key functions:
    explain_prediction — compute SHAP values via TreeExplainer.
    feature_importance_ranking — sorted (feature, importance) list.
    top_factors — top-N contributing features for a single prediction.
    generate_text_explanation — natural-language prediction explanation.
    get_shap_summary_data — data dict suitable for bar/beeswarm plots.

Author: NIFTY-50 Intelligence Team
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

try:
    import shap
except ImportError:
    shap = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ===================================================================
# 1. Explain Prediction (SHAP TreeExplainer)
# ===================================================================

def explain_prediction(
    model: Any,
    X: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
) -> Optional[np.ndarray]:
    """Compute SHAP values for the given model and input data.

    Uses ``shap.TreeExplainer`` which is compatible with XGBoost,
    LightGBM, CatBoost, and scikit-learn tree ensembles.

    If *model* is a wrapper / ensemble object, the function attempts to
    locate the first tree-based sub-model (via common attribute names).

    Args:
        model: A fitted tree-based model or an object containing one.
        X: Feature matrix (DataFrame or array-like).
        feature_names: Optional list of feature names.  If ``None``,
            inferred from ``X.columns`` when *X* is a DataFrame.

    Returns:
        np.ndarray of SHAP values with shape ``(n_samples, n_features)``.
        Returns ``None`` if SHAP computation fails.
    """
    if shap is None:
        logger.error("shap package is not installed — cannot compute SHAP values")
        return None

    try:
        # Try to unwrap ensemble / wrapper objects
        actual_model = _unwrap_model(model)

        explainer = shap.TreeExplainer(actual_model)
        shap_values = explainer.shap_values(X)

        # For binary classification shap may return a list [class_0, class_1]
        if isinstance(shap_values, list):
            # Use the positive-class (index 1) SHAP values
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        logger.info(
            "SHAP values computed: %d samples × %d features",
            shap_values.shape[0],
            shap_values.shape[1],
        )
        return shap_values

    except Exception as exc:  # noqa: BLE001
        logger.error("SHAP computation failed: %s", exc)
        return None


def _unwrap_model(model: Any) -> Any:
    """Attempt to extract a tree-based estimator from a wrapper object.

    Checks common attribute names used by ensemble wrappers or pipelines:
    ``xgb_model``, ``rf_model``, ``estimator``, ``best_estimator_``.

    Args:
        model: Possibly-wrapped model.

    Returns:
        The inner tree-based model, or the original object unchanged.
    """
    for attr in ("xgb_model", "rf_model", "estimator", "best_estimator_"):
        inner = getattr(model, attr, None)
        if inner is not None and hasattr(inner, "predict"):
            logger.debug("Unwrapped model via attribute '%s'", attr)
            return inner
    return model


# ===================================================================
# 2. Feature Importance Ranking
# ===================================================================

def feature_importance_ranking(
    model: Any,
    feature_names: List[str],
) -> List[Tuple[str, float]]:
    """Extract and sort feature importances from a fitted model.

    Args:
        model: A fitted model exposing ``feature_importances_``
            (e.g. scikit-learn trees, XGBoost).
        feature_names: List of feature names matching the model's
            training features.

    Returns:
        List of ``(feature_name, importance)`` tuples sorted in
        descending order of importance.  Returns an empty list if the
        model does not expose feature importances.
    """
    try:
        actual_model = _unwrap_model(model)
        importances = actual_model.feature_importances_
    except AttributeError:
        logger.warning("Model does not expose feature_importances_")
        return []

    if len(importances) != len(feature_names):
        logger.warning(
            "Mismatch: %d importances vs %d feature names",
            len(importances),
            len(feature_names),
        )
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    paired = list(zip(feature_names, importances.tolist()))
    paired.sort(key=lambda t: t[1], reverse=True)
    return paired


# ===================================================================
# 3. Top Contributing Factors
# ===================================================================

def top_factors(
    shap_values: np.ndarray,
    feature_names: List[str],
    instance_idx: int = 0,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """Identify the top-N most influential features for one prediction.

    Args:
        shap_values: SHAP values array ``(n_samples, n_features)``.
        feature_names: Corresponding feature names.
        instance_idx: Row index of the prediction to explain (default 0).
        top_n: Number of top features to return (default 5).

    Returns:
        List of dicts, each with keys:
        ``feature``, ``shap_value``, ``direction`` (``'bullish'`` or
        ``'bearish'``), ``abs_impact``.
    """
    if shap_values is None:
        return []

    try:
        instance_shap = shap_values[instance_idx]
    except IndexError:
        logger.error("instance_idx %d out of range (max %d)", instance_idx, len(shap_values) - 1)
        return []

    # Pair features with SHAP values
    abs_vals = np.abs(instance_shap)
    top_indices = np.argsort(abs_vals)[::-1][:top_n]

    factors: List[Dict[str, Any]] = []
    for idx in top_indices:
        sv = float(instance_shap[idx])
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        factors.append(
            {
                "feature": name,
                "shap_value": sv,
                "direction": "bullish" if sv > 0 else "bearish",
                "abs_impact": abs(sv),
            }
        )

    return factors


# ===================================================================
# 4. Human-Readable Text Explanation
# ===================================================================

def generate_text_explanation(
    shap_values: np.ndarray,
    feature_names: List[str],
    prediction: float,
    instance_idx: int = 0,
) -> str:
    """Generate a natural-language explanation for one prediction.

    Example output::

        The model predicts UP with 72% confidence. Key factors:
        RSI_14 (bearish, −0.15), SMA_50 crossover (bullish, +0.23), ...

    Args:
        shap_values: SHAP values array.
        feature_names: Feature names.
        prediction: Model output (probability of positive class, 0–1).
        instance_idx: Index of the instance to explain (default 0).

    Returns:
        Human-readable explanation string.
    """
    if shap_values is None:
        return "Explanation unavailable — SHAP values could not be computed."

    direction = "UP" if prediction >= 0.5 else "DOWN"
    confidence = prediction if prediction >= 0.5 else 1.0 - prediction

    factors = top_factors(shap_values, feature_names, instance_idx=instance_idx, top_n=5)

    if not factors:
        return (
            f"The model predicts {direction} with {confidence:.0%} confidence. "
            "No feature-level explanation is available."
        )

    # Build factor descriptions
    parts: List[str] = []
    for f in factors:
        sign = "+" if f["shap_value"] > 0 else ""
        parts.append(f'{f["feature"]} ({f["direction"]}, {sign}{f["shap_value"]:.3f})')

    factors_str = ", ".join(parts)

    return (
        f"The model predicts {direction} with {confidence:.0%} confidence. "
        f"Key factors: {factors_str}."
    )


# ===================================================================
# 5. SHAP Summary Data (for plotting)
# ===================================================================

def get_shap_summary_data(
    shap_values: np.ndarray,
    feature_names: List[str],
) -> Dict[str, Any]:
    """Prepare summary data suitable for bar or beeswarm SHAP plots.

    Args:
        shap_values: SHAP values array ``(n_samples, n_features)``.
        feature_names: Feature names.

    Returns:
        Dict with keys:
        - ``feature_names``: list of feature names sorted by importance.
        - ``mean_abs_shap``: corresponding mean |SHAP| values.
    """
    if shap_values is None:
        return {"feature_names": [], "mean_abs_shap": []}

    mean_abs = np.abs(shap_values).mean(axis=0)

    if len(mean_abs) != len(feature_names):
        feature_names = [f"feature_{i}" for i in range(len(mean_abs))]

    # Sort descending
    order = np.argsort(mean_abs)[::-1]
    sorted_names = [feature_names[i] for i in order]
    sorted_vals = [float(mean_abs[i]) for i in order]

    return {
        "feature_names": sorted_names,
        "mean_abs_shap": sorted_vals,
    }
