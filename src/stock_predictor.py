"""
Stock Direction Predictor for NIFTY-50 Investment Intelligence Platform.

This module implements an ensemble ML classifier that combines XGBoost and
Random Forest to predict next-day stock price direction (up / down).  It
includes walk-forward (rolling window) validation, feature importance
extraction, and a convenience end-to-end training pipeline.

Key classes:
    StockPredictor — ensemble classifier with train / predict / evaluate.

Key functions:
    train_and_evaluate — full pipeline from raw data to results dict.

Author: NIFTY-50 Intelligence Team
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    classification_report,
)

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None  # graceful degradation if xgboost not installed

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ===================================================================
# StockPredictor class
# ===================================================================

class StockPredictor:
    """Ensemble stock-direction classifier using XGBoost + Random Forest.

    The two models independently predict the probability that the next day's
    return is positive.  Their probabilities are averaged for the final
    prediction, providing a simple but effective model-blending strategy.

    Attributes:
        xgb_model: The XGBClassifier instance (or ``None`` if xgboost is
            not installed).
        rf_model: The RandomForestClassifier instance.
        is_fitted: Whether ``train()`` has been called successfully.
        feature_names: Feature names recorded during training.
    """

    def __init__(self) -> None:
        """Initialise both sub-models with tuned default hyper-parameters."""

        # --- XGBoost -------------------------------------------------------
        if XGBClassifier is not None:
            self.xgb_model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )
        else:
            self.xgb_model = None
            logger.warning(
                "xgboost is not installed — falling back to RandomForest only"
            )

        # --- Random Forest -------------------------------------------------
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

        self.is_fitted: bool = False
        self.feature_names: Optional[List[str]] = None

    # ------------------------------------------------------------------ train
    def train(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
    ) -> None:
        """Fit both sub-models on the training data.

        Args:
            X_train: Feature matrix (samples × features).
            y_train: Binary target array (1 = up, 0 = down).
        """
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)

        # Fit Random Forest
        self.rf_model.fit(X_train, y_train)
        logger.info("RandomForest fitted — training samples: %d", len(y_train))

        # Fit XGBoost (if available)
        if self.xgb_model is not None:
            self.xgb_model.fit(X_train, y_train)
            logger.info("XGBoost fitted — training samples: %d", len(y_train))

        self.is_fitted = True

    # -------------------------------------------------------------- predict
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return ensemble binary predictions (0 or 1).

        Averages predicted probabilities from both models and applies a
        threshold of 0.5.

        Args:
            X: Feature matrix.

        Returns:
            np.ndarray of shape ``(n_samples,)`` with 0/1 predictions.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    # -------------------------------------------------------- predict_proba
    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return ensemble predicted probabilities for the positive class.

        Args:
            X: Feature matrix.

        Returns:
            np.ndarray of shape ``(n_samples,)`` — probability of class 1.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained — call train() first")

        rf_proba = self.rf_model.predict_proba(X)[:, 1]

        if self.xgb_model is not None:
            xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
            avg_proba = (rf_proba + xgb_proba) / 2.0
        else:
            avg_proba = rf_proba

        return avg_proba

    # ------------------------------------------------------------- evaluate
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute a comprehensive evaluation metrics dictionary.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels (0 / 1).
            y_proba: Predicted probabilities for the positive class
                (optional — needed for ROC-AUC).

        Returns:
            Dict with keys: accuracy, precision, recall, f1,
            directional_accuracy, and optionally roc_auc.
        """
        metrics: Dict[str, float] = {}
        try:
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
            metrics["precision"] = float(
                precision_score(y_true, y_pred, zero_division=0)
            )
            metrics["recall"] = float(
                recall_score(y_true, y_pred, zero_division=0)
            )
            metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
            metrics["directional_accuracy"] = metrics["accuracy"]

            if y_proba is not None:
                try:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
                except ValueError:
                    logger.warning("ROC-AUC could not be computed (single class?)")

        except Exception as exc:  # noqa: BLE001
            logger.error("Evaluation error: %s", exc)

        return metrics

    # ----------------------------------------- walk_forward_validation
    def walk_forward_validation(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        train_window: int = 252,
        test_window: int = 21,
    ) -> pd.DataFrame:
        """Rolling-window (walk-forward) back-test.

        The dataset is split into consecutive windows: the model is trained
        on the previous *train_window* rows and then predicts the next
        *test_window* rows.  The window then slides forward by
        *test_window* and the process repeats.

        Args:
            df: Full DataFrame containing features, target, and optionally
                a ``Date`` column.
            feature_cols: Column names to use as features.
            target_col: Name of the target column.
            train_window: Number of rows in each training window
                (default 252 ≈ 1 trading year).
            test_window: Number of rows in each test window
                (default 21 ≈ 1 trading month).

        Returns:
            pd.DataFrame with columns: Date (if available), Actual,
            Predicted, Predicted_Proba.
        """
        if len(df) < train_window + test_window:
            logger.warning(
                "Insufficient data for walk-forward validation "
                "(need ≥ %d rows, got %d)",
                train_window + test_window,
                len(df),
            )
            return pd.DataFrame(
                columns=["Date", "Actual", "Predicted", "Predicted_Proba"]
            )

        results: List[Dict[str, Any]] = []
        n = len(df)
        step = 0

        for start in range(0, n - train_window - test_window + 1, test_window):
            train_end = start + train_window
            test_end = min(train_end + test_window, n)

            X_tr = df.iloc[start:train_end][feature_cols]
            y_tr = df.iloc[start:train_end][target_col]
            X_te = df.iloc[train_end:test_end][feature_cols]
            y_te = df.iloc[train_end:test_end][target_col]

            try:
                self.train(X_tr, y_tr)
                preds = self.predict(X_te)
                probas = self.predict_proba(X_te)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Walk-forward step %d failed: %s", step, exc)
                step += 1
                continue

            for i, idx in enumerate(range(train_end, test_end)):
                row: Dict[str, Any] = {
                    "Actual": int(y_te.iloc[i]),
                    "Predicted": int(preds[i]),
                    "Predicted_Proba": float(probas[i]),
                }
                if "Date" in df.columns:
                    row["Date"] = df.iloc[idx]["Date"]
                results.append(row)

            step += 1
            if step % 5 == 0:
                logger.info("Walk-forward: completed %d steps", step)

        logger.info("Walk-forward validation: %d total predictions", len(results))
        return pd.DataFrame(results)

    # ------------------------------------------------ get_feature_importance
    def get_feature_importance(self) -> List[Tuple[str, float]]:
        """Return ensemble-averaged feature importances sorted descending.

        Averages the Gini / gain importances from both sub-models.  If
        only one model is available, returns that model's importances.

        Returns:
            List of ``(feature_name, importance)`` tuples sorted by
            importance (highest first).

        Raises:
            RuntimeError: If the model has not been trained.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained — call train() first")

        rf_imp = self.rf_model.feature_importances_

        if self.xgb_model is not None:
            xgb_imp = self.xgb_model.feature_importances_
            avg_imp = (rf_imp + xgb_imp) / 2.0
        else:
            avg_imp = rf_imp

        # Pair with names
        names = self.feature_names or [f"feature_{i}" for i in range(len(avg_imp))]
        paired = list(zip(names, avg_imp.tolist()))
        paired.sort(key=lambda t: t[1], reverse=True)
        return paired


# ===================================================================
# End-to-end convenience function
# ===================================================================

def train_and_evaluate(data_dir: str = "data/") -> Tuple[Dict[str, Any], StockPredictor]:
    """Full pipeline: load data → engineer features → train → evaluate.

    If the specified *data_dir* does not exist or contains no CSV files,
    the function falls back to synthetic sample data so the pipeline can
    be demonstrated without the real Kaggle dataset.

    Args:
        data_dir: Directory containing raw stock CSV files.

    Returns:
        Tuple of ``(results_dict, trained_predictor)`` where
        *results_dict* contains evaluation metrics, sample predictions,
        and feature importances.
    """
    # Late import to avoid circular dependency at module level
    from src.data_processing import (  # noqa: WPS433
        load_all_stocks,
        clean_data,
        add_technical_indicators,
        calculate_returns,
        prepare_ml_features,
        get_sample_data,
    )

    # ---- 1. Load data ---------------------------------------------------
    try:
        raw_df = load_all_stocks(data_dir)
        logger.info("Loaded real data from %s", data_dir)
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("Could not load data from %s: %s — using sample data", data_dir, exc)
        raw_df = get_sample_data()

    # ---- 2. Clean -------------------------------------------------------
    df = clean_data(raw_df)

    # ---- 3. Technical indicators + returns ------------------------------
    df = add_technical_indicators(df)
    df = calculate_returns(df)

    # ---- 4. Feature engineering -----------------------------------------
    X, y = prepare_ml_features(df)

    # ---- 5. Train / test split ------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, shuffle=False,  # time-series: no shuffle
    )
    logger.info("Train: %d samples | Test: %d samples", len(X_train), len(X_test))

    # ---- 6. Train model -------------------------------------------------
    predictor = StockPredictor()
    predictor.train(X_train, y_train)

    # ---- 7. Evaluate ----------------------------------------------------
    y_pred = predictor.predict(X_test)
    y_proba = predictor.predict_proba(X_test)
    metrics = predictor.evaluate(y_test.values, y_pred, y_proba)
    feature_imp = predictor.get_feature_importance()

    # ---- 8. Classification report (logged) ------------------------------
    logger.info(
        "\n%s",
        classification_report(y_test, y_pred, target_names=["DOWN", "UP"]),
    )

    results: Dict[str, Any] = {
        "metrics": metrics,
        "feature_importances": feature_imp[:15],  # top 15
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X.shape[1],
        "sample_predictions": {
            "actual": y_test.values[:20].tolist(),
            "predicted": y_pred[:20].tolist(),
            "probabilities": y_proba[:20].tolist(),
        },
    }

    logger.info("Pipeline complete — accuracy: %.4f", metrics.get("accuracy", 0))
    return results, predictor
