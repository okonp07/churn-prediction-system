from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from app.core.config import get_settings
from src.features.build_features import ChurnFeatureBuilder
from src.models.evaluate import compute_metrics
from src.models.select_model import compute_selection_score
from src.models.target_manager import TargetManager

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:  # pragma: no cover - exercised indirectly when dependency is absent
    LGBMClassifier = None
    LGBMRegressor = None


@dataclass
class CandidateResult:
    name: str
    estimator: Any
    cv_metrics: dict[str, float]
    validation_metrics: dict[str, Any]
    selection_score: float


def build_preprocessor(
    numerical_columns: list[str],
    categorical_columns: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=numeric_steps), numerical_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )


def build_candidate_estimators(
    target_manager: TargetManager,
    numerical_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, Pipeline]:
    settings = get_settings()
    is_classification = target_manager.task_type != "regression"

    if is_classification:
        n_classes = max(len(target_manager.classes_), 2)
        objective = "multi:softprob" if n_classes > 2 else "binary:logistic"
        xgb_params = {
            "objective": objective,
            "n_estimators": 350,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 3,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "reg_lambda": 2.0,
            "random_state": settings.random_seed,
            "eval_metric": "mlogloss" if n_classes > 2 else "logloss",
            "tree_method": "hist",
            "n_jobs": -1,
        }
        if n_classes > 2:
            xgb_params["num_class"] = n_classes
        xgb_estimator = XGBClassifier(**xgb_params)

        candidates = {
            "logistic_regression": Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor(numerical_columns, categorical_columns, scale_numeric=True)),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1_500,
                            class_weight="balanced",
                            solver="lbfgs",
                        ),
                    ),
                ]
            ),
            "random_forest": Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor(numerical_columns, categorical_columns, scale_numeric=False)),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=450,
                            max_depth=18,
                            min_samples_leaf=4,
                            class_weight="balanced_subsample",
                            random_state=settings.random_seed,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            "xgboost": Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor(numerical_columns, categorical_columns, scale_numeric=False)),
                    ("model", xgb_estimator),
                ]
            ),
        }
        if LGBMClassifier is not None:
            lgbm_params = {
                "objective": "multiclass" if n_classes > 2 else "binary",
                "n_estimators": 350,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "subsample": 0.85,
                "colsample_bytree": 0.8,
                "min_child_samples": 30,
                "class_weight": "balanced",
                "random_state": settings.random_seed,
            }
            if n_classes > 2:
                lgbm_params["num_class"] = n_classes
            candidates["lightgbm"] = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor(numerical_columns, categorical_columns, scale_numeric=False)),
                    ("model", LGBMClassifier(**lgbm_params)),
                ]
            )
        return candidates

    candidates = {
        "elastic_net": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(numerical_columns, categorical_columns, scale_numeric=True)),
                ("model", ElasticNet(alpha=0.001, l1_ratio=0.3, random_state=settings.random_seed)),
            ]
        ),
        "random_forest_regressor": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(numerical_columns, categorical_columns, scale_numeric=False)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=450,
                        max_depth=18,
                        min_samples_leaf=4,
                        random_state=settings.random_seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "xgboost_regressor": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(numerical_columns, categorical_columns, scale_numeric=False)),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=350,
                        learning_rate=0.05,
                        max_depth=6,
                        min_child_weight=3,
                        subsample=0.85,
                        colsample_bytree=0.8,
                        reg_lambda=2.0,
                        random_state=settings.random_seed,
                        objective="reg:squarederror",
                        tree_method="hist",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
    if LGBMRegressor is not None:
        candidates["lightgbm_regressor"] = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(numerical_columns, categorical_columns, scale_numeric=False)),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=350,
                        learning_rate=0.05,
                        num_leaves=31,
                        subsample=0.85,
                        colsample_bytree=0.8,
                        min_child_samples=30,
                        random_state=settings.random_seed,
                    ),
                ),
            ]
        )
    return candidates


def split_training_data(
    features: pd.DataFrame,
    target: np.ndarray,
    target_manager: TargetManager,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    settings = get_settings()
    stratify = target if target_manager.task_type != "regression" else None
    return train_test_split(
        features,
        target,
        test_size=settings.validation_split,
        random_state=settings.random_seed,
        stratify=stratify,
    )


def cross_validate_estimator(
    estimator: Pipeline,
    features: pd.DataFrame,
    target: np.ndarray,
    target_manager: TargetManager,
) -> dict[str, float]:
    settings = get_settings()
    if target_manager.task_type == "regression":
        splitter = KFold(
            n_splits=settings.cross_validation_folds,
            shuffle=True,
            random_state=settings.random_seed,
        )
    else:
        splitter = StratifiedKFold(
            n_splits=settings.cross_validation_folds,
            shuffle=True,
            random_state=settings.random_seed,
        )

    fold_scores: list[dict[str, float]] = []
    for train_idx, val_idx in splitter.split(features, target):
        fold_estimator = clone(estimator)
        X_train_fold = features.iloc[train_idx]
        X_val_fold = features.iloc[val_idx]
        y_train_fold = target[train_idx]
        y_val_fold = target[val_idx]
        fold_estimator.fit(X_train_fold, y_train_fold)
        fold_predictions = fold_estimator.predict(X_val_fold)
        fold_probabilities = (
            fold_estimator.predict_proba(X_val_fold)
            if hasattr(fold_estimator, "predict_proba")
            else None
        )
        fold_metrics = compute_metrics(y_val_fold, fold_predictions, fold_probabilities, target_manager)
        fold_scores.append(
            {
                key: float(value)
                for key, value in fold_metrics.items()
                if isinstance(value, (int, float, np.floating))
            }
        )

    averaged_scores: dict[str, float] = {}
    for key in fold_scores[0]:
        averaged_scores[key] = float(np.mean([score[key] for score in fold_scores]))
    return averaged_scores


def train_and_select_model(
    features: pd.DataFrame,
    target: np.ndarray,
    feature_builder: ChurnFeatureBuilder,
    target_manager: TargetManager,
) -> tuple[CandidateResult, list[CandidateResult], dict[str, Any]]:
    X_train, X_val, y_train, y_val = split_training_data(features, target, target_manager)
    candidate_estimators = build_candidate_estimators(
        target_manager,
        feature_builder.numerical_columns_ + feature_builder.binary_columns_,
        feature_builder.categorical_columns_,
    )

    results: list[CandidateResult] = []
    for name, estimator in candidate_estimators.items():
        cv_metrics = cross_validate_estimator(estimator, X_train, y_train, target_manager)
        fitted_estimator = clone(estimator).fit(X_train, y_train)
        validation_predictions = fitted_estimator.predict(X_val)
        validation_probabilities = (
            fitted_estimator.predict_proba(X_val) if hasattr(fitted_estimator, "predict_proba") else None
        )
        validation_metrics = compute_metrics(
            y_val, validation_predictions, validation_probabilities, target_manager
        )
        selection_score = compute_selection_score(validation_metrics)
        results.append(
            CandidateResult(
                name=name,
                estimator=fitted_estimator,
                cv_metrics=cv_metrics,
                validation_metrics=validation_metrics,
                selection_score=selection_score,
            )
        )

    best_result = max(results, key=lambda item: item.selection_score)
    holdout_payload = {
        "y_validation": y_val,
        "validation_features": X_val,
        "validation_probabilities": best_result.estimator.predict_proba(X_val)
        if hasattr(best_result.estimator, "predict_proba")
        else None,
        "validation_predictions": best_result.estimator.predict(X_val),
    }
    return best_result, results, holdout_payload
