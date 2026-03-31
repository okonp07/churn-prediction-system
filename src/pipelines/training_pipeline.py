from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from app.core.config import PROJECT_ROOT, get_settings
from app.core.logger import setup_logger
from app.utils.helpers import flatten_metric_payload
from src.data.load_data import load_test_data, load_train_data
from src.data.target_normalization import (
    describe_target_normalization,
    normalize_target_frame,
    resolve_target_normalization_map,
)
from src.data.validate_data import validate_inference_frame, validate_training_frame
from src.features.build_features import ChurnFeatureBuilder
from src.models.evaluate import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_target_distribution,
    save_json,
)
from src.models.predict import score_dataframe
from src.models.target_manager import TargetManager
from src.models.train import train_and_select_model


LOGGER = setup_logger()


def _extract_feature_importance(bundle_model: Any) -> pd.DataFrame:
    preprocessor = bundle_model.named_steps["preprocessor"]
    estimator = bundle_model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(estimator, "feature_importances_"):
        importance_values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        coefficients = estimator.coef_
        importance_values = np.mean(np.abs(coefficients), axis=0) if coefficients.ndim > 1 else np.abs(coefficients)
    else:
        importance_values = np.zeros(len(feature_names))

    importance_frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance_values,
        }
    ).sort_values("importance", ascending=False)
    importance_frame["base_feature"] = importance_frame["feature"].apply(_base_feature_name)
    grouped = (
        importance_frame.groupby("base_feature", as_index=False)["importance"].sum().sort_values(
            "importance", ascending=False
        )
    )
    return grouped


def _base_feature_name(transformed_name: str) -> str:
    stripped = transformed_name.split("__", 1)[-1]
    prefixes = [
        "gender",
        "region_category",
        "membership_category",
        "joined_through_referral",
        "preferred_offer_types",
        "medium_of_operation",
        "internet_option",
        "used_special_discount",
        "offer_application_preference",
        "past_complaint",
        "complaint_status",
        "feedback",
        "customer_tenure_bucket",
        "visit_time_segment",
        "engagement_segment",
        "spend_segment",
    ]
    for prefix in sorted(prefixes, key=len, reverse=True):
        if stripped == prefix or stripped.startswith(prefix + "_"):
            return prefix
    return stripped


def run_training_pipeline(
    train_path: str | Path | None = None,
    test_path: str | Path | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    train_frame = pd.read_csv(train_path) if train_path else load_train_data()
    test_frame = pd.read_csv(test_path) if test_path else load_test_data()

    train_report = validate_training_frame(train_frame, settings.target_column)
    test_report = validate_inference_frame(test_frame)
    if not train_report.is_valid:
        raise ValueError(f"Training data validation failed: {train_report.to_dict()}")
    if not test_report.is_valid:
        raise ValueError(f"Test data validation failed: {test_report.to_dict()}")

    normalization_map = resolve_target_normalization_map(settings.target_normalization_map)
    normalized_train_frame = normalize_target_frame(train_frame, settings.target_column, normalization_map)
    normalization_summary = describe_target_normalization(
        train_frame[settings.target_column],
        normalized_train_frame[settings.target_column],
        normalization_map,
    )

    target_manager = TargetManager().fit(
        normalized_train_frame[settings.target_column],
        original_target=train_frame[settings.target_column],
        normalization_map=normalization_map,
    )
    LOGGER.info("Detected task type: %s", target_manager.task_type)

    feature_builder = ChurnFeatureBuilder()
    features = feature_builder.fit_transform(normalized_train_frame.drop(columns=[settings.target_column]))
    encoded_target = target_manager.transform(normalized_train_frame[settings.target_column])

    best_result, all_results, holdout_payload = train_and_select_model(
        features=features,
        target=encoded_target,
        feature_builder=feature_builder,
        target_manager=target_manager,
    )

    final_model = clone(best_result.estimator).fit(features, encoded_target)
    model_bundle = {
        "model": final_model,
        "feature_builder": feature_builder,
        "target_manager": target_manager,
        "metadata": {
            "project_name": settings.project_name,
            "project_version": settings.project_version,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "task_detection": target_manager.metadata(),
            "target_normalization": normalization_summary,
            "reference_date": str(feature_builder.reference_date.date()),
            "validation_metrics": flatten_metric_payload(best_result.validation_metrics),
            "candidate_models": [
                {
                    "name": result.name,
                    "selection_score": result.selection_score,
                    "cv_metrics": flatten_metric_payload(result.cv_metrics),
                    "validation_metrics": flatten_metric_payload(result.validation_metrics),
                }
                for result in all_results
            ],
            "feature_summary": feature_builder.get_feature_summary(),
            "best_model_name": best_result.name,
        },
    }

    settings.bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, settings.bundle_path)

    comparison_frame = pd.DataFrame(
        [
            {
                "model_name": result.name,
                "selection_score": result.selection_score,
                **{f"cv_{key}": value for key, value in result.cv_metrics.items()},
                **{f"val_{key}": value for key, value in result.validation_metrics.items() if not isinstance(value, list)},
            }
            for result in all_results
        ]
    ).sort_values("selection_score", ascending=False)
    comparison_frame.to_csv(settings.comparison_path, index=False)

    feature_importance = _extract_feature_importance(final_model)
    feature_importance.to_csv(settings.feature_importance_path, index=False)

    summary_payload = {
        "training_validation_report": train_report.to_dict(),
        "test_validation_report": test_report.to_dict(),
        "best_model_name": best_result.name,
        "best_model_metrics": flatten_metric_payload(best_result.validation_metrics),
        "candidate_model_count": len(all_results),
        "task_detection": target_manager.metadata(),
        "target_normalization": normalization_summary,
        "feature_summary": feature_builder.get_feature_summary(),
    }
    save_json(summary_payload, settings.metrics_path)

    plot_target_distribution(
        normalized_train_frame[settings.target_column],
        PROJECT_ROOT / "artifacts" / "plots" / "target_distribution.png",
    )

    if target_manager.task_type != "regression":
        plot_confusion_matrix(
            best_result.validation_metrics["confusion_matrix"],
            [str(item) for item in target_manager.classes_],
            PROJECT_ROOT / "artifacts" / "plots" / "confusion_matrix.png",
        )

        val_proba = holdout_payload["validation_probabilities"]
        if val_proba is not None:
            high_risk_cutoff = max(len(target_manager.classes_) - settings.high_risk_top_n_classes, 0)
            high_risk_truth = (holdout_payload["y_validation"] >= high_risk_cutoff).astype(int)
            high_risk_score = val_proba[:, high_risk_cutoff:].sum(axis=1)
            plot_calibration_curve(
                high_risk_truth,
                high_risk_score,
                PROJECT_ROOT / "artifacts" / "plots" / "high_risk_calibration.png",
            )

    test_predictions = score_dataframe(model_bundle, test_frame)
    output_batch_path = PROJECT_ROOT / settings.batch_prediction_output
    output_batch_path.parent.mkdir(parents=True, exist_ok=True)
    test_predictions.to_csv(output_batch_path, index=False)
    test_predictions.head(5).to_json(settings.prediction_preview_path, orient="records", indent=2)

    return {
        "bundle_path": str(settings.bundle_path),
        "metrics_path": str(settings.metrics_path),
        "comparison_path": str(settings.comparison_path),
        "feature_importance_path": str(settings.feature_importance_path),
        "sample_predictions_path": str(output_batch_path),
        "best_model_name": best_result.name,
        "validation_metrics": flatten_metric_payload(best_result.validation_metrics),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the churn prediction system.")
    parser.add_argument("--train-path", default=None, help="Optional path to the training CSV.")
    parser.add_argument("--test-path", default=None, help="Optional path to the test CSV.")
    args = parser.parse_args()

    summary = run_training_pipeline(args.train_path, args.test_path)
    print(summary)


if __name__ == "__main__":
    main()
