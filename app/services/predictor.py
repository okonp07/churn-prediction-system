from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd

from app.core.config import get_settings
from app.core.logger import setup_logger
from app.services.explainer import ExplainerService
from app.services.recommender import RecommendationEngine
from src.models.predict import load_bundle, score_dataframe


LOGGER = setup_logger()


class PredictorService:
    def __init__(self, bundle_path: str | None = None):
        self.settings = get_settings()
        self.bundle = load_bundle(bundle_path or self.settings.bundle_path)
        self.feature_builder = self.bundle["feature_builder"]
        self.target_manager = self.bundle["target_manager"]
        self.model = self.bundle["model"]
        self.explainer = ExplainerService(self.bundle)
        self.recommender = RecommendationEngine(self.feature_builder.learned_thresholds)

    def predict_record(self, record: dict[str, Any]) -> dict[str, Any]:
        input_frame = pd.DataFrame([record])
        summary_frame = score_dataframe(self.bundle, input_frame)
        row = summary_frame.iloc[0].to_dict()
        engineered_profile = self.feature_builder.transform(input_frame).iloc[0].to_dict()

        predicted_class_index = self.target_manager.encode_label(row["predicted_class"])
        driver_details = self._business_risk_drivers(record, engineered_profile)
        if len(driver_details) < 3:
            explanation_details = self.explainer.local_driver_details(input_frame, predicted_class_index)
            seen_messages = {item["message"] for item in driver_details}
            for detail in explanation_details:
                if not self._is_business_friendly_explanation(detail, record):
                    continue
                if detail["message"] in seen_messages:
                    continue
                driver_details.append(detail)
                seen_messages.add(detail["message"])
                if len(driver_details) >= 3:
                    break

        recommendations = self.recommender.recommend(
            raw_profile=record,
            engineered_profile=engineered_profile,
            risk_band=row["risk_band"],
            top_driver_messages=[item["message"] for item in driver_details],
        )

        probability_breakdown = {
            str(label): float(row.get(f"prob_class_{label}", 0.0)) for label in self.target_manager.classes_
        }

        return {
            "customer_id": row.get("customer_id"),
            "predicted_class": row["predicted_class"],
            "predicted_label": row["predicted_label"],
            "risk_score": float(row["risk_score"]),
            "risk_band": row["risk_band"],
            "confidence": None if pd.isna(row["confidence"]) else float(row["confidence"]),
            "top_risk_drivers": [item["message"] for item in driver_details],
            "driver_details": driver_details,
            "recommendations": recommendations,
            "probability_breakdown": probability_breakdown,
        }

    def predict_batch(self, records: list[dict[str, Any]] | pd.DataFrame) -> list[dict[str, Any]]:
        frame = pd.DataFrame(records) if isinstance(records, list) else records.copy()
        predictions: list[dict[str, Any]] = []
        for _, record in frame.iterrows():
            predictions.append(self.predict_record(record.to_dict()))
        return predictions

    def recommend_only(
        self,
        record: dict[str, Any],
        risk_band: str | None = None,
        top_risk_drivers: list[str] | None = None,
    ) -> dict[str, Any]:
        engineered = self.feature_builder.transform(pd.DataFrame([record])).iloc[0].to_dict()
        resolved_risk_band = risk_band or "Medium"
        recommendations = self.recommender.recommend(
            raw_profile=record,
            engineered_profile=engineered,
            risk_band=resolved_risk_band,
            top_driver_messages=top_risk_drivers,
        )
        return {
            "risk_band": resolved_risk_band,
            "recommendations": recommendations,
        }

    def model_info(self) -> dict[str, Any]:
        metadata = self.bundle["metadata"]
        return {
            "project_name": metadata["project_name"],
            "project_version": metadata["project_version"],
            "trained_at": metadata["trained_at"],
            "task_detection": metadata["task_detection"],
            "target_normalization": metadata.get("target_normalization"),
            "reference_date": metadata["reference_date"],
            "best_model_name": metadata["best_model_name"],
            "validation_metrics": metadata["validation_metrics"],
            "feature_summary": metadata["feature_summary"],
            "global_feature_importance": self.explainer.global_feature_importance(),
        }

    @staticmethod
    def _is_business_friendly_explanation(detail: dict[str, Any], raw_profile: dict[str, Any]) -> bool:
        allowed_features = {
            "days_since_last_login",
            "avg_time_spent",
            "avg_transaction_value",
            "complaint_status",
            "feedback",
            "points_in_wallet",
            "membership_category",
            "customer_tenure_bucket",
        }
        feature = detail.get("feature")
        if feature not in allowed_features:
            return False
        if feature == "feedback" and raw_profile.get("feedback") not in {
            "Poor Product Quality",
            "Poor Website",
            "Poor Customer Service",
            "Too many ads",
            "No reason specified",
        }:
            return False
        return True

    def _business_risk_drivers(
        self,
        raw_profile: dict[str, Any],
        engineered_profile: dict[str, Any],
    ) -> list[dict[str, Any]]:
        thresholds = self.feature_builder.learned_thresholds
        driver_details: list[dict[str, Any]] = []

        def add_driver(feature: str, value: Any, message: str) -> None:
            driver_details.append(
                {
                    "feature": feature,
                    "feature_value": value,
                    "direction": "up",
                    "message": message,
                }
            )

        days_since_login = engineered_profile.get("days_since_last_login")
        if pd.notna(days_since_login) and days_since_login >= thresholds.get("activity_login_gap", 14):
            add_driver(
                "days_since_last_login",
                days_since_login,
                f"Long time since last login ({days_since_login:.0f} days) is a strong churn signal.",
            )

        avg_time_spent = engineered_profile.get("avg_time_spent")
        if pd.notna(avg_time_spent) and avg_time_spent <= thresholds.get("avg_time_spent_low", 60):
            add_driver(
                "avg_time_spent",
                avg_time_spent,
                f"Low average time spent ({avg_time_spent:.1f}) suggests weak engagement.",
            )

        avg_transaction_value = engineered_profile.get("avg_transaction_value")
        if pd.notna(avg_transaction_value) and avg_transaction_value <= thresholds.get(
            "avg_transaction_value_low", 15_000
        ):
            add_driver(
                "avg_transaction_value",
                avg_transaction_value,
                f"Low average transaction value ({avg_transaction_value:.2f}) points to declining commercial value.",
            )

        complaint_status = raw_profile.get("complaint_status")
        if str(raw_profile.get("past_complaint", "")).lower() == "yes" and complaint_status in {
            "Unsolved",
            "No Information Available",
        }:
            add_driver(
                "complaint_status",
                complaint_status,
                f"Past complaint with status '{complaint_status}' is elevating churn risk.",
            )

        feedback = raw_profile.get("feedback")
        if feedback in {
            "Poor Product Quality",
            "Poor Website",
            "Poor Customer Service",
            "Too many ads",
            "No reason specified",
        }:
            add_driver("feedback", feedback, f"Negative feedback ('{feedback}') is a direct churn warning sign.")

        points_in_wallet = engineered_profile.get("points_in_wallet")
        if pd.isna(points_in_wallet) or (
            pd.notna(points_in_wallet) and points_in_wallet <= thresholds.get("wallet_low", 620)
        ):
            add_driver(
                "points_in_wallet",
                points_in_wallet,
                "Low or missing wallet points suggest weak loyalty-program engagement.",
            )

        avg_frequency_login_days = engineered_profile.get("avg_frequency_login_days")
        if pd.notna(avg_frequency_login_days) and avg_frequency_login_days >= 18:
            add_driver(
                "avg_frequency_login_days",
                avg_frequency_login_days,
                f"Customers logging in only every {avg_frequency_login_days:.0f} days are showing reduced habit strength.",
            )

        deduplicated: list[dict[str, Any]] = []
        seen_features: set[str] = set()
        for item in driver_details:
            if item["feature"] in seen_features:
                continue
            seen_features.add(item["feature"])
            deduplicated.append(item)
            if len(deduplicated) >= 3:
                break
        return deduplicated


@lru_cache(maxsize=1)
def get_predictor_service() -> PredictorService:
    LOGGER.info("Loading predictor service from bundle at %s", get_settings().bundle_path)
    return PredictorService()
