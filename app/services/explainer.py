from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

try:
    import shap
except ImportError:  # pragma: no cover - exercised indirectly when dependency is absent
    shap = None


TREE_MODEL_NAMES = {"XGBClassifier", "XGBRegressor", "LGBMClassifier", "LGBMRegressor", "RandomForestClassifier", "RandomForestRegressor"}


class ExplainerService:
    def __init__(self, bundle: dict[str, Any]):
        self.bundle = bundle
        self.model_pipeline = bundle["model"]
        self.feature_builder = bundle["feature_builder"]
        self.target_manager = bundle["target_manager"]
        self.preprocessor = self.model_pipeline.named_steps["preprocessor"]
        self.estimator = self.model_pipeline.named_steps["model"]

    @lru_cache(maxsize=1)
    def _get_shap_explainer(self) -> Any | None:
        if shap is None:
            return None
        try:
            estimator_name = self.estimator.__class__.__name__
            if estimator_name in TREE_MODEL_NAMES:
                return shap.TreeExplainer(self.estimator)
            if hasattr(self.estimator, "coef_"):
                background = np.zeros((1, len(self.preprocessor.get_feature_names_out())))
                return shap.LinearExplainer(self.estimator, background)
        except Exception:
            return None
        return None

    def local_driver_details(
        self,
        raw_frame: pd.DataFrame,
        predicted_class_index: int,
        top_n: int = 3,
    ) -> list[dict[str, Any]]:
        feature_frame = self.feature_builder.transform(raw_frame)
        transformed = self.preprocessor.transform(feature_frame)
        explainer = self._get_shap_explainer()

        if explainer is None:
            return self._fallback_driver_details(feature_frame.iloc[0].to_dict(), top_n=top_n)

        try:
            shap_values = explainer.shap_values(transformed)
        except Exception:
            return self._fallback_driver_details(feature_frame.iloc[0].to_dict(), top_n=top_n)

        if isinstance(shap_values, list):
            if len(shap_values) > 1:
                weights = np.linspace(0.0, 1.0, len(shap_values))
                local_values = np.tensordot(weights, np.stack([item[0] for item in shap_values]), axes=(0, 0))
            else:
                local_values = shap_values[predicted_class_index][0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            if shap_values.shape[-1] > 1:
                weights = np.linspace(0.0, 1.0, shap_values.shape[-1])
                local_values = shap_values[0] @ weights
            else:
                local_values = shap_values[0, :, predicted_class_index]
        else:
            local_values = shap_values[0]

        feature_names = self.preprocessor.get_feature_names_out()
        details_frame = pd.DataFrame(
            {
                "transformed_feature": feature_names,
                "importance": np.abs(local_values),
                "direction": np.sign(local_values),
            }
        )
        details_frame["base_feature"] = details_frame["transformed_feature"].apply(self._base_feature_name)
        grouped = (
            details_frame.groupby("base_feature", as_index=False)
            .agg({"importance": "sum", "direction": "mean"})
            .sort_values("importance", ascending=False)
        )
        raw_features = feature_frame.iloc[0].to_dict()
        grouped = grouped[grouped["direction"] > 0]

        translated_details: list[dict[str, Any]] = []
        for _, row in grouped.iterrows():
            feature_name = row["base_feature"]
            feature_value = raw_features.get(feature_name)
            if self._skip_feature(feature_name, feature_value):
                continue
            translated_details.append(self._translate_driver(feature_name, feature_value, row["direction"]))
            if len(translated_details) >= top_n:
                break

        if translated_details:
            return translated_details
        return self._fallback_driver_details(feature_frame.iloc[0].to_dict(), top_n=top_n)

    def global_feature_importance(self, top_n: int = 15) -> list[dict[str, Any]]:
        preprocessor = self.model_pipeline.named_steps["preprocessor"]
        estimator = self.model_pipeline.named_steps["model"]
        feature_names = preprocessor.get_feature_names_out()

        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            coefficients = estimator.coef_
            importances = np.mean(np.abs(coefficients), axis=0) if coefficients.ndim > 1 else np.abs(coefficients)
        else:
            importances = np.zeros(len(feature_names))

        importance_frame = pd.DataFrame({"feature": feature_names, "importance": importances})
        importance_frame["base_feature"] = importance_frame["feature"].apply(self._base_feature_name)
        grouped = (
            importance_frame.groupby("base_feature", as_index=False)["importance"]
            .sum()
            .sort_values("importance", ascending=False)
            .head(top_n)
        )
        return grouped.to_dict(orient="records")

    def _fallback_driver_details(self, row: dict[str, Any], top_n: int) -> list[dict[str, Any]]:
        candidates = [
            ("days_since_last_login", row.get("days_since_last_login")),
            ("avg_time_spent", row.get("avg_time_spent")),
            ("avg_transaction_value", row.get("avg_transaction_value")),
            ("complaint_status", row.get("complaint_status")),
            ("feedback", row.get("feedback")),
        ]
        heuristics: list[dict[str, Any]] = []
        for feature_name, feature_value in candidates:
            if self._skip_feature(feature_name, feature_value):
                continue
            heuristics.append(self._translate_driver(feature_name, feature_value, 1.0))
            if len(heuristics) >= top_n:
                break
        return heuristics

    @staticmethod
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

    def _translate_driver(self, feature_name: str, feature_value: Any, direction: float) -> dict[str, Any]:
        direction_word = "increasing" if direction >= 0 else "reducing"
        message = f"{feature_name.replace('_', ' ').title()} is {direction_word} churn risk."

        if feature_name == "days_since_last_login" and feature_value is not None:
            message = f"Long time since last login ({feature_value:.0f} days) is increasing churn risk."
        elif feature_name == "avg_time_spent" and feature_value is not None:
            message = f"Average time spent ({feature_value:.1f}) suggests weaker engagement and higher churn risk."
        elif feature_name == "avg_transaction_value" and feature_value is not None:
            message = f"Transaction value ({feature_value:.2f}) is materially influencing the current risk estimate."
        elif feature_name == "complaint_status":
            message = f"Complaint status '{feature_value}' is strongly influencing the risk estimate."
        elif feature_name == "feedback":
            message = f"Customer feedback '{feature_value}' is one of the strongest risk signals."
        elif feature_name == "membership_category":
            message = f"Membership category '{feature_value}' contributes meaningfully to the predicted risk."
        elif feature_name == "points_in_wallet" and feature_value is not None:
            message = f"Wallet points ({feature_value:.2f}) are shaping the churn risk estimate."

        return {
            "feature": feature_name,
            "feature_value": None if pd.isna(feature_value) else feature_value,
            "direction": "up" if direction >= 0 else "down",
            "message": message,
        }

    @staticmethod
    def _skip_feature(feature_name: str, feature_value: Any) -> bool:
        if pd.isna(feature_value):
            return True
        if feature_name.endswith("_flag") and float(feature_value) == 0.0:
            return True
        return False
