from __future__ import annotations

import pandas as pd

from app.services.predictor import PredictorService
from src.data.target_normalization import normalize_target_series


def test_target_normalization_maps_legacy_labels_to_business_scale() -> None:
    raw_target = pd.Series([-1, 1, 2, 3, 4, 5, None], name="churn_risk_score")
    normalized = normalize_target_series(raw_target, {"-1": 1, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5})

    assert normalized.dropna().tolist() == [1, 1, 2, 3, 4, 5]


def test_model_metadata_reports_normalized_classes(ensure_trained_bundle) -> None:
    predictor = PredictorService()
    model_info = predictor.model_info()

    assert model_info["task_detection"]["classes"] == [1, 2, 3, 4, 5]
    assert model_info["target_normalization"]["normalized_classes"] == [1, 2, 3, 4, 5]
    assert model_info["target_normalization"]["lowest_risk_class"] == 1
    assert model_info["target_normalization"]["highest_risk_class"] == 5
