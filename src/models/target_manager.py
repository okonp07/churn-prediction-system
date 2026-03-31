from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TargetManager:
    task_type: str = ""
    classes_: list[Any] = field(default_factory=list)
    source_classes_: list[Any] = field(default_factory=list)
    target_min_: float | None = None
    target_max_: float | None = None
    normalization_map_: dict[Any, Any] = field(default_factory=dict)
    strategy_details_: dict[str, Any] = field(default_factory=dict)

    def fit(
        self,
        target: pd.Series,
        original_target: pd.Series | None = None,
        normalization_map: dict[Any, Any] | None = None,
    ) -> "TargetManager":
        clean_target = target.dropna()
        source_target = original_target.dropna() if original_target is not None else clean_target
        unique_values = np.sort(clean_target.unique())
        source_unique_values = np.sort(source_target.unique())
        is_numeric = pd.api.types.is_numeric_dtype(clean_target)
        integer_like = is_numeric and np.allclose(unique_values, np.round(unique_values))
        self.source_classes_ = source_unique_values.tolist()
        self.normalization_map_ = dict(normalization_map or {})

        if len(unique_values) == 2:
            self.task_type = "binary_classification"
            self.classes_ = unique_values.tolist()
        elif is_numeric and integer_like and len(unique_values) <= 12:
            self.task_type = "ordinal_multiclass_classification"
            self.classes_ = unique_values.tolist()
        elif len(unique_values) <= 20:
            self.task_type = "multiclass_classification"
            self.classes_ = unique_values.tolist()
        else:
            self.task_type = "regression"
            self.target_min_ = float(clean_target.min())
            self.target_max_ = float(clean_target.max())

        self.strategy_details_ = {
            "source_unique_values": source_unique_values.tolist(),
            "unique_values": unique_values.tolist(),
            "unique_count": int(len(unique_values)),
            "is_numeric": bool(is_numeric),
            "integer_like": bool(integer_like),
            "normalization_applied": bool(
                self.normalization_map_ and source_unique_values.tolist() != unique_values.tolist()
            ),
            "normalization_map": {
                str(source_value): mapped_value
                for source_value, mapped_value in self.normalization_map_.items()
                if source_value in source_unique_values and source_value != mapped_value
            },
            "strategy": self.task_type,
            "notes": self._strategy_note(),
        }
        return self

    def transform(self, target: pd.Series) -> np.ndarray:
        if self.task_type == "regression":
            return target.to_numpy(dtype=float)
        mapping = {label: index for index, label in enumerate(self.classes_)}
        return target.map(mapping).to_numpy(dtype=int)

    def inverse_transform(self, values: np.ndarray | list[int | float]) -> np.ndarray:
        if self.task_type == "regression":
            return np.asarray(values, dtype=float)
        classes = np.asarray(self.classes_)
        indices = np.asarray(values, dtype=int)
        return classes[indices]

    def encode_label(self, label: Any) -> int | float:
        if self.task_type == "regression":
            return float(label)
        return {value: idx for idx, value in enumerate(self.classes_)}[label]

    def decode_label(self, encoded: int | float) -> Any:
        if self.task_type == "regression":
            return float(encoded)
        return self.classes_[int(encoded)]

    def normalized_score_from_prediction(
        self, probabilities: np.ndarray | None = None, predictions: np.ndarray | None = None
    ) -> np.ndarray:
        if self.task_type == "regression":
            assert predictions is not None
            value_range = max((self.target_max_ or 1.0) - (self.target_min_ or 0.0), 1e-9)
            return np.clip((np.asarray(predictions) - (self.target_min_ or 0.0)) / value_range, 0, 1)

        assert probabilities is not None
        class_positions = np.arange(probabilities.shape[1], dtype=float)
        score = probabilities @ class_positions
        denominator = max(len(self.classes_) - 1, 1)
        return np.clip(score / denominator, 0, 1)

    def risk_band(self, score: float) -> str:
        if score < 0.34:
            return "Low"
        if score < 0.67:
            return "Medium"
        return "High"

    def label_name(self, raw_value: Any) -> str:
        if self.task_type == "regression":
            return f"Predicted score {raw_value:.2f}"
        return f"Risk Tier {raw_value}"

    def metadata(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "classes": self.classes_,
            "source_classes": self.source_classes_ or self.classes_,
            "target_min": self.target_min_,
            "target_max": self.target_max_,
            "normalization_map": {
                str(source_value): mapped_value
                for source_value, mapped_value in self.normalization_map_.items()
                if source_value != mapped_value
            },
            "strategy_details": self.strategy_details_,
        }

    def _strategy_note(self) -> str:
        normalization_applied = bool(self.normalization_map_ and (self.source_classes_ or self.classes_) != self.classes_)
        if self.task_type == "ordinal_multiclass_classification":
            if normalization_applied:
                changed_pairs = ", ".join(
                    f"{source}->{mapped}"
                    for source, mapped in self.normalization_map_.items()
                    if source != mapped
                )
                return (
                    "Original target labels were normalized before modeling "
                    f"({changed_pairs}) so the deployed system exposes a cleaner business-facing risk scale. "
                    "The normalized target remains numeric, ordered, and low cardinality, so the system uses "
                    "ordinal-aware metrics during model selection."
                )
            return (
                "Target is numeric, integer-like, and low cardinality, so the system preserves the ordered "
                "risk meaning and uses ordinal-aware metrics during model selection."
            )
        if self.task_type == "binary_classification":
            return "Target has two distinct values, so the system uses binary classification."
        if self.task_type == "multiclass_classification":
            return "Target has a limited number of classes, so the system uses multiclass classification."
        return "Target has many distinct values, so the system uses regression."
