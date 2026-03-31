from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"


@dataclass(slots=True)
class AppSettings:
    project_name: str
    project_version: str
    random_seed: int
    target_column: str
    id_columns: list[str]
    leakage_columns: list[str]
    negative_feedback_values: list[str]
    validation_split: float
    cross_validation_folds: int
    high_risk_top_n_classes: int
    feature_importance_top_n: int
    batch_prediction_output: str
    target_normalization_map: dict[str, int] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    project_root: Path = PROJECT_ROOT

    @property
    def bundle_path(self) -> Path:
        return self.project_root / self.artifacts["bundle_path"]

    @property
    def metrics_path(self) -> Path:
        return self.project_root / self.artifacts["metrics_path"]

    @property
    def comparison_path(self) -> Path:
        return self.project_root / self.artifacts["comparison_path"]

    @property
    def feature_importance_path(self) -> Path:
        return self.project_root / self.artifacts["feature_importance_path"]

    @property
    def prediction_preview_path(self) -> Path:
        return self.project_root / self.artifacts["prediction_preview_path"]

    @property
    def raw_data_dir(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.project_root / "data" / "processed"


def _load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = Path(os.getenv("CHURN_CONFIG_PATH", config_path or DEFAULT_CONFIG_PATH))
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


@lru_cache(maxsize=1)
def get_settings(config_path: Path | None = None) -> AppSettings:
    config = _load_config(config_path)
    return AppSettings(**config)
