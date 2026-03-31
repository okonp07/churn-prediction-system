from __future__ import annotations

import pandas as pd

from app.core.config import get_settings
from src.features.build_features import ChurnFeatureBuilder
from src.data.target_normalization import normalize_target_frame, resolve_target_normalization_map
from src.models.target_manager import TargetManager


def prepare_training_matrices(
    df: pd.DataFrame,
    target_manager: TargetManager | None = None,
) -> tuple[ChurnFeatureBuilder, TargetManager, pd.DataFrame, pd.Series]:
    settings = get_settings()
    normalization_map = resolve_target_normalization_map(settings.target_normalization_map)
    normalized_df = normalize_target_frame(df, settings.target_column, normalization_map)
    target_manager = target_manager or TargetManager().fit(
        normalized_df[settings.target_column],
        original_target=df[settings.target_column],
        normalization_map=normalization_map,
    )
    feature_builder = ChurnFeatureBuilder()
    features = feature_builder.fit_transform(normalized_df.drop(columns=[settings.target_column]))
    target = pd.Series(
        target_manager.transform(normalized_df[settings.target_column]),
        name=settings.target_column,
    )
    return feature_builder, target_manager, features, target


def prepare_inference_matrix(df: pd.DataFrame, feature_builder: ChurnFeatureBuilder) -> pd.DataFrame:
    return feature_builder.transform(df)
