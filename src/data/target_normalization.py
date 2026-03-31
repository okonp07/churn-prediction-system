from __future__ import annotations

from typing import Any, Mapping

import pandas as pd


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return value
        try:
            numeric = float(stripped)
        except ValueError:
            return value
        return int(numeric) if numeric.is_integer() else numeric
    return value


def resolve_target_normalization_map(raw_mapping: Mapping[Any, Any] | None) -> dict[Any, Any]:
    if not raw_mapping:
        return {}
    return {_coerce_scalar(key): _coerce_scalar(value) for key, value in raw_mapping.items()}


def normalize_target_series(
    target: pd.Series,
    raw_mapping: Mapping[Any, Any] | None,
) -> pd.Series:
    mapping = resolve_target_normalization_map(raw_mapping)
    if not mapping:
        return target.copy()

    normalized = target.copy()
    non_null_mask = normalized.notna()
    normalized.loc[non_null_mask] = normalized.loc[non_null_mask].map(lambda value: mapping.get(value, value))
    return normalized


def normalize_target_frame(
    df: pd.DataFrame,
    target_column: str,
    raw_mapping: Mapping[Any, Any] | None,
) -> pd.DataFrame:
    normalized = df.copy()
    if target_column in normalized.columns:
        normalized[target_column] = normalize_target_series(normalized[target_column], raw_mapping)
    return normalized


def describe_target_normalization(
    original_target: pd.Series,
    normalized_target: pd.Series,
    raw_mapping: Mapping[Any, Any] | None,
) -> dict[str, Any]:
    mapping = resolve_target_normalization_map(raw_mapping)
    source_classes = sorted(pd.Series(original_target).dropna().unique().tolist())
    normalized_classes = sorted(pd.Series(normalized_target).dropna().unique().tolist())
    effective_mapping = {
        str(source_value): mapped_value
        for source_value, mapped_value in mapping.items()
        if source_value in source_classes and source_value != mapped_value
    }

    if effective_mapping:
        note = (
            "The raw churn labels were normalized to a business-friendly 1-to-5 scale before training. "
            "The legacy '-1' tier is merged into score '1', so score 1 now represents the lowest-risk customers."
        )
    else:
        note = "No target-label normalization was applied before training."

    return {
        "applied": bool(effective_mapping),
        "source_classes": source_classes,
        "normalized_classes": normalized_classes,
        "mapping": effective_mapping,
        "notes": note,
    }
