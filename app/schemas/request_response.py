from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CustomerPayload(BaseModel):
    customer_id: str | None = None
    Name: str | None = None
    age: int | None = None
    gender: str | None = None
    security_no: str | None = None
    region_category: str | None = None
    membership_category: str | None = None
    joining_date: str | None = None
    joined_through_referral: str | None = None
    referral_id: str | None = None
    preferred_offer_types: str | None = None
    medium_of_operation: str | None = None
    internet_option: str | None = None
    last_visit_time: str | None = None
    days_since_last_login: float | None = None
    avg_time_spent: float | None = None
    avg_transaction_value: float | None = None
    avg_frequency_login_days: str | float | None = None
    points_in_wallet: float | None = None
    used_special_discount: str | None = None
    offer_application_preference: str | None = None
    past_complaint: str | None = None
    complaint_status: str | None = None
    feedback: str | None = None


class BatchPredictRequest(BaseModel):
    records: list[CustomerPayload]


class DriverDetail(BaseModel):
    feature: str
    feature_value: Any | None = None
    direction: str
    message: str


class PredictionResponse(BaseModel):
    customer_id: str | None = None
    predicted_class: Any
    predicted_label: str
    risk_score: float
    risk_band: str
    confidence: float | None = None
    top_risk_drivers: list[str]
    driver_details: list[DriverDetail]
    recommendations: list[str]
    probability_breakdown: dict[str, float] = Field(default_factory=dict)


class BatchPredictionResponse(BaseModel):
    prediction_count: int
    predictions: list[PredictionResponse]


class RecommendationRequest(BaseModel):
    customer: CustomerPayload
    risk_band: str | None = None
    top_risk_drivers: list[str] | None = None


class RecommendationResponse(BaseModel):
    risk_band: str
    recommendations: list[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    project_name: str
    version: str


class ModelInfoResponse(BaseModel):
    project_name: str
    project_version: str
    trained_at: str
    task_detection: dict[str, Any]
    target_normalization: dict[str, Any] | None = None
    reference_date: str
    best_model_name: str
    validation_metrics: dict[str, Any]
    feature_summary: dict[str, Any]
    global_feature_importance: list[dict[str, Any]]
