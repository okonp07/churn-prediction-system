from __future__ import annotations

import pytest


pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from app.api.main import app
from src.data.load_data import load_train_data


def test_health_endpoint(ensure_trained_bundle) -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"


def test_predict_endpoint(ensure_trained_bundle) -> None:
    client = TestClient(app)
    sample = load_train_data().drop(columns=["churn_risk_score"]).iloc[0].to_dict()
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    payload = response.json()
    assert "risk_score" in payload
    assert "recommendations" in payload
