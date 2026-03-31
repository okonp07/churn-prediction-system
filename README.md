# Churn Prediction System

[![Live Streamlit App](https://img.shields.io/badge/Live%20App-Open%20Streamlit-black?style=for-the-badge&logo=streamlit&logoColor=white)](https://churn-prediction-system-d26clfwqqdxokbwmysq7ek.streamlit.app/)

## Live Demo

[Launch the Streamlit app](https://churn-prediction-system-d26clfwqqdxokbwmysq7ek.streamlit.app/)

This project upgrades the original notebook into a production-style churn prediction platform. It trains a stronger leakage-safe model, exposes predictions through FastAPI, provides a Streamlit interface for stakeholders, explains predictions, and returns deterministic retention recommendations.

## Architecture Summary

- `src/` contains data validation, feature engineering, target strategy detection, model training, evaluation, selection, and reusable inference helpers.
- `app/` contains production services for prediction, explanation, recommendation generation, configuration, and the FastAPI application.
- `frontend/streamlit_app.py` is the stakeholder-facing demo UI.
- `artifacts/` stores the trained model bundle, metrics, plots, and sample outputs.
- `notebooks/` contains a teaching notebook that walks through the full productionized workflow.

## Target Strategy

The system automatically inspects `churn_risk_score` before modeling:

- Binary target: binary classification
- Low-cardinality integer target: ordinal-aware multiclass classification
- Higher-cardinality target: regression

For this dataset, the raw target values were originally `[-1, 1, 2, 3, 4, 5]`. The production pipeline normalizes that into a cleaner business-facing scale of `[1, 2, 3, 4, 5]` by merging the legacy `-1` tier into score `1`, where `1` is the lowest churn risk and `5` is the highest. The system then treats the target as ordinal multiclass classification and uses weighted F1, quadratic weighted kappa, and ordinal MAE during model selection.

## Project Structure

```text
churn_prediction_system/
├── app/
│   ├── api/main.py
│   ├── core/
│   ├── schemas/request_response.py
│   ├── services/
│   └── utils/helpers.py
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── pipelines/training_pipeline.py
├── frontend/streamlit_app.py
├── artifacts/
├── data/raw/
├── notebooks/exploratory_analysis.ipynb
├── tests/
├── requirements.txt
├── Dockerfile
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Train the Model

```bash
python -m src.pipelines.training_pipeline
```

This command will:

- validate the data
- detect the appropriate ML task type
- engineer production-safe features
- benchmark multiple models
- select the strongest model using business-aware metrics
- save the reusable model bundle to `artifacts/model/churn_model_bundle.joblib`
- export metrics, plots, and sample predictions into `artifacts/`

## Serve the API

```bash
uvicorn app.api.main:app --reload
```

Important endpoints:

- `GET /health`
- `GET /model-info`
- `POST /predict`
- `POST /predict_batch`
- `POST /recommend`

## Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

## Deploy to Streamlit Community Cloud

This repository is now organized for Streamlit Community Cloud:

- root entrypoint: `streamlit_app.py`
- runtime dependencies: `requirements.txt`
- optional local/dev dependencies: `requirements-dev.txt`
- app configuration: `.streamlit/config.toml`

Deployment steps:

1. Push this folder to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and click `Create app`.
3. Select your repository and branch.
4. Set the entrypoint file to `streamlit_app.py`.
5. In `Advanced settings`, choose Python `3.11` for the safest package compatibility.
6. Deploy.

Streamlit Community Cloud docs:

- [Deploy your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [File organization](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization)
- [App dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)

## CLI Inference

```bash
python -m src.models.predict --input data/raw/test.csv --output artifacts/sample_outputs/cli_predictions.csv
```

## Example API Requests

Single prediction:

```json
{
  "customer_id": "demo-customer-001",
  "Name": "Ava Johnson",
  "age": 35,
  "gender": "F",
  "security_no": "SEC-DEMO-001",
  "region_category": "City",
  "membership_category": "Premium Membership",
  "joining_date": "2017-05-10",
  "joined_through_referral": "No",
  "referral_id": "CID-REF-001",
  "preferred_offer_types": "Gift Vouchers/Coupons",
  "medium_of_operation": "Smartphone",
  "internet_option": "Wi-Fi",
  "last_visit_time": "21:15:00",
  "days_since_last_login": 19,
  "avg_time_spent": 48.0,
  "avg_transaction_value": 12500.0,
  "avg_frequency_login_days": "12",
  "points_in_wallet": 240.0,
  "used_special_discount": "Yes",
  "offer_application_preference": "No",
  "past_complaint": "Yes",
  "complaint_status": "Unsolved",
  "feedback": "Poor Customer Service"
}
```

Batch JSON scoring:

```json
{
  "records": [
    {
      "customer_id": "demo-customer-001",
      "Name": "Ava Johnson",
      "age": 35,
      "gender": "F",
      "security_no": "SEC-DEMO-001",
      "region_category": "City",
      "membership_category": "Premium Membership",
      "joining_date": "2017-05-10",
      "joined_through_referral": "No",
      "referral_id": "CID-REF-001",
      "preferred_offer_types": "Gift Vouchers/Coupons",
      "medium_of_operation": "Smartphone",
      "internet_option": "Wi-Fi",
      "last_visit_time": "21:15:00",
      "days_since_last_login": 19,
      "avg_time_spent": 48.0,
      "avg_transaction_value": 12500.0,
      "avg_frequency_login_days": "12",
      "points_in_wallet": 240.0,
      "used_special_discount": "Yes",
      "offer_application_preference": "No",
      "past_complaint": "Yes",
      "complaint_status": "Unsolved",
      "feedback": "Poor Customer Service"
    }
  ]
}
```

## Testing

```bash
pytest
```

## Assumptions

- The raw `-1` target value is merged into churn score `1` so the deployed system exposes a cleaner and more interpretable `1` to `5` business scale.
- Tenure is computed against the dataset snapshot reference date learned during training so inference stays consistent with model training.
- Recommendations combine model signals with deterministic business rules to stay readable and reliable in demos.
