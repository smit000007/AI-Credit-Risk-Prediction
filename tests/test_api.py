"""Integration tests for the FastAPI Credit Risk Prediction endpoints."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_and_model_info_endpoints():
    """Verify /health and /model-info return valid metadata."""
    res_health = client.get("/health")
    assert res_health.status_code == 200
    data = res_health.json()
    assert data["status"] == "healthy"
    assert data["feature_count"] >= 15

    res_info = client.get("/model-info")
    assert res_info.status_code == 200
    info = res_info.json()
    assert "best_model" in info
    assert len(info["metrics_comparison"]) == 4


def test_predict_endpoint_low_and_high_risk():
    """Verify /predict returns Risk Tier, Probability, Credit Score, and SHAP explanations."""
    low_risk_payload = {
        "RevolvingUtilizationOfUnsecuredLines": 0.12,
        "age": 50,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.20,
        "MonthlyIncome": 9000.0,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 2,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 1.0,
    }
    res_low = client.post("/predict", json=low_risk_payload)
    assert res_low.status_code == 200
    body_low = res_low.json()
    assert body_low["risk_tier"] == "Low"
    assert 0.0 <= body_low["default_probability"] < 0.20
    assert len(body_low["shap_reasons"]) == 6

    high_risk_payload = {
        "RevolvingUtilizationOfUnsecuredLines": 1.18,
        "age": 25,
        "NumberOfTime30-59DaysPastDueNotWorse": 3,
        "DebtRatio": 0.88,
        "MonthlyIncome": 2200.0,
        "NumberOfOpenCreditLinesAndLoans": 4,
        "NumberOfTimes90DaysLate": 3,
        "NumberRealEstateLoansOrLines": 0,
        "NumberOfTime60-89DaysPastDueNotWorse": 2,
        "NumberOfDependents": 3.0,
    }
    res_high = client.post("/predict", json=high_risk_payload)
    assert res_high.status_code == 200
    body_high = res_high.json()
    assert body_high["default_probability"] > body_low["default_probability"]
    assert body_high["risk_tier"] in ("Medium", "High")
