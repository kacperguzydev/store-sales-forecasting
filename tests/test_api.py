from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_check():
    """The root endpoint should confirm the API is running."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_valid_response():
    """A valid prediction request should return a non-negative sales prediction."""
    payload = {
        "store_nbr": 1,
        "family": "BEVERAGES",
        "date": "2017-08-20",
        "onpromotion": 5
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_sales"] >= 0
    assert data["store_nbr"] == 1
    assert data["family"] == "BEVERAGES"


def test_predict_zero_forced_family_returns_zero():
    """A family with no sales in the past year should be zero-forced."""
    payload = {
        "store_nbr": 1,
        "family": "BOOKS",
        "date": "2017-08-20",
        "onpromotion": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["predicted_sales"] < 1.0


def test_predict_invalid_family_returns_404():
    """An unknown product family should return a 404 error, not crash the API."""
    payload = {
        "store_nbr": 1,
        "family": "NOT_A_REAL_FAMILY",
        "date": "2017-08-20",
        "onpromotion": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 404


def test_predict_invalid_date_format_returns_400():
    """A malformed date string should return a 400 error, not crash the API."""
    payload = {
        "store_nbr": 1,
        "family": "BEVERAGES",
        "date": "not-a-date",
        "onpromotion": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400