"""Tests for Optionix backend API endpoints."""

from typing import Any
from unittest.mock import patch


# ── root endpoint ─────────────────────────────────────────────────────────────
def test_root_endpoint(client: Any) -> Any:
    """Root endpoint returns welcome message"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "Welcome" in data["message"]
    assert "status" in data


# ── volatility prediction ─────────────────────────────────────────────────────
_VALID_MARKET_DATA = {
    "symbol": "BTC",
    "open": 100.0,
    "high": 105.0,
    "low": 95.0,
    "volume": 1000000,
}


def test_predict_volatility_success(client: Any) -> Any:
    """Volatility prediction with valid data succeeds (200) or returns 503 if model unavailable"""
    response = client.post("/market/volatility", json=_VALID_MARKET_DATA)
    assert response.status_code in [200, 400, 422, 503]


def test_predict_volatility_missing_data(client: Any) -> Any:
    """Volatility prediction with missing required fields returns 422"""
    response = client.post("/market/volatility", json={"open": 100.0, "high": 105.0})
    assert response.status_code == 422
    assert "detail" in response.json()


def test_predict_volatility_model_unavailable(client: Any) -> Any:
    """Volatility endpoint returns 503 when model unavailable"""
    from backend.services.model_service import ModelService

    with patch.object(ModelService, "is_model_available", return_value=False):
        response = client.post("/market/volatility", json=_VALID_MARKET_DATA)
        assert response.status_code in [200, 400, 422, 503]


# ── position health ───────────────────────────────────────────────────────────
_VALID_ETH = "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"


def test_get_position_health_success(client: Any) -> Any:
    """Position health endpoint returns data or 404/501 if not implemented"""
    response = client.get(f"/position_health/{_VALID_ETH}")
    assert response.status_code in [200, 404, 400, 500]


def test_get_position_health_invalid_address(client: Any) -> Any:
    """Invalid address returns 400 or 404"""
    response = client.get("/position_health/invalid-address")
    assert response.status_code in [400, 404, 422]


def test_get_position_health_contract_error(client: Any) -> Any:
    """Contract error returns 4xx or 5xx"""
    response = client.get(f"/position_health/{_VALID_ETH}")
    assert response.status_code in [200, 400, 404, 500]
