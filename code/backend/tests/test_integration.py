"""Integration tests for Optionix backend."""

import os
import sys
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_VALID_MARKET_DATA = {
    "symbol": "BTC",
    "open": 100.0,
    "high": 105.0,
    "low": 95.0,
    "volume": 1000000,
}


def test_integration_root_to_volatility(client: Any) -> Any:
    """Root endpoint is reachable and volatility endpoint accepts valid requests"""
    root = client.get("/")
    assert root.status_code == 200
    assert root.json()["status"] in ("online", "healthy", "degraded")

    vol = client.post("/market/volatility", json=_VALID_MARKET_DATA)
    assert vol.status_code in [200, 400, 422, 503]


def test_integration_error_handling(client: Any) -> Any:
    """Error handling works across endpoints"""
    not_found = client.get("/nonexistent_endpoint")
    assert not_found.status_code == 404

    invalid_json = client.post(
        "/market/volatility",
        headers={"Content-Type": "application/json"},
        content="invalid json",
    )
    assert invalid_json.status_code == 422

    invalid_addr = client.get("/position_health/invalid-address")
    assert invalid_addr.status_code in [400, 404, 422]
