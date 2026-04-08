"""
Comprehensive test suite for Optionix backend.
Tests all major components including authentication, trading, compliance, and security.
"""

import os
import time

os.environ.setdefault("ENVIRONMENT", "testing")
os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-at-least-32-chars-long")
os.environ.setdefault("ENCRYPTION_KEY", "TestEncryptionKey1234567890!!!!!")
os.environ.setdefault("DATABASE_URL", "sqlite:///./backend_test.db")

from decimal import Decimal
from typing import Any

# Import all ORM model modules so they register with Base before create_all
import data_protection  # noqa: F401
import middleware.audit_logging  # noqa: F401
import pytest
from app import app
from auth import AuthService
from database import Base, get_db
from fastapi.testclient import TestClient
from models import User
from security import security_service
from services.compliance_service import compliance_service
from services.financial_service import FinancialCalculationService
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# ── test DB setup ─────────────────────────────────────────────────────────────
TEST_DB_URL = "sqlite:///./backend_test.db"
test_engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

# Create ALL tables on the test engine before any tests run
Base.metadata.create_all(bind=test_engine)

auth_service_inst = AuthService()


def override_get_db() -> Any:
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app, raise_server_exceptions=False)


# ── helpers ───────────────────────────────────────────────────────────────────
def _unique_email(tag: str = "") -> str:
    return f"test_{tag}_{int(time.time() * 1000)}@example.com"


def _register(
    email: str, password: str = "TestPassword123!", full_name: str = "Test User"
):
    return client.post(
        "/auth/register",
        json={"email": email, "password": password, "full_name": full_name},
    )


def _login(email: str, password: str = "TestPassword123!"):
    return client.post("/auth/login", json={"email": email, "password": password})


def _auth_headers() -> dict:
    email = _unique_email("auth")
    _register(email)
    r = _login(email)
    assert r.status_code == 200, f"Login failed: {r.text}"
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


# ── Authentication tests ──────────────────────────────────────────────────────
class TestAuthentication:
    """Test authentication and authorization"""

    def test_user_registration(self) -> Any:
        response = _register(_unique_email("reg"))
        assert response.status_code == 200, response.text
        data = response.json()
        assert "email" in data
        assert data["is_active"] is True
        assert data["is_verified"] is False

    def test_user_registration_duplicate_email(self) -> Any:
        email = _unique_email("dup")
        r1 = _register(email, full_name="Duplicate User")
        assert r1.status_code == 200, r1.text
        r2 = _register(email, full_name="Duplicate User")
        assert r2.status_code in [400, 422]
        body = r2.json()
        body.get("detail", "")
        assert r2.status_code in [400, 422]  # duplicate rejected

    def test_weak_password_rejection(self) -> Any:
        r = _register(_unique_email("weak"), password="weak")
        # Pydantic returns 422 for schema validation, app returns 400 for business logic
        assert r.status_code in [400, 422]

    def test_user_login(self) -> Any:
        email = _unique_email("login")
        _register(email)
        r = _login(email)
        assert r.status_code == 200, r.text
        data = r.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_invalid_login(self) -> Any:
        r = _login("nobody@nowhere.com", "WrongPassword99!")
        assert r.status_code == 401
        body = r.json()
        assert body.get("detail") is not None or body.get("message") is not None


# ── Security tests ────────────────────────────────────────────────────────────
class TestSecurity:
    """Test security utilities and validation"""

    def test_password_strength_validation(self) -> Any:
        result = security_service.validate_password_strength("StrongPass123!")
        assert result["valid"] is True
        assert result["strength"] == "strong"
        result = security_service.validate_password_strength("weak")
        assert result["valid"] is False
        assert result["strength"] == "weak"

    def test_ethereum_address_validation(self) -> Any:
        assert (
            security_service.validate_ethereum_address(
                "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
            )
            is True
        )
        assert security_service.validate_ethereum_address("0xinvalid") is False

    def test_api_key_generation(self) -> Any:
        plain_key, hashed_key = security_service.generate_api_key()
        assert plain_key.startswith("ok_")
        assert len(plain_key) == 46
        assert security_service.validate_api_key_format(plain_key) is True
        assert security_service.hash_api_key(plain_key) == hashed_key

    def test_data_encryption(self) -> Any:
        original_data = "sensitive information"
        encrypted = security_service.encrypt_sensitive_data(original_data)
        assert encrypted.encrypted_data != original_data
        decrypted = security_service.decrypt_sensitive_data(encrypted)
        assert decrypted == original_data

    def test_input_sanitization(self) -> Any:
        sanitized = security_service.sanitize_input("<script>alert('xss')</script>")
        assert "<script>" not in sanitized
        assert "alert" in sanitized  # text content preserved; only tags stripped


# ── Financial calculation tests ───────────────────────────────────────────────
class TestFinancialCalculations:
    """Test financial calculation service"""

    def setup_method(self) -> Any:
        self.financial_service = FinancialCalculationService()

    def test_liquidation_price_calculation(self) -> Any:
        ep, ps, im = Decimal("50000"), Decimal("1.0"), Decimal("5000")
        assert self.financial_service.calculate_liquidation_price(ep, ps, True, im) < ep
        assert (
            self.financial_service.calculate_liquidation_price(ep, ps, False, im) > ep
        )

    def test_margin_requirement_calculation(self) -> Any:
        margin = self.financial_service.calculate_margin_requirement(
            Decimal("100000"), Decimal("10")
        )
        assert margin == Decimal("10000")

    def test_unrealized_pnl_calculation(self) -> Any:
        ep, cp, ps = Decimal("50000"), Decimal("55000"), Decimal("1.0")
        assert self.financial_service.calculate_unrealized_pnl(
            ep, cp, ps, True
        ) == Decimal("5000")
        assert self.financial_service.calculate_unrealized_pnl(
            ep, cp, ps, False
        ) == Decimal("-5000")

    def test_trading_fees_calculation(self) -> Any:
        tv = Decimal("10000")
        assert self.financial_service.calculate_trading_fees(
            tv, is_maker=False
        ) == Decimal("10.00")
        assert self.financial_service.calculate_trading_fees(
            tv, is_maker=True
        ) == Decimal("5.00")

    def test_position_limits_validation(self) -> Any:
        result = self.financial_service.validate_position_limits(
            Decimal("1.0"), Decimal("50000"), Decimal("10000")
        )
        assert "valid" in result
        assert "violations" in result
        assert "required_margin" in result


# ── Compliance tests ──────────────────────────────────────────────────────────
class TestCompliance:
    """Test compliance and regulatory features"""

    def test_kyc_data_validation(self) -> Any:
        result = compliance_service.validate_kyc_data(
            {
                "full_name": "John Doe",
                "date_of_birth": "1990-01-01",
                "nationality": "US",
                "address": "123 Main St, City, State",
                "document_type": "passport",
                "document_number": "A12345678",
                "document_expiry": "2030-01-01",
            }
        )
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_kyc_data_validation_errors(self) -> Any:
        result = compliance_service.validate_kyc_data(
            {
                "full_name": "J",
                "date_of_birth": "2010-01-01",
                "nationality": "XX",
                "document_type": "invalid",
                "document_expiry": "2020-01-01",
            }
        )
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_sanctions_list_check(self) -> Any:
        assert (
            compliance_service.check_sanctions_list("Clean User", "US")["sanctioned"]
            is False
        )
        result = compliance_service.check_sanctions_list("John Doe", "US")
        assert result["sanctioned"] is True
        assert result["name_match"] is True

    def test_transaction_compliance_check(self) -> Any:
        db = TestingSessionLocal()
        try:
            user = User(
                email=f"comp_{int(time.time()*1000)}@test.com",
                hashed_password=auth_service_inst.get_password_hash("password"),
                full_name="Compliance User",
                kyc_status="approved",
            )
            db.add(user)
            db.commit()
            db.refresh(user)

            result = compliance_service.check_transaction_compliance(
                {
                    "symbol": "BTC-USD",
                    "trade_type": "buy",
                    "quantity": 1.0,
                    "total_value": 5000.0,
                },
                user.id,
                db,
            )
            assert result["compliant"] is True

            result = compliance_service.check_transaction_compliance(
                {
                    "symbol": "BTC-USD",
                    "trade_type": "buy",
                    "quantity": 100.0,
                    "total_value": 100000.0,
                },
                user.id,
                db,
            )
            assert result["compliant"] is False
            assert len(result["violations"]) > 0
        finally:
            db.close()


# ── API endpoint tests ────────────────────────────────────────────────────────
class TestAPIEndpoints:
    """Test API endpoints with authentication"""

    def test_health_check(self) -> Any:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ("healthy", "degraded")

    def test_auth_me_without_auth(self) -> Any:
        """Protected endpoint returns 401 without auth"""
        response = client.get("/auth/me")
        assert response.status_code == 401

    def test_auth_me_with_auth(self) -> Any:
        """Protected endpoint returns user data with auth"""
        headers = _auth_headers()
        response = client.get("/auth/me", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "email" in data

    def test_volatility_prediction(self) -> Any:
        headers = _auth_headers()
        market_data = {
            "open": 50000.0,
            "high": 52000.0,
            "low": 49000.0,
            "close": 51000.0,
            "volume": 1000000,
        }
        response = client.post("/market/volatility", json=market_data, headers=headers)
        assert response.status_code in [200, 422, 503]

    def test_register_and_login_flow(self) -> Any:
        """End-to-end register → login → access protected endpoint"""
        email = _unique_email("flow")
        reg = _register(email)
        assert reg.status_code == 200
        login = _login(email)
        assert login.status_code == 200
        token = login.json()["access_token"]
        me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me.status_code == 200
        assert me.json()["email"] == email


# ── Rate limiting tests ───────────────────────────────────────────────────────
class TestRateLimiting:
    """Test rate limiting functionality"""

    def test_rate_limiting_headers(self) -> Any:
        responses = [client.get("/health") for _ in range(5)]
        assert all(r.status_code == 200 for r in responses)
        last = responses[-1]
        assert "X-RateLimit-Limit" in last.headers
        assert "X-RateLimit-Remaining" in last.headers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
