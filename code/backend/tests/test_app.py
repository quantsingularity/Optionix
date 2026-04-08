"""Backend API tests for Optionix platform."""

import unittest
from typing import Any

from app import app
from data_protection import DataProcessingLog  # noqa: register ORM models
from database import Base, get_db
from fastapi.testclient import TestClient
from middleware.audit_logging import AuditLog as StructLog  # noqa
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

_engine = create_engine(
    "sqlite:///./test_app.db",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
_Session = sessionmaker(bind=_engine)
Base.metadata.create_all(bind=_engine)


def _override_get_db():
    db = _Session()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = _override_get_db
_client = TestClient(app, raise_server_exceptions=False)


class TestBackendAPI(unittest.TestCase):

    def setUp(self) -> Any:
        self.client = _client
        self.auth_token = None

    def _register_and_login(self, suffix: str = ""):
        import time

        email = f"testapp_{suffix}_{int(time.time()*1000)}@x.com"
        self.client.post(
            "/auth/register",
            json={
                "email": email,
                "password": "TestPassword123!",
                "full_name": "Test User",
            },
        )
        r = self.client.post(
            "/auth/login", json={"email": email, "password": "TestPassword123!"}
        )
        if r.status_code == 200:
            self.auth_token = r.json().get("access_token")
        return email

    def get_auth_headers(self) -> Any:
        return {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}

    def test_health_check(self) -> Any:
        """Test the health check endpoint"""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["status"], ("healthy", "degraded"))
        self.assertIn("services", data)

    def test_root_endpoint(self) -> Any:
        """Test root endpoint"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("status", data)

    def test_authentication(self) -> Any:
        """Test authentication endpoints"""
        import time

        email = f"auth_{int(time.time()*1000)}@x.com"
        reg = self.client.post(
            "/auth/register",
            json={
                "email": email,
                "password": "TestPassword123!",
                "full_name": "Auth User",
            },
        )
        self.assertEqual(reg.status_code, 200)

        r = self.client.post(
            "/auth/login", json={"email": email, "password": "TestPassword123!"}
        )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("access_token", data)
        self.auth_token = data["access_token"]

        bad = self.client.post(
            "/auth/login", json={"email": "no@no.com", "password": "Bad99!"}
        )
        self.assertEqual(bad.status_code, 401)

    def test_user_management(self) -> Any:
        """Test user management via /auth/register and /auth/me"""
        import time

        email = f"mgmt_{int(time.time()*1000)}@x.com"
        r = self.client.post(
            "/auth/register",
            json={
                "email": email,
                "password": "TestPassword123!",
                "full_name": "Mgmt User",
            },
        )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("email", data)

        # Unauthenticated access to protected endpoint
        unauth = self.client.get("/auth/me")
        self.assertEqual(unauth.status_code, 401)

        # Authenticated access
        self._register_and_login("mgmt2")
        me = self.client.get("/auth/me", headers=self.get_auth_headers())
        self.assertEqual(me.status_code, 200)

    def test_error_handling(self) -> Any:
        """Test error handling"""
        not_found = self.client.get("/nonexistent_endpoint_xyz")
        self.assertEqual(not_found.status_code, 404)

        bad_json = self.client.post(
            "/auth/login",
            headers={"Content-Type": "application/json"},
            content="not json at all",
        )
        self.assertEqual(bad_json.status_code, 422)


if __name__ == "__main__":
    unittest.main()
