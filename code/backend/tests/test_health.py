"""
Health service and endpoint tests.
"""

from app.services.health_service import HealthService


class TestHealthService:
    def test_database_healthy(self, db):
        svc = HealthService()
        status = svc.check_database(db)
        assert status == "healthy"

    def test_redis_returns_string(self):
        svc = HealthService()
        status = svc.check_redis()
        assert isinstance(status, str)
        assert status in ("healthy", "unhealthy", "unavailable")

    def test_full_status_has_required_keys(self, db):
        from app.services.blockchain_service import BlockchainService
        from app.services.model_service import ModelService

        svc = HealthService()
        result = svc.get_full_status(db, BlockchainService(), ModelService())
        assert "status" in result
        assert "services" in result
        assert result["status"] in ("healthy", "degraded")
        for key in ("database", "blockchain", "model", "redis"):
            assert key in result["services"]

    def test_full_status_overall_healthy_when_all_ok(self, db):
        from app.services.blockchain_service import BlockchainService
        from app.services.model_service import ModelService

        svc = HealthService()
        result = svc.get_full_status(db, BlockchainService(), ModelService())
        # Database should always be healthy in tests
        assert result["services"]["database"] == "healthy"


class TestHealthEndpoint:
    def test_health_endpoint_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_status(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] in ("healthy", "degraded")

    def test_health_contains_all_services(self, client):
        resp = client.get("/health")
        services = resp.json()["services"]
        for key in (
            "database",
            "blockchain",
            "model",
            "redis",
            "security_services",
            "audit_logging",
        ):
            assert key in services

    def test_health_security_features_populated(self, client):
        resp = client.get("/health")
        features = resp.json().get("security_features", {})
        assert features.get("rbac_enabled") is True
        assert features.get("encryption_enabled") is True
        assert features.get("mfa_enabled") is True
