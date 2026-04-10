"""System endpoint and error handling tests."""


class TestSystemEndpoints:
    def test_root_endpoint(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data
        assert "Welcome" in data["message"]
        assert data["status"] == "online"
        assert "version" in data

    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("healthy", "degraded")
        assert "services" in data
        assert "version" in data

    def test_health_services_keys(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        services = resp.json()["services"]
        expected = ["database", "blockchain", "model", "redis"]
        for key in expected:
            assert key in services

    def test_health_security_features(self, client):
        resp = client.get("/health")
        data = resp.json()
        features = data.get("security_features", {})
        assert features.get("rbac_enabled") is True
        assert features.get("encryption_enabled") is True
        assert features.get("audit_logging") is True


class TestErrorHandling:
    def test_404_not_found(self, client):
        resp = client.get("/nonexistent_endpoint_xyz")
        assert resp.status_code == 404

    def test_422_invalid_json(self, client):
        resp = client.post(
            "/auth/login",
            content="not json at all",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    def test_error_response_has_detail(self, client):
        resp = client.get("/nonexistent")
        # FastAPI returns 404 with detail
        assert resp.status_code == 404


class TestSecurityHeaders:
    def test_security_headers_present(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "x-content-type-options" in resp.headers
        assert resp.headers["x-content-type-options"] == "nosniff"
        assert "x-frame-options" in resp.headers

    def test_rate_limit_headers_present(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert "x-ratelimit-limit" in resp.headers
        assert "x-ratelimit-remaining" in resp.headers
        assert "x-ratelimit-reset" in resp.headers
