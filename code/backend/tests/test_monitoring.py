"""
Transaction monitoring and SAR generation tests.
"""

import time

import pytest
from app.services.compliance_service import ComplianceService


@pytest.fixture
def svc():
    return ComplianceService()


class TestTransactionMonitoring:
    def test_no_trades_not_suspicious(self, svc, db):
        from app.auth import auth_service
        from app.models import User

        user = User(
            email=f"mon_{int(time.time()*1000)}@test.com",
            hashed_password=auth_service.get_password_hash("pw"),
            full_name="Monitor User",
        )
        db.add(user)
        db.commit()

        result = svc.monitor_transaction_patterns(user.id, db)
        assert result["suspicious"] is False
        assert result["trade_count"] == 0

    def test_monitoring_returns_required_keys(self, svc, db):
        from app.auth import auth_service
        from app.models import User

        user = User(
            email=f"mon2_{int(time.time()*1000)}@test.com",
            hashed_password=auth_service.get_password_hash("pw"),
            full_name="Monitor User 2",
        )
        db.add(user)
        db.commit()

        result = svc.monitor_transaction_patterns(user.id, db)
        for key in ("suspicious", "alerts", "total_volume", "trade_count"):
            assert key in result


class TestSARGeneration:
    def test_sar_generated_for_valid_user(self, svc, db):
        from app.auth import auth_service
        from app.models import User

        user = User(
            email=f"sar_{int(time.time()*1000)}@test.com",
            hashed_password=auth_service.get_password_hash("pw"),
            full_name="SAR User",
        )
        db.add(user)
        db.commit()

        sar = svc.generate_sar_report(
            user.id,
            {"type": "high_volume", "message": "Test suspicious activity"},
            db,
        )
        assert "report_id" in sar
        assert sar["report_id"].startswith("SAR_")
        assert "user_info" in sar
        assert "regulatory_requirements" in sar

    def test_sar_for_missing_user_raises(self, svc, db):
        with pytest.raises(ValueError):
            svc.generate_sar_report(999999, {"type": "test"}, db)
