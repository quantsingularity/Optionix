"""Model and auth service unit tests."""

import time

import pytest
from app.auth import AuthService, Permission, rbac_service


class TestAuthService:
    def test_password_hash_and_verify(self):
        svc = AuthService()
        pwd = "TestPassword123!"
        hashed = svc.get_password_hash(pwd)
        assert hashed != pwd
        assert svc.verify_password(pwd, hashed) is True

    def test_wrong_password_fails(self):
        svc = AuthService()
        hashed = svc.get_password_hash("correct")
        assert svc.verify_password("wrong", hashed) is False

    def test_access_token_creation(self):
        svc = AuthService()
        token = svc.create_access_token({"sub": "user-123", "role": "trader"})
        assert isinstance(token, str)
        assert len(token) > 0

    def test_access_token_verifiable(self):
        svc = AuthService()
        data = {"sub": "user-123", "role": "trader", "email": "t@t.com"}
        token = svc.create_access_token(data)
        payload = svc.verify_token(token)
        assert payload is not None
        assert payload["sub"] == "user-123"
        assert payload["type"] == "access"

    def test_refresh_token_type(self):
        svc = AuthService()
        token = svc.create_refresh_token({"sub": "user-123"})
        payload = svc.verify_token(token)
        assert payload["type"] == "refresh"

    def test_invalid_token_returns_none(self):
        svc = AuthService()
        assert svc.verify_token("not.a.valid.token") is None

    def test_failed_attempt_tracking(self):
        svc = AuthService()
        key = "test_login_127.0.0.1"
        svc.clear_failed_attempts(key)

        result = svc.check_failed_attempts(key)
        assert result["locked"] is False
        assert result["count"] == 0

    def test_lockout_after_max_attempts(self):
        svc = AuthService()
        key = f"lockout_test_{int(time.time()*1000)}"
        for _ in range(5):
            svc.record_failed_attempt(key)
        result = svc.check_failed_attempts(key)
        assert result["locked"] is True

    def test_clear_resets_attempts(self):
        svc = AuthService()
        key = f"clear_test_{int(time.time()*1000)}"
        for _ in range(3):
            svc.record_failed_attempt(key)
        svc.clear_failed_attempts(key)
        result = svc.check_failed_attempts(key)
        assert result["count"] == 0
        assert result["locked"] is False


class TestRBACService:
    def test_trader_can_execute_trade(self):
        assert (
            rbac_service.check_permission("trader", Permission.EXECUTE_TRADE.value)
            is True
        )

    def test_trader_cannot_manage_system(self):
        assert (
            rbac_service.check_permission("trader", Permission.MANAGE_SYSTEM.value)
            is False
        )

    def test_admin_has_many_permissions(self):
        perms = rbac_service.get_user_permissions("admin")
        assert len(perms) > 0

    def test_super_admin_has_all_permissions(self):
        perms = rbac_service.get_user_permissions("super_admin")
        for perm in Permission:
            assert perm.value in perms

    def test_viewer_limited_permissions(self):
        perms = rbac_service.get_user_permissions("viewer")
        assert Permission.DELETE_USER.value not in perms

    def test_unknown_role_has_no_permissions(self):
        perms = rbac_service.get_user_permissions("unknown_role")
        assert perms == []

    def test_compliance_officer_can_approve_kyc(self):
        assert (
            rbac_service.check_permission(
                "compliance_officer", Permission.APPROVE_KYC.value
            )
            is True
        )


class TestUserModel:
    def test_user_creation(self, db):
        pass

        from app.auth import auth_service as svc
        from app.models import User

        user = User(
            email=f"model_{int(time.time()*1000)}@test.com",
            hashed_password=svc.get_password_hash("password"),
            full_name="Model Test User",
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        assert user.id is not None
        assert user.user_id is not None
        assert user.is_active is True
        assert user.is_verified is False
        assert user.role == "trader"
        assert user.kyc_status == "pending"
        assert user.mfa_enabled is False
        assert user.risk_score == 0

    def test_unique_user_id_per_user(self, db):
        from app.auth import auth_service as svc
        from app.models import User

        ts = int(time.time() * 1000)
        user1 = User(
            email=f"uid1_{ts}@test.com",
            hashed_password=svc.get_password_hash("pw"),
            full_name="User One",
        )
        user2 = User(
            email=f"uid2_{ts}@test.com",
            hashed_password=svc.get_password_hash("pw"),
            full_name="User Two",
        )
        db.add_all([user1, user2])
        db.commit()

        assert user1.user_id != user2.user_id

    def test_duplicate_email_rejected(self, db):
        from app.auth import auth_service as svc
        from app.models import User
        from sqlalchemy.exc import IntegrityError

        email = f"dup_{int(time.time()*1000)}@test.com"
        u1 = User(
            email=email, hashed_password=svc.get_password_hash("pw"), full_name="U1"
        )
        u2 = User(
            email=email, hashed_password=svc.get_password_hash("pw"), full_name="U2"
        )
        db.add(u1)
        db.commit()
        db.add(u2)
        with pytest.raises(IntegrityError):
            db.commit()
