"""
Authentication routes for Optionix platform.
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from ..auth import UserRole, auth_service, get_current_user, log_auth_event
from ..config import settings
from ..data_protection import data_protection_service
from ..database import get_db
from ..models import User
from ..schemas import TokenResponse, UserCreate, UserLogin, UserResponse
from ..security import security_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse)
async def register_user(
    user_data: UserCreate, request: Request, db: Session = Depends(get_db)
):
    """User registration with comprehensive security checks"""
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "")

    failed_attempts = auth_service.check_failed_attempts(f"register_{client_ip}")
    if failed_attempts["locked"]:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many registration attempts. Please try again later.",
        )

    try:
        existing_user = db.query(User).filter(User.email == user_data.email).first()
        if existing_user:
            auth_service.record_failed_attempt(f"register_{client_ip}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )

        password_validation = security_service.validate_password_strength(
            user_data.password
        )
        if not password_validation["valid"]:
            auth_service.record_failed_attempt(f"register_{client_ip}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Password validation failed: {', '.join(password_validation['issues'])}",
            )

        sanitized_data = security_service.sanitize_input(user_data.model_dump())
        hashed_password = auth_service.get_password_hash(user_data.password)

        user = User(
            email=sanitized_data["email"],
            hashed_password=hashed_password,
            full_name=sanitized_data["full_name"],
            is_active=True,
            is_verified=False,
            kyc_status="pending",
            role=UserRole.TRADER.value,
            mfa_enabled=False,
            risk_score=0,
            data_retention_consent=sanitized_data.get("data_retention_consent", False),
            marketing_consent=sanitized_data.get("marketing_consent", False),
            data_processing_consent=sanitized_data.get(
                "data_processing_consent", False
            ),
            consent_date=(
                datetime.now(timezone.utc).replace(tzinfo=None)
                if sanitized_data.get("data_processing_consent")
                else None
            ),
        )

        db.add(user)
        db.commit()
        db.refresh(user)

        data_protection_service.create_data_processing_log(
            db=db,
            data_subject_id=str(user.id),
            processing_activity="user_registration",
            data_types=["personal_information", "contact_details"],
            legal_basis="contract",
            purpose="Account creation and service provision",
            user_id=user.id,
            retention_period=2555,
            consent_given=user.data_processing_consent,
        )

        log_auth_event(
            db, user.id, "user_registration", client_ip, user_agent, "success"
        )
        auth_service.clear_failed_attempts(f"register_{client_ip}")

        return UserResponse.model_validate(user)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        auth_service.record_failed_attempt(f"register_{client_ip}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed",
        )


@router.post("/login", response_model=TokenResponse)
async def login_user(
    user_data: UserLogin, request: Request, db: Session = Depends(get_db)
):
    """User login with JWT token issuance and MFA support"""
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "")

    failed = auth_service.check_failed_attempts(f"login_{client_ip}")
    if failed["locked"]:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed login attempts. Please try again later.",
        )

    user = db.query(User).filter(User.email == user_data.email).first()
    if not user or not auth_service.verify_password(
        user_data.password, user.hashed_password
    ):
        auth_service.record_failed_attempt(f"login_{client_ip}")
        log_auth_event(db, None, "login_failed", client_ip, user_agent, "failure")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Account is disabled"
        )

    if user.is_locked:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is locked. Please contact support.",
        )

    if user.mfa_enabled and not user_data.mfa_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="MFA token required"
        )

    auth_service.clear_failed_attempts(f"login_{client_ip}")

    token_data = {"sub": str(user.user_id), "role": user.role, "email": user.email}
    access_token = auth_service.create_access_token(token_data)
    refresh_token = auth_service.create_refresh_token(token_data)

    user.last_login = datetime.now(timezone.utc).replace(tzinfo=None)
    db.commit()

    log_auth_event(db, user.id, "login_success", client_ip, user_agent, "success")

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60,
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get the currently authenticated user's profile"""
    user = db.query(User).filter(User.user_id == current_user.get("sub")).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )
    return UserResponse.model_validate(user)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_access_token(request: Request):
    """Refresh access token using a valid refresh token"""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token missing",
        )
    refresh_token = auth_header.split(" ")[1]
    payload = auth_service.verify_token(refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )
    token_data = {
        "sub": payload["sub"],
        "role": payload.get("role"),
        "email": payload.get("email"),
    }
    new_access_token = auth_service.create_access_token(token_data)
    new_refresh_token = auth_service.create_refresh_token(token_data)
    return TokenResponse(
        access_token=new_access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60,
    )
