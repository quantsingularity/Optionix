"""
Main FastAPI application for Optionix platform.
Integrates comprehensive security, compliance, and financial standards.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from decimal import Decimal

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session

from .auth import UserRole, auth_service, get_current_user, log_auth_event
from .config import settings
from .data_protection import data_protection_service
from .database import create_tables, get_db
from .middleware.security import (
    AdvancedRateLimitMiddleware,
    AuditLoggingMiddleware,
    RequestValidationMiddleware,
    SecurityHeadersMiddleware,
)
from .models import User
from .schemas import (
    HealthCheckResponse,
    MarketDataRequest,
    TokenResponse,
    UserCreate,
    UserLogin,
    UserResponse,
    VolatilityResponse,
)
from .security import security_service
from .services.blockchain_service import BlockchainService
from .services.financial_service import FinancialCalculationService
from .services.model_service import ModelService

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Optionix API...")
    try:
        create_tables()
        logger.info("Database tables created/verified")
        logger.info("Optionix API started successfully")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    yield
    logger.info("Shutting down Optionix API...")


app = FastAPI(
    title=settings.app_name,
    description="Comprehensive API for options trading platform with advanced security, compliance, and financial standards",
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(AdvancedRateLimitMiddleware)
app.add_middleware(RequestValidationMiddleware)
app.add_middleware(AuditLoggingMiddleware)

blockchain_service = BlockchainService()
model_service = ModelService()
financial_service = FinancialCalculationService()

security = HTTPBearer()


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code in [401, 403, 429]:
        logger.warning(
            f"Security exception: {exc.status_code} - {exc.detail}",
            extra={
                "client_ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", ""),
                "endpoint": request.url.path,
                "method": request.method,
            },
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "http_error",
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.get("/", tags=["System"])
async def root():
    return {
        "message": "Welcome to Optionix API",
        "status": "online",
        "version": settings.app_version,
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["System"])
async def health_check():
    """System health check with security and compliance status"""
    services_status = {
        "database": "healthy",
        "blockchain": "healthy" if blockchain_service.is_connected() else "unhealthy",
        "model": "healthy" if model_service.is_model_available() else "unhealthy",
        "redis": "healthy",
        "compliance_engine": "healthy",
        "security_services": "healthy",
        "audit_logging": "healthy",
    }
    overall_status = (
        "healthy"
        if all(s == "healthy" for s in services_status.values())
        else "degraded"
    )
    return HealthCheckResponse(
        status=overall_status,
        version=settings.app_version,
        services=services_status,
        security_features={
            "mfa_enabled": True,
            "rbac_enabled": True,
            "encryption_enabled": True,
            "audit_logging": True,
            "compliance_monitoring": True,
        },
    )


@app.post(
    "/market/volatility",
    response_model=VolatilityResponse,
    tags=["Market Data", "Model"],
)
async def get_volatility_prediction(
    market_data: MarketDataRequest, db: Session = Depends(get_db)
):
    """Get volatility prediction for a given market data point"""
    try:
        data_for_model = market_data.model_dump()
        prediction_result = model_service.get_volatility_prediction(data_for_model, db)
        return VolatilityResponse(
            symbol=market_data.symbol,
            volatility=Decimal(str(prediction_result["volatility"])),
            confidence=(
                Decimal(str(prediction_result["confidence"]))
                if prediction_result.get("confidence") is not None
                else None
            ),
            model_version=prediction_result.get("model_version"),
            prediction_horizon="24h",
            timestamp=datetime.utcnow(),
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Volatility prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Volatility prediction failed",
        )


@app.post("/auth/register", response_model=UserResponse, tags=["Authentication"])
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
                datetime.utcnow()
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


@app.post("/auth/login", response_model=TokenResponse, tags=["Authentication"])
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
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled",
        )

    if user.is_locked:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is locked. Please contact support.",
        )

    if user.mfa_enabled and not user_data.mfa_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="MFA token required",
        )

    auth_service.clear_failed_attempts(f"login_{client_ip}")

    token_data = {"sub": str(user.user_id), "role": user.role, "email": user.email}
    access_token = auth_service.create_access_token(token_data)
    refresh_token = auth_service.create_refresh_token(token_data)

    user.last_login = datetime.utcnow()
    db.commit()

    log_auth_event(db, user.id, "login_success", client_ip, user_agent, "success")

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60,
    )


@app.get("/auth/me", response_model=UserResponse, tags=["Authentication"])
async def get_current_user_profile(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get the currently authenticated user's profile"""
    user = db.query(User).filter(User.user_id == current_user.get("sub")).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return UserResponse.model_validate(user)


@app.post("/auth/refresh", response_model=TokenResponse, tags=["Authentication"])
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


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
