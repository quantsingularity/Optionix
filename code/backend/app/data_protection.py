"""
Data protection module for Optionix.
Provides field-level encryption, data masking, and GDPR compliance features.
"""

import base64
import hashlib
import json
import logging
import secrets
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text
from sqlalchemy.orm import Session

from .config import settings
from .models import Base, User

logger = logging.getLogger(__name__)


class DataClassification(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class PIIType(str, Enum):
    NAME = "name"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    SSN = "ssn"
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"
    BANK_ACCOUNT = "bank_account"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    BIOMETRIC = "biometric"


class EncryptedField:
    """Descriptor for encrypted database fields"""

    def __init__(
        self, classification: DataClassification = DataClassification.CONFIDENTIAL
    ) -> None:
        self.classification = classification
        self.encryption_service: Optional[Any] = None

    def __set_name__(self, owner: Any, name: Any) -> None:
        self.name = name
        self.private_name = f"_{name}"

    def __get__(self, obj: Any, objtype: Any = None) -> Any:
        if obj is None:
            return self
        encrypted_value = getattr(obj, self.private_name, None)
        if encrypted_value is None:
            return None
        if self.encryption_service is None:
            self.encryption_service = DataProtectionService()
        try:
            return self.encryption_service.decrypt_field(encrypted_value)
        except Exception as e:
            logger.error(f"Failed to decrypt field {self.name}: {e}")
            return None

    def __set__(self, obj: Any, value: Any) -> None:
        if value is None:
            setattr(obj, self.private_name, None)
            return
        if self.encryption_service is None:
            self.encryption_service = DataProtectionService()
        try:
            encrypted_value = self.encryption_service.encrypt_field(value)
            setattr(obj, self.private_name, encrypted_value)
        except Exception as e:
            logger.error(f"Failed to encrypt field {self.name}: {e}")
            setattr(obj, self.private_name, value)


class DataRetentionPolicy(Base):
    """Data retention policies for different data types"""

    __tablename__ = "data_retention_policies"
    id = Column(Integer, primary_key=True)
    data_type = Column(String(100), nullable=False, unique=True)
    retention_period_days = Column(Integer, nullable=False)
    classification = Column(String(50), nullable=False)
    auto_delete = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DataProcessingLog(Base):
    """Log of data processing activities for GDPR compliance"""

    __tablename__ = "data_processing_logs"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=True)
    data_subject_id = Column(String(100), nullable=False)
    processing_activity = Column(String(100), nullable=False)
    data_types = Column(Text, nullable=False)
    legal_basis = Column(String(100), nullable=False)
    purpose = Column(Text, nullable=False)
    retention_period = Column(Integer, nullable=True)
    third_party_sharing = Column(Boolean, default=False)
    third_parties = Column(Text, nullable=True)
    consent_given = Column(Boolean, default=False)
    consent_date = Column(DateTime, nullable=True)
    consent_withdrawn = Column(Boolean, default=False)
    consent_withdrawn_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class DataSubjectRequest(Base):
    """GDPR data subject requests"""

    __tablename__ = "data_subject_requests"
    id = Column(Integer, primary_key=True)
    request_id = Column(String(100), unique=True, nullable=False)
    user_id = Column(Integer, nullable=True)
    email = Column(String(255), nullable=False)
    request_type = Column(String(50), nullable=False)
    status = Column(String(50), default="pending")
    description = Column(Text, nullable=True)
    verification_method = Column(String(100), nullable=True)
    verification_completed = Column(Boolean, default=False)
    response_data = Column(Text, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DataProtectionService:
    """Service for data protection, encryption, and GDPR compliance"""

    def __init__(self) -> None:
        self._fernet: Optional[Fernet] = None
        self._initialize_encryption()

    def _initialize_encryption(self) -> None:
        try:
            key_material = settings.encryption_key.encode()
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b"optionix_data_protection_salt",
                iterations=100000,
            )
            derived_key = base64.urlsafe_b64encode(kdf.derive(key_material))
            self._fernet = Fernet(derived_key)
            logger.info("Data protection service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize data protection: {e}")
            raise

    def encrypt_field(self, data: str) -> str:
        if not data or not self._fernet:
            return data
        return self._fernet.encrypt(data.encode()).decode()

    def decrypt_field(self, encrypted_data: str) -> str:
        if not encrypted_data or not self._fernet:
            return encrypted_data
        return self._fernet.decrypt(encrypted_data.encode()).decode()

    def mask_pii(self, data: str, pii_type: PIIType) -> str:
        if not data:
            return data
        if pii_type == PIIType.EMAIL:
            parts = data.split("@")
            if len(parts) == 2:
                local = parts[0]
                masked_local = (
                    local[0] + "*" * (len(local) - 1) if len(local) > 1 else "*"
                )
                return f"{masked_local}@{parts[1]}"
        elif pii_type in (PIIType.SSN,):
            return "***-**-" + data[-4:] if len(data) >= 4 else "***"
        elif pii_type in (PIIType.CREDIT_CARD,):
            return "**** **** **** " + data[-4:] if len(data) >= 4 else "****"
        elif pii_type == PIIType.PHONE:
            return "***-***-" + data[-4:] if len(data) >= 4 else "***"
        # Default: show first and last character
        if len(data) <= 2:
            return "*" * len(data)
        return data[0] + "*" * (len(data) - 2) + data[-1]

    def anonymize_user_data(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        anonymized = user_data.copy()
        pii_fields = {
            "email": PIIType.EMAIL,
            "phone": PIIType.PHONE,
            "ssn": PIIType.SSN,
        }
        for field, pii_type in pii_fields.items():
            if field in anonymized and anonymized[field]:
                anonymized[field] = self.mask_pii(str(anonymized[field]), pii_type)
        if "full_name" in anonymized:
            name_parts = str(anonymized["full_name"]).split()
            anonymized["full_name"] = (
                name_parts[0][0] + "*** " + name_parts[-1][0] + "***"
                if len(name_parts) > 1
                else name_parts[0][0] + "***"
            )
        return anonymized

    def create_data_processing_log(
        self,
        db: Session,
        data_subject_id: str,
        processing_activity: str,
        data_types: List[str],
        legal_basis: str,
        purpose: str,
        user_id: Optional[int] = None,
        retention_period: Optional[int] = None,
        consent_given: bool = False,
    ) -> DataProcessingLog:
        try:
            log_entry = DataProcessingLog(
                user_id=user_id,
                data_subject_id=data_subject_id,
                processing_activity=processing_activity,
                data_types=json.dumps(data_types),
                legal_basis=legal_basis,
                purpose=purpose,
                retention_period=retention_period,
                consent_given=consent_given,
                consent_date=datetime.utcnow() if consent_given else None,
            )
            db.add(log_entry)
            db.commit()
            db.refresh(log_entry)
            return log_entry
        except Exception as e:
            logger.error(f"Failed to create data processing log: {e}")
            db.rollback()
            raise

    def export_user_data(self, user_id: int, db: Session) -> Dict[str, Any]:
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise ValueError("User not found")
            export_data = {
                "export_date": datetime.utcnow().isoformat(),
                "user_id": user.user_id,
                "email": user.email,
                "full_name": user.full_name,
                "role": user.role,
                "kyc_status": user.kyc_status,
                "account_created": (
                    user.created_at.isoformat() if user.created_at else None
                ),
                "last_login": user.last_login.isoformat() if user.last_login else None,
                "consents": {
                    "data_processing": user.data_processing_consent,
                    "marketing": user.marketing_consent,
                    "data_retention": user.data_retention_consent,
                    "consent_date": (
                        user.consent_date.isoformat() if user.consent_date else None
                    ),
                },
            }
            return export_data
        except Exception as e:
            logger.error(f"Failed to export user data: {e}")
            raise

    def request_data_deletion(
        self, user_id: int, db: Session, reason: str = "user_request"
    ) -> Dict[str, Any]:
        try:
            request_id = f"DEL_{user_id}_{secrets.token_hex(8)}"
            deletion_request = DataSubjectRequest(
                request_id=request_id,
                user_id=user_id,
                email="",
                request_type="erasure",
                status="pending",
                description=reason,
            )
            db.add(deletion_request)
            db.commit()
            return {
                "request_id": request_id,
                "status": "pending",
                "estimated_completion": (
                    datetime.utcnow() + timedelta(days=30)
                ).isoformat(),
            }
        except Exception as e:
            logger.error(f"Failed to create deletion request: {e}")
            db.rollback()
            raise

    def hash_identifier(self, identifier: str) -> str:
        return hashlib.sha256(identifier.encode()).hexdigest()


data_protection_service = DataProtectionService()
