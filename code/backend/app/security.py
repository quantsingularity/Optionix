"""
Security Service for Optionix Platform.
Implements encryption, password validation, API key management, and audit logging.
"""

import base64
import hashlib
import logging
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import bcrypt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from .config import settings

logger = logging.getLogger(__name__)


class SecurityLevel(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class EncryptionStandard(str, Enum):
    AES_256_GCM = "aes_256_gcm"
    FERNET = "fernet"
    RSA_4096 = "rsa_4096"
    CHACHA20_POLY1305 = "chacha20_poly1305"


class ComplianceFramework(str, Enum):
    GDPR = "gdpr"
    UK_GDPR = "uk_gdpr"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    GLBA = "glba"
    NYCRR_500 = "nycrr_500"
    CCPA = "ccpa"


@dataclass
class SecurityContext:
    user_id: str
    session_id: str
    ip_address: str
    user_agent: str
    security_level: SecurityLevel
    permissions: List[str]
    mfa_verified: bool
    timestamp: datetime


@dataclass
class EncryptionResult:
    encrypted_data: str
    encryption_method: str
    key_id: str
    timestamp: datetime
    checksum: str
    encryption_key: Optional[bytes] = None


class SecurityService:
    """Security service implementing financial industry standards"""

    def __init__(self) -> None:
        self._master_key: Optional[bytes] = None
        self._fernet: Optional[Fernet] = None
        self._encryption_keys: Dict[str, bytes] = {}
        self._session_store: Dict[str, Dict[str, Any]] = {}
        self._failed_attempts: Dict[str, Dict[str, Any]] = {}
        self._rate_limits: Dict[str, Dict[str, Any]] = {}
        self._audit_log: List[Dict[str, Any]] = []
        self._initialize_security()

    def _initialize_security(self) -> None:
        try:
            self._load_master_key()
            self._initialize_encryption_keys()
            logger.info("Security service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize security service: {e}")
            raise

    def _load_master_key(self) -> None:
        try:
            key_material = settings.secret_key.encode()
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b"optionix_master_salt_2024_v2",
                iterations=200000,
            )
            derived_key = kdf.derive(key_material)
            self._master_key = base64.urlsafe_b64encode(derived_key)
            self._fernet = Fernet(self._master_key)
        except Exception as e:
            logger.error(f"Failed to load master key: {e}")
            raise

    def _initialize_encryption_keys(self) -> None:
        try:
            self._encryption_keys = {
                "pii_data": Fernet.generate_key(),
                "financial_data": Fernet.generate_key(),
                "audit_logs": Fernet.generate_key(),
                "session_data": Fernet.generate_key(),
                "backup_data": Fernet.generate_key(),
            }
        except Exception as e:
            logger.error(f"Failed to initialize encryption keys: {e}")
            raise

    def encrypt_field(self, data: str) -> str:
        if not data:
            return data
        key = self._encryption_keys["pii_data"]
        fernet = Fernet(key)
        return fernet.encrypt(data.encode()).decode()

    def decrypt_field(self, encrypted_data: str) -> str:
        if not encrypted_data:
            return encrypted_data
        key = self._encryption_keys["pii_data"]
        fernet = Fernet(key)
        return fernet.decrypt(encrypted_data.encode()).decode()

    def encrypt_sensitive_data(self, data: str) -> EncryptionResult:
        try:
            key = Fernet.generate_key()
            fernet = Fernet(key)
            encrypted = fernet.encrypt(data.encode()).decode()
            checksum = hashlib.sha256(data.encode()).hexdigest()
            return EncryptionResult(
                encrypted_data=encrypted,
                encryption_method="FERNET",
                key_id=secrets.token_hex(16),
                timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
                checksum=checksum,
                encryption_key=key,
            )
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise

    def decrypt_sensitive_data(self, result: EncryptionResult) -> str:
        try:
            if result.encryption_key is None:
                raise ValueError("Encryption key not available in result")
            fernet = Fernet(result.encryption_key)
            return fernet.decrypt(result.encrypted_data.encode()).decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise

    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        issues = []
        if len(password) < 12:
            issues.append("Password must be at least 12 characters long")
        if not re.search("[a-z]", password):
            issues.append("Password must contain lowercase letters")
        if not re.search("[A-Z]", password):
            issues.append("Password must contain uppercase letters")
        if not re.search("\\d", password):
            issues.append("Password must contain numbers")
        if not re.search('[!@#$%^&*(),.?":{}|<>]', password):
            issues.append("Password must contain special characters")
        common_patterns = [
            "(.)\\1{3,}",
            "(0123|1234|2345|3456|4567|5678|6789|7890)",
            "(abcd|bcde|cdef|defg|efgh|fghi|ghij|hijk|ijkl|jklm|klmn|lmno|mnop|nopq|opqr|pqrs|qrst|rstu|stuv|tuvw|uvwx|vwxy|wxyz)",
        ]
        for pattern in common_patterns:
            if re.search(pattern, password.lower()):
                issues.append("Password contains common patterns")
                break

        is_valid = len(issues) == 0
        strength = "strong" if is_valid else ("medium" if len(issues) <= 1 else "weak")
        return {
            "valid": is_valid,
            "issues": issues,
            "strength": strength,
            "strength_score": max(0, 100 - len(issues) * 20),
        }

    def sanitize_input(
        self, data: Union[str, Dict[str, Any]]
    ) -> Union[str, Dict[str, Any]]:
        _SQL_RE = re.compile(
            r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|EXEC|UNION|SELECT|TRUNCATE)\b",
            re.IGNORECASE,
        )

        def _clean(value: str) -> str:
            out = re.sub(
                '[<>"' + "'" + r";\\ ]",
                lambda m: "" if m.group() in "<>\"'\\;" else m.group(),
                value,
            )
            out = re.sub('[<>"' + "'" + r";\\]", "", value)
            out = _SQL_RE.sub("", out)
            return out[:1000]

        if isinstance(data, str):
            return _clean(data)
        elif isinstance(data, dict):
            return {k: _clean(v) if isinstance(v, str) else v for k, v in data.items()}
        return data

    def validate_ethereum_address(self, address: str) -> bool:
        return bool(re.match(r"^0x[0-9a-fA-F]{40}$", address))

    def generate_api_key(self) -> Tuple[str, str]:
        prefix = settings.api_key_prefix
        raw = secrets.token_urlsafe(32)
        plain_key = f"{prefix}{raw}"[:46].ljust(46, "x")[:46]
        hashed_key = self.hash_api_key(plain_key)
        return plain_key, hashed_key

    def validate_api_key_format(self, api_key: str) -> bool:
        return (
            isinstance(api_key, str)
            and api_key.startswith(settings.api_key_prefix)
            and len(api_key) == 46
        )

    def hash_api_key(self, api_key: str) -> str:
        return hashlib.sha256(api_key.encode()).hexdigest()

    def check_rate_limit(
        self, user_id: str, action: str = "default", limit: int = 100
    ) -> bool:
        key = f"{user_id}:{action}"
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        window_start = now - timedelta(minutes=1)
        if key not in self._rate_limits:
            self._rate_limits[key] = {"requests": [], "blocked": False}
        entry = self._rate_limits[key]
        entry["requests"] = [t for t in entry["requests"] if t > window_start]
        if len(entry["requests"]) >= limit:
            return False
        entry["requests"].append(now)
        return True

    def log_audit_event(self, event_data: Dict[str, Any]) -> bool:
        try:
            event_data["logged_at"] = (
                datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
            )
            self._audit_log.append(event_data)
            logger.info(f"Audit event logged: {event_data.get('action', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Audit logging failed: {e}")
            return False

    def get_audit_log_hash(self, event_data: Dict[str, Any]) -> str:
        import json

        serialized = json.dumps(event_data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def generate_secure_token(self, length: int = 32) -> str:
        return secrets.token_urlsafe(length)

    def hash_password(self, password: str) -> str:
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password.encode(), salt).decode()

    def verify_password(self, password: str, hashed: str) -> bool:
        try:
            return bcrypt.checkpw(password.encode(), hashed.encode())
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            return False


security_service = SecurityService()
