"""
Audit logging service for Optionix backend.
Provides comprehensive audit trail for all critical operations.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import structlog

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    _structlog_available = True
except ImportError:
    _structlog_available = False


class AuditLog:
    """Lightweight structured audit log entry (non-ORM)"""

    def __init__(self, action: str, **kwargs: Any) -> None:
        self.action = action
        self.timestamp = datetime.utcnow().isoformat()
        self.data = kwargs


class AuditLogger:
    """Service for comprehensive audit logging"""

    def __init__(self) -> None:
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="audit")

    def __del__(self) -> None:
        try:
            self.executor.shutdown(wait=False)
        except Exception:
            pass

    def log_event(
        self,
        action: str,
        user_id: Optional[int] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_data: Optional[Dict[str, Any]] = None,
        response_data: Optional[Dict[str, Any]] = None,
        status: str = "success",
        error_message: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an audit event asynchronously"""
        event = {
            "action": action,
            "user_id": user_id,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if additional_context:
            event.update(additional_context)

        self.executor.submit(self._write_log, event)

    def _write_log(self, event: Dict[str, Any]) -> None:
        try:
            logger.info(f"AUDIT: {json.dumps(event, default=str)}")
        except Exception as e:
            logger.error(f"Audit write failed: {e}")

    def log_trade_event(
        self,
        action: str,
        trade_id: str,
        user_id: int,
        symbol: str,
        trade_type: str,
        quantity: float,
        price: float,
        status: str = "success",
    ) -> None:
        self.log_event(
            action=action,
            user_id=user_id,
            resource_type="trade",
            resource_id=trade_id,
            status=status,
            additional_context={
                "symbol": symbol,
                "trade_type": trade_type,
                "quantity": quantity,
                "price": price,
            },
        )

    def log_security_event(
        self,
        action: str,
        user_id: Optional[int],
        ip_address: str,
        success: bool,
        details: Optional[str] = None,
    ) -> None:
        self.log_event(
            action=action,
            user_id=user_id,
            ip_address=ip_address,
            status="success" if success else "failure",
            error_message=details if not success else None,
        )


audit_logger = AuditLogger()
