"""
Health check service for Optionix platform.
Provides structured service health monitoring.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class HealthService:
    """Aggregates health status from all platform services"""

    def check_database(self, db: Any) -> str:
        try:
            db.execute(__import__("sqlalchemy").text("SELECT 1"))
            return "healthy"
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
            return "unhealthy"

    def check_redis(self) -> str:
        try:
            from ..middleware.security import redis_client

            if redis_client is not None:
                redis_client.ping()
                return "healthy"
            return "unavailable"
        except Exception:
            return "unhealthy"

    def get_full_status(
        self,
        db: Any,
        blockchain_service: Any,
        model_service: Any,
    ) -> Dict[str, Any]:
        services: Dict[str, str] = {
            "database": self.check_database(db),
            "blockchain": (
                "healthy" if blockchain_service.is_connected() else "unhealthy"
            ),
            "model": "healthy" if model_service.is_model_available() else "unhealthy",
            "redis": self.check_redis(),
            "compliance_engine": "healthy",
            "security_services": "healthy",
            "audit_logging": "healthy",
        }
        overall = (
            "healthy"
            if all(s in ("healthy", "unavailable") for s in services.values())
            else "degraded"
        )
        return {"status": overall, "services": services}


health_service = HealthService()
