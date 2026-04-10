"""
Market data routes for Optionix platform.
"""

import logging
from datetime import datetime, timezone
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..database import get_db
from ..schemas import MarketDataRequest, VolatilityResponse
from ..services.model_service import ModelService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/market", tags=["Market Data"])

_model_service: ModelService | None = None


def get_model_service() -> ModelService:
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service


@router.post("/volatility", response_model=VolatilityResponse)
async def get_volatility_prediction(
    market_data: MarketDataRequest,
    db: Session = Depends(get_db),
    model_service: ModelService = Depends(get_model_service),
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
            timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Volatility prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Volatility prediction failed",
        )
