"""API routes package for Optionix platform"""

from fastapi import APIRouter

from .auth import router as auth_router
from .market import router as market_router

api_router = APIRouter()
api_router.include_router(auth_router)
api_router.include_router(market_router)

__all__ = ["api_router"]
