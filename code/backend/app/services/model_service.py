"""
ML model service module for Optionix platform.
Handles machine learning model interactions with robust validation and security.
"""

import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
from sqlalchemy.orm import Session

from ..config import settings
from ..models import MarketData

logger = logging.getLogger(__name__)

try:
    import joblib

    _joblib_available = True
except ImportError:
    _joblib_available = False
    logger.warning("joblib not available; model loading disabled")


class ModelService:
    """Service for handling ML model predictions with security and validation"""

    def __init__(self) -> None:
        self.model: Optional[Any] = None
        self.model_metadata: Dict[str, Any] = {}
        self.feature_scaler: Optional[Any] = None
        self.model_hash: Optional[str] = None
        self._load_model()
        self._load_feature_scaler()

    def _load_model(self) -> None:
        try:
            model_path = settings.model_path
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found at {model_path}")
                self.model = None
                return
            if not _joblib_available:
                self.model = None
                return
            self.model = joblib.load(model_path)
            self._load_model_metadata(model_path)
            logger.info("Volatility model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading volatility model: {e}")
            self.model = None

    def _load_model_metadata(self, model_path: str) -> None:
        try:
            metadata_path = model_path.replace(".pkl", "_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    self.model_metadata = json.load(f)
            else:
                self.model_metadata = {"version": settings.model_version}
        except Exception as e:
            logger.warning(f"Could not load model metadata: {e}")
            self.model_metadata = {"version": settings.model_version}

    def _load_feature_scaler(self) -> None:
        try:
            scaler_path = os.path.join(
                os.path.dirname(settings.model_path), "feature_scaler.pkl"
            )
            if os.path.exists(scaler_path) and _joblib_available:
                self.feature_scaler = joblib.load(scaler_path)
                logger.info("Feature scaler loaded successfully")
            else:
                self.feature_scaler = None
        except Exception as e:
            logger.error(f"Error loading feature scaler: {e}")
            self.feature_scaler = None

    def _verify_model_integrity(self, model_path: str) -> bool:
        try:
            with open(model_path, "rb") as f:
                data = f.read()
            file_hash = hashlib.sha256(data).hexdigest()
            self.model_hash = file_hash
            return True
        except Exception as e:
            logger.error(f"Model integrity check failed: {e}")
            return False

    def is_model_available(self) -> bool:
        return self.model is not None

    def _prepare_features(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Prepare feature vector from market data"""
        open_price = float(market_data.get("open", 0))
        high_price = float(market_data.get("high", 0))
        low_price = float(market_data.get("low", 0))
        close_price = float(market_data.get("close") or market_data.get("open", 0))
        volume = float(market_data.get("volume", 0))

        price_range = high_price - low_price if high_price > low_price else 0.0001
        price_change = (close_price - open_price) / open_price if open_price > 0 else 0
        high_low_ratio = high_price / low_price if low_price > 0 else 1
        volume_normalized = np.log1p(volume)

        features = np.array(
            [
                [
                    open_price,
                    high_price,
                    low_price,
                    close_price,
                    volume,
                    price_range,
                    price_change,
                    high_low_ratio,
                    volume_normalized,
                ]
            ]
        )
        return features

    def _fallback_volatility(self, market_data: Dict[str, Any]) -> float:
        """Statistical fallback volatility when model is unavailable"""
        high = float(market_data.get("high", 1))
        low = float(market_data.get("low", 1))
        open_price = float(market_data.get("open", 1))
        if open_price <= 0:
            return 0.02
        parkinson_vol = np.sqrt(np.log(high / low) ** 2 / (4 * np.log(2)))
        return float(min(max(parkinson_vol, 0.001), 2.0))

    def get_volatility_prediction(
        self, market_data: Dict[str, Any], db: Session
    ) -> Dict[str, Any]:
        """Get volatility prediction for market data"""
        try:
            self._validate_market_data(market_data)

            if self.model is None:
                volatility = self._fallback_volatility(market_data)
                return {
                    "volatility": volatility,
                    "confidence": None,
                    "model_version": "fallback_statistical",
                    "prediction_timestamp": datetime.utcnow().isoformat(),
                    "features_used": list(market_data.keys()),
                }

            features = self._prepare_features(market_data)
            if self.feature_scaler is not None:
                features = self.feature_scaler.transform(features)

            prediction = self.model.predict(features)[0]
            volatility = float(max(0.001, min(prediction, 2.0)))

            confidence: Optional[float] = None
            if hasattr(self.model, "predict_proba"):
                try:
                    proba = self.model.predict_proba(features)[0]
                    confidence = float(max(proba))
                except Exception:
                    pass

            self._store_prediction(market_data, volatility, db)

            return {
                "volatility": volatility,
                "confidence": confidence,
                "model_version": self.model_metadata.get(
                    "version", settings.model_version
                ),
                "prediction_timestamp": datetime.utcnow().isoformat(),
                "features_used": list(market_data.keys()),
            }
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise ValueError(f"Prediction failed: {str(e)}")

    def _validate_market_data(self, market_data: Dict[str, Any]) -> None:
        required = ["open", "high", "low", "volume"]
        for field in required:
            if field not in market_data:
                raise ValueError(f"Missing required field: {field}")
            val = float(market_data[field])
            if field != "volume" and val <= 0:
                raise ValueError(f"Field {field} must be positive")
            if val < 0:
                raise ValueError(f"Field {field} cannot be negative")

        open_p = float(market_data["open"])
        high_p = float(market_data["high"])
        low_p = float(market_data["low"])

        if high_p < low_p:
            raise ValueError("High price must be >= low price")
        if high_p < open_p * 0.5 or low_p > open_p * 2:
            raise ValueError("Price values appear invalid (high/low too far from open)")

    def _store_prediction(
        self, market_data: Dict[str, Any], volatility: float, db: Session
    ) -> None:
        try:
            symbol = market_data.get("symbol", "UNKNOWN")
            market_record = MarketData(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                open_price=float(market_data.get("open", 0)),
                high_price=float(market_data.get("high", 0)),
                low_price=float(market_data.get("low", 0)),
                close_price=float(
                    market_data.get("close") or market_data.get("open", 0)
                ),
                volume=float(market_data.get("volume", 0)),
                volatility=volatility,
            )
            db.add(market_record)
            db.commit()
        except Exception as e:
            logger.warning(f"Could not store market data: {e}")
            db.rollback()

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_available": self.model is not None,
            "model_version": self.model_metadata.get("version", settings.model_version),
            "scaler_available": self.feature_scaler is not None,
            "model_hash": self.model_hash,
        }
