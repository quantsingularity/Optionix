import hashlib
import json
import logging
import os
from datetime import datetime, timezone
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
    def __init__(self) -> None:
        self.model: Optional[Any] = None
        self.model_metadata: Dict[str, Any] = {}
        self.feature_scaler: Optional[Any] = None
        self.model_hash: Optional[str] = None
        self._initialize_service()

    def _initialize_service(self) -> None:
        model_path = settings.model_path
        if os.path.exists(model_path) and self._verify_model_integrity(model_path):
            self._load_model(model_path)
            self._load_feature_scaler(model_path)
        else:
            logger.warning(
                "Model initialization skipped: file missing or integrity failed"
            )

    def _load_model(self, model_path: str) -> None:
        try:
            if _joblib_available:
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

    def _load_feature_scaler(self, model_path: str) -> None:
        try:
            scaler_path = os.path.join(
                os.path.dirname(model_path), "feature_scaler.pkl"
            )
            if os.path.exists(scaler_path) and _joblib_available:
                self.feature_scaler = joblib.load(scaler_path)
                logger.info("Feature scaler loaded successfully")
        except Exception as e:
            logger.error(f"Error loading feature scaler: {e}")
            self.feature_scaler = None

    def _verify_model_integrity(self, model_path: str) -> bool:
        try:
            with open(model_path, "rb") as f:
                data = f.read()
            self.model_hash = hashlib.sha256(data).hexdigest()
            return True
        except Exception as e:
            logger.error(f"Model integrity check failed: {e}")
            return False

    def is_model_available(self) -> bool:
        return self.model is not None

    def _prepare_features(self, data: Dict[str, float]) -> np.ndarray:
        price_range = data["high"] - data["low"] if data["high"] > data["low"] else 1e-6
        price_change = (data["close"] - data["open"]) / data["open"]
        high_low_ratio = data["high"] / data["low"]

        features = np.array(
            [
                [
                    data["open"],
                    data["high"],
                    data["low"],
                    data["close"],
                    data["volume"],
                    price_range,
                    price_change,
                    high_low_ratio,
                ]
            ]
        )
        return features

    def _fallback_volatility(self, data: Dict[str, float]) -> float:
        parkinson_vol = np.sqrt(
            np.log(data["high"] / data["low"]) ** 2 / (4 * np.log(2))
        )
        return float(min(max(parkinson_vol, 0.001), 2.0))

    def get_volatility_prediction(
        self, market_data: Dict[str, Any], db: Session
    ) -> Dict[str, Any]:
        try:
            cleaned_data = self._validate_and_clean_data(market_data)
            ts = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

            if self.model is None:
                vol = self._fallback_volatility(cleaned_data)
                return {
                    "volatility": vol,
                    "confidence": None,
                    "model_version": "fallback_statistical",
                    "prediction_timestamp": ts,
                    "features_used": list(cleaned_data.keys()),
                }

            features = self._prepare_features(cleaned_data)
            if self.feature_scaler is not None:
                features = self.feature_scaler.transform(features)

            prediction = self.model.predict(features)[0]
            volatility = float(max(0.001, min(prediction, 2.0)))

            confidence = None
            if hasattr(self.model, "predict_proba"):
                try:
                    proba = self.model.predict_proba(features)[0]
                    confidence = float(max(proba))
                except:
                    pass

            self._store_prediction(
                market_data.get("symbol", "UNKNOWN"), cleaned_data, volatility, db
            )

            return {
                "volatility": volatility,
                "confidence": confidence,
                "model_version": self.model_metadata.get(
                    "version", settings.model_version
                ),
                "prediction_timestamp": ts,
                "features_used": list(cleaned_data.keys()),
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise ValueError(f"Prediction failed: {str(e)}")

    def _validate_and_clean_data(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        required = ["open", "high", "low", "volume"]
        cleaned = {}
        for field in required:
            if field not in market_data:
                raise ValueError(f"Missing required field: {field}")
            cleaned[field] = float(market_data[field])
            if cleaned[field] < 0:
                raise ValueError(f"Field {field} cannot be negative")
            if field != "volume" and cleaned[field] <= 0:
                raise ValueError(f"Field {field} must be positive")

        cleaned["close"] = float(market_data.get("close") or cleaned["open"])
        if cleaned["high"] < cleaned["low"]:
            raise ValueError("High price must be >= low price")
        return cleaned

    def _store_prediction(
        self, symbol: str, data: Dict[str, float], vol: float, db: Session
    ) -> None:
        try:
            record = MarketData(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
                open_price=data["open"],
                high_price=data["high"],
                low_price=data["low"],
                close_price=data["close"],
                volume=data["volume"],
                volatility=vol,
            )
            db.add(record)
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
