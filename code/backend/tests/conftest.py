"""
Test configuration and fixtures for Optionix backend tests.
"""

import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

os.environ.setdefault("ENVIRONMENT", "testing")
os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-at-least-32-chars-long")
os.environ.setdefault("ENCRYPTION_KEY", "TestEncryptionKey1234567890!!!!!")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")

from unittest.mock import MagicMock, patch

from app import app
from database import Base, get_db

SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db():
    Base.metadata.create_all(bind=engine)
    db_session = TestingSessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db):
    def override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def mock_model_service(client):
    """Mock the model service for testing"""
    mock = MagicMock()
    mock.is_model_available.return_value = True
    mock.predict_volatility.return_value = 0.25
    with patch("backend.app.model_service", mock):
        yield mock


@pytest.fixture
def mock_blockchain_service(client):
    """Mock the blockchain service for testing"""
    mock = MagicMock()
    mock.is_connected.return_value = True
    mock.is_valid_address.return_value = True
    mock.get_position_health.return_value = {
        "address": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
        "size": 1.0,
        "is_long": True,
        "entry_price": 50000.0,
        "liquidation_price": 45000.0,
    }
    with patch("backend.app.blockchain_service", mock):
        yield mock
