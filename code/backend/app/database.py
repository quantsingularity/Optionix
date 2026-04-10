import logging
from typing import Generator, Type

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool, Pool, QueuePool, StaticPool

from .config import settings
from .models import Base

logger = logging.getLogger(__name__)

database_url = settings.database_url
poolclass: Type[Pool]
if database_url.startswith("sqlite"):
    poolclass = StaticPool if ":memory:" in database_url else NullPool
    connect_args = {"check_same_thread": False}
    pool_pre_ping = False
else:
    poolclass = QueuePool
    connect_args = {}
    pool_pre_ping = True


def _make_engine(url: str, pc: Type[Pool], ca: dict, ping: bool):
    kwargs: dict = dict(
        poolclass=pc, pool_pre_ping=ping, echo=settings.debug, connect_args=ca
    )
    if pc == QueuePool:
        kwargs["pool_size"] = settings.database_pool_size
        kwargs["max_overflow"] = settings.database_max_overflow
    return create_engine(url, **kwargs)


try:
    engine = _make_engine(database_url, poolclass, connect_args, pool_pre_ping)
except Exception as e:
    logger.error(f"Failed to create database engine: {e}")
    logger.warning("Falling back to SQLite in-memory database")
    engine = create_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables() -> None:
    """Create all database tables"""
    global engine, SessionLocal
    try:
        with engine.connect():
            pass
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        logger.warning("Falling back to SQLite in-memory database")
        engine = create_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully (SQLite)")


def get_db() -> Generator[Session, None, None]:
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session() -> Session:
    """Get a database session for direct use"""
    return SessionLocal()
