"""
Database configuration and session management.
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool, StaticPool
import logging
import os

from app.config import DATABASE_URL

logger = logging.getLogger(__name__)

# Determine if we're using SQLite or PostgreSQL
is_sqlite = DATABASE_URL.startswith("sqlite")

# Create SQLAlchemy engine with appropriate configuration
if is_sqlite:
    # SQLite configuration
    engine = create_engine(
        DATABASE_URL,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},  # Allow SQLite to be used in multiple threads
        echo=False,  # Set to True for SQL query logging
    )
else:
    # PostgreSQL configuration
    engine = create_engine(
        DATABASE_URL,
        poolclass=NullPool,  # Disable connection pooling for serverless
        echo=False,  # Set to True for SQL query logging
    )

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class for declarative models
Base = declarative_base()

def get_db():
    """
    Dependency to get database session.
    Yields a database session and ensures it's closed after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """
    Initialize database by creating all tables.
    """
    try:
        # Import all models to ensure they're registered with Base
        try:
            from app.models.db import Scenario, PredictionHistory
        except ImportError:
            # Models might not exist yet, that's okay for development
            logger.warning("Database models not found, skipping model import")
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info(f"Database tables created successfully using {'SQLite' if is_sqlite else 'PostgreSQL'}")
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")
        # Don't raise in development mode to allow the app to start
        if not is_sqlite:
            raise