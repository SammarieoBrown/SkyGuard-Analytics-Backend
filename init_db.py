"""
Initialize the database by creating all tables.
"""
import sys
from pathlib import Path
import logging

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.database import init_db, engine
from app.models import Scenario, PredictionHistory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Initialize the database."""
    try:
        logger.info("Initializing database...")
        logger.info(f"Connecting to database: {engine.url}")
        
        # Initialize database tables
        init_db()
        
        # Test connection
        from sqlalchemy import text
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            logger.info("Database connection successful")
            
        # List created tables
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        logger.info(f"Created tables: {tables}")
        
        logger.info("Database initialization complete!")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()