import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

_engine = None


def get_engine():
    """Returns a singleton SQLAlchemy engine, created lazily."""
    global _engine
    if _engine is None:
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise RuntimeError("DATABASE_URL is not set. Copy .env.example to .env and fill it in.")
        _engine = create_engine(db_url, pool_pre_ping=True)
    return _engine


def test_connection():
    """Simple sanity check used by scripts/test_connection.py"""
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version();"))
        return result.scalar()


def get_schema_summary():
    """
    Introspects the public schema: table names, columns, types.
    This will be fed to the LLM as context for text-to-SQL generation in Phase 1.
    """
    engine = get_engine()
    query = text("""
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position;
    """)
    schema = {}
    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()
        for table_name, column_name, data_type in rows:
            schema.setdefault(table_name, []).append((column_name, data_type))
    return schema
