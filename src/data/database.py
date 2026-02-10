"""Database connection and schema utilities."""

import os
import duckdb
from pathlib import Path

DATABASE_PATH = os.getenv("DATABASE_PATH", "data/enterprise.duckdb")


def get_connection() -> duckdb.DuckDBPyConnection:
    """Get a connection to the DuckDB database."""
    db_path = Path(DATABASE_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))


def get_schema_info() -> str:
    """Get database schema information for the AI agent."""
    conn = get_connection()

    schema_info = []

    # Get all tables
    tables = conn.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main'
    """).fetchall()

    for (table_name,) in tables:
        # Get columns for each table
        columns = conn.execute(f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
        """).fetchall()

        col_info = ", ".join([f"{col} ({dtype})" for col, dtype in columns])
        schema_info.append(f"Table: {table_name}\n  Columns: {col_info}")

        # Get sample values for categorical columns
        sample_query = f"SELECT * FROM {table_name} LIMIT 3"
        samples = conn.execute(sample_query).fetchdf()
        if not samples.empty:
            row_count = conn.execute(f'SELECT COUNT(*) FROM {table_name}').fetchone()[0]
            schema_info.append(f"  Total rows: {row_count}")

    conn.close()
    return "\n\n".join(schema_info)


def execute_query(sql: str) -> tuple:
    """Execute a SQL query and return results as DataFrame."""
    conn = get_connection()
    try:
        result = conn.execute(sql).fetchdf()
        conn.close()
        return result, None
    except Exception as e:
        conn.close()
        return None, str(e)


def get_table_samples() -> dict:
    """Get sample data from each table for context."""
    conn = get_connection()
    samples = {}

    tables = conn.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main'
    """).fetchall()

    for (table_name,) in tables:
        df = conn.execute(f"SELECT * FROM {table_name} LIMIT 5").fetchdf()
        samples[table_name] = df.to_dict('records')

    conn.close()
    return samples
