"""Data Loader - Loads real API data into the database."""

import duckdb
import pandas as pd
from pathlib import Path
from datetime import datetime
import os

from .yahoo_finance import YahooFinanceConnector, fetch_stock_data
from .market_data import MarketDataConnector, fetch_market_data

DATABASE_PATH = os.getenv("DATABASE_PATH", "data/enterprise.duckdb")


class DataLoader:
    """Loads data from various sources into the database."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DATABASE_PATH
        self._ensure_db_exists()

    def _ensure_db_exists(self):
        """Ensure database directory exists."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self):
        """Get database connection."""
        return duckdb.connect(self.db_path)

    def create_real_data_tables(self):
        """Create tables for real API data."""
        conn = self._get_connection()

        # Companies (from Yahoo Finance)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS companies (
                ticker VARCHAR PRIMARY KEY,
                company_name VARCHAR,
                sector VARCHAR,
                industry VARCHAR,
                country VARCHAR,
                market_cap BIGINT,
                employees INTEGER,
                website VARCHAR,
                description VARCHAR,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Stock Prices (historical)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stock_prices (
                date DATE,
                ticker VARCHAR,
                open DECIMAL(12,4),
                high DECIMAL(12,4),
                low DECIMAL(12,4),
                close DECIMAL(12,4),
                volume BIGINT,
                PRIMARY KEY(date, ticker)
            )
        """)

        # Company Financials (quarterly)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS company_financials (
                ticker VARCHAR,
                quarter_date DATE,
                total_revenue DECIMAL(20,2),
                gross_profit DECIMAL(20,2),
                operating_income DECIMAL(20,2),
                net_income DECIMAL(20,2),
                PRIMARY KEY(ticker, quarter_date)
            )
        """)

        # Market Indices
        conn.execute("""
            CREATE TABLE IF NOT EXISTS market_indices (
                date DATE,
                symbol VARCHAR,
                index_name VARCHAR,
                open DECIMAL(12,4),
                high DECIMAL(12,4),
                low DECIMAL(12,4),
                close DECIMAL(12,4),
                volume BIGINT,
                PRIMARY KEY(date, symbol)
            )
        """)

        # Sector Performance
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sector_performance (
                date DATE,
                symbol VARCHAR,
                sector VARCHAR,
                open DECIMAL(12,4),
                high DECIMAL(12,4),
                low DECIMAL(12,4),
                close DECIMAL(12,4),
                volume BIGINT,
                PRIMARY KEY(date, symbol)
            )
        """)

        # Currency Rates
        conn.execute("""
            CREATE TABLE IF NOT EXISTS currency_rates (
                date DATE,
                symbol VARCHAR,
                currency_pair VARCHAR,
                open DECIMAL(10,6),
                high DECIMAL(10,6),
                low DECIMAL(10,6),
                close DECIMAL(10,6),
                PRIMARY KEY(date, symbol)
            )
        """)

        # Commodities
        conn.execute("""
            CREATE TABLE IF NOT EXISTS commodities (
                date DATE,
                symbol VARCHAR,
                commodity VARCHAR,
                open DECIMAL(12,4),
                high DECIMAL(12,4),
                low DECIMAL(12,4),
                close DECIMAL(12,4),
                volume BIGINT,
                PRIMARY KEY(date, symbol)
            )
        """)

        # Data refresh log
        conn.execute("""
            CREATE TABLE IF NOT EXISTS data_refresh_log (
                refresh_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                table_name VARCHAR,
                rows_loaded INTEGER,
                status VARCHAR,
                error_message VARCHAR
            )
        """)

        conn.close()
        print("Created real data tables")

    def _load_dataframe(self, conn, table_name: str, df: pd.DataFrame, unique_cols: list = None):
        """Load a DataFrame into a table, handling duplicates."""
        if df.empty:
            print(f"  No data to load for {table_name}")
            return 0

        # Create temp table and insert
        temp_table = f"temp_{table_name}"

        try:
            # Register DataFrame and insert
            conn.register('df_temp', df)

            if unique_cols:
                # Delete existing records that will be replaced
                unique_condition = " AND ".join([f"t.{col} = s.{col}" for col in unique_cols])
                conn.execute(f"""
                    DELETE FROM {table_name} t
                    WHERE EXISTS (
                        SELECT 1 FROM df_temp s
                        WHERE {unique_condition}
                    )
                """)

            # Insert new data
            cols = ", ".join(df.columns)
            conn.execute(f"INSERT INTO {table_name} ({cols}) SELECT {cols} FROM df_temp")

            rows = len(df)
            print(f"  Loaded {rows} rows into {table_name}")
            return rows

        except Exception as e:
            print(f"  Error loading {table_name}: {e}")
            return 0

    def load_stock_data(self, tickers: list = None, period: str = "1y"):
        """Load stock data from Yahoo Finance."""
        print("\n" + "="*50)
        print("LOADING STOCK DATA")
        print("="*50)

        data = fetch_stock_data(tickers, period)
        conn = self._get_connection()

        # Load companies
        if not data['companies'].empty:
            companies = data['companies'].copy()
            companies['last_updated'] = datetime.now()
            self._load_dataframe(conn, 'companies', companies, ['ticker'])

        # Load stock prices
        if not data['stock_prices'].empty:
            self._load_dataframe(conn, 'stock_prices', data['stock_prices'], ['date', 'ticker'])

        # Load financials
        if not data['financials'].empty:
            self._load_dataframe(conn, 'company_financials', data['financials'], ['ticker', 'quarter_date'])

        self._log_refresh(conn, 'stock_data', len(data.get('stock_prices', [])), 'success')
        conn.close()

    def load_market_data(self, period: str = "1y"):
        """Load market indices and economic data."""
        print("\n" + "="*50)
        print("LOADING MARKET DATA")
        print("="*50)

        data = fetch_market_data(period)
        conn = self._get_connection()

        # Load market indices
        if not data['market_indices'].empty:
            self._load_dataframe(conn, 'market_indices', data['market_indices'], ['date', 'symbol'])

        # Load sector performance
        if not data['sector_performance'].empty:
            self._load_dataframe(conn, 'sector_performance', data['sector_performance'], ['date', 'symbol'])

        # Load currency rates
        if not data['currency_rates'].empty:
            self._load_dataframe(conn, 'currency_rates', data['currency_rates'], ['date', 'symbol'])

        # Load commodities
        if not data['commodities'].empty:
            self._load_dataframe(conn, 'commodities', data['commodities'], ['date', 'symbol'])

        self._log_refresh(conn, 'market_data', len(data.get('market_indices', [])), 'success')
        conn.close()

    def _log_refresh(self, conn, table_name: str, rows: int, status: str, error: str = None):
        """Log data refresh."""
        conn.execute("""
            INSERT INTO data_refresh_log (table_name, rows_loaded, status, error_message)
            VALUES (?, ?, ?, ?)
        """, [table_name, rows, status, error])

    def load_all_data(self, period: str = "1y"):
        """Load all real data from APIs."""
        print("\n" + "="*60)
        print("  LOADING ALL REAL DATA FROM PUBLIC APIs")
        print("="*60)

        # Create tables if not exist
        self.create_real_data_tables()

        # Load stock data
        self.load_stock_data(period=period)

        # Load market data
        self.load_market_data(period=period)

        # Print summary
        self._print_summary()

    def _print_summary(self):
        """Print data loading summary."""
        conn = self._get_connection()

        print("\n" + "="*50)
        print("DATABASE SUMMARY")
        print("="*50)

        tables = [
            'companies', 'stock_prices', 'company_financials',
            'market_indices', 'sector_performance', 'currency_rates', 'commodities',
            'customers', 'products', 'sales', 'sales_reps', 'monthly_metrics'
        ]

        for table in tables:
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                print(f"  {table}: {count:,} rows")
            except:
                pass

        conn.close()

    def get_refresh_history(self) -> pd.DataFrame:
        """Get data refresh history."""
        conn = self._get_connection()
        df = conn.execute("""
            SELECT * FROM data_refresh_log
            ORDER BY refresh_time DESC
            LIMIT 20
        """).fetchdf()
        conn.close()
        return df


def refresh_all_data(period: str = "1y"):
    """Convenience function to refresh all data."""
    loader = DataLoader()
    loader.load_all_data(period)
    return loader


if __name__ == "__main__":
    refresh_all_data()
