#!/usr/bin/env python3
"""
Data Refresh Script - Load real data from public APIs.

Usage:
    python refresh_data.py              # Refresh all data (1 year history)
    python refresh_data.py --period 2y  # Refresh with 2 years history
    python refresh_data.py --stocks     # Refresh only stock data
    python refresh_data.py --market     # Refresh only market data
    python refresh_data.py --synthetic  # Also regenerate synthetic data
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Refresh data from public APIs")
    parser.add_argument(
        "--period",
        default="1y",
        help="Historical data period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y (default: 1y)"
    )
    parser.add_argument(
        "--stocks",
        action="store_true",
        help="Only refresh stock data"
    )
    parser.add_argument(
        "--market",
        action="store_true",
        help="Only refresh market/economic data"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Also regenerate synthetic enterprise data"
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Specific stock tickers to fetch (e.g., AAPL MSFT GOOGL)"
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("  OPERATIONS DATA AI AGENT - DATA REFRESH")
    print("="*60)

    # Import here to avoid issues if dependencies not installed
    from src.connectors.data_loader import DataLoader

    loader = DataLoader()

    # Create tables first
    loader.create_real_data_tables()

    if args.stocks:
        print(f"\nRefreshing STOCK data (period: {args.period})...")
        loader.load_stock_data(tickers=args.tickers, period=args.period)

    elif args.market:
        print(f"\nRefreshing MARKET data (period: {args.period})...")
        loader.load_market_data(period=args.period)

    else:
        print(f"\nRefreshing ALL data (period: {args.period})...")
        loader.load_all_data(period=args.period)

    if args.synthetic:
        print("\nRegenerating synthetic enterprise data...")
        from src.data.seed_data import (
            create_tables, generate_products, generate_customers,
            generate_sales_reps, generate_sales, generate_monthly_metrics
        )
        import duckdb

        conn = duckdb.connect(loader.db_path)

        # Clear existing synthetic data
        for table in ['customers', 'products', 'sales', 'sales_reps', 'monthly_metrics']:
            try:
                conn.execute(f"DELETE FROM {table}")
            except:
                pass

        create_tables(conn)
        generate_products(conn)
        generate_customers(conn, n=500)
        generate_sales_reps(conn, n=25)
        generate_sales(conn, n=5000)
        generate_monthly_metrics(conn)
        conn.close()

    # Print final summary
    loader._print_summary()

    print("\n" + "="*60)
    print("  DATA REFRESH COMPLETE!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
