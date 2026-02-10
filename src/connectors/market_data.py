"""Market Data Connector - Economic indicators and market indices."""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
import requests


class MarketDataConnector:
    """Connector for market indices and economic data."""

    # Major market indices
    INDICES = {
        '^GSPC': 'S&P 500',
        '^DJI': 'Dow Jones',
        '^IXIC': 'NASDAQ',
        '^RUT': 'Russell 2000',
        '^VIX': 'VIX Volatility',
        '^FTSE': 'FTSE 100',
        '^N225': 'Nikkei 225',
        '^HSI': 'Hang Seng',
    }

    # Sector ETFs for sector analysis
    SECTOR_ETFS = {
        'XLK': 'Technology',
        'XLF': 'Financials',
        'XLV': 'Healthcare',
        'XLE': 'Energy',
        'XLI': 'Industrials',
        'XLP': 'Consumer Staples',
        'XLY': 'Consumer Discretionary',
        'XLU': 'Utilities',
        'XLB': 'Materials',
        'XLRE': 'Real Estate',
    }

    # Currency pairs
    CURRENCIES = {
        'EURUSD=X': 'EUR/USD',
        'GBPUSD=X': 'GBP/USD',
        'USDJPY=X': 'USD/JPY',
        'USDCHF=X': 'USD/CHF',
        'AUDUSD=X': 'AUD/USD',
    }

    # Commodities
    COMMODITIES = {
        'GC=F': 'Gold',
        'SI=F': 'Silver',
        'CL=F': 'Crude Oil',
        'NG=F': 'Natural Gas',
        'BTC-USD': 'Bitcoin',
        'ETH-USD': 'Ethereum',
    }

    def get_index_data(self, period: str = "2y") -> pd.DataFrame:
        """
        Get historical data for major market indices.

        Returns:
            DataFrame with index performance data
        """
        all_data = []

        print("Fetching market indices...")
        for symbol, name in self.INDICES.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)

                if hist.empty:
                    continue

                hist = hist.reset_index()
                hist['symbol'] = symbol
                hist['index_name'] = name
                hist.columns = [c.lower().replace(' ', '_') for c in hist.columns]

                cols = ['date', 'symbol', 'index_name', 'open', 'high', 'low', 'close', 'volume']
                available_cols = [c for c in cols if c in hist.columns]
                all_data.append(hist[available_cols])

                print(f"  Fetched {name}")

            except Exception as e:
                print(f"  Error fetching {name}: {e}")

        if not all_data:
            return pd.DataFrame()

        df = pd.concat(all_data, ignore_index=True)
        df['date'] = pd.to_datetime(df['date']).dt.date
        return df

    def get_sector_performance(self, period: str = "1y") -> pd.DataFrame:
        """
        Get sector ETF performance data.

        Returns:
            DataFrame with sector performance
        """
        all_data = []

        print("Fetching sector ETFs...")
        for symbol, sector in self.SECTOR_ETFS.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)

                if hist.empty:
                    continue

                hist = hist.reset_index()
                hist['symbol'] = symbol
                hist['sector'] = sector
                hist.columns = [c.lower().replace(' ', '_') for c in hist.columns]

                cols = ['date', 'symbol', 'sector', 'open', 'high', 'low', 'close', 'volume']
                available_cols = [c for c in cols if c in hist.columns]
                all_data.append(hist[available_cols])

                print(f"  Fetched {sector}")

            except Exception as e:
                print(f"  Error fetching {sector}: {e}")

        if not all_data:
            return pd.DataFrame()

        df = pd.concat(all_data, ignore_index=True)
        df['date'] = pd.to_datetime(df['date']).dt.date
        return df

    def get_currency_rates(self, period: str = "1y") -> pd.DataFrame:
        """
        Get currency exchange rate data.

        Returns:
            DataFrame with currency rates
        """
        all_data = []

        print("Fetching currency rates...")
        for symbol, pair in self.CURRENCIES.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)

                if hist.empty:
                    continue

                hist = hist.reset_index()
                hist['symbol'] = symbol
                hist['currency_pair'] = pair
                hist.columns = [c.lower().replace(' ', '_') for c in hist.columns]

                cols = ['date', 'symbol', 'currency_pair', 'open', 'high', 'low', 'close']
                available_cols = [c for c in cols if c in hist.columns]
                all_data.append(hist[available_cols])

                print(f"  Fetched {pair}")

            except Exception as e:
                print(f"  Error fetching {pair}: {e}")

        if not all_data:
            return pd.DataFrame()

        df = pd.concat(all_data, ignore_index=True)
        df['date'] = pd.to_datetime(df['date']).dt.date
        return df

    def get_commodity_prices(self, period: str = "1y") -> pd.DataFrame:
        """
        Get commodity and crypto prices.

        Returns:
            DataFrame with commodity prices
        """
        all_data = []

        print("Fetching commodities and crypto...")
        for symbol, name in self.COMMODITIES.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)

                if hist.empty:
                    continue

                hist = hist.reset_index()
                hist['symbol'] = symbol
                hist['commodity'] = name
                hist.columns = [c.lower().replace(' ', '_') for c in hist.columns]

                cols = ['date', 'symbol', 'commodity', 'open', 'high', 'low', 'close', 'volume']
                available_cols = [c for c in cols if c in hist.columns]
                all_data.append(hist[available_cols])

                print(f"  Fetched {name}")

            except Exception as e:
                print(f"  Error fetching {name}: {e}")

        if not all_data:
            return pd.DataFrame()

        df = pd.concat(all_data, ignore_index=True)
        df['date'] = pd.to_datetime(df['date']).dt.date
        return df

    def get_market_summary(self) -> pd.DataFrame:
        """
        Get current market summary with key metrics.

        Returns:
            DataFrame with current market data
        """
        summary = []

        all_symbols = {**self.INDICES, **self.SECTOR_ETFS}

        print("Fetching market summary...")
        for symbol, name in all_symbols.items():
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="5d")

                if hist.empty:
                    continue

                current = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
                change_pct = ((current - prev) / prev * 100) if prev else 0

                summary.append({
                    'symbol': symbol,
                    'name': name,
                    'current_price': round(current, 2),
                    'previous_close': round(prev, 2),
                    'change_percent': round(change_pct, 2),
                    'volume': hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0,
                })

            except Exception as e:
                print(f"  Error: {e}")

        return pd.DataFrame(summary)


def fetch_market_data(period: str = "1y") -> dict:
    """
    Fetch all market data in one call.

    Returns:
        Dictionary with DataFrames for each data type
    """
    connector = MarketDataConnector()

    print("\n" + "="*50)
    print("FETCHING MARKET DATA")
    print("="*50 + "\n")

    indices = connector.get_index_data(period)
    sectors = connector.get_sector_performance(period)
    currencies = connector.get_currency_rates(period)
    commodities = connector.get_commodity_prices(period)

    return {
        'market_indices': indices,
        'sector_performance': sectors,
        'currency_rates': currencies,
        'commodities': commodities,
    }
