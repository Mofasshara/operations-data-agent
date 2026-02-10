"""Yahoo Finance Connector - Pulls real stock and financial data."""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional


class YahooFinanceConnector:
    """Connector for Yahoo Finance API to get real market data."""

    # Popular tech companies for demo
    DEFAULT_TICKERS = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "NVDA", "TSLA", "JPM", "V", "JNJ",
        "WMT", "PG", "HD", "BAC", "DIS"
    ]

    def __init__(self, tickers: List[str] = None):
        """
        Initialize with a list of stock tickers.

        Args:
            tickers: List of stock symbols (e.g., ["AAPL", "MSFT"])
        """
        self.tickers = tickers or self.DEFAULT_TICKERS

    def get_stock_prices(self, period: str = "2y") -> pd.DataFrame:
        """
        Get historical stock prices for all tickers.

        Args:
            period: Time period - 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max

        Returns:
            DataFrame with columns: date, ticker, open, high, low, close, volume
        """
        all_data = []

        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)

                if hist.empty:
                    continue

                hist = hist.reset_index()
                hist['ticker'] = ticker
                hist.columns = [c.lower().replace(' ', '_') for c in hist.columns]

                # Select relevant columns
                cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
                available_cols = [c for c in cols if c in hist.columns]
                hist = hist[available_cols]

                all_data.append(hist)
                print(f"  Fetched {len(hist)} records for {ticker}")

            except Exception as e:
                print(f"  Error fetching {ticker}: {e}")

        if not all_data:
            return pd.DataFrame()

        df = pd.concat(all_data, ignore_index=True)
        df['date'] = pd.to_datetime(df['date']).dt.date
        return df

    def get_company_info(self) -> pd.DataFrame:
        """
        Get company information for all tickers.

        Returns:
            DataFrame with company details
        """
        companies = []

        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                companies.append({
                    'ticker': ticker,
                    'company_name': info.get('longName', info.get('shortName', ticker)),
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    'country': info.get('country', 'Unknown'),
                    'market_cap': info.get('marketCap', 0),
                    'employees': info.get('fullTimeEmployees', 0),
                    'website': info.get('website', ''),
                    'description': info.get('longBusinessSummary', '')[:500] if info.get('longBusinessSummary') else '',
                })
                print(f"  Fetched info for {ticker}")

            except Exception as e:
                print(f"  Error fetching info for {ticker}: {e}")

        return pd.DataFrame(companies)

    def get_financials(self) -> pd.DataFrame:
        """
        Get quarterly financial data (revenue, profit, etc.)

        Returns:
            DataFrame with financial metrics
        """
        financials = []

        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)

                # Get quarterly financials
                quarterly = stock.quarterly_financials
                if quarterly.empty:
                    continue

                for col in quarterly.columns:
                    quarter_date = col.date() if hasattr(col, 'date') else col

                    row = {
                        'ticker': ticker,
                        'quarter_date': quarter_date,
                        'total_revenue': quarterly.loc['Total Revenue', col] if 'Total Revenue' in quarterly.index else None,
                        'gross_profit': quarterly.loc['Gross Profit', col] if 'Gross Profit' in quarterly.index else None,
                        'operating_income': quarterly.loc['Operating Income', col] if 'Operating Income' in quarterly.index else None,
                        'net_income': quarterly.loc['Net Income', col] if 'Net Income' in quarterly.index else None,
                    }
                    financials.append(row)

                print(f"  Fetched financials for {ticker}")

            except Exception as e:
                print(f"  Error fetching financials for {ticker}: {e}")

        df = pd.DataFrame(financials)
        if not df.empty:
            df['quarter_date'] = pd.to_datetime(df['quarter_date']).dt.date
        return df

    def get_dividends(self) -> pd.DataFrame:
        """
        Get dividend history for all tickers.

        Returns:
            DataFrame with dividend payments
        """
        dividends = []

        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                div = stock.dividends

                if div.empty:
                    continue

                div_df = div.reset_index()
                div_df.columns = ['date', 'dividend']
                div_df['ticker'] = ticker
                dividends.append(div_df)

            except Exception as e:
                print(f"  Error fetching dividends for {ticker}: {e}")

        if not dividends:
            return pd.DataFrame()

        df = pd.concat(dividends, ignore_index=True)
        df['date'] = pd.to_datetime(df['date']).dt.date
        return df[['date', 'ticker', 'dividend']]

    def get_current_prices(self) -> pd.DataFrame:
        """
        Get current stock prices and key metrics.

        Returns:
            DataFrame with current price data
        """
        prices = []

        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                prices.append({
                    'ticker': ticker,
                    'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                    'previous_close': info.get('previousClose', 0),
                    'open': info.get('open', 0),
                    'day_high': info.get('dayHigh', 0),
                    'day_low': info.get('dayLow', 0),
                    'volume': info.get('volume', 0),
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'dividend_yield': info.get('dividendYield', 0),
                    '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                    '52_week_low': info.get('fiftyTwoWeekLow', 0),
                })

            except Exception as e:
                print(f"  Error fetching current price for {ticker}: {e}")

        return pd.DataFrame(prices)


# Convenience function
def fetch_stock_data(tickers: List[str] = None, period: str = "1y") -> dict:
    """
    Fetch all stock data in one call.

    Args:
        tickers: List of stock symbols
        period: Historical data period

    Returns:
        Dictionary with DataFrames for each data type
    """
    connector = YahooFinanceConnector(tickers)

    print("Fetching stock data from Yahoo Finance...")

    print("\n1. Fetching company info...")
    companies = connector.get_company_info()

    print("\n2. Fetching historical prices...")
    prices = connector.get_stock_prices(period)

    print("\n3. Fetching financials...")
    financials = connector.get_financials()

    print("\n4. Fetching current prices...")
    current = connector.get_current_prices()

    return {
        'companies': companies,
        'stock_prices': prices,
        'financials': financials,
        'current_prices': current,
    }
