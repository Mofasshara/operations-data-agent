# Operations Data AI Agent

An intelligent AI agent that lets you talk to your enterprise data using natural language. Ask questions, get insights, view charts, and receive predictions.

## Features

- **Natural Language Queries**: Ask questions like "What were our top products last quarter?"
- **Automatic Visualizations**: Generates charts for trends, comparisons, and distributions
- **Predictions & Forecasting**: Get sales forecasts and trend predictions
- **Actionable Insights**: Receive recommendations based on data analysis
- **Real Market Data**: Live stock prices, market indices, and economic indicators from Yahoo Finance

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/Mofasshara/operations-data-agent.git
cd operations-data-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 3. Initialize Database

```bash
# Option A: Synthetic data only (quick start)
python -m src.data.seed_data

# Option B: Load real market data from Yahoo Finance
python refresh_data.py

# Option C: Load both synthetic + real data
python refresh_data.py --synthetic
```

### 4. Run the App

```bash
streamlit run src/ui/app.py
```

## Data Sources

### Synthetic Enterprise Data (Faker)
| Table | Rows | Description |
|-------|------|-------------|
| customers | 500 | Fake company profiles |
| products | 12 | Product catalog |
| sales | 5,000 | Transaction history |
| sales_reps | 25 | Sales team |
| monthly_metrics | 25 | Aggregated KPIs |

### Real Market Data (Yahoo Finance API)
| Table | Source | Description |
|-------|--------|-------------|
| companies | Yahoo Finance | Real company info (AAPL, MSFT, etc.) |
| stock_prices | Yahoo Finance | Historical stock prices |
| company_financials | Yahoo Finance | Quarterly revenue, profit |
| market_indices | Yahoo Finance | S&P 500, Dow Jones, NASDAQ, etc. |
| sector_performance | Yahoo Finance | Technology, Healthcare, Energy ETFs |
| currency_rates | Yahoo Finance | EUR/USD, GBP/USD, etc. |
| commodities | Yahoo Finance | Gold, Oil, Bitcoin, etc. |

## Example Queries

### Enterprise Data
- "Show me total revenue by region"
- "What are the top 10 products by sales?"
- "Compare Q1 vs Q2 revenue"
- "Forecast sales for the next 3 months"

### Market Data
- "Show me Apple's stock price trend"
- "Compare tech sector vs healthcare performance"
- "What's the S&P 500 trend this year?"
- "Show me Bitcoin price history"
- "Which companies have the highest market cap?"

## Refreshing Data

```bash
# Refresh all real data (1 year history)
python refresh_data.py

# Refresh with 2 years of history
python refresh_data.py --period 2y

# Refresh only stock data
python refresh_data.py --stocks

# Refresh only market indices
python refresh_data.py --market

# Refresh specific stocks
python refresh_data.py --tickers AAPL MSFT GOOGL TSLA
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │Yahoo Finance│  │   Faker     │  │  Future:    │             │
│  │  (Real API) │  │ (Synthetic) │  │ Salesforce  │             │
│  └──────┬──────┘  └──────┬──────┘  └─────────────┘             │
└─────────┼────────────────┼──────────────────────────────────────┘
          │                │
          └────────┬───────┘
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                         DuckDB Database                         │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐      │
│  │ customers │ │  sales    │ │stock_prices│ │ indices   │      │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘      │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LangGraph Agents                           │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐      │
│  │SQL Agent  │ │Viz Agent  │ │Predict    │ │Insight    │      │
│  │           │ │(Plotly)   │ │(Prophet)  │ │Agent      │      │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘      │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Streamlit UI                               │
│                   http://localhost:8501                         │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
operations-data-agent/
├── src/
│   ├── agents/
│   │   ├── orchestrator.py       # Main LangGraph workflow
│   │   ├── sql_agent.py          # Text-to-SQL
│   │   ├── visualization_agent.py
│   │   └── prediction_agent.py
│   ├── connectors/               # NEW: Real data connectors
│   │   ├── yahoo_finance.py      # Stock & financial data
│   │   ├── market_data.py        # Indices, currencies, commodities
│   │   └── data_loader.py        # Load data into database
│   ├── data/
│   │   ├── database.py           # Database connection
│   │   └── seed_data.py          # Synthetic data generator
│   └── ui/
│       └── app.py                # Streamlit interface
├── docs/
│   └── ARCHITECTURE.md           # Detailed architecture diagrams
├── refresh_data.py               # CLI to refresh real data
├── requirements.txt
└── README.md
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Agent Framework | LangGraph + LangChain |
| LLM | OpenAI GPT-4o-mini |
| Database | DuckDB |
| Visualizations | Plotly |
| Forecasting | Prophet |
| Real Data | Yahoo Finance (yfinance) |
| Synthetic Data | Faker |
| Web UI | Streamlit |

## License

MIT
