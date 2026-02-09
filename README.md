# Operations Data AI Agent

An intelligent AI agent that lets you talk to your enterprise data using natural language. Ask questions, get insights, view charts, and receive predictions.

## Features

- **Natural Language Queries**: Ask questions like "What were our top products last quarter?"
- **Automatic Visualizations**: Generates charts for trends, comparisons, and distributions
- **Predictions & Forecasting**: Get sales forecasts and trend predictions
- **Actionable Insights**: Receive recommendations based on data analysis

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
python -m src.data.seed_data
```

### 4. Run the App

```bash
streamlit run src/ui/app.py
```

## Example Queries

- "Show me total revenue by region"
- "What are the top 10 products by sales?"
- "Compare Q1 vs Q2 revenue"
- "Forecast sales for the next 3 months"
- "Which customers have the highest order value?"
- "Show me the monthly sales trend"

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Streamlit  │────▶│  LangGraph  │────▶│   DuckDB    │
│     UI      │     │   Agents    │     │  Database   │
└─────────────┘     └──────┬──────┘     └─────────────┘
                           │
                    ┌──────┴──────┐
                    ▼             ▼
              ┌──────────┐  ┌──────────┐
              │  Plotly  │  │  Prophet │
              │  Charts  │  │ Forecast │
              └──────────┘  └──────────┘
```

## Project Structure

```
operations-data-agent/
├── src/
│   ├── agents/
│   │   ├── orchestrator.py    # Main agent workflow
│   │   ├── sql_agent.py       # Text-to-SQL
│   │   ├── visualization_agent.py
│   │   └── prediction_agent.py
│   ├── data/
│   │   ├── database.py        # Database connection
│   │   └── seed_data.py       # Sample data generator
│   ├── ui/
│   │   └── app.py             # Streamlit interface
│   └── utils/
│       └── helpers.py
├── config/
├── tests/
├── requirements.txt
└── README.md
```

## Tech Stack

- **LangGraph** - Agent orchestration
- **OpenAI GPT-4o** - Natural language understanding
- **DuckDB** - Fast analytical database
- **Plotly** - Interactive visualizations
- **Prophet** - Time series forecasting
- **Streamlit** - Web interface

## License

MIT
