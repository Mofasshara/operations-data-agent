"""Guardrails Agent - Validates queries and provides helpful responses."""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
from ..data.database import get_connection


class QueryGuardrails:
    """Validates queries and provides helpful error handling."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)

    def get_available_data_summary(self) -> str:
        """Get a summary of what data is available in the database."""
        conn = get_connection()

        summary_parts = []

        # Check each table and get sample values
        tables_info = {
            'customers': {'key_cols': ['region', 'segment', 'industry'], 'desc': 'Customer company data'},
            'products': {'key_cols': ['product_name', 'category'], 'desc': 'Product catalog'},
            'sales': {'key_cols': ['region', 'channel'], 'desc': 'Sales transactions'},
            'sales_reps': {'key_cols': ['region'], 'desc': 'Sales representatives'},
            'stock_prices': {'key_cols': ['ticker'], 'desc': 'Historical stock prices'},
            'companies': {'key_cols': ['ticker', 'sector'], 'desc': 'Company information'},
            'company_financials': {'key_cols': ['ticker'], 'desc': 'Quarterly financials'},
            'market_indices': {'key_cols': ['index_name'], 'desc': 'Market indices (S&P 500, etc.)'},
            'sector_performance': {'key_cols': ['sector'], 'desc': 'Sector ETF performance'},
            'currency_rates': {'key_cols': ['currency_pair'], 'desc': 'Currency exchange rates'},
            'commodities': {'key_cols': ['commodity'], 'desc': 'Commodity prices'},
        }

        for table, info in tables_info.items():
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                if count > 0:
                    # Get unique values for key columns
                    key_values = []
                    for col in info['key_cols']:
                        try:
                            values = conn.execute(f"SELECT DISTINCT {col} FROM {table} LIMIT 10").fetchall()
                            values = [str(v[0]) for v in values if v[0]]
                            if values:
                                key_values.append(f"{col}: {', '.join(values[:5])}")
                        except:
                            pass

                    summary_parts.append(f"- {table} ({count} rows): {info['desc']}")
                    if key_values:
                        summary_parts.append(f"    Available: {'; '.join(key_values)}")
            except:
                pass

        conn.close()
        return "\n".join(summary_parts) if summary_parts else "No data available"

    def check_query_relevance(self, question: str, schema: str) -> dict:
        """Check if the question is relevant to the available data."""

        prompt = ChatPromptTemplate.from_template("""You are a data assistant. Analyze if this question can be answered with the available database.

Available Database Schema:
{schema}

User Question: {question}

Respond in this exact format:
IS_RELEVANT: <true or false>
CONFIDENCE: <high, medium, or low>
REASON: <brief explanation>
SUGGESTED_TABLES: <comma-separated list of tables that might help, or "none">
""")

        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"question": question, "schema": schema})

        # Parse response
        result = {
            "is_relevant": True,
            "confidence": "medium",
            "reason": "",
            "suggested_tables": []
        }

        for line in response.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace('_', '')
                value = value.strip()

                if 'isrelevant' in key:
                    result['is_relevant'] = value.lower() == 'true'
                elif 'confidence' in key:
                    result['confidence'] = value.lower()
                elif 'reason' in key:
                    result['reason'] = value
                elif 'suggestedtables' in key:
                    if value.lower() != 'none':
                        result['suggested_tables'] = [t.strip() for t in value.split(',')]

        return result

    def handle_empty_result(self, question: str, sql_query: str) -> str:
        """Generate a helpful message when query returns no results."""

        # Get available data summary
        available_data = self.get_available_data_summary()

        prompt = ChatPromptTemplate.from_template("""The user asked a question but got no results. Provide a helpful response.

User Question: {question}
SQL Query Executed: {sql_query}
Result: No data found

Available Data in Database:
{available_data}

Provide a helpful response that:
1. Acknowledges no data was found
2. Explains possible reasons (data not loaded, wrong filter, etc.)
3. Suggests what data IS available that might be relevant
4. Suggests how to load more data if applicable

Keep the response concise (3-5 sentences).
""")

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "question": question,
            "sql_query": sql_query,
            "available_data": available_data
        })

    def handle_irrelevant_query(self, question: str) -> str:
        """Generate a response for questions outside the data scope."""

        available_data = self.get_available_data_summary()

        return f"""I can only answer questions about the data in my database. Your question doesn't seem to match the available data.

**What I can help with:**
{available_data}

**Example questions you can ask:**
- "Show me total revenue by region"
- "What are the top products by sales?"
- "Show me Apple stock price trend"
- "Compare sales by channel"

Please try rephrasing your question to match the available data."""

    def get_data_availability(self, entity_type: str, entity_value: str) -> dict:
        """Check if specific data is available (e.g., is 'TSLA' in stock_prices?)."""
        conn = get_connection()

        checks = {
            'ticker': [
                ('stock_prices', 'ticker'),
                ('companies', 'ticker'),
                ('company_financials', 'ticker'),
            ],
            'region': [
                ('sales', 'region'),
                ('customers', 'region'),
            ],
            'product': [
                ('products', 'product_name'),
                ('sales', 'product_id'),
            ],
        }

        results = {"found": False, "tables": [], "suggestions": []}

        if entity_type in checks:
            for table, column in checks[entity_type]:
                try:
                    count = conn.execute(f"""
                        SELECT COUNT(*) FROM {table}
                        WHERE LOWER({column}) LIKE LOWER('%{entity_value}%')
                    """).fetchone()[0]

                    if count > 0:
                        results["found"] = True
                        results["tables"].append(table)
                except:
                    pass

            # If not found, get suggestions
            if not results["found"]:
                for table, column in checks[entity_type]:
                    try:
                        available = conn.execute(f"""
                            SELECT DISTINCT {column} FROM {table} LIMIT 10
                        """).fetchall()
                        results["suggestions"].extend([v[0] for v in available if v[0]])
                    except:
                        pass

        conn.close()
        return results


class ResponseEnhancer:
    """Enhances responses with context and suggestions."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.3)

    def enhance_response(self, question: str, data: pd.DataFrame, base_response: str) -> str:
        """Add helpful context to the response."""

        if data is None or data.empty:
            return base_response

        # Add data quality notes if applicable
        notes = []

        # Check for nulls
        null_cols = data.columns[data.isnull().any()].tolist()
        if null_cols:
            notes.append(f"Note: Some columns have missing values: {', '.join(null_cols)}")

        # Check date range if date column exists
        date_cols = [c for c in data.columns if 'date' in c.lower()]
        if date_cols:
            try:
                min_date = pd.to_datetime(data[date_cols[0]]).min()
                max_date = pd.to_datetime(data[date_cols[0]]).max()
                notes.append(f"Data range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
            except:
                pass

        if notes:
            return base_response + "\n\n---\n" + "\n".join(notes)

        return base_response
