"""SQL Agent - Converts natural language to SQL queries."""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re


SQL_GENERATION_PROMPT = """You are an expert SQL analyst. Your job is to convert natural language questions into SQL queries for a DuckDB database.

## Database Schema:
{schema}

## Rules:
1. Write valid DuckDB SQL syntax
2. Use appropriate aggregations (SUM, AVG, COUNT, etc.) when asked for totals or averages
3. Use DATE functions for time-based queries (DATE_TRUNC, EXTRACT, etc.)
4. Always include ORDER BY for ranked results
5. Use LIMIT for "top N" queries
6. For percentage calculations, multiply by 100
7. Use table aliases for clarity in JOINs
8. Return ONLY the SQL query, no explanations

## Common Patterns:
- "last month" = WHERE sale_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') AND sale_date < DATE_TRUNC('month', CURRENT_DATE)
- "last quarter" = last 3 months
- "year over year" = compare current period to same period last year
- "by region/product/etc" = GROUP BY that column

## User Question:
{question}

## SQL Query:"""


class SQLAgent:
    """Agent that converts natural language to SQL and executes queries."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.prompt = ChatPromptTemplate.from_template(SQL_GENERATION_PROMPT)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate_sql(self, question: str, schema: str) -> str:
        """Generate SQL from natural language question."""
        response = self.chain.invoke({
            "question": question,
            "schema": schema
        })

        # Clean up the response - extract just the SQL
        sql = self._clean_sql(response)
        return sql

    def _clean_sql(self, text: str) -> str:
        """Extract and clean SQL from LLM response."""
        # Remove markdown code blocks if present
        text = re.sub(r'```sql\s*', '', text)
        text = re.sub(r'```\s*', '', text)

        # Remove any explanatory text before SELECT
        if 'SELECT' in text.upper():
            idx = text.upper().find('SELECT')
            text = text[idx:]

        # Remove trailing explanations after the query
        # Find the last semicolon or end of query
        text = text.strip()
        if ';' in text:
            text = text[:text.rfind(';') + 1]

        return text.strip()

    def validate_sql(self, sql: str) -> tuple[bool, str]:
        """Basic SQL validation."""
        sql_upper = sql.upper()

        # Check for dangerous operations
        dangerous = ['DROP', 'DELETE', 'TRUNCATE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
        for keyword in dangerous:
            if keyword in sql_upper:
                return False, f"Query contains forbidden keyword: {keyword}"

        # Check it starts with SELECT
        if not sql_upper.strip().startswith('SELECT'):
            return False, "Query must start with SELECT"

        return True, "Valid"


QUERY_ANALYSIS_PROMPT = """Analyze this user question and determine:
1. What type of query is this? (data_retrieval, trend_analysis, comparison, prediction, recommendation)
2. Does it need a visualization? If so, what type? (bar, line, pie, table, none)
3. Does it need forecasting/prediction?

Question: {question}

Respond in this exact format:
QUERY_TYPE: <type>
VISUALIZATION: <chart_type or none>
NEEDS_FORECAST: <true or false>
"""


class QueryAnalyzer:
    """Analyzes queries to determine routing."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.prompt = ChatPromptTemplate.from_template(QUERY_ANALYSIS_PROMPT)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def analyze(self, question: str) -> dict:
        """Analyze the question and return routing info."""
        response = self.chain.invoke({"question": question})

        # Parse the response
        result = {
            "query_type": "data_retrieval",
            "visualization": "table",
            "needs_forecast": False
        }

        for line in response.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip().lower()

                if 'query_type' in key:
                    result['query_type'] = value
                elif 'visualization' in key:
                    result['visualization'] = value if value != 'none' else None
                elif 'forecast' in key:
                    result['needs_forecast'] = value == 'true'

        return result
