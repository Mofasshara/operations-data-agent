"""Visualization Agent - Creates charts from query results."""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


CHART_CONFIG_PROMPT = """You are a data visualization expert. Given a dataframe and user question, determine the best chart configuration.

DataFrame columns: {columns}
DataFrame sample (first 3 rows): {sample}
User question: {question}

Respond in this exact format (use actual column names from the dataframe):
CHART_TYPE: <bar, line, pie, scatter, area, histogram>
X_COLUMN: <column name for x-axis>
Y_COLUMN: <column name for y-axis>
COLOR_COLUMN: <column name for color grouping, or none>
TITLE: <descriptive chart title>
"""


class VisualizationAgent:
    """Creates visualizations from query results."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.prompt = ChatPromptTemplate.from_template(CHART_CONFIG_PROMPT)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def create_chart(self, df: pd.DataFrame, question: str, chart_hint: str = None) -> go.Figure:
        """Create an appropriate chart for the data."""

        if df.empty or len(df.columns) < 1:
            return None

        # Get chart configuration from LLM
        config = self._get_chart_config(df, question)

        # Override with hint if provided
        if chart_hint and chart_hint != 'table':
            config['chart_type'] = chart_hint

        # Create the chart
        try:
            fig = self._build_chart(df, config)
            return fig
        except Exception as e:
            print(f"Chart creation error: {e}")
            # Fallback to simple bar chart
            return self._create_fallback_chart(df, config.get('title', 'Results'))

    def _get_chart_config(self, df: pd.DataFrame, question: str) -> dict:
        """Get chart configuration from LLM."""
        columns = list(df.columns)
        sample = df.head(3).to_string()

        response = self.chain.invoke({
            "columns": columns,
            "sample": sample,
            "question": question
        })

        # Parse response
        config = {
            "chart_type": "bar",
            "x_column": columns[0] if columns else None,
            "y_column": columns[1] if len(columns) > 1 else columns[0],
            "color_column": None,
            "title": "Query Results"
        }

        for line in response.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace('_', '')
                value = value.strip()

                if 'charttype' in key:
                    config['chart_type'] = value.lower()
                elif 'xcolumn' in key or 'x column' in key:
                    if value.lower() != 'none' and value in df.columns:
                        config['x_column'] = value
                elif 'ycolumn' in key or 'y column' in key:
                    if value.lower() != 'none' and value in df.columns:
                        config['y_column'] = value
                elif 'colorcolumn' in key or 'color column' in key:
                    if value.lower() != 'none' and value in df.columns:
                        config['color_column'] = value
                elif 'title' in key:
                    config['title'] = value

        return config

    def _build_chart(self, df: pd.DataFrame, config: dict) -> go.Figure:
        """Build the chart based on configuration."""
        chart_type = config['chart_type']
        x = config['x_column']
        y = config['y_column']
        color = config['color_column']
        title = config['title']

        # Ensure we have valid columns
        if x not in df.columns:
            x = df.columns[0]
        if y not in df.columns:
            y = df.columns[-1] if len(df.columns) > 1 else df.columns[0]

        chart_builders = {
            'bar': lambda: px.bar(df, x=x, y=y, color=color, title=title),
            'line': lambda: px.line(df, x=x, y=y, color=color, title=title, markers=True),
            'pie': lambda: px.pie(df, names=x, values=y, title=title),
            'scatter': lambda: px.scatter(df, x=x, y=y, color=color, title=title),
            'area': lambda: px.area(df, x=x, y=y, color=color, title=title),
            'histogram': lambda: px.histogram(df, x=x, title=title),
        }

        builder = chart_builders.get(chart_type, chart_builders['bar'])
        fig = builder()

        # Style the chart
        fig.update_layout(
            template="plotly_white",
            font=dict(family="Arial", size=12),
            title_font_size=16,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40),
        )

        return fig

    def _create_fallback_chart(self, df: pd.DataFrame, title: str) -> go.Figure:
        """Create a simple fallback chart."""
        if len(df.columns) >= 2:
            fig = px.bar(df, x=df.columns[0], y=df.columns[1], title=title)
        else:
            fig = px.bar(df, x=df.index, y=df.columns[0], title=title)

        fig.update_layout(template="plotly_white")
        return fig

    def create_forecast_chart(
        self,
        historical: pd.DataFrame,
        forecast: pd.DataFrame,
        date_col: str,
        value_col: str,
        title: str = "Forecast"
    ) -> go.Figure:
        """Create a chart showing historical data and forecast."""
        fig = go.Figure()

        # Historical data
        fig.add_trace(go.Scatter(
            x=historical[date_col],
            y=historical[value_col],
            mode='lines+markers',
            name='Historical',
            line=dict(color='#2E86AB', width=2),
        ))

        # Forecast
        if 'yhat' in forecast.columns:
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#F18F01', width=2, dash='dash'),
            ))

            # Confidence interval
            if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
                fig.add_trace(go.Scatter(
                    x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
                    y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
                    fill='toself',
                    fillcolor='rgba(241, 143, 1, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval',
                ))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=value_col,
            template="plotly_white",
            hovermode='x unified',
        )

        return fig
