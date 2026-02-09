"""Prediction Agent - Time series forecasting and predictions."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Prophet import with fallback
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not installed. Using simple forecasting fallback.")


class PredictionAgent:
    """Handles forecasting and predictions."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.3)

    def forecast_timeseries(
        self,
        df: pd.DataFrame,
        date_column: str,
        value_column: str,
        periods: int = 90
    ) -> tuple[pd.DataFrame, dict]:
        """
        Forecast future values using Prophet or fallback method.

        Returns:
            tuple: (forecast_df, metrics_dict)
        """
        # Prepare data for Prophet format
        prophet_df = df[[date_column, value_column]].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        prophet_df = prophet_df.sort_values('ds')

        if PROPHET_AVAILABLE:
            return self._prophet_forecast(prophet_df, periods)
        else:
            return self._simple_forecast(prophet_df, periods)

    def _prophet_forecast(self, df: pd.DataFrame, periods: int) -> tuple[pd.DataFrame, dict]:
        """Use Prophet for forecasting."""
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
        )

        model.fit(df)

        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        # Calculate metrics
        metrics = self._calculate_metrics(df, forecast)

        return forecast, metrics

    def _simple_forecast(self, df: pd.DataFrame, periods: int) -> tuple[pd.DataFrame, dict]:
        """Simple linear trend forecast as fallback."""
        df = df.copy()
        df['ds_numeric'] = (df['ds'] - df['ds'].min()).dt.days

        # Fit linear regression
        x = df['ds_numeric'].values
        y = df['y'].values

        # Simple linear regression
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
        intercept = (np.sum(y) - slope * np.sum(x)) / n

        # Generate future dates
        last_date = df['ds'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')

        # Calculate predictions
        future_numeric = (future_dates - df['ds'].min()).days
        predictions = slope * future_numeric + intercept

        # Create forecast dataframe
        forecast = pd.DataFrame({
            'ds': pd.concat([df['ds'], pd.Series(future_dates)]),
            'yhat': np.concatenate([slope * x + intercept, predictions]),
        })

        # Add confidence intervals (simple percentage)
        std = df['y'].std()
        forecast['yhat_lower'] = forecast['yhat'] - 1.96 * std
        forecast['yhat_upper'] = forecast['yhat'] + 1.96 * std

        metrics = {
            'trend': 'increasing' if slope > 0 else 'decreasing',
            'slope_per_day': slope,
            'forecast_avg': predictions.mean(),
        }

        return forecast, metrics

    def _calculate_metrics(self, historical: pd.DataFrame, forecast: pd.DataFrame) -> dict:
        """Calculate forecast metrics and insights."""
        hist_mean = historical['y'].mean()
        hist_std = historical['y'].std()

        # Future only
        future_mask = forecast['ds'] > historical['ds'].max()
        future_forecast = forecast[future_mask]

        if len(future_forecast) == 0:
            return {}

        forecast_mean = future_forecast['yhat'].mean()
        forecast_end = future_forecast['yhat'].iloc[-1]
        forecast_start = future_forecast['yhat'].iloc[0]

        growth = ((forecast_end - forecast_start) / forecast_start * 100) if forecast_start != 0 else 0
        change_from_hist = ((forecast_mean - hist_mean) / hist_mean * 100) if hist_mean != 0 else 0

        return {
            'historical_mean': round(hist_mean, 2),
            'forecast_mean': round(forecast_mean, 2),
            'expected_growth_percent': round(growth, 2),
            'change_from_historical': round(change_from_hist, 2),
            'forecast_min': round(future_forecast['yhat'].min(), 2),
            'forecast_max': round(future_forecast['yhat'].max(), 2),
        }

    def generate_insights(self, df: pd.DataFrame, question: str, forecast_metrics: dict = None) -> str:
        """Generate natural language insights about the data."""

        # Build context
        context_parts = []

        if not df.empty:
            context_parts.append(f"Data summary: {len(df)} rows")
            for col in df.select_dtypes(include=[np.number]).columns[:3]:
                context_parts.append(f"  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, avg={df[col].mean():.2f}")

        if forecast_metrics:
            context_parts.append(f"Forecast metrics: {forecast_metrics}")

        context = "\n".join(context_parts)

        prompt = ChatPromptTemplate.from_template("""You are a business analyst. Based on the data and question, provide 2-3 key insights and actionable recommendations.

Data Context:
{context}

User Question: {question}

Provide insights in this format:
## Key Insights
- [insight 1]
- [insight 2]
- [insight 3]

## Recommendations
- [recommendation 1]
- [recommendation 2]
""")

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question})


class TrendAnalyzer:
    """Analyzes trends in data."""

    @staticmethod
    def detect_trend(df: pd.DataFrame, value_column: str) -> dict:
        """Detect trend direction and strength."""
        if len(df) < 2:
            return {"trend": "insufficient_data"}

        values = df[value_column].values

        # Simple linear regression for trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        # Calculate percent change
        pct_change = ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0

        # Determine trend strength
        if abs(pct_change) < 5:
            strength = "stable"
        elif abs(pct_change) < 15:
            strength = "moderate"
        else:
            strength = "strong"

        return {
            "trend": "increasing" if slope > 0 else "decreasing",
            "strength": strength,
            "percent_change": round(pct_change, 2),
            "slope": round(slope, 4),
        }

    @staticmethod
    def detect_anomalies(df: pd.DataFrame, value_column: str, threshold: float = 2.0) -> pd.DataFrame:
        """Detect anomalies using z-score method."""
        values = df[value_column]
        mean = values.mean()
        std = values.std()

        if std == 0:
            return pd.DataFrame()

        z_scores = (values - mean) / std
        anomalies = df[abs(z_scores) > threshold].copy()
        anomalies['z_score'] = z_scores[abs(z_scores) > threshold]

        return anomalies

    @staticmethod
    def compare_periods(
        df: pd.DataFrame,
        date_column: str,
        value_column: str,
        period: str = 'month'
    ) -> dict:
        """Compare current period to previous period."""
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])

        if period == 'month':
            current_start = pd.Timestamp.now().replace(day=1)
            previous_start = (current_start - pd.DateOffset(months=1))
            previous_end = current_start - timedelta(days=1)
        elif period == 'quarter':
            current_start = pd.Timestamp.now().replace(day=1, month=((pd.Timestamp.now().month - 1) // 3) * 3 + 1)
            previous_start = current_start - pd.DateOffset(months=3)
            previous_end = current_start - timedelta(days=1)
        else:  # year
            current_start = pd.Timestamp.now().replace(day=1, month=1)
            previous_start = current_start - pd.DateOffset(years=1)
            previous_end = current_start - timedelta(days=1)

        current_data = df[df[date_column] >= current_start][value_column]
        previous_data = df[(df[date_column] >= previous_start) & (df[date_column] <= previous_end)][value_column]

        current_total = current_data.sum()
        previous_total = previous_data.sum()

        change_pct = ((current_total - previous_total) / previous_total * 100) if previous_total != 0 else 0

        return {
            "current_period_total": round(current_total, 2),
            "previous_period_total": round(previous_total, 2),
            "change_percent": round(change_pct, 2),
            "change_direction": "up" if change_pct > 0 else "down" if change_pct < 0 else "flat",
        }
