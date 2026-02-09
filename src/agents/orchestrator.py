"""Main orchestrator - Coordinates all agents using LangGraph."""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
import pandas as pd
import operator

from .sql_agent import SQLAgent, QueryAnalyzer
from .visualization_agent import VisualizationAgent
from .prediction_agent import PredictionAgent, TrendAnalyzer
from ..data.database import get_schema_info, execute_query


class AgentState(TypedDict):
    """State passed between agents."""
    question: str
    query_type: str
    visualization_type: str | None
    needs_forecast: bool
    schema: str
    sql_query: str | None
    sql_error: str | None
    data: pd.DataFrame | None
    chart: object | None
    forecast: pd.DataFrame | None
    forecast_metrics: dict | None
    insights: str | None
    response: str | None
    messages: Annotated[list, operator.add]


class DataAgent:
    """Main agent that orchestrates the workflow."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.sql_agent = SQLAgent(model_name)
        self.query_analyzer = QueryAnalyzer(model_name)
        self.viz_agent = VisualizationAgent(model_name)
        self.prediction_agent = PredictionAgent(model_name)
        self.trend_analyzer = TrendAnalyzer()

        # Build the workflow graph
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("generate_sql", self._generate_sql)
        workflow.add_node("execute_query", self._execute_query)
        workflow.add_node("create_visualization", self._create_visualization)
        workflow.add_node("generate_forecast", self._generate_forecast)
        workflow.add_node("generate_insights", self._generate_insights)
        workflow.add_node("compose_response", self._compose_response)

        # Define edges
        workflow.set_entry_point("analyze_query")
        workflow.add_edge("analyze_query", "generate_sql")
        workflow.add_edge("generate_sql", "execute_query")

        # Conditional routing after query execution
        workflow.add_conditional_edges(
            "execute_query",
            self._route_after_query,
            {
                "visualize": "create_visualization",
                "forecast": "generate_forecast",
                "insights": "generate_insights",
                "respond": "compose_response",
            }
        )

        workflow.add_edge("create_visualization", "generate_insights")
        workflow.add_edge("generate_forecast", "create_visualization")
        workflow.add_edge("generate_insights", "compose_response")
        workflow.add_edge("compose_response", END)

        return workflow.compile()

    def _analyze_query(self, state: AgentState) -> AgentState:
        """Analyze the user's question to determine intent."""
        analysis = self.query_analyzer.analyze(state["question"])
        schema = get_schema_info()

        return {
            **state,
            "query_type": analysis["query_type"],
            "visualization_type": analysis["visualization"],
            "needs_forecast": analysis["needs_forecast"],
            "schema": schema,
            "messages": [f"Analyzing query: {state['question']}"],
        }

    def _generate_sql(self, state: AgentState) -> AgentState:
        """Generate SQL from the question."""
        sql = self.sql_agent.generate_sql(state["question"], state["schema"])

        # Validate
        is_valid, error = self.sql_agent.validate_sql(sql)

        if not is_valid:
            return {
                **state,
                "sql_query": None,
                "sql_error": error,
                "messages": [f"SQL validation failed: {error}"],
            }

        return {
            **state,
            "sql_query": sql,
            "sql_error": None,
            "messages": [f"Generated SQL: {sql[:100]}..."],
        }

    def _execute_query(self, state: AgentState) -> AgentState:
        """Execute the SQL query."""
        if state.get("sql_error"):
            return {**state, "data": None}

        result, error = execute_query(state["sql_query"])

        if error:
            return {
                **state,
                "data": None,
                "sql_error": error,
                "messages": [f"Query execution error: {error}"],
            }

        return {
            **state,
            "data": result,
            "messages": [f"Query returned {len(result)} rows"],
        }

    def _route_after_query(self, state: AgentState) -> str:
        """Determine next step after query execution."""
        if state.get("sql_error") or state.get("data") is None:
            return "respond"

        if state.get("needs_forecast"):
            return "forecast"

        if state.get("visualization_type"):
            return "visualize"

        return "insights"

    def _create_visualization(self, state: AgentState) -> AgentState:
        """Create visualization if applicable."""
        if state.get("data") is None or state["data"].empty:
            return state

        # Check if we have forecast data to visualize
        if state.get("forecast") is not None and not state["forecast"].empty:
            # Create forecast chart
            historical = state["data"]
            date_col = [c for c in historical.columns if 'date' in c.lower() or 'month' in c.lower()]
            value_col = [c for c in historical.columns if c not in date_col]

            if date_col and value_col:
                chart = self.viz_agent.create_forecast_chart(
                    historical,
                    state["forecast"],
                    date_col[0],
                    value_col[0],
                    title="Forecast vs Historical"
                )
                return {**state, "chart": chart}

        # Regular chart
        chart = self.viz_agent.create_chart(
            state["data"],
            state["question"],
            state.get("visualization_type")
        )

        return {
            **state,
            "chart": chart,
            "messages": [f"Created {state.get('visualization_type', 'chart')} visualization"],
        }

    def _generate_forecast(self, state: AgentState) -> AgentState:
        """Generate forecast if requested."""
        if state.get("data") is None or state["data"].empty:
            return state

        df = state["data"]

        # Find date and value columns
        date_cols = [c for c in df.columns if 'date' in c.lower() or 'month' in c.lower()]
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        if not date_cols or not numeric_cols:
            return {
                **state,
                "messages": [f"Cannot forecast: need date and numeric columns"],
            }

        date_col = date_cols[0]
        value_col = numeric_cols[0]

        # Determine forecast periods from question
        periods = 90  # default 3 months
        question_lower = state["question"].lower()
        if "week" in question_lower:
            periods = 7
        elif "month" in question_lower:
            periods = 30
        elif "quarter" in question_lower:
            periods = 90
        elif "year" in question_lower:
            periods = 365

        forecast, metrics = self.prediction_agent.forecast_timeseries(
            df, date_col, value_col, periods
        )

        return {
            **state,
            "forecast": forecast,
            "forecast_metrics": metrics,
            "messages": [f"Generated {periods}-day forecast"],
        }

    def _generate_insights(self, state: AgentState) -> AgentState:
        """Generate insights about the data."""
        if state.get("data") is None or state["data"].empty:
            return state

        insights = self.prediction_agent.generate_insights(
            state["data"],
            state["question"],
            state.get("forecast_metrics")
        )

        return {
            **state,
            "insights": insights,
            "messages": ["Generated insights"],
        }

    def _compose_response(self, state: AgentState) -> AgentState:
        """Compose the final response."""
        parts = []

        # Handle errors
        if state.get("sql_error"):
            parts.append(f"**Error:** {state['sql_error']}")
            if state.get("sql_query"):
                parts.append(f"\n**SQL Attempted:**\n```sql\n{state['sql_query']}\n```")
        else:
            # Show SQL query
            if state.get("sql_query"):
                parts.append(f"**Query:**\n```sql\n{state['sql_query']}\n```")

            # Show data summary
            if state.get("data") is not None and not state["data"].empty:
                df = state["data"]
                parts.append(f"\n**Results:** {len(df)} rows returned")

            # Show forecast metrics
            if state.get("forecast_metrics"):
                metrics = state["forecast_metrics"]
                parts.append("\n**Forecast Summary:**")
                for key, value in metrics.items():
                    parts.append(f"- {key.replace('_', ' ').title()}: {value}")

            # Show insights
            if state.get("insights"):
                parts.append(f"\n{state['insights']}")

        return {
            **state,
            "response": "\n".join(parts),
        }

    def run(self, question: str) -> dict:
        """Run the agent with a question."""
        initial_state: AgentState = {
            "question": question,
            "query_type": "",
            "visualization_type": None,
            "needs_forecast": False,
            "schema": "",
            "sql_query": None,
            "sql_error": None,
            "data": None,
            "chart": None,
            "forecast": None,
            "forecast_metrics": None,
            "insights": None,
            "response": None,
            "messages": [],
        }

        result = self.workflow.invoke(initial_state)

        return {
            "response": result.get("response", "No response generated"),
            "data": result.get("data"),
            "chart": result.get("chart"),
            "sql_query": result.get("sql_query"),
            "forecast": result.get("forecast"),
        }
