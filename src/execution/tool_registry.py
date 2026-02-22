"""Natural language to analytics function mapping.

Maps user questions to pre-built analytics functions using regex
pattern matching. This is a deterministic, fast alternative to
LLM-based function calling — appropriate for a system with a
known set of capabilities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd


@dataclass
class Tool:
    """A registered analytics tool."""

    name: str
    description: str
    patterns: list[str]  # Regex patterns to match
    handler: Callable[[], pd.DataFrame]
    example_questions: list[str] = field(default_factory=list)


@dataclass
class ToolResult:
    """Result of executing a tool."""

    tool_name: str
    description: str
    data: pd.DataFrame
    matched_pattern: str


class ToolRegistry:
    """Registry mapping natural language questions to analytics functions."""

    def __init__(self) -> None:
        self.tools: list[Tool] = []

    def register(
        self,
        name: str,
        description: str,
        patterns: list[str],
        handler: Callable[[], pd.DataFrame],
        example_questions: list[str] | None = None,
    ) -> None:
        """Register a new tool."""
        self.tools.append(
            Tool(
                name=name,
                description=description,
                patterns=patterns,
                handler=handler,
                example_questions=example_questions or [],
            )
        )

    def match(self, question: str) -> Tool | None:
        """Find the best matching tool for a natural language question.

        Tries each tool's regex patterns (case-insensitive).
        Returns first match, or None.
        """
        question_lower = question.lower()
        for tool in self.tools:
            for pattern in tool.patterns:
                if re.search(pattern, question_lower):
                    return tool
        return None

    def execute(self, question: str) -> ToolResult | None:
        """Match and execute a tool. Returns ToolResult or None."""
        tool = self.match(question)
        if tool is None:
            return None

        data = tool.handler()
        matched = next(
            (p for p in tool.patterns if re.search(p, question.lower())),
            "",
        )

        return ToolResult(
            tool_name=tool.name,
            description=tool.description,
            data=data,
            matched_pattern=matched,
        )

    def list_tools(self) -> list[dict[str, Any]]:
        """List all registered tools with descriptions and examples."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "example_questions": t.example_questions,
            }
            for t in self.tools
        ]


def build_default_registry(store: Any) -> ToolRegistry:
    """Build the default tool registry with all analytics functions.

    Args:
        store: A TradeStore instance with data loaded.

    Returns:
        Configured ToolRegistry.
    """
    registry = ToolRegistry()

    registry.register(
        name="venue_performance",
        description="Average execution quality by trading venue",
        patterns=[r"venue", r"exchange", r"where.*executed", r"routing"],
        handler=store.avg_is_by_venue,
        example_questions=[
            "Which venue has the best execution quality?",
            "Compare IS across exchanges",
        ],
    )

    registry.register(
        name="worst_executions",
        description="Worst executions by implementation shortfall",
        patterns=[r"worst", r"bad.*execution", r"outlier", r"highest.*is"],
        handler=lambda: store.worst_executions(20),
        example_questions=[
            "Show me the worst executions",
            "Which orders had the highest IS?",
        ],
    )

    registry.register(
        name="is_decomposition",
        description="Implementation Shortfall component breakdown",
        patterns=[r"implementation shortfall", r"\bis\b.*decomp", r"cost.*breakdown", r"delay.*cost"],
        handler=store.is_decomposition_summary,
        example_questions=[
            "Break down the implementation shortfall components",
            "What is the average delay cost?",
        ],
    )

    registry.register(
        name="fill_rates",
        description="Fill rate analysis by order type",
        patterns=[r"fill rate", r"completion", r"unfilled", r"order type"],
        handler=store.fill_rate_by_order_type,
        example_questions=[
            "What is the fill rate by order type?",
            "How many orders are partially filled?",
        ],
    )

    registry.register(
        name="symbol_analysis",
        description="Execution quality by symbol/stock",
        patterns=[r"symbol", r"stock", r"ticker", r"by name", r"aapl|msft|googl|nvda"],
        handler=store.avg_is_by_symbol,
        example_questions=[
            "Which stock has the worst execution quality?",
            "Show IS by symbol",
        ],
    )

    registry.register(
        name="daily_summary",
        description="Daily execution quality trends",
        patterns=[r"daily", r"day.*by.*day", r"trend", r"over time"],
        handler=store.daily_summary,
        example_questions=[
            "Show daily execution quality trends",
            "How has IS changed over time?",
        ],
    )

    registry.register(
        name="venue_market_share",
        description="Fill volume distribution across venues",
        patterns=[r"market share", r"volume.*venue", r"distribution"],
        handler=store.venue_market_share,
        example_questions=[
            "What is the venue market share?",
            "How is fill volume distributed?",
        ],
    )

    return registry
