"""DuckDuckGo search tool for the research agent. No API key needed."""

from typing import Type

from crewai.tools import BaseTool
from ddgs import DDGS
from pydantic import BaseModel, Field


class SearchInput(BaseModel):
    """Defines what the agent must pass to this tool."""
    query: str = Field(..., description="A short web search query, e.g. 'solid state battery breakthroughs 2026'")


class WebSearchTool(BaseTool):
    name: str = "Web Search"
    description: str = (
        "Searches the public web using DuckDuckGo and returns the top results "
        "as numbered titles, URLs and text snippets. "
        "Use this whenever you need facts, news, statistics or sources. "
        "Call it several times with different queries to cover a topic properly."
    )
    args_schema: Type[BaseModel] = SearchInput
    max_results: int = 5

    def _run(self, query: str) -> str:
        try:
            results = DDGS().text(query, max_results=self.max_results)
        except Exception as exc:
            return f"Search failed for '{query}': {exc}. Try a shorter, simpler query."

        if not results:
            return f"No results found for '{query}'. Try different keywords."

        blocks = []
        for i, r in enumerate(results, start=1):
            blocks.append(
                f"{i}. {r.get('title', 'Untitled')}\n"
                f"   URL: {r.get('href', 'n/a')}\n"
                f"   Snippet: {r.get('body', '')}"
            )
        return "\n\n".join(blocks)
