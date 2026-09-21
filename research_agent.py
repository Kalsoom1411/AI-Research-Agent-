"""Single CrewAI agent that researches a topic and writes a report."""

import os

# These must be set BEFORE crewai is imported.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_STORAGE_DIR", "/tmp/crewai")

from crewai import Agent, Crew, LLM, Process, Task  # noqa: E402

from search_tool import WebSearchTool  # noqa: E402

# Groq's model id is "openai/gpt-oss-120b".
# CrewAI needs a "groq/" provider prefix in front of it.
MODEL_ID = "groq/openai/gpt-oss-120b"


def build_crew(topic: str, api_key: str, max_results: int = 5, temperature: float = 0.4) -> Crew:
    llm = LLM(
        model=MODEL_ID,
        api_key=api_key,
        temperature=temperature,
        max_tokens=8000,
    )

    researcher = Agent(
        role="Senior Research Analyst",
        goal=f"Research '{topic}' thoroughly using the web and write an accurate, well-sourced report.",
        backstory=(
            "You are a meticulous analyst with fifteen years of experience turning scattered "
            "web sources into clear briefing documents. You never state a fact you have not "
            "verified through a search, you always note when sources disagree, and you write "
            "in plain language that a non-expert can follow."
        ),
        tools=[WebSearchTool(max_results=max_results)],
        llm=llm,
        verbose=True,
        allow_delegation=False,   # single agent: nobody to delegate to
        max_iter=12,              # ceiling on think/search loops
    )

    research_task = Task(
        description=(
            "Write a research report on this topic: {topic}\n\n"
            "Follow this process:\n"
            "1. Run at least 3 different web searches with varied wording to gather sources. "
            "Search for background, for recent developments, and for criticism or counterarguments.\n"
            "2. Read the snippets and identify the key facts, figures and dates.\n"
            "3. Note where sources disagree or where evidence is thin.\n"
            "4. Write the final report.\n\n"
            "Rules: base every claim on your search results. If you could not verify something, "
            "say so explicitly instead of guessing."
        ),
        expected_output=(
            "A markdown report on {topic} with exactly these sections:\n"
            "# <Report title>\n"
            "## Executive Summary  (3-4 sentences)\n"
            "## Background  (what this is and why it matters)\n"
            "## Key Findings  (5-8 bullet points, each with a concrete fact or figure)\n"
            "## Current Developments  (the most recent news you found)\n"
            "## Challenges and Open Questions\n"
            "## Conclusion\n"
            "## Sources  (numbered list of the actual URLs you used)\n\n"
            "Output raw markdown only. Do not wrap the whole report in a code fence."
        ),
        agent=researcher,
        markdown=True,
    )

    return Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential,
        verbose=True,
        memory=False,   # keep it off: memory pulls in a vector DB and an embeddings key
    )


def run_research(topic: str, api_key: str, max_results: int = 5, temperature: float = 0.4) -> str:
    crew = build_crew(topic, api_key, max_results, temperature)
    result = crew.kickoff(inputs={"topic": topic})
    return str(result)
