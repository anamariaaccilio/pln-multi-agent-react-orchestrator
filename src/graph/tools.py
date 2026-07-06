"""
Tools available to the Researcher Agent.

Defines LangGraph-compatible tool wrappers for:
  - knowledge_base_search: queries the local WikiQA ChromaDB vector store.
  - web_search: queries DuckDuckGo for live web results.

These tools are invoked by the tool_node in the graph when the researcher
decides an action.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List


@dataclass
class Tool:
    """A simple tool descriptor that the researcher can invoke."""

    name: str
    description: str
    fn: Callable[..., str]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _knowledge_base_search(question: str, top_k: int = 5) -> str:
    """Search the local WikiQA vector store and return formatted results.
    
    Also stores raw structured results in _last_kb_results for the tool_node
    to pass as retrieved_context to downstream nodes.
    """
    from src.retriever.interface import resolve_retriever
    from src.config import DEFAULT_CONFIG

    retrieve_context = resolve_retriever(DEFAULT_CONFIG.use_fallback_retriever)
    results = retrieve_context(question, top_k=top_k)
    print('[KB TOOL] ', results, '\n')

    # Store structured results for downstream use (avoids re-querying)
    global _last_kb_results
    _last_kb_results = results

    if not results:
        return "No se encontraron resultados relevantes en la base de conocimiento."

    lines = []
    for i, chunk in enumerate(results, 1):
        score = chunk.get("score", 0.0)
        source = chunk.get("source", "unknown")
        content = chunk.get("content", "")
        lines.append(f"[{i}] (score={score:.2f}, source={source})\n{content}")
    return "\n\n".join(lines)


# Module-level storage for structured KB results
_last_kb_results: list = []


def get_last_kb_results() -> list:
    """Return the structured results from the last knowledge_base_search call."""
    return _last_kb_results


def _web_search(query: str, max_results: int = 3) -> str:
    """Search the web via DuckDuckGo and return formatted results."""
    try:
        from ddgs import DDGS
    except ImportError:
        return (
            "ERROR: duckduckgo-search no esta instalado. "
            "Ejecuta: pip install duckduckgo-search"
        )

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            print('[DDGS TOOL] ', results, '\n')
    except Exception as e:
        print('[DDGS TOOL] [ERROR], ', e, '\n')
        return f"ERROR en busqueda web: {e}"

    if not results:
        return "No se encontraron resultados en la web para esta consulta."

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        body = r.get("body", "")
        href = r.get("href", "")
        lines.append(f"[{i}] {title}\n{body}\nURL: {href}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOL_KNOWLEDGE_BASE = Tool(
    name="knowledge_base_search",
    description=(
        "Busca en la base de conocimiento vectorial local (WikiQA/ChromaDB). "
        "Usa esta herramienta para encontrar informacion factual sobre temas "
        "cubiertos por el dataset. Parametros: question (str), top_k (int, default 5)."
    ),
    fn=_knowledge_base_search,
)

TOOL_WEB_SEARCH = Tool(
    name="web_search",
    description=(
        "Busca en internet via DuckDuckGo. Usa esta herramienta cuando la "
        "base de conocimiento local no tiene informacion suficiente, o para "
        "obtener datos actualizados. Parametros: query (str), max_results (int, default 3)."
    ),
    fn=_web_search,
)

RESEARCHER_TOOLS: List[Tool] = [TOOL_KNOWLEDGE_BASE, TOOL_WEB_SEARCH]


def get_tool_by_name(name: str) -> Tool | None:
    """Look up a tool by name."""
    for tool in RESEARCHER_TOOLS:
        if tool.name == name:
            return tool
    return None


def format_tools_description() -> str:
    """Return a formatted description of all available tools for the prompt."""
    lines = []
    for tool in RESEARCHER_TOOLS:
        lines.append(f"- {tool.name}: {tool.description}")
    return "\n".join(lines)
