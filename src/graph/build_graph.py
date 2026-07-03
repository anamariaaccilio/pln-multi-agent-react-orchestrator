"""Construccion del grafo LangGraph: Investigador -> Auditor -> Redactor.

    START
      |
      v
   researcher
      |
      v
    auditor --route_after_audit-->  "writer"              --> writer --> END
                               |---> "researcher" (ciclo)  --> researcher
                               '---> "writer_with_warning" --> writer --> END
"""
from __future__ import annotations

from functools import partial
from typing import Optional

from langgraph.graph import END, StateGraph

from src.agents.auditor import auditor_node
from src.agents.researcher import researcher_node
from src.agents.state import AgentState
from src.agents.writer import writer_node
from src.config import AgentConfig, DEFAULT_CONFIG
from src.graph.routes import route_after_audit


def build_agent_graph(config: Optional[AgentConfig] = None, llm_fn=None):
    """
    Construye y compila el grafo LangGraph del módulo de orquestación.

    Args:
        config: AgentConfig con los umbrales/flags a usar. Por defecto DEFAULT_CONFIG.
        llm_fn: funcion generate_llm_response(prompt) a inyectar en el researcher_node
            (util para pasar un LLM real sin tocar el codigo del nodo).

    Returns:
        Grafo compilado (invocable con .invoke(initial_state)).
    """
    cfg = config or DEFAULT_CONFIG
    graph = StateGraph(AgentState)

    graph.add_node("researcher", partial(researcher_node, agent_config=cfg, llm_fn=llm_fn))
    graph.add_node("auditor", partial(auditor_node, agent_config=cfg))
    graph.add_node("writer", partial(writer_node, agent_config=cfg))

    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "auditor")
    graph.add_conditional_edges(
        "auditor",
        route_after_audit,
        {
            "writer": "writer",
            "writer_with_warning": "writer",
            "researcher": "researcher",
        },
    )
    graph.add_edge("writer", END)

    return graph.compile()
