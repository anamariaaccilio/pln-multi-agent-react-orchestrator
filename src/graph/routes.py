"""Logica de enrutamiento condicional del grafo LangGraph.

Equivale a la "politica" en un esquema de control secuencial: dado el estado
actual (observacion del auditor o del researcher), decide cual es la siguiente accion/nodo.
"""
from __future__ import annotations

from src.agents.state import AgentState


def route_after_researcher(state: AgentState) -> str:
    """
    Decide the next node after the researcher runs.

    - If the researcher produced tool_calls (plan phase) -> go to tool_node
    - If the researcher finished synthesizing (no pending tool calls) -> go to auditor
    """
    tool_calls = state.get("tool_calls", [])
    if tool_calls:
        return "tool_node"
    return "auditor"


def route_after_audit(state: AgentState) -> str:
    """
    Devuelve la clave de la siguiente arista, ya calculada por auditor_node
    en state["route_decision"]:

      - "writer"               -> audit_passed = True
      - "researcher"           -> audit_passed = False y quedan iteraciones
      - "writer_with_warning"  -> se alcanzo MAX_ITERATIONS sin aprobar

    LangGraph usa el string devuelto para elegir la arista de salida definida
    en add_conditional_edges() (ver src/graph/build_graph.py).
    """
    return state.get("route_decision", "writer")
