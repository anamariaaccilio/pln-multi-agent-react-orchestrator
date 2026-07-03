"""Logica de enrutamiento condicional del grafo LangGraph.

Equivale a la "politica" en un esquema de control secuencial: dado el estado
actual (observacion del auditor), decide cual es la siguiente accion/nodo.
"""
from __future__ import annotations

from src.agents.state import AgentState


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
