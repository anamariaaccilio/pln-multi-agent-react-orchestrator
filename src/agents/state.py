"""Definicion del estado global compartido por el grafo LangGraph (AgentState).

LangGraph pasa este diccionario de nodo a nodo: cada nodo (researcher_node,
auditor_node, writer_node) recibe el AgentState completo y devuelve solo las
claves que actualiza; LangGraph fusiona ese resultado sobre el estado previo.
"""
from __future__ import annotations

from typing import List, TypedDict


class TraceEvent(TypedDict):
    """Un evento de traza ReAct: Thought -> Action -> Observation."""

    agent: str
    thought: str
    action: str
    observation: str


class EvidenceChunk(TypedDict):
    """Un fragmento de evidencia devuelto por retrieve_context()."""

    content: str
    source: str
    score: float


class ToolCall(TypedDict):
    """A pending tool invocation from the researcher."""

    tool_name: str
    args: dict


class ToolResult(TypedDict):
    """The result of executing a tool."""

    tool_name: str
    observation: str


class AgentState(TypedDict, total=False):
    """Estado compartido entre researcher_node, auditor_node y writer_node."""

    # Entrada
    question: str

    # Salida del Agente Investigador
    retrieved_context: List[EvidenceChunk]
    draft_answer: str
    evidence_list: List[str]
    limitations: str

    # Tool interaction (researcher <-> tool_node)
    tool_calls: List[ToolCall]
    tool_results: List[ToolResult]
    researcher_step: str  # "plan" | "synthesize"

    # Salida del Agente Auditor
    audit_passed: bool
    audit_feedback: str
    missing_info: str
    evidence_score: float
    hallucination_risk: float
    route_decision: str

    # Salida del Agente Redactor
    final_answer: str
    confidence_level: str
    warnings: List[str]

    # Control de flujo y trazabilidad
    iterations: int
    trace: List[TraceEvent]
    system_type: str
