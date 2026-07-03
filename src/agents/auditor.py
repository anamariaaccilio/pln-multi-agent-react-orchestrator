"""Agente Auditor de Hechos: evalua evidencia y decide aprobar o rechazar el borrador.

Guardrails contra alucinaciones: en vez de confiar en que el propio LLM se
autoevalue, el auditor calcula evidence_score y hallucination_risk con una
heuristica lexica independiente (interseccion de palabras clave entre el
borrador y el contexto). Es un proxy ligero de "faithfulness", no un NLI real;
ver docs/architecture.md para como sustituirlo por un juez basado en LLM.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from src.agents.prompts import AUDITOR_PROMPT
from src.agents.state import AgentState, EvidenceChunk
from src.config import DEFAULT_CONFIG
from src.utils.trace import add_trace

_STOPWORDS = {
    "de", "la", "el", "en", "y", "a", "los", "las", "un", "una", "que", "es",
    "por", "para", "con", "su", "sus", "se", "del", "al", "lo", "como", "fue",
    "esta", "este", "esa", "ese",
}


def _tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-zA-ZáéíóúñÁÉÍÓÚÑ0-9]+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


def estimate_evidence_score(draft_answer: str, context: List[EvidenceChunk]) -> float:
    """Proporcion de palabras clave del borrador que tambien aparecen en el contexto."""
    if not context or not draft_answer.strip():
        return 0.0
    context_tokens = set()
    for chunk in context:
        context_tokens.update(_tokenize(chunk["content"]))
    draft_tokens = _tokenize(draft_answer)
    if not draft_tokens:
        return 0.0
    supported = sum(1 for t in draft_tokens if t in context_tokens)
    return round(supported / len(draft_tokens), 2)


def estimate_hallucination_risk(evidence_score: float, context: List[EvidenceChunk]) -> float:
    """Riesgo de alucinacion: complemento del evidence_score, maximo si no hay contexto."""
    if not context:
        return 1.0
    return round(max(0.0, 1.0 - evidence_score), 2)


def auditor_node(state: AgentState, agent_config=None) -> Dict[str, Any]:
    """
    Nodo LangGraph del Agente Auditor.

    Nota: el parametro se llama `agent_config` (no `config`) para evitar que
    LangGraph inyecte su propio RunnableConfig (ver researcher_node).

    Prompt: AUDITOR_PROMPT (ver src/agents/prompts.py).

    Reglas de aprobacion:
      - Si no hay contexto -> rechazar.
      - Si evidence_score < MIN_EVIDENCE_SCORE -> rechazar.
      - Si hallucination_risk > MAX_HALLUCINATION_RISK -> rechazar.
      - En otro caso -> aprobar.
      - Si se alcanzo MAX_ITERATIONS sin aprobar -> ruta a writer_with_warning.

    Thought -> Action -> Observation:
      - Thought: debo verificar si el borrador esta sustentado en el contexto.
      - Action: estimate_evidence_score + estimate_hallucination_risk.
      - Observation: decision de auditoria (aprobado/rechazado) y motivo.
    """
    cfg = agent_config or DEFAULT_CONFIG
    context = state.get("retrieved_context", [])
    draft_answer = state.get("draft_answer", "")
    iterations = state.get("iterations", 0)
    trace = list(state.get("trace", []))

    trace = add_trace(
        trace,
        agent="auditor",
        thought="Debo verificar si el borrador esta sustentado en el contexto.",
        action="estimate_evidence_score(draft_answer, context)",
        observation="Calculando evidence_score y hallucination_risk.",
    )

    evidence_score = estimate_evidence_score(draft_answer, context)
    hallucination_risk = estimate_hallucination_risk(evidence_score, context)

    missing_info = ""
    audit_passed = True

    if not context:
        audit_passed = False
        missing_info = "No se recupero contexto para la pregunta."
        audit_feedback = "Rechazado: no hay evidencia disponible en el contexto recuperado."
    elif evidence_score < cfg.min_evidence_score:
        audit_passed = False
        missing_info = "El borrador incluye afirmaciones no respaldadas por el contexto."
        audit_feedback = (
            f"Rechazado: evidence_score={evidence_score} < {cfg.min_evidence_score}. "
            "Reformula la respuesta usando unicamente el contexto proporcionado."
        )
    elif hallucination_risk > cfg.max_hallucination_risk:
        audit_passed = False
        missing_info = "Riesgo de alucinacion por encima del umbral permitido."
        audit_feedback = (
            f"Rechazado: hallucination_risk={hallucination_risk} > {cfg.max_hallucination_risk}."
        )
    else:
        audit_feedback = "Aprobado: la respuesta esta razonablemente sustentada en el contexto."

    reached_max_iterations = iterations >= cfg.max_iterations
    if audit_passed:
        route_decision = "writer"
    elif reached_max_iterations:
        route_decision = "writer_with_warning"
    else:
        route_decision = "researcher"

    trace = add_trace(
        trace,
        agent="auditor",
        thought="Decidiendo ruta segun evidence_score, hallucination_risk e iteraciones.",
        action=f"route_after_audit(audit_passed={audit_passed}, iterations={iterations})",
        observation=f"Decision de ruta: {route_decision}.",
    )

    return {
        "audit_passed": audit_passed,
        "audit_feedback": audit_feedback,
        "missing_info": missing_info,
        "evidence_score": evidence_score,
        "hallucination_risk": hallucination_risk,
        "route_decision": route_decision,
        "trace": trace,
    }


__all__ = ["auditor_node", "AUDITOR_PROMPT", "estimate_evidence_score", "estimate_hallucination_risk"]
