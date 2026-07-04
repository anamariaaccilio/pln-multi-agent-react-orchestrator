"""Agente Redactor: produce la respuesta final con evidencia y nivel de confianza."""
from __future__ import annotations

from typing import Any, Dict, List

from src.agents.prompts import WRITER_PROMPT
from src.agents.state import AgentState
from src.config import DEFAULT_CONFIG
from src.utils.trace import add_trace


def _confidence_level(evidence_score: float, hallucination_risk: float, forced_by_max_iter: bool) -> str:
    if forced_by_max_iter or evidence_score < 0.5:
        return "Bajo"
    if evidence_score >= 0.70 and hallucination_risk <= 0.30:
        return "Alto"
    return "Medio"


def writer_node(state: AgentState, agent_config=None) -> Dict[str, Any]:
    """
    Nodo LangGraph del Agente Redactor.

    Prompt: WRITER_PROMPT (ver src/agents/prompts.py).

    Thought -> Action -> Observation:
      - Thought: debo comunicar la respuesta final con evidencia y confianza.
      - Action: format_final_answer(draft_answer, evidence, audit_result).
      - Observation: respuesta final generada (con o sin advertencia).

    Nota: el parametro se llama `agent_config` (no `config`) para evitar que
    LangGraph inyecte su propio RunnableConfig (ver researcher_node).
    """
    cfg = agent_config or DEFAULT_CONFIG
    context = state.get("retrieved_context", [])
    draft_answer = state.get("draft_answer", "")
    evidence_score = state.get("evidence_score", 0.0)
    hallucination_risk = state.get("hallucination_risk", 1.0)
    route_decision = state.get("route_decision", "")
    trace = list(state.get("trace", []))

    trace = add_trace(
        trace,
        agent="writer",
        thought="Debo comunicar la respuesta final con evidencia y nivel de confianza.",
        action="format_final_answer(draft_answer, evidence, audit_result)",
        observation="Construyendo respuesta final estructurada.",
    )

    warnings: List[str] = []
    forced_by_max_iterations = route_decision == "writer_with_warning"
    if forced_by_max_iterations:
        warnings.append(
            f"Se alcanzo MAX_ITERATIONS ({cfg.max_iterations}) sin aprobacion completa del auditor."
        )

    no_evidence = not context or evidence_score == 0.0
    if no_evidence:
        warnings.append("No hay evidencia suficiente en el contexto recuperado.")

    confidence = _confidence_level(evidence_score, hallucination_risk, forced_by_max_iterations)

    print(f"\n[WRITER] Confidence: {confidence} | Warnings: {len(warnings)}")
    print(f"[WRITER] Evidence score: {evidence_score:.2f} | Route: {route_decision}\n")

    if no_evidence:
        final_answer = (
            "No hay evidencia suficiente en el contexto recuperado para "
            "responder con seguridad."
        )
    else:
        evidence_bullets = "\n".join(f"- {c['content']}" for c in context)
        warning_text = "\n".join(warnings) if warnings else "Ninguna."
        final_answer = (
            f"Respuesta final:\n{draft_answer.strip()}\n\n"
            f"Evidencia usada:\n{evidence_bullets}\n\n"
            f"Nivel de confianza:\n{confidence}\n\n"
            f"Advertencia:\n{warning_text}"
        )

    trace = add_trace(
        trace,
        agent="writer",
        thought="Respuesta final lista para entrega y evaluacion.",
        action="return final_answer",
        observation=f"Nivel de confianza: {confidence}. Advertencias: {len(warnings)}.",
    )

    return {
        "final_answer": final_answer,
        "confidence_level": confidence,
        "warnings": warnings,
        "trace": trace,
    }


__all__ = ["writer_node", "WRITER_PROMPT"]
