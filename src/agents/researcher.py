"""Agente Investigador: recupera evidencia y redacta una respuesta preliminar."""
from __future__ import annotations

from typing import Any, Dict

from src.agents.prompts import RESEARCHER_PROMPT, build_researcher_prompt
from src.agents.state import AgentState
from src.config import DEFAULT_CONFIG
from src.llm.interface import resolve_llm
from src.retriever.interface import resolve_retriever
from src.utils.trace import add_trace


def researcher_node(state: AgentState, agent_config=None, llm_fn=None) -> Dict[str, Any]:
    """
    Nodo LangGraph del Agente Investigador.

    Prompt: RESEARCHER_PROMPT (ver src/agents/prompts.py).

    Thought -> Action -> Observation:
      - Thought: necesito recuperar evidencia antes de responder.
      - Action: retrieve_context(question, top_k) + generate_llm_response(prompt).
      - Observation: numero de fragmentos recuperados y longitud del borrador.

    Nota: el parametro se llama `agent_config` (no `config`) a proposito.
    LangGraph inyecta automaticamente su propio RunnableConfig en cualquier
    parametro de nodo literalmente llamado `config`, lo que pisaria nuestro
    AgentConfig si usaramos ese nombre.
    """
    cfg = agent_config or DEFAULT_CONFIG
    question = state["question"]
    iterations = state.get("iterations", 0)
    trace = list(state.get("trace", []))

    retrieve_context = resolve_retriever(cfg.use_fallback_retriever)
    retrieved = retrieve_context(question, top_k=cfg.top_k)

    trace = add_trace(
        trace,
        agent="researcher",
        thought="Necesito recuperar evidencia antes de responder.",
        action=f"retrieve_context(question, top_k={cfg.top_k})",
        observation=f"Se recuperaron {len(retrieved)} fragmentos relevantes.",
    )

    generate = llm_fn or resolve_llm(cfg.use_fallback_llm, cfg.llm_provider)
    prior_feedback = state.get("audit_feedback", "") if state.get("iterations", 0) > 0 else ""
    prompt = build_researcher_prompt(question, retrieved, prior_feedback)
    draft_answer = generate(prompt)

    max_score = max((c["score"] for c in retrieved), default=0.0)
    limitations = (
        "Contexto limitado o de baja relevancia; la respuesta puede ser incompleta."
        if not retrieved or max_score < 0.5
        else ""
    )

    trace = add_trace(
        trace,
        agent="researcher",
        thought="Generando respuesta preliminar apoyada unicamente en el contexto.",
        action="generate_llm_response(researcher_prompt)",
        observation=f"Borrador generado ({len(draft_answer)} caracteres).",
    )

    return {
        "retrieved_context": retrieved,
        "draft_answer": draft_answer,
        "evidence_list": [c["content"] for c in retrieved],
        "limitations": limitations,
        "iterations": iterations + 1,
        "trace": trace,
        "system_type": cfg.system_type,
    }


__all__ = ["researcher_node", "RESEARCHER_PROMPT"]
