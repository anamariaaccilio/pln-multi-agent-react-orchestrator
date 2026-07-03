"""Función principal del módulo de orquestación: multi_agent_rag(question)."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from src.agents.state import AgentState
from src.config import AgentConfig, DEFAULT_CONFIG
from src.graph.build_graph import build_agent_graph
from src.retriever.interface import register_retriever

# Cache de grafos compilados por combinacion de configuracion, para no
# recompilar el grafo LangGraph en cada llamada (por ejemplo, en un notebook
# que llama a multi_agent_rag() muchas veces sobre el mismo dataset).
_COMPILED_GRAPH_CACHE: Dict[str, Any] = {}


def _build_initial_state(question: str, system_type: str) -> AgentState:
    return {
        "question": question,
        "retrieved_context": [],
        "draft_answer": "",
        "evidence_list": [],
        "limitations": "",
        "audit_passed": False,
        "audit_feedback": "",
        "missing_info": "",
        "evidence_score": 0.0,
        "hallucination_risk": 1.0,
        "route_decision": "",
        "final_answer": "",
        "confidence_level": "Bajo",
        "warnings": [],
        "iterations": 0,
        "trace": [],
        "system_type": system_type,
    }


def multi_agent_rag(
    question: str,
    retriever: Optional[Callable[..., Any]] = None,
    llm: Optional[Callable[[str], str]] = None,
    config: Optional[AgentConfig] = None,
) -> Dict[str, Any]:
    """
    Punto de entrada principal del sistema Multi-Agent ReAct del módulo de orquestación.

    Args:
        question: pregunta del usuario en lenguaje natural.
        retriever: función retrieve_context(question, top_k) real del Data & Retrieval Layer.
            Si se provee, se registra automaticamente y se desactiva el modo
            fallback del retriever para esta llamada.
        llm: función generate_llm_response(prompt) -> str real (local_llm,
            gemini_llm, o cualquier función propia). Si se provee, se usa en
            lugar del fallback para esta llamada.
        config: AgentConfig para sobreescribir umbrales/flags. Por defecto
            usa src.config.DEFAULT_CONFIG.

    Returns:
        Diccionario con: question, retrieved_context, draft_answer,
        audit_passed, audit_feedback, missing_info, final_answer, iterations,
        trace, evidence_score, hallucination_risk, confidence_level,
        warnings, system_type.
    """
    cfg = config or DEFAULT_CONFIG
    if retriever is not None:
        register_retriever(retriever)
        cfg = cfg.with_overrides(use_fallback_retriever=False)
    if llm is not None:
        cfg = cfg.with_overrides(use_fallback_llm=False)

    graph_key = f"{cfg.use_fallback_retriever}-{cfg.use_fallback_llm}-{cfg.max_iterations}-{id(llm)}"
    if graph_key not in _COMPILED_GRAPH_CACHE:
        _COMPILED_GRAPH_CACHE[graph_key] = build_agent_graph(config=cfg, llm_fn=llm)
    compiled_graph = _COMPILED_GRAPH_CACHE[graph_key]

    initial_state = _build_initial_state(question, cfg.system_type)

    try:
        # recursion_limit generoso: cada ciclo researcher->auditor consume 2
        # pasos internos de LangGraph; MAX_ITERATIONS ciclos + margen de writer.
        recursion_limit = (cfg.max_iterations + 2) * 4
        final_state = compiled_graph.invoke(
            initial_state, config={"recursion_limit": recursion_limit}
        )
    except Exception as exc:
        return {
            **initial_state,
            "final_answer": f"Error ejecutando el grafo multi-agente: {exc}",
            "warnings": [f"pipeline_error: {exc}"],
        }

    return dict(final_state)
