"""Agente Investigador: decide qué herramientas usar, luego sintetiza una respuesta.

Flow in the graph:
  researcher (plan) -> tool_node -> researcher (synthesize) -> auditor

The researcher operates in two steps:
  1. "plan": Decides which tools to call (knowledge base, web search, or both).
  2. "synthesize": After tool_node executes the tools, uses the observations
     to draft an answer.
"""
from __future__ import annotations

from typing import Any, Dict

from src.agents.prompts import RESEARCHER_PROMPT, build_researcher_react_prompt, build_researcher_synthesize_prompt
from src.agents.state import AgentState
from src.config import AgentConfig, DEFAULT_CONFIG
from src.graph.tool_node import parse_tool_calls
from src.graph.tools import format_tools_description
from src.llm.interface import resolve_llm
from src.utils.trace import add_trace


def researcher_node(state: AgentState, agent_config=None, llm_fn=None) -> Dict[str, Any]:
    """
    Nodo LangGraph del Agente Investigador (ReAct con herramientas).

    Opera en dos fases controladas por state["researcher_step"]:
      - "plan" (default): genera THOUGHT + ACTION para invocar herramientas.
      - "synthesize": usa las observaciones de las herramientas para redactar
        la respuesta preliminar.

    Nota: el parámetro se llama `agent_config` (no `config`) a propósito.
    LangGraph inyecta automáticamente su propio RunnableConfig en cualquier
    parámetro de nodo literalmente llamado `config`.
    """
    cfg = agent_config or DEFAULT_CONFIG
    question = state["question"]
    iterations = state.get("iterations", 0)
    trace = list(state.get("trace", []))
    step = state.get("researcher_step", "plan")

    print(f"\n[RESEARCHER] Step: {step} | Iteration: {iterations} | Question: {question[:80]}")

    generate = llm_fn or resolve_llm(cfg.use_fallback_llm, cfg.llm_provider)

    if step == "synthesize":
        return _synthesize_step(state, cfg, question, iterations, trace, generate)
    else:
        return _plan_step(state, cfg, question, iterations, trace, generate)


def _plan_step(
    state: AgentState,
    cfg: AgentConfig,
    question: str,
    iterations: int,
    trace: list,
    generate,
) -> Dict[str, Any]:
    """
    Phase 1: The researcher decides which tools to call.

    Generates a ReAct-style THOUGHT + ACTION output. The tool calls are
    parsed and stored in state for tool_node to execute.

    If the auditor already rejected a previous attempt (iterations > 0),
    web_search is always included alongside knowledge_base_search to
    ensure the researcher escalates its search strategy.
    """
    prior_feedback = state.get("audit_feedback", "") if iterations > 0 else ""
    tools_desc = format_tools_description()

    prompt = build_researcher_react_prompt(question, tools_desc, prior_feedback)
    llm_output = generate(prompt)

    # Handle empty LLM response gracefully
    if not llm_output:
        print("[RESEARCHER] WARNING: LLM returned empty response, defaulting to KB search")
        llm_output = ""

    # Parse tool calls from the LLM output
    tool_calls = parse_tool_calls(llm_output)

    print(f"[RESEARCHER] LLM raw output:\n{llm_output[:300]}")
    print(f"[RESEARCHER] Parsed tool_calls: {[c['tool_name'] for c in tool_calls]}")

    # If the LLM didn't produce valid tool calls, default to knowledge base search
    if not tool_calls:
        tool_calls = [{"tool_name": "knowledge_base_search", "args": {"question": question, "top_k": cfg.top_k}}]

    # If auditor rejected a previous attempt, ensure web_search is included
    # so the researcher doesn't just repeat the same KB-only strategy
    if prior_feedback:
        tool_names = [c["tool_name"] for c in tool_calls]
        if "web_search" not in tool_names:
            tool_calls.append({
                "tool_name": "web_search",
                "args": {"query": question, "max_results": 3},
            })
        print(f"[RESEARCHER] Prior feedback detected, ensuring web_search is included")

    print(f"[RESEARCHER] Final tool_calls: {[c['tool_name'] for c in tool_calls]}\n")

    trace = add_trace(
        trace,
        agent="researcher",
        thought="Necesito decidir qué herramientas usar para responder la pregunta.",
        action=f"Herramientas seleccionadas: {[c['tool_name'] for c in tool_calls]}",
        observation=f"Se planificaron {len(tool_calls)} llamadas a herramientas.",
    )

    return {
        "tool_calls": tool_calls,
        "researcher_step": "synthesize",
        "iterations": iterations + 1,
        "trace": trace,
        "system_type": cfg.system_type,
    }


def _synthesize_step(
    state: AgentState,
    cfg: AgentConfig,
    question: str,
    iterations: int,
    trace: list,
    generate,
) -> Dict[str, Any]:
    """
    Phase 2: After tools have been executed, synthesize a draft answer
    from the observations.
    """
    tool_results = state.get("tool_results", [])

    # Build context from tool observations
    observations_text = ""
    evidence_list = []

    for result in tool_results:
        tool_name = result.get("tool_name", "")
        observation = result.get("observation", "")
        observations_text += f"\n--- {tool_name} ---\n{observation}\n"

        # Collect all observations as evidence
        if observation and "No se encontraron" not in observation and "ERROR" not in observation:
            evidence_list.append(observation)

    # Build retrieved_context from ALL tool results (KB + web)
    # so the auditor evaluates against everything the researcher used
    retrieved_context = state.get("retrieved_context", [])

    # Also add web search results as context chunks for the auditor
    for result in tool_results:
        if result.get("tool_name") == "web_search":
            observation = result.get("observation", "")
            if observation and "No se encontraron" not in observation and "ERROR" not in observation:
                # Parse web results into chunks for the auditor
                for block in observation.split("\n\n"):
                    block = block.strip()
                    if block and not block.startswith("URL:"):
                        retrieved_context.append({
                            "content": block,
                            "source": "web_search",
                            "score": 0.7,  # Default score for web results
                        })

    prompt = build_researcher_synthesize_prompt(question, observations_text)
    draft_answer = generate(prompt)

    # Handle empty LLM response
    if not draft_answer:
        print("[RESEARCHER] WARNING: LLM returned empty synthesis, using fallback")
        draft_answer = "Insufficient evidence to provide an answer based on the available context."

    print(f"[RESEARCHER] Synthesized draft_answer ({len(draft_answer)} chars):")
    print(f"  {draft_answer[:200]}...\n")

    max_score = max((c.get("score", 0.0) for c in retrieved_context), default=0.0)
    limitations = (
        "Contexto limitado o de baja relevancia; la respuesta puede ser incompleta."
        if not retrieved_context or max_score < 0.5
        else ""
    )

    trace = add_trace(
        trace,
        agent="researcher",
        thought="Sintetizando respuesta a partir de las observaciones de las herramientas.",
        action="generate_llm_response(researcher_synthesize_prompt)",
        observation=f"Borrador generado ({len(draft_answer)} caracteres).",
    )

    return {
        "retrieved_context": retrieved_context,
        "draft_answer": draft_answer,
        "evidence_list": evidence_list or [c.get("content", "") for c in retrieved_context],
        "limitations": limitations,
        "researcher_step": "plan",  # Reset for next iteration if auditor rejects
        "tool_calls": [],
        "tool_results": [],
        "trace": trace,
        "system_type": cfg.system_type,
    }


__all__ = ["researcher_node", "RESEARCHER_PROMPT"]
