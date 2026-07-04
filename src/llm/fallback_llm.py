"""LLM de respaldo (fallback), sin dependencias externas, para desarrollar y
probar el grafo sin GPU ni API keys.

No interpreta lenguaje natural: aplica reglas simples sobre el prompt
(formato producido por src/agents/prompts.py) para producir respuestas
deterministas y reproducibles. Es suficiente para ejercitar el grafo
LangGraph, el enrutamiento condicional, los guardrails del Fact Auditor y
las trazas ReAct.

Para generación real, usar src/llm/local_llm.py (recomendado por el
enunciado: Mistral-7B-Instruct / Llama-3-8B-Instruct cuantizado) o
src/llm/gemini_llm.py (opcional, para desarrollo rápido).
"""
from __future__ import annotations

import json
import re


def fallback_generate_llm_response(prompt: str, **kwargs) -> str:
    """Punto de entrada del backend fallback: fallback_generate_llm_response(prompt) -> str."""

    # --- Handle ReAct planning prompts (tool selection phase) ---
    if "HERRAMIENTAS DISPONIBLES:" in prompt and "THOUGHT y ACTION:" in prompt:
        return _handle_react_plan(prompt)

    # --- Handle synthesis prompts (after tool execution) ---
    if "OBSERVACIONES DE HERRAMIENTAS:" in prompt:
        return _handle_react_synthesize(prompt)

    # --- Legacy: original researcher/writer prompt format ---
    question_match = re.search(r"PREGUNTA:\s*(.*)", prompt)
    context_match = re.search(r"CONTEXTO:\n(.*?)\n\n", prompt, re.DOTALL)

    question = question_match.group(1).strip() if question_match else ""
    context_block = context_match.group(1).strip() if context_match else ""

    if not context_block or context_block == "(sin contexto recuperado)":
        return "No encuentro evidencia suficiente en el contexto para responder esta pregunta."

    sentences = re.split(r"(?<=[.])\s+", context_block)
    grounded_facts = [s for s in sentences if len(s) > 15][:2]
    answer = " ".join(grounded_facts) if grounded_facts else context_block[:200]

    # Simula una alucinacion controlada cuando el escenario es de baja evidencia,
    # para poder demostrar el rechazo del Fact Auditor de forma reproducible.
    if "baja_evidencia" in question.lower():
        answer += (
            " Ademas, esta estructura fue disenada originalmente para ser "
            "desmontada en 1909, un dato que no aparece en el contexto anterior."
        )

    return answer.strip()


def _handle_react_plan(prompt: str) -> str:
    """
    Fallback for the ReAct planning phase.

    Always calls knowledge_base_search. If the question contains hints that
    web search would help, also calls web_search.
    """
    question_match = re.search(r"PREGUNTA:\s*(.*)", prompt)
    question = question_match.group(1).strip() if question_match else "unknown"

    # Default: always search the knowledge base
    actions = [
        f'THOUGHT: Necesito buscar informacion sobre la pregunta en la base de conocimiento.\n'
        f'ACTION: {{"tool_name": "knowledge_base_search", "args": {{"question": "{question}", "top_k": 5}}}}'
    ]

    # If auditor feedback mentions needing more info, also search the web
    if "FEEDBACK PREVIO DEL AUDITOR" in prompt:
        actions.append(
            f'THOUGHT: El auditor pidio mas informacion, buscare en la web tambien.\n'
            f'ACTION: {{"tool_name": "web_search", "args": {{"query": "{question}", "max_results": 3}}}}'
        )

    return "\n\n".join(actions)


def _handle_react_synthesize(prompt: str) -> str:
    """
    Fallback for the synthesis phase: extract key info from tool observations.
    """
    question_match = re.search(r"PREGUNTA:\s*(.*)", prompt)
    obs_match = re.search(r"OBSERVACIONES DE HERRAMIENTAS:\n(.*?)\n\nRESPUESTA:", prompt, re.DOTALL)

    question = question_match.group(1).strip() if question_match else ""
    observations = obs_match.group(1).strip() if obs_match else ""

    if not observations or "No se encontraron" in observations:
        return "No encuentro evidencia suficiente en las herramientas para responder esta pregunta."

    # Extract content from observations (take first meaningful sentences)
    sentences = re.split(r"(?<=[.])\s+", observations)
    # Filter out metadata lines (score=, source=, URL:, ---)
    content_sentences = [
        s for s in sentences
        if len(s) > 15
        and not s.startswith("[")
        and not s.startswith("---")
        and "score=" not in s
        and "URL:" not in s
    ]

    grounded_facts = content_sentences[:3]
    answer = " ".join(grounded_facts) if grounded_facts else observations[:300]

    # Simulate controlled hallucination for the "baja_evidencia" scenario
    if "baja_evidencia" in question.lower():
        answer += (
            " Ademas, esta estructura fue disenada originalmente para ser "
            "desmontada en 1909, un dato que no aparece en el contexto anterior."
        )

    return answer.strip()
