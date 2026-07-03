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

import re


def fallback_generate_llm_response(prompt: str, **kwargs) -> str:
    """Punto de entrada del backend fallback: fallback_generate_llm_response(prompt) -> str."""
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
