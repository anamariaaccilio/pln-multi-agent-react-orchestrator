"""Prompts y constructores de prompt para cada agente (ReAct, corto y auditable)."""
from __future__ import annotations

from typing import List

from src.agents.state import EvidenceChunk
from src.utils.formatting import format_context_block

RESEARCHER_PROMPT = (
    "Eres un agente investigador. Tu tarea es buscar evidencia relevante para "
    "responder la pregunta del usuario. Usa unicamente el contexto recuperado. "
    "No inventes informacion. Si el contexto es insuficiente, indica que falta."
)

AUDITOR_PROMPT = (
    "Eres un agente auditor de hechos. Tu tarea es verificar si la respuesta "
    "propuesta esta respaldada por el contexto recuperado. Rechaza cualquier "
    "afirmacion que no pueda sustentarse en la evidencia."
)

WRITER_PROMPT = (
    "Eres un agente redactor. Tu tarea es producir la respuesta final para el "
    "usuario usando unicamente la evidencia aprobada. Indica el nivel de "
    "confianza y cualquier advertencia relevante."
)


def build_researcher_prompt(
    question: str,
    context: List[EvidenceChunk],
    prior_feedback: str = "",
) -> str:
    """Construye el prompt del Researcher Agent. El marcador [TASK=...] es leido por fallback_llm."""
    feedback_block = (
        f"\nFEEDBACK PREVIO DEL AUDITOR (corrige en base a esto):\n{prior_feedback}\n"
        if prior_feedback
        else ""
    )
    return (
        f"[TASK=RESEARCHER]\n{RESEARCHER_PROMPT}\n\n"
        f"PREGUNTA: {question}\n\n"
        f"CONTEXTO:\n{format_context_block(context)}\n"
        f"{feedback_block}\n"
        "RESPUESTA:"
    )


def build_writer_prompt(question: str, draft_answer: str, context: List[EvidenceChunk]) -> str:
    """Construye el prompt del Redactor (no usado por el fallback, pensado para LLM real)."""
    return (
        f"[TASK=WRITER]\n{WRITER_PROMPT}\n\n"
        f"PREGUNTA: {question}\n\n"
        f"BORRADOR APROBADO:\n{draft_answer}\n\n"
        f"CONTEXTO:\n{format_context_block(context)}\n\n"
        "RESPUESTA FINAL:"
    )
