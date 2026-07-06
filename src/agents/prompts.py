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

RESEARCHER_REACT_PROMPT = (
    "Eres un agente investigador con acceso a herramientas. Tu tarea es decidir "
    "que herramientas usar para buscar evidencia relevante y responder la pregunta "
    "del usuario.\n\n"
    "INSTRUCCIONES DE FORMATO (OBLIGATORIO):\n"
    "Responde EXACTAMENTE con lineas THOUGHT y ACTION. Cada ACTION debe ser un JSON "
    "valido en UNA SOLA LINEA con el formato:\n"
    "ACTION: {\"tool_name\": \"<nombre_herramienta>\", \"args\": {\"param\": \"valor\"}}\n\n"
    "Puedes emitir multiples pares THOUGHT/ACTION si necesitas varias herramientas.\n"
    "No escribas nada mas. No expliques. Solo THOUGHT y ACTION.\n\n"
    "EJEMPLO DE RESPUESTA CORRECTA:\n"
    "THOUGHT: Necesito buscar en la base de conocimiento sobre el tema.\n"
    "ACTION: {\"tool_name\": \"knowledge_base_search\", \"args\": {\"question\": \"mi pregunta\", \"top_k\": 5}}\n"
    "THOUGHT: Tambien necesito informacion actualizada de la web.\n"
    "ACTION: {\"tool_name\": \"web_search\", \"args\": {\"query\": \"mi consulta\", \"max_results\": 3}}"
)

RESEARCHER_SYNTHESIZE_PROMPT = (
    "Eres un agente investigador. Tienes los resultados de las herramientas "
    "que usaste. Sintetiza una respuesta clara y precisa basada UNICAMENTE "
    "en las observaciones obtenidas. No inventes informacion. Si la evidencia "
    "es insuficiente, indicalo explicitamente.\n\n"
    "IMPORTANTE: Responde en el MISMO IDIOMA que las observaciones/evidencia. "
    "Si la evidencia esta en ingles, responde en ingles. "
    "Usa frases y vocabulario directamente del contexto para maximizar la "
    "fidelidad. No uses markdown ni formato especial, solo texto plano."
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


def build_researcher_react_prompt(
    question: str,
    tools_description: str,
    prior_feedback: str = "",
) -> str:
    """
    Build the ReAct planning prompt for the researcher (Phase 1: tool selection).

    The researcher sees available tools and decides which to call.
    """
    feedback_block = (
        f"\nFEEDBACK PREVIO DEL AUDITOR (corrige en base a esto):\n{prior_feedback}\n"
        if prior_feedback
        else ""
    )
    return (
        f"[TASK=RESEARCHER]\n{RESEARCHER_REACT_PROMPT}\n\n"
        f"HERRAMIENTAS DISPONIBLES:\n{tools_description}\n\n"
        f"PREGUNTA: {question}\n"
        f"{feedback_block}\n"
        "Responde con THOUGHT y ACTION:"
    )


def build_researcher_synthesize_prompt(
    question: str,
    observations: str,
) -> str:
    """
    Build the synthesis prompt for the researcher (Phase 2: draft answer from observations).
    """
    return (
        f"[TASK=RESEARCHER]\n{RESEARCHER_SYNTHESIZE_PROMPT}\n\n"
        f"PREGUNTA: {question}\n\n"
        f"OBSERVACIONES DE HERRAMIENTAS:\n{observations}\n\n"
        "RESPUESTA:"
    )


def build_researcher_prompt(
    question: str,
    context: List[EvidenceChunk],
    prior_feedback: str = "",
) -> str:
    """Construye el prompt del Researcher Agent (legacy, usado por fallback_llm)."""
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
