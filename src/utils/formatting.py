"""Utilidades de formato de texto compartidas entre agentes."""
from __future__ import annotations

from typing import List

from src.agents.state import EvidenceChunk


def format_context_block(context: List[EvidenceChunk]) -> str:
    """Convierte la lista de fragmentos recuperados en un bloque de texto para el prompt."""
    if not context:
        return "(sin contexto recuperado)"
    lines = []
    for i, chunk in enumerate(context, start=1):
        lines.append(
            f"[{i}] (score={chunk['score']:.2f}, fuente={chunk['source']}) {chunk['content']}"
        )
    return "\n".join(lines)
