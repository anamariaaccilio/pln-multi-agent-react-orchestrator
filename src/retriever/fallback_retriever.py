"""Retriever de respaldo (fallback) para desarrollar y probar el módulo de
orquestación sin depender del Data & Retrieval Layer.

Simula dos escenarios deterministas para poder demostrar el ciclo completo
del grafo sin conexión a una base vectorial real:

  - Escenario por defecto: contexto bueno y relevante -> el Fact Auditor aprueba.
  - Escenario "baja_evidencia": contexto pobre -> el Fact Auditor rechaza y
    dispara el ciclo de corrección (ver examples/run_multi_agent_demo.py).
"""
from __future__ import annotations

from typing import Any, Dict, List

_FALLBACK_KNOWLEDGE_BASE: Dict[str, List[Dict[str, Any]]] = {
    "default": [
        {
            "content": (
                "La Torre Eiffel fue construida entre 1887 y 1889 por el ingeniero "
                "Gustave Eiffel como entrada monumental a la Exposicion Universal de Paris."
            ),
            "source": "wikiqa_doc_0142",
            "score": 0.91,
        },
        {
            "content": (
                "La torre mide 330 metros de altura incluyendo antenas y fue la "
                "estructura mas alta del mundo hasta 1930."
            ),
            "source": "wikiqa_doc_0143",
            "score": 0.87,
        },
        {
            "content": (
                "Actualmente la Torre Eiffel recibe cerca de 7 millones de visitantes "
                "al ano y es uno de los monumentos pagos mas visitados del mundo."
            ),
            "source": "wikiqa_doc_0144",
            "score": 0.79,
        },
    ],
    "low_evidence": [
        {
            "content": (
                "Existen numerosas estructuras metalicas construidas en Europa "
                "durante el siglo XIX con fines expositivos."
            ),
            "source": "wikiqa_doc_9001",
            "score": 0.42,
        },
    ],
}


def fallback_retrieve_context(question: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Simula retrieve_context(question, top_k) con contenido determinista.

    Si la pregunta contiene la palabra clave 'baja_evidencia', devuelve
    fragmentos deliberadamente pobres para poder demostrar el rechazo del
    Fact Auditor y el ciclo de corrección de forma reproducible.
    """
    key = "low_evidence" if "baja_evidencia" in question.lower() else "default"
    chunks = _FALLBACK_KNOWLEDGE_BASE[key]
    return chunks[:top_k] if top_k else list(chunks)
