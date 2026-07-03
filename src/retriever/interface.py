"""
Contrato oficial del retriever, compartido con el Data & Retrieval Layer.

El Data & Retrieval Layer debe entregar una función con esta firma exacta:

    def retrieve_context(question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        ...

que consulte su base vectorial local (FAISS o ChromaDB) sobre WikiQA /
Financial-QA y devuelva una lista de fragmentos con este formato:

    [
        {"content": "...", "source": "...", "score": 0.85},
        ...
    ]

Mientras el Data & Retrieval Layer no entregue su implementación final, el
módulo de orquestación usa `src.retriever.fallback_retriever.fallback_retrieve_context`
como sustituto, controlado por `USE_FALLBACK_RETRIEVER` en `src/config.py`.

Para integrar el retriever real, ver docs/integration_with_retriever.md.
"""
from __future__ import annotations

from typing import Callable, List, Optional, TypedDict


class RetrievedChunk(TypedDict):
    content: str
    source: str
    score: float


RetrieverFn = Callable[..., List[RetrievedChunk]]

_real_retriever: Optional[RetrieverFn] = None


def register_retriever(fn: RetrieverFn) -> None:
    """Inyecta el retriever real del Data & Retrieval Layer (retrieve_context(question, top_k))."""
    global _real_retriever
    _real_retriever = fn


def get_registered_retriever() -> Optional[RetrieverFn]:
    """Devuelve el retriever real registrado, o None si aun no se registro ninguno."""
    return _real_retriever


def resolve_retriever(use_fallback: bool) -> RetrieverFn:
    """Selecciona la función retrieve_context activa según configuración."""
    from src.retriever.fallback_retriever import fallback_retrieve_context

    if use_fallback:
        return fallback_retrieve_context

    if _real_retriever is None:
        raise RuntimeError(
            "USE_FALLBACK_RETRIEVER=False pero no se registro ningun retriever real. "
            "Llama a register_retriever(retrieve_context) con la función del "
            "Data & Retrieval Layer antes de invocar multi_agent_rag(). "
            "Ver docs/integration_with_retriever.md."
        )
    return _real_retriever
