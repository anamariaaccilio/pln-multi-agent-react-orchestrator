"""
Contrato oficial del LLM usado por el módulo de orquestación.

Cualquier backend (fallback, local cuantizado, Gemini, u otro) debe exponer
una función con esta firma:

    def generate_llm_response(prompt: str, **kwargs) -> str:
        ...

El modo principal de arquitectura está preparado para un LLM local cuantizado
(Mistral-7B-Instruct o Llama-3-8B-Instruct en 4 bits, ver src/llm/local_llm.py),
tal como pide el enunciado. Gemini (src/llm/gemini_llm.py) queda como
proveedor opcional para desarrollo rápido sin GPU, y el fallback
(src/llm/fallback_llm.py) es solo para pruebas locales sin dependencias
externas ni costo.

El módulo de orquestación nunca depende de un único proveedor: los agentes
llaman a `generate_llm_response`, no a `gemini_llm` ni a `local_llm`
directamente. El proveedor activo se resuelve aquí, en un solo lugar.
"""
from __future__ import annotations

from typing import Callable, Optional

LLMFn = Callable[..., str]

_real_llm: Optional[LLMFn] = None


def register_llm(fn: LLMFn) -> None:
    """Inyecta un LLM real (local, Gemini, u otro) para toda la sesión."""
    global _real_llm
    _real_llm = fn


def get_registered_llm() -> Optional[LLMFn]:
    """Devuelve el LLM real registrado, o None si aun no se registro ninguno."""
    return _real_llm


def resolve_llm(use_fallback: bool, provider: str = "fallback") -> LLMFn:
    """
    Selecciona la función generate_llm_response activa.

    - use_fallback=True             -> siempre usa el fallback determinista.
    - use_fallback=False + provider -> resuelve "local" o "gemini" automaticamente.
    - use_fallback=False + registro manual -> usa register_llm(fn) si se llamo antes.
    """
    from src.llm.fallback_llm import fallback_generate_llm_response

    if use_fallback:
        return fallback_generate_llm_response

    if _real_llm is not None:
        return _real_llm

    if provider == "local":
        from src.llm.local_llm import generate_llm_response as local_generate_llm_response

        return local_generate_llm_response

    if provider == "gemini":
        from src.llm.gemini_llm import generate_llm_response as gemini_generate_llm_response

        return gemini_generate_llm_response

    raise RuntimeError(
        "USE_FALLBACK_LLM=False pero no hay LLM real disponible: no se paso "
        "llm_fn a multi_agent_rag(), no se llamo a register_llm(), y "
        f"provider='{provider}' no es 'local' ni 'gemini'. "
        "Ver docs/how_to_run.md."
    )
