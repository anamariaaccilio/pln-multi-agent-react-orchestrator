"""Backend LLM opcional via Gemini API, para desarrollo rápido sin GPU.

El enunciado pide priorizar inferencia local cuantizada (ver local_llm.py);
Gemini queda como proveedor opcional, no como dependencia obligatoria del
módulo de orquestación.

Requiere:
    pip install google-generativeai
    export GOOGLE_API_KEY=...
"""
from __future__ import annotations

import os


def generate_llm_response(prompt: str, model: str = "gemini-2.5-flash", **kwargs) -> str:
    """Genera una respuesta con Gemini. Requiere la variable de entorno GOOGLE_API_KEY."""
    # import google.generativeai as genai

    # genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    # gemini_model = genai.GenerativeModel(model)
    # response = gemini_model.generate_content(prompt)
    # return response.text.strip()

    from google import genai

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        **kwargs,
    )

    return response.text.strip()