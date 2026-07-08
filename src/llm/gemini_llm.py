"""Backend LLM via API: Gemini o Nemotron (OpenRouter).

Proveedores soportados:
  - "gemini": usa google-generativeai (requiere GOOGLE_API_KEY)
  - "nemotron": usa OpenRouter con modelo Nemotron (requiere OPENROUTER_API_KEY)

Requiere:
    pip install google-generativeai openai python-dotenv
    Variables de entorno en .env:
      GOOGLE_API_KEY=...
      OPENROUTER_API_KEY=...
      NEMOTRON_CHAT_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
"""
from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()


def generate_llm_response(prompt: str, **kwargs) -> str:
    """
    Genera una respuesta usando el provider configurado.

    Selecciona Nemotron (OpenRouter) si OPENROUTER_API_KEY está disponible,
    o Gemini si GOOGLE_API_KEY está disponible. Prioriza OpenRouter para
    evitar consumir quota de Gemini.
    """
    if os.environ.get("OPENROUTER_API_KEY"):
        return _generate_nemotron(prompt, **kwargs)
    elif os.environ.get("GOOGLE_API_KEY"):
        return _generate_gemini(prompt, **kwargs)
    else:
        raise RuntimeError(
            "No hay API key disponible. Configura OPENROUTER_API_KEY o "
            "GOOGLE_API_KEY en el archivo .env"
        )


def _generate_nemotron(prompt: str, **kwargs) -> str:
    """Genera una respuesta via OpenRouter con Nemotron."""
    from openai import OpenAI

    model = os.environ.get("NEMOTRON_CHAT_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    import time
    max_retries = 2

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.2),
                max_tokens=kwargs.get("max_tokens", 1024),
            )

            # Safety: handle empty/None responses from the API
            if not response.choices:
                print(f"[NEMOTRON] WARNING: No choices returned (attempt {attempt + 1}/{max_retries + 1})")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s
                    continue
                # Final attempt failed — try Gemini as fallback
                if os.environ.get("GOOGLE_API_KEY"):
                    print("[NEMOTRON] Falling back to Gemini...")
                    return _generate_gemini(prompt, **kwargs)
                return ""

            content = response.choices[0].message.content
            if content is None:
                print(f"[NEMOTRON] WARNING: content is None. Finish reason: {response.choices[0].finish_reason}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                if os.environ.get("GOOGLE_API_KEY"):
                    print("[NEMOTRON] Falling back to Gemini...")
                    return _generate_gemini(prompt, **kwargs)
                return ""

            return content.strip()

        except Exception as e:
            print(f"[NEMOTRON] ERROR (attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            # Final attempt failed — try Gemini as fallback
            if os.environ.get("GOOGLE_API_KEY"):
                print("[NEMOTRON] Falling back to Gemini...")
                return _generate_gemini(prompt, **kwargs)
            return ""

    return ""


def _generate_gemini(prompt: str, **kwargs) -> str:
    """Genera una respuesta con Gemini."""
    from google import genai

    # Default alineado con config/settings.yaml (llm.model_name). No hay wiring
    # actual entre AgentConfig.llm_model_name y esta funcion (los agentes llaman
    # generate(prompt) sin pasar model), asi que el default de aca es el que
    # realmente rige. gemini-2.5-flash (sin "-lite") tiene una cuota gratuita
    # mucho mas restrictiva y se agota rapido en un batch de 50 preguntas.
    model = kwargs.get("model", "gemini-2.5-flash-lite")
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    return response.text.strip()
