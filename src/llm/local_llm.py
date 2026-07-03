"""Interfaz para un LLM local cuantizado (Mistral-7B-Instruct, Llama-3-8B-Instruct, ...).

No se optimiza el modelo aqui: se deja una interfaz clara (generate_llm_response)
para que pueda sustituirse por cualquier modelo de Hugging Face cargado en 4 bits.

Requisitos (Colab con GPU T4 o superior):
    pip install transformers accelerate bitsandbytes torch
"""
from __future__ import annotations

_MODEL = None
_TOKENIZER = None


def _lazy_load(model_name: str):
    """Carga perezosa del modelo cuantizado (solo se ejecuta la primera vez)."""
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return _MODEL, _TOKENIZER

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    _MODEL = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quant_config, device_map="auto"
    )
    return _MODEL, _TOKENIZER


def generate_llm_response(
    prompt: str,
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    max_new_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    """
    Genera una respuesta con un LLM local cuantizado en 4 bits.

    Para usar Llama-3-8B-Instruct en vez de Mistral-7B-Instruct, basta con
    cambiar model_name a "meta-llama/Meta-Llama-3-8B-Instruct" (requiere
    aceptar la licencia en Hugging Face y hacer login con un token valido).
    """
    model, tokenizer = _lazy_load(model_name)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
    )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text[len(prompt):].strip() if text.startswith(prompt) else text.strip()
