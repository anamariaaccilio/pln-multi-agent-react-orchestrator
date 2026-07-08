"""
Rotacion de multiples API keys de Gemini para esquivar el limite diario del
free tier (20 requests/dia/modelo/proyecto). Cada persona del equipo crea su
propia key (cuenta personal de Google, sin tarjeta); este modulo las prueba
en orden y salta a la siguiente automaticamente cuando la actual se queda
sin cuota (429 / RESOURCE_EXHAUSTED).

Rotacion a nivel de LLAMADA individual (no de pregunta completa): si
multi_agent_rag hace 3 llamadas al LLM para una pregunta y la 2da choca con
el limite, solo esa llamada reintenta con la siguiente key — no se rehace
toda la pregunta desde cero, así no se desperdicia cuota ya gastada.

El indice de la key activa se persiste en un archivo de estado
(outputs/evaluation_ready/.key_rotation_state.json), asi que si el proceso
se corta y se vuelve a correr, no vuelve a probar desde la key #1 (que ya
sabemos que esta agotada) sino que sigue desde la ultima que funcionaba.

Configuracion en .env (agregar junto a GOOGLE_API_KEY):
    GOOGLE_API_KEYS=key_de_persona_a,key_de_persona_b,key_de_persona_c

Si GOOGLE_API_KEYS no esta seteado, se usa GOOGLE_API_KEY (una sola key,
comportamiento normal sin rotacion).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT_DIR / "outputs" / "evaluation_ready" / ".key_rotation_state.json"


class AllKeysExhausted(RuntimeError):
    """Se probaron todas las keys disponibles y ninguna tiene cuota."""


def _load_keys() -> List[str]:
    from dotenv import load_dotenv

    load_dotenv()
    multi = os.environ.get("GOOGLE_API_KEYS", "").strip()
    if multi:
        keys = [k.strip() for k in multi.split(",") if k.strip()]
        if keys:
            return keys
    single = os.environ.get("GOOGLE_API_KEY", "").strip()
    if single:
        return [single]
    raise RuntimeError(
        "No hay ninguna key configurada. Agrega GOOGLE_API_KEYS (varias, separadas "
        "por coma) o GOOGLE_API_KEY (una sola) en .env."
    )


class KeyRotator:
    def __init__(self, keys: Optional[List[str]] = None, state_path: Path = STATE_PATH):
        self.keys = keys or _load_keys()
        self.state_path = state_path
        self.index = self._load_state()
        print(f"[key_rotation] {len(self.keys)} key(s) disponibles. Empezando en key #{self.index + 1}.")

    def _load_state(self) -> int:
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
                idx = int(data.get("index", 0))
                return max(0, min(idx, len(self.keys) - 1))
            except Exception:
                return 0
        return 0

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps({"index": self.index, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}),
            encoding="utf-8",
        )

    def reset(self) -> None:
        """Volver a empezar desde la key #1 (usar cuando sabemos que las cuotas ya resetearon, ej. al otro dia)."""
        self.index = 0
        self._save_state()

    def _rotate(self) -> bool:
        """Avanza a la siguiente key. Devuelve False si ya dimos toda la vuelta."""
        next_index = self.index + 1
        if next_index >= len(self.keys):
            return False
        self.index = next_index
        self._save_state()
        print(f"[key_rotation] Cuota agotada. Rotando a key #{self.index + 1}/{len(self.keys)}.")
        return True

    def generate(self, prompt: str, **kwargs) -> str:
        """Compatible con el contrato generate_llm_response(prompt) -> str."""
        from src.llm.gemini_llm import _generate_gemini

        tried = 0
        last_exc: Optional[Exception] = None
        # Arranca en la key activa actual; si falla por cuota, prueba las
        # siguientes (sin repetir las que ya sabemos agotadas en esta vuelta).
        while tried < len(self.keys):
            os.environ["GOOGLE_API_KEY"] = self.keys[self.index]
            try:
                return _generate_gemini(prompt, **kwargs)
            except Exception as exc:
                msg = str(exc)
                is_quota = "429" in msg or "RESOURCE_EXHAUSTED" in msg
                if not is_quota:
                    raise
                last_exc = exc
                tried += 1
                if not self._rotate():
                    break

        raise AllKeysExhausted(
            f"Las {len(self.keys)} key(s) disponibles se quedaron sin cuota. "
            f"Ultimo error: {last_exc}"
        )


__all__ = ["KeyRotator", "AllKeysExhausted"]
