"""Utilidades de trazabilidad ReAct (Thought -> Action -> Observation)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def add_trace(
    trace: List[Dict[str, Any]],
    agent: str,
    thought: str,
    action: str,
    observation: str,
) -> List[Dict[str, Any]]:
    """
    Agrega un evento de traza y devuelve una NUEVA lista (no muta la original).

    El thought debe ser breve y operativo (una frase auditable), no una
    cadena de razonamiento extensa: solo sirve para justificar la decision
    del agente, no para que el agente "piense en voz alta".
    """
    event: Dict[str, Any] = {
        "agent": agent,
        "thought": thought,
        "action": action,
        "observation": observation,
    }
    return [*trace, event]


def print_trace(result: Dict[str, Any]) -> None:
    """Imprime la traza ReAct de un resultado de multi_agent_rag() de forma legible."""
    trace = result.get("trace", [])
    for i, event in enumerate(trace, start=1):
        print(f"[{i}] ({event['agent']}) THOUGHT: {event['thought']}")
        print(f"     ACTION: {event['action']}")
        print(f"     OBSERVATION: {event['observation']}")


def save_trace(result: Dict[str, Any], path: str) -> None:
    """Guarda la traza y metadatos minimos del resultado en un archivo JSON."""
    payload = {
        "question": result.get("question"),
        "iterations": result.get("iterations"),
        "audit_passed": result.get("audit_passed"),
        "trace": result.get("trace", []),
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
