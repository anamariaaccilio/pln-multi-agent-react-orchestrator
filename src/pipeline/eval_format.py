"""Conversión del resultado de multi_agent_rag() al formato esperado por el módulo de evaluación."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def convert_to_eval_format(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convierte el resultado de multi_agent_rag() al formato plano que el módulo de evaluación
    usa para evaluar con RAGAS o LLM-as-a-Judge (ver docs/integration_evaluation.md).
    """
    contexts = [c["content"] for c in result.get("retrieved_context", [])]
    return {
        "question": result["question"],
        "answer": result["final_answer"],
        "contexts": contexts,
        "system_type": result.get("system_type", "multi_agent_react"),
        "audit_passed": result.get("audit_passed", False),
        "evidence_score": result.get("evidence_score", 0.0),
        # None (no 1.0) cuando el sistema no calcula esta metrica (p.ej. naive_rag,
        # que no tiene auditor): 1.0 significaria "riesgo maximo", lo cual es
        # enganoso para un sistema que simplemente no mide esto.
        "hallucination_risk": result.get("hallucination_risk"),
    }


def save_eval_record(result: Dict[str, Any], path: str) -> None:
    """Guarda un registro individual en outputs/evaluation_ready/ como JSON."""
    record = convert_to_eval_format(result)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def append_eval_jsonl(result: Dict[str, Any], path: str) -> None:
    """Agrega un registro a un archivo .jsonl acumulado (util para lotes de preguntas)."""
    record = convert_to_eval_format(result)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
