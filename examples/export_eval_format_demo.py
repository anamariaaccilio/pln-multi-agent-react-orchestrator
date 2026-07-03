"""Demuestra la exportacion del resultado de multi_agent_rag() al formato
plano que consume el Evaluation Layer (RAGAS / LLM-as-a-Judge).

Ejecutar directamente desde la raíz del repo:
    python examples/export_eval_format_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.eval_format import append_eval_jsonl, convert_to_eval_format
from src.pipeline.multi_agent_rag import multi_agent_rag

QUESTIONS = [
    "Cuando y por quien fue construida la Torre Eiffel?",
    "baja_evidencia: para que fue disenada originalmente esta estructura?",
]

OUTPUT_PATH = "outputs/evaluation_ready/multi_agent_react_demo.jsonl"


def main() -> None:
    for question in QUESTIONS:
        result = multi_agent_rag(question)
        eval_item = convert_to_eval_format(result)

        print("=" * 70)
        print("question:", eval_item["question"])
        print("system_type:", eval_item["system_type"])
        print("audit_passed:", eval_item["audit_passed"])
        print("evidence_score:", eval_item["evidence_score"])
        print("hallucination_risk:", eval_item["hallucination_risk"])
        print("contexts:", len(eval_item["contexts"]), "fragmentos")

        append_eval_jsonl(result, OUTPUT_PATH)

    print(f"\nRegistros exportados en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
