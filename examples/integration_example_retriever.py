"""Demuestra como conectar un retriever real del Data & Retrieval Layer sin
tocar ningun archivo de src/agents/.

Ejecutar directamente desde la raíz del repo:
    python examples/integration_example_retriever.py

Este ejemplo simula un retriever "real" (por ejemplo, uno respaldado por
FAISS/ChromaDB sobre WikiQA) con una función local que respeta el contrato
exacto documentado en docs/integration_with_retriever.md:

    def retrieve_context(question: str, top_k: int = 5) -> list[dict]:
        return [{"content": "...", "source": "...", "score": 0.85}, ...]

Al reemplazar el retriever, ni researcher.py, ni auditor.py, ni writer.py
cambian: solo se pasa `retriever=...` a multi_agent_rag().
"""
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.multi_agent_rag import multi_agent_rag
from src.utils.trace import print_trace


def retrieve_context(question: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Simula el retriever real del Data & Retrieval Layer (WikiQA + FAISS/ChromaDB)."""
    corpus = [
        {
            "content": (
                "Los guardrails en un sistema RAG son verificaciones automaticas que "
                "rechazan una respuesta si no esta suficientemente sustentada en el "
                "contexto recuperado, reduciendo el riesgo de alucinaciones."
            ),
            "source": "corpus_real_doc_001",
            "score": 0.88,
        },
        {
            "content": (
                "Un guardrail tipico calcula un evidence_score o hallucination_risk "
                "y fuerza un ciclo de correccion cuando el umbral no se cumple."
            ),
            "source": "corpus_real_doc_002",
            "score": 0.81,
        },
    ]
    return corpus[:top_k]


if __name__ == "__main__":
    question = "Como ayudan los guardrails a reducir alucinaciones en un sistema RAG?"

    result = multi_agent_rag(question, retriever=retrieve_context)

    print("=" * 70)
    print("INTEGRACION CON RETRIEVER REAL (simulado)")
    print("=" * 70)
    print(result["final_answer"])
    print(f"\naudit_passed={result['audit_passed']}  evidence_score={result['evidence_score']}")

    print("\n--- Traza ReAct ---")
    print_trace(result)
