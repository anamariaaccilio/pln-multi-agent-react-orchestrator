"""Demo end-to-end del módulo de orquestación en modo fallback (sin dependencias externas).

Ejecutar directamente desde la raíz del repo:
    python examples/run_multi_agent_demo.py

Muestra el diagrama del grafo y los dos casos obligatorios:
  - Un ejemplo APROBADO (buen contexto, el Fact Auditor aprueba en la primera pasada).
  - Un ejemplo RECHAZADO (contexto pobre, ciclo de corrección y advertencia final).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph.build_graph import build_agent_graph
from src.graph.visualize import visualize_graph
from src.pipeline.eval_format import convert_to_eval_format
from src.pipeline.multi_agent_rag import multi_agent_rag
from src.utils.trace import print_trace, save_trace


def run_approved_example() -> None:
    question = "Cuando y por quien fue construida la Torre Eiffel?"
    result = multi_agent_rag(question)

    print("=" * 70)
    print("EJEMPLO APROBADO")
    print("=" * 70)
    print(result["final_answer"])
    print(f"\naudit_passed={result['audit_passed']}  iterations={result['iterations']}")
    print(f"evidence_score={result['evidence_score']}  hallucination_risk={result['hallucination_risk']}")

    print("\n--- Traza ReAct ---")
    print_trace(result)
    save_trace(result, "outputs/traces/trace_approved.json")

    print("\n--- Formato de evaluacion ---")
    print(convert_to_eval_format(result))


def run_rejected_example() -> None:
    # La palabra clave 'baja_evidencia' activa el escenario pobre del retriever fallback.
    question = "baja_evidencia: para que fue disenada originalmente esta estructura?"
    result = multi_agent_rag(question)

    print("\n" + "=" * 70)
    print("EJEMPLO RECHAZADO / CORREGIDO CON ADVERTENCIA")
    print("=" * 70)
    print(result["final_answer"])
    print(f"\naudit_passed={result['audit_passed']}  iterations={result['iterations']}")
    print(f"evidence_score={result['evidence_score']}  hallucination_risk={result['hallucination_risk']}")
    print(f"warnings={result['warnings']}")

    print("\n--- Traza ReAct ---")
    print_trace(result)
    save_trace(result, "outputs/traces/trace_rejected.json")

    print("\n--- Formato de evaluacion ---")
    print(convert_to_eval_format(result))


def main() -> None:
    graph = build_agent_graph()
    print("Diagrama Mermaid del grafo:\n")
    print(visualize_graph(graph))
    print()

    run_approved_example()
    run_rejected_example()


if __name__ == "__main__":
    main()
