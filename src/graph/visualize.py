"""Visualizacion del grafo: diagrama Mermaid + intento de export PNG nativo de LangGraph."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

MERMAID_DIAGRAM = """graph TD
    A[User Question] --> B[Researcher Agent]
    B -->|tool_calls| C[Tool Node]
    C -->|observations| B
    B -->|draft_answer| D[Fact Auditor Agent]
    D -->|Approved| E[Writer Agent]
    D -->|Rejected and iterations available| B
    D -->|Max iterations reached| E
    E --> F[Final Answer]

    C -.->|knowledge_base_search| G[(ChromaDB / WikiQA)]
    C -.->|web_search| H((DuckDuckGo))
"""


def visualize_graph(compiled_graph=None, output_dir: str = "outputs/graph") -> str:
    """
    Guarda el diagrama Mermaid en outputs/graph/graph_diagram.mmd y devuelve su
    texto. Si se pasa el grafo compilado de LangGraph, intenta ademas exportar
    un PNG nativo (requiere dependencias extra tipo pygraphviz/mermaid-cli;
    si fallan, se ignora silenciosamente porque no es critico para el pipeline).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mermaid_path = out_dir / "graph_diagram.mmd"
    mermaid_path.write_text(MERMAID_DIAGRAM, encoding="utf-8")

    if compiled_graph is not None:
        try:
            png_bytes = compiled_graph.get_graph().draw_mermaid_png()
            (out_dir / "graph_diagram.png").write_bytes(png_bytes)
        except Exception:
            pass

    return MERMAID_DIAGRAM


def print_graph_ascii(compiled_graph) -> None:
    """Imprime una representacion ASCII del grafo si el metodo esta disponible."""
    try:
        compiled_graph.get_graph().print_ascii()
    except Exception as exc:
        print("No se pudo generar ASCII del grafo:", exc)
        print(MERMAID_DIAGRAM)
