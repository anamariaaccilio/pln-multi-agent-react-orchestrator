"""
Genera graficos + tabla resumen + borrador de conclusiones a partir de
outputs/evaluation_ready/comparison_metrics.csv y los *_per_question.csv
producidos por evaluation/metrics.py.

Salida en outputs/evaluation_ready/report/:
    - latency_comparison.png
    - quality_metrics_comparison.png
    - latency_distribution.png
    - report.md   (tablas + conclusiones a completar)

Uso:
    python -m evaluation.report
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

OUTPUT_DIR = ROOT_DIR / "outputs" / "evaluation_ready"
REPORT_DIR = OUTPUT_DIR / "report"

# Paleta categorica fija (colorblind-safe, orden fijo: naive_rag = slot1 azul,
# multi_agent_react = slot8 naranja) — ver skill dataviz / references/palette.md.
COLOR_NAIVE = "#2a78d6"
COLOR_MULTI = "#eb6834"
SYSTEM_ORDER = ["naive_rag", "multi_agent_react"]
SYSTEM_LABELS = {"naive_rag": "Naive RAG", "multi_agent_react": "Multi-Agent ReAct"}
SYSTEM_COLORS = {"naive_rag": COLOR_NAIVE, "multi_agent_react": COLOR_MULTI}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#c9c8c2",
    "axes.grid": True,
    "grid.color": "#e6e5e0",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "font.size": 11,
    "text.color": "#0b0b0b",
    "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
})


def _bar_with_labels(ax, labels, values, colors, fmt="{:.2f}"):
    bars = ax.bar(labels, values, color=colors, width=0.5)
    for bar, value in zip(bars, values):
        if value is None or pd.isna(value):
            continue
        ax.annotate(
            fmt.format(value),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#0b0b0b",
        )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return bars


def plot_latency_comparison(comparison: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ordered = comparison.set_index("system_type").loc[SYSTEM_ORDER]
    labels = [SYSTEM_LABELS[s] for s in SYSTEM_ORDER]
    colors = [SYSTEM_COLORS[s] for s in SYSTEM_ORDER]
    _bar_with_labels(ax, labels, ordered["avg_latency_seconds"], colors, fmt="{:.1f}s")
    ax.set_ylabel("Latencia promedio (segundos / pregunta)")
    ax.set_title("Naive RAG vs Multi-Agent ReAct — Latencia")
    fig.tight_layout()
    out_path = REPORT_DIR / "latency_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_quality_metrics(comparison: pd.DataFrame) -> Path:
    ordered = comparison.set_index("system_type").loc[SYSTEM_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    # Panel 1: token-F1 vs expected_answer, aplica a ambos sistemas. Se usa
    # la version "core" (solo el texto redactado por el modelo, sin el
    # bloque de evidencia que el Writer agrega en Multi-Agent ReAct) porque
    # la version cruda penaliza injustamente a quien incluye mas contexto
    # citado en la respuesta.
    ax = axes[0]
    labels = [SYSTEM_LABELS[s] for s in SYSTEM_ORDER]
    colors = [SYSTEM_COLORS[s] for s in SYSTEM_ORDER]
    _bar_with_labels(ax, labels, ordered["avg_token_f1_core_vs_expected"], colors, fmt="{:.2f}")
    ax.set_title("Token-F1 vs expected_answer\n(solo texto redactado, sin evidencia citada)", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    # Panel 2: audit_pass_rate solo existe para multi_agent_react (no tiene
    # sentido graficarlo junto a naive_rag, que no tiene auditor: dejaria un
    # hueco NaN engañoso). Se muestra como metrica propia de un solo sistema.
    ax = axes[1]
    multi_row = ordered.loc["multi_agent_react"]
    _bar_with_labels(
        ax, [SYSTEM_LABELS["multi_agent_react"]], [multi_row["audit_pass_rate"]],
        [SYSTEM_COLORS["multi_agent_react"]], fmt="{:.2f}",
    )
    ax.set_title("Tasa de aprobacion del auditor\n(solo Multi-Agent ReAct)", fontsize=10)
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    fig.suptitle("Naive RAG vs Multi-Agent ReAct — Calidad", y=1.02)
    fig.tight_layout()
    out_path = REPORT_DIR / "quality_metrics_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_latency_distribution(naive_df: pd.DataFrame, multi_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = [naive_df["latency_seconds"].dropna(), multi_df["latency_seconds"].dropna()]
    bp = ax.boxplot(
        data,
        tick_labels=[SYSTEM_LABELS[s] for s in SYSTEM_ORDER],
        patch_artist=True,
        widths=0.5,
        medianprops={"color": "#0b0b0b"},
    )
    for patch, system in zip(bp["boxes"], SYSTEM_ORDER):
        patch.set_facecolor(SYSTEM_COLORS[system])
        patch.set_alpha(0.75)
    ax.set_ylabel("Latencia por pregunta (segundos)")
    ax.set_title("Distribucion de latencia por pregunta")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out_path = REPORT_DIR / "latency_distribution.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def build_conclusions_draft(comparison: pd.DataFrame) -> str:
    row = comparison.set_index("system_type")
    naive, multi = row.loc["naive_rag"], row.loc["multi_agent_react"]

    def pct_diff(a, b):
        if a in (None, 0) or pd.isna(a) or b is None or pd.isna(b):
            return "N/D"
        return f"{(b - a) / a * 100:+.0f}%"

    lines = [
        "## Conclusiones (borrador — completar con analisis cualitativo)",
        "",
        f"- **Latencia**: Multi-Agent ReAct tardo en promedio {multi['avg_latency_seconds']}s/pregunta "
        f"vs {naive['avg_latency_seconds']}s/pregunta de Naive RAG "
        f"({pct_diff(naive['avg_latency_seconds'], multi['avg_latency_seconds'])}).",
        f"- **Correctness (token-F1 vs expected_answer, solo texto redactado)**: Naive RAG = {naive['avg_token_f1_core_vs_expected']}, "
        f"Multi-Agent ReAct = {multi['avg_token_f1_core_vs_expected']} "
        f"({pct_diff(naive['avg_token_f1_core_vs_expected'], multi['avg_token_f1_core_vs_expected'])}). "
        f"[Nota: la version cruda que incluye el bloque \"Evidencia usada\" del Writer da "
        f"{naive['avg_token_f1_vs_expected']} vs {multi['avg_token_f1_vs_expected']} — mucho mas baja para "
        f"Multi-Agent porque ese bloque vuelca el contexto crudo completo, no texto redactado por el modelo; "
        f"por eso se usa la version 'core' para comparar.]",
        f"- **Guardrail propio**: el auditor aprobo el {multi['audit_pass_rate']:.0%} de los borradores en la primera(s) "
        f"iteracion(es) (promedio {multi['avg_iterations']} iteraciones/pregunta); "
        f"hallucination_risk promedio = {multi['avg_hallucination_risk']}.",
        f"- **Errores**: Naive RAG tuvo {int(naive['n_errors'])} fallos, Multi-Agent ReAct {int(multi['n_errors'])} "
        f"(de {int(naive['n_questions'])} preguntas cada uno).",
        "",
        "> TODO (manual): agregar 2-3 ejemplos cualitativos de "
        "`naive_rag_per_question.csv` / `multi_agent_react_per_question.csv` donde "
        "el guardrail del auditor evito una alucinacion evidente, y 1-2 casos donde "
        "el costo extra de latencia del multi-agente no se tradujo en mejor respuesta.",
        "",
        "> Si se corrio `evaluation/metrics.py --with-ragas`, agregar aqui la "
        "comparacion de `outputs/evaluation_ready/ragas_metrics.csv` "
        "(faithfulness, answer_relevancy, context_precision).",
    ]
    return "\n".join(lines)


def main() -> None:
    comparison_path = OUTPUT_DIR / "comparison_metrics.csv"
    if not comparison_path.exists():
        raise SystemExit("No existe comparison_metrics.csv. Corre 'python -m evaluation.metrics' primero.")

    comparison = pd.read_csv(comparison_path)
    naive_df = pd.read_csv(OUTPUT_DIR / "naive_rag_per_question.csv")
    multi_df = pd.read_csv(OUTPUT_DIR / "multi_agent_react_per_question.csv")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    latency_png = plot_latency_comparison(comparison)
    quality_png = plot_quality_metrics(comparison)
    dist_png = plot_latency_distribution(naive_df, multi_df)

    conclusions = build_conclusions_draft(comparison)

    report_md = [
        "# Reporte de evaluacion — Naive RAG vs Multi-Agent ReAct",
        "",
        f"50 preguntas de WikiQA (`50_preguntas_wikiqa.csv`). Metricas propias "
        f"calculadas por `evaluation/metrics.py` (sin costo adicional de API).",
        "",
        "## Tabla resumen",
        "",
        comparison.to_markdown(index=False),
        "",
        "## Graficos",
        "",
        f"![Latencia]({latency_png.name})",
        "",
        f"![Calidad]({quality_png.name})",
        "",
        f"![Distribucion de latencia]({dist_png.name})",
        "",
        conclusions,
    ]

    report_path = REPORT_DIR / "report.md"
    report_path.write_text("\n".join(report_md), encoding="utf-8")

    print("Graficos guardados en:", REPORT_DIR)
    print("Reporte:", report_path)


if __name__ == "__main__":
    main()
