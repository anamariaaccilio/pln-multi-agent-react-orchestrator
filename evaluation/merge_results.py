"""
Combina los archivos .jsonl de varias personas (cada una corrio una porcion
de las 50 preguntas con su propia API key, via --suffix en run_batch.py) en
el archivo canonico que usan evaluation/metrics.py y evaluation/report.py.

Ejemplo: si Ana, Beto y Caro corrieron cada uno un tercio de las preguntas
de naive_rag con --suffix ana / beto / caro, quedan:
    outputs/evaluation_ready/naive_rag_ana.jsonl
    outputs/evaluation_ready/naive_rag_beto.jsonl
    outputs/evaluation_ready/naive_rag_caro.jsonl

Este script los junta en:
    outputs/evaluation_ready/naive_rag.jsonl

Antes de correrlo, cada persona debe mandar (git, drive, lo que sea) su
archivo *_<suffix>.jsonl para que queden todos juntos en esta carpeta.

Uso:
    python -m evaluation.merge_results --system naive
    python -m evaluation.merge_results --system multi
    python -m evaluation.merge_results --system both
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

OUTPUT_DIR = ROOT_DIR / "outputs" / "evaluation_ready"


def merge_system(system_name: str) -> None:
    canonical_path = OUTPUT_DIR / f"{system_name}.jsonl"
    part_paths = sorted(OUTPUT_DIR.glob(f"{system_name}_*.jsonl"))

    if not part_paths:
        print(f"[{system_name}] No se encontraron archivos {system_name}_<suffix>.jsonl para mergear.")
        return

    print(f"[{system_name}] Mergeando {len(part_paths)} archivo(s): {[p.name for p in part_paths]}")

    by_question: Dict[str, Dict[str, Any]] = {}

    # Si ya existe un archivo canonico previo, lo incluimos tambien como base.
    all_paths = ([canonical_path] if canonical_path.exists() else []) + part_paths

    for path in all_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                question = record.get("question")
                if not question:
                    continue
                existing = by_question.get(question)
                # Preferir un registro exitoso sobre uno con error, sin importar de que archivo venga.
                if existing is None or (existing.get("error") and not record.get("error")):
                    by_question[question] = record

    with open(canonical_path, "w", encoding="utf-8") as f:
        for record in by_question.values():
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    n_errors = sum(1 for r in by_question.values() if r.get("error"))
    print(f"[{system_name}] {len(by_question)} preguntas unicas en {canonical_path.name} ({n_errors} con error).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=["naive", "multi", "both"], default="both")
    args = parser.parse_args()

    systems = {
        "naive": ["naive_rag"],
        "multi": ["multi_agent_react"],
        "both": ["naive_rag", "multi_agent_react"],
    }[args.system]

    for system_name in systems:
        merge_system(system_name)


if __name__ == "__main__":
    main()
