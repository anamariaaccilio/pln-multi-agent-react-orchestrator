"""
Corre las 50 preguntas de evaluacion (50_preguntas_wikiqa.csv) contra
Naive RAG y/o Multi-Agent ReAct, y guarda los resultados en
outputs/evaluation_ready/{naive_rag,multi_agent_react}.jsonl

Resumible: si el proceso se corta a mitad de camino (rate limit, error de
red, etc.), volver a correrlo sigue donde quedo en vez de repetir preguntas
ya resueltas.

Requiere GOOGLE_API_KEY (o GOOGLE_API_KEYS, ver abajo) en un archivo .env en
la raiz del repo (ver .env.example) y que config/settings.yaml tenga
provider: "gemini" y retriever.use_fallback: false (ya es el default del
repo).

Uso:
    python -m evaluation.run_batch --system both
    python -m evaluation.run_batch --system naive --n 5      # prueba rapida
    python -m evaluation.run_batch --system multi --sleep 3  # mas lento, mas seguro con rate limits

Rotacion automatica de multiples API keys (recomendado si el free tier de
Gemini se queda corto: 20 requests/dia/key). En .env:

    GOOGLE_API_KEYS=key1,key2,key3,key4

El script prueba la key activa en cada llamada al LLM; si esa key se quedo
sin cuota (429), rota automaticamente a la siguiente SIN rehacer la
pregunta desde cero (solo reintenta esa llamada puntual). El progreso queda
grabado en outputs/evaluation_ready/.key_rotation_state.json para no volver
a probar keys ya agotadas en la proxima corrida. Si se agotan TODAS las
keys, el batch se detiene solo (no sigue reintentando en loop) y deja todo
lo ya resuelto guardado; correr el mismo comando de nuevo mas tarde (con
mas keys agregadas a GOOGLE_API_KEYS, o al otro dia cuando resetee la
cuota) continua exactamente donde quedo.

Alternativa/complemento — repartir rangos entre varias personas/maquinas,
cada una con su propia key, escribiendo a un archivo con sufijo propio para
no pisarse, y mergeando despues con evaluation/merge_results.py:

    python -m evaluation.run_batch --system naive --start 0 --end 17 --suffix ana
    python -m evaluation.run_batch --system naive --start 17 --end 34 --suffix beto
    python -m evaluation.merge_results --system naive
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Set

# Algunos prints de otras capas (p.ej. retrieval_pipeline.py) usan caracteres
# unicode (✓) que la consola de Windows (cp1252) no puede codificar. Forzamos
# UTF-8 en stdout/stderr para que el batch no muera por un simple print().
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import functools

import pandas as pd
from tqdm import tqdm

from evaluation.key_rotation import KeyRotator
from src.pipeline.eval_format import convert_to_eval_format
from src.pipeline.multi_agent_rag import multi_agent_rag
from src.pipeline.naive_rag import naive_rag
from src.retriever.interface import register_retriever

QUESTIONS_CSV = ROOT_DIR / "50_preguntas_wikiqa.csv"
OUTPUT_DIR = ROOT_DIR / "outputs" / "evaluation_ready"


def _already_done(path: Path) -> Set[str]:
    """IDs (question text) ya presentes en el jsonl de salida, para resumir."""
    if not path.exists():
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("error"):
                    continue  # se reintentara en la proxima corrida
                done.add(record["question"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _append_record(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_system(
    system_name: str,
    run_fn,
    questions: pd.DataFrame,
    sleep_seconds: float,
    retries: int,
    rotator: KeyRotator,
    out_name: str | None = None,
) -> bool:
    """Devuelve False si se detuvo antes de tiempo (varias preguntas seguidas agotaron todas las keys), True si termino la porcion pedida."""
    out_path = OUTPUT_DIR / f"{out_name or system_name}.jsonl"
    done = _already_done(out_path)

    pending = questions[~questions["question"].isin(done)]
    if pending.empty:
        print(f"[{system_name}] Ya estan las {len(questions)} preguntas resueltas en {out_path.name}. Nada que hacer.")
        return True

    print(f"[{system_name}] {len(done)} ya resueltas, {len(pending)} pendientes.")

    # Si varias preguntas SEGUIDAS agotan las 7 keys incluso despues de
    # esperar y resetear la rotacion, es una señal fuerte de que es la cuota
    # DIARIA (no la de por-minuto, que se recupera sola en ~60s) la que esta
    # agotada, y no tiene sentido seguir moliendo. En ese caso frenamos todo
    # el batch. Una pregunta resuelta con exito resetea este contador.
    consecutive_full_exhaustions = 0
    MAX_CONSECUTIVE_FULL_EXHAUSTIONS = 3

    for _, row in tqdm(pending.iterrows(), total=len(pending), desc=system_name):
        question = row["question"]
        expected_answer = row.get("expected_answer", "")
        qid = row.get("id", "")

        attempt = 0
        result = None
        elapsed = 0.0
        while attempt <= retries:
            start = time.perf_counter()
            try:
                candidate = run_fn(question)
                elapsed = time.perf_counter() - start
                # naive_rag/multi_agent_rag no relanzan errores del LLM (p.ej.
                # un 503 transitorio de Gemini, o AllKeysExhausted del
                # rotador): los devuelven como texto dentro de final_answer,
                # envueltos por su propio try/except interno. Sin este
                # chequeo, esas respuestas invalidas quedarian guardadas
                # como si fueran correctas.
                answer_text = candidate.get("final_answer", "")
                if answer_text.startswith("Error generating response") or answer_text.startswith(
                    "Error ejecutando el grafo multi-agente"
                ):
                    raise RuntimeError(answer_text)
                result = candidate
                break
            except Exception as exc:  # rate limit puntual, timeout, 503, keys agotadas, etc.
                elapsed = time.perf_counter() - start
                attempt += 1
                msg = str(exc)
                all_keys_exhausted = "se quedaron sin cuota" in msg  # texto de AllKeysExhausted
                is_rate_limit = all_keys_exhausted or "429" in msg or "RESOURCE_EXHAUSTED" in msg

                if all_keys_exhausted:
                    consecutive_full_exhaustions += 1
                    print(f"\n[{system_name}] Las {len(rotator.keys)} keys se quedaron sin cuota (intento "
                          f"{consecutive_full_exhaustions}/{MAX_CONSECUTIVE_FULL_EXHAUSTIONS} seguido).")
                    if consecutive_full_exhaustions >= MAX_CONSECUTIVE_FULL_EXHAUSTIONS:
                        print(f"[{system_name}] Parece cuota DIARIA agotada (no se recupera esperando). "
                              f"Deteniendo el batch. Ya guardado: {len(done)} preguntas.")
                        _dedupe_jsonl(out_path)
                        return False
                    wait = 65  # ventana tipica de reset de un limite por-minuto
                    print(f"[{system_name}] Esperando {wait}s y reseteando a la key #1 por si es limite por-minuto...")
                    time.sleep(wait)
                    rotator.reset()
                    continue

                wait = min(90, 20 * attempt) if is_rate_limit else min(30, 2 ** attempt)
                print(f"\n[{system_name}] ERROR en '{question[:60]}...' (intento {attempt}/{retries}): {exc}")
                if attempt <= retries:
                    print(f"[{system_name}] Reintentando en {wait}s...")
                    time.sleep(wait)

        if result is not None:
            consecutive_full_exhaustions = 0

        if result is None:
            # Se agotaron los reintentos: registrar el fallo y seguir con la siguiente pregunta.
            record = {
                "id": qid,
                "question": question,
                "answer": "",
                "contexts": [],
                "system_type": system_name,
                "audit_passed": None,
                "evidence_score": None,
                "hallucination_risk": None,
                "expected_answer": expected_answer,
                "latency_seconds": round(elapsed, 2),
                "iterations": None,
                "error": True,
            }
        else:
            record = convert_to_eval_format(result)
            record["id"] = qid
            record["expected_answer"] = expected_answer
            record["latency_seconds"] = round(elapsed, 2)
            record["iterations"] = result.get("iterations", 1)
            record["error"] = False

        _append_record(out_path, record)

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    _dedupe_jsonl(out_path)
    return True


def _dedupe_jsonl(path: Path) -> None:
    """Colapsa lineas repetidas por pregunta (de reintentos en corridas
    distintas), prefiriendo el registro exitoso mas reciente sobre errores
    viejos, para que evaluation/metrics.py no cuente la misma pregunta 2 veces.
    """
    if not path.exists():
        return
    by_question: Dict[str, Dict[str, Any]] = {}
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
            if existing is None or existing.get("error") and not record.get("error"):
                by_question[question] = record

    with open(path, "w", encoding="utf-8") as f:
        for record in by_question.values():
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=["naive", "multi", "both"], default="both")
    parser.add_argument("--n", type=int, default=None, help="Solo correr las primeras N preguntas (pruebas rapidas).")
    parser.add_argument("--start", type=int, default=None, help="Fila inicial (0-indexed, inclusive) del CSV a correr.")
    parser.add_argument("--end", type=int, default=None, help="Fila final (0-indexed, exclusive) del CSV a correr.")
    parser.add_argument(
        "--suffix", type=str, default=None,
        help="Sufijo para el archivo de salida (ej. --suffix ana -> naive_rag_ana.jsonl). "
             "Util para que cada persona del equipo corra su porcion sin pisar el archivo de otros.",
    )
    parser.add_argument("--sleep", type=float, default=1.0, help="Segundos de espera entre preguntas (evita rate limits).")
    parser.add_argument("--retries", type=int, default=2, help="Reintentos por pregunta ante error/rate limit.")
    parser.add_argument(
        "--reset-keys", action="store_true",
        help="Volver a empezar la rotacion desde la primera key de GOOGLE_API_KEYS "
             "(usar cuando sabemos que las cuotas ya resetearon, ej. al otro dia).",
    )
    args = parser.parse_args()

    if not QUESTIONS_CSV.exists():
        raise SystemExit(f"No se encontro {QUESTIONS_CSV}. Corre preparar_50_preguntas.py primero.")

    # Registra el retriever real (WikiQA + ChromaDB) del Data & Retrieval Layer.
    # config/settings.yaml tiene retriever.use_fallback=false, asi que sin esto
    # naive_rag()/multi_agent_rag() fallan con "no se registro ningun retriever real".
    from src.retriever.retrieval_pipeline import retrieve_context

    register_retriever(retrieve_context)

    rotator = KeyRotator()
    if args.reset_keys:
        rotator.reset()

    run_fns = {
        "naive_rag": functools.partial(naive_rag, llm=rotator.generate),
        "multi_agent_react": functools.partial(multi_agent_rag, llm=rotator.generate),
    }

    questions = pd.read_csv(QUESTIONS_CSV)

    if args.start is not None or args.end is not None:
        start = args.start or 0
        end = args.end if args.end is not None else len(questions)
        questions = questions.iloc[start:end]
        print(f"Rango seleccionado: filas [{start}:{end}) -> {len(questions)} preguntas.")

    if args.n:
        questions = questions.head(args.n)

    systems_to_run = {
        "naive": ["naive_rag"],
        "multi": ["multi_agent_react"],
        "both": ["naive_rag", "multi_agent_react"],
    }[args.system]

    for system_name in systems_to_run:
        out_name = f"{system_name}_{args.suffix}" if args.suffix else system_name
        completed = run_system(
            system_name, run_fns[system_name], questions, args.sleep, args.retries, rotator, out_name=out_name
        )
        if not completed:
            print(
                "\nSe agotaron todas las keys disponibles antes de terminar. "
                "Agrega mas keys a GOOGLE_API_KEYS en .env y volve a correr el mismo comando "
                "(retoma automaticamente donde quedo)."
            )
            break

    print("\nListo. Resultados en:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
