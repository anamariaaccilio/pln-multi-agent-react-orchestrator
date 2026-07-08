"""
Metricas de comparacion Naive RAG vs Multi-Agent ReAct.

Dos familias de metricas:

1. Metricas propias, gratis (no gastan mas llamadas al LLM): calculadas a
   partir de lo que ya guardo evaluation/run_batch.py en los .jsonl
   (latencia, iteraciones, evidence_score, hallucination_risk, audit_passed,
   longitud de respuesta, overlap lexico contra expected_answer de WikiQA).

2. RAGAS (opcional, --with-ragas): faithfulness / answer_relevancy /
   context_precision usando Gemini como juez. Consume llamadas adicionales
   a la API, por eso queda detras de un flag explicito.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

OUTPUT_DIR = ROOT_DIR / "outputs" / "evaluation_ready"

_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "to", "is", "was", "were", "are",
    "and", "or", "for", "with", "by", "at", "as", "that", "this", "it",
    "de", "la", "el", "en", "y", "a", "los", "las", "un", "una", "que",
}


def _tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-zA-ZáéíóúñÁÉÍÓÚÑ0-9]+", str(text).lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


def token_f1(prediction: str, reference: str) -> float:
    """F1 de solapamiento de tokens entre la respuesta y expected_answer.

    Proxy barato de 'correctness' cuando hay ground truth (como WikiQA),
    sin gastar llamadas a LLM. No reemplaza a un juicio semantico real
    (RAGAS / LLM-as-a-Judge), pero es reproducible y gratis.
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_set, ref_set = set(pred_tokens), set(ref_tokens)
    overlap = pred_set & ref_set
    if not overlap:
        return 0.0
    precision = len(overlap) / len(pred_set)
    recall = len(overlap) / len(ref_set)
    return round(2 * precision * recall / (precision + recall), 4)


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"No existe {path}. Corre evaluation/run_batch.py primero.")
    records = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    return pd.DataFrame(records)


def extract_core_answer(text: str) -> str:
    """Extrae solo la respuesta real del formato del Writer Agent, sin el
    bloque de 'Evidencia usada' (que vuelca el contexto crudo completo).

    El Writer devuelve:
        Respuesta final:
        <respuesta>

        Evidencia usada:
        - <chunk de contexto completo, ~1200 chars c/u>
        ...

    Comparar token-F1 contra ese bloque completo penaliza injustamente a
    Multi-Agent ReAct: infla el denominador con texto que ni siquiera
    redacto el modelo (es el contexto recuperado, repetido). naive_rag no
    tiene este problema porque su 'answer' ya es solo la respuesta.
    """
    text = text or ""
    if text.startswith("Respuesta final:"):
        core = text[len("Respuesta final:"):]
        for marker in ("\n\nEvidencia usada:", "\nEvidencia usada:"):
            idx = core.find(marker)
            if idx != -1:
                core = core[:idx]
                break
        return core.strip()
    return text.strip()


def compute_dataframe_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["answer_length_chars"] = df["answer"].fillna("").str.len()
    df["answer_core"] = df["answer"].fillna("").apply(extract_core_answer)
    df["core_answer_length_chars"] = df["answer_core"].str.len()
    df["num_contexts"] = df["contexts"].apply(lambda c: len(c) if isinstance(c, list) else 0)
    # Metrica "cruda" (incluye el bloque de evidencia si lo hay) y la
    # "justa" (solo el texto que redacto el modelo). Se reportan ambas para
    # que quede claro el efecto del formato del Writer, pero la "core" es
    # la que corresponde usar para comparar correctness entre sistemas.
    df["token_f1_vs_expected"] = df.apply(
        lambda r: token_f1(r.get("answer", ""), r.get("expected_answer", "")), axis=1
    )
    df["token_f1_core_vs_expected"] = df.apply(
        lambda r: token_f1(r["answer_core"], r.get("expected_answer", "")), axis=1
    )
    return df


def summarize(df: pd.DataFrame, system_name: str) -> Dict[str, float]:
    n = len(df)
    n_errors = int(df["error"].sum()) if "error" in df.columns else 0
    ok = df[df["error"] == False] if "error" in df.columns else df  # noqa: E712

    summary = {
        "system_type": system_name,
        "n_questions": n,
        "n_errors": n_errors,
        "avg_latency_seconds": round(ok["latency_seconds"].mean(), 2) if "latency_seconds" in ok else None,
        "avg_iterations": round(ok["iterations"].mean(), 2) if "iterations" in ok else None,
        "avg_evidence_score": round(ok["evidence_score"].dropna().mean(), 3) if "evidence_score" in ok and ok["evidence_score"].notna().any() else None,
        "avg_hallucination_risk": round(ok["hallucination_risk"].dropna().mean(), 3) if "hallucination_risk" in ok and ok["hallucination_risk"].notna().any() else None,
        "audit_pass_rate": round(ok["audit_passed"].dropna().mean(), 3) if "audit_passed" in ok and ok["audit_passed"].notna().any() else None,
        "avg_answer_length_chars": round(ok["answer_length_chars"].mean(), 1),
        "avg_core_answer_length_chars": round(ok["core_answer_length_chars"].mean(), 1) if "core_answer_length_chars" in ok else None,
        "avg_num_contexts": round(ok["num_contexts"].mean(), 2),
        # "vs_expected" (cruda) queda para transparencia/debug: incluye el
        # bloque de evidencia del Writer y penaliza injustamente a los
        # sistemas que lo agregan. "core_vs_expected" es la comparable.
        "avg_token_f1_vs_expected": round(ok["token_f1_vs_expected"].mean(), 4),
        "avg_token_f1_core_vs_expected": round(ok["token_f1_core_vs_expected"].mean(), 4) if "token_f1_core_vs_expected" in ok else None,
    }
    return summary


GIVEUP_PHRASES = ("does not contain", "no contiene", "not contain enough", "insufficient", "no hay evidencia")


def paired_analysis(naive_df: pd.DataFrame, multi_df: pd.DataFrame) -> Dict[str, object]:
    """
    Compara Token-F1 (core) pregunta-por-pregunta entre ambos sistemas.

    Un promedio agregado (ver summarize()) puede esconder que la diferencia
    no es significativa, o que hay subgrupos con comportamientos opuestos
    que se cancelan entre si. Esta funcion responde eso con un diseño
    pareado (misma pregunta, dos sistemas):

    - Test de significancia: t de Student pareada + Wilcoxon signed-rank
      (no parametrico, mas apropiado si la distribucion no es normal).
    - Conteo win/loss/tie por pregunta.
    - Desglose "Naive se rinde" (usa lenguaje de evidencia insuficiente,
      ver GIVEUP_PHRASES) vs "Naive intenta responder": el patron esperado
      es que Multi-Agent (con acceso a web_search) rinda mejor en el primer
      subgrupo, y que la diferencia se diluya o invierta en el segundo.
    """
    from scipy import stats

    merged = naive_df.merge(multi_df, on="question", suffixes=("_naive", "_multi"))
    n = len(merged)
    if n < 2:
        return {"n": n, "note": "Muy pocas preguntas en comun para un test pareado."}

    naive_scores = merged["token_f1_core_vs_expected_naive"]
    multi_scores = merged["token_f1_core_vs_expected_multi"]
    diff = multi_scores - naive_scores

    t_stat, t_p = stats.ttest_rel(multi_scores, naive_scores)
    try:
        w_stat, w_p = stats.wilcoxon(multi_scores, naive_scores)
    except ValueError:
        w_stat, w_p = None, None

    result = {
        "n": n,
        "naive_mean": round(naive_scores.mean(), 4),
        "multi_mean": round(multi_scores.mean(), 4),
        "mean_diff_multi_minus_naive": round(diff.mean(), 4),
        "paired_ttest_t": round(t_stat, 4),
        "paired_ttest_p": round(t_p, 4),
        "wilcoxon_stat": round(w_stat, 4) if w_stat is not None else None,
        "wilcoxon_p": round(w_p, 4) if w_p is not None else None,
        "n_multi_better": int((diff > 0.01).sum()),
        "n_naive_better": int((diff < -0.01).sum()),
        "n_tied": int((diff.abs() <= 0.01).sum()),
    }

    gives_up = naive_df["answer"].fillna("").str.lower().apply(
        lambda a: any(p in a for p in GIVEUP_PHRASES)
    )
    giveup_questions = set(naive_df.loc[gives_up, "question"])
    is_giveup = merged["question"].isin(giveup_questions)

    result["n_naive_gives_up"] = int(is_giveup.sum())
    result["giveup_subset_naive_mean"] = round(naive_scores[is_giveup].mean(), 4) if is_giveup.any() else None
    result["giveup_subset_multi_mean"] = round(multi_scores[is_giveup].mean(), 4) if is_giveup.any() else None
    result["non_giveup_subset_naive_mean"] = round(naive_scores[~is_giveup].mean(), 4) if (~is_giveup).any() else None
    result["non_giveup_subset_multi_mean"] = round(multi_scores[~is_giveup].mean(), 4) if (~is_giveup).any() else None

    return result


def run_ragas(naive_df: pd.DataFrame, multi_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluacion opcional con RAGAS (faithfulness, answer_relevancy,
    context_precision) usando Gemini como LLM juez. Gasta llamadas API
    adicionales: 3 metricas x 2 sistemas x N preguntas.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from datasets import Dataset
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    except ImportError as exc:
        raise SystemExit(
            "Faltan dependencias para --with-ragas. Instala: "
            "pip install ragas langchain-google-genai\n"
            f"Detalle: {exc}"
        )

    judge_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.0)
    judge_embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    results = []
    for name, df in [("naive_rag", naive_df), ("multi_agent_react", multi_df)]:
        ok = df[df["error"] == False] if "error" in df.columns else df  # noqa: E712
        dataset = Dataset.from_pandas(
            ok[["question", "answer", "contexts"]].reset_index(drop=True)
        )
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=judge_llm,
            embeddings=judge_embeddings,
        )
        row = result.to_pandas().mean(numeric_only=True).to_dict()
        row["system_type"] = name
        results.append(row)

    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-ragas", action="store_true", help="Tambien correr RAGAS (gasta llamadas API extra).")
    args = parser.parse_args()

    naive_df = compute_dataframe_metrics(load_jsonl(OUTPUT_DIR / "naive_rag.jsonl"))
    multi_df = compute_dataframe_metrics(load_jsonl(OUTPUT_DIR / "multi_agent_react.jsonl"))

    naive_summary = summarize(naive_df, "naive_rag")
    multi_summary = summarize(multi_df, "multi_agent_react")

    comparison = pd.DataFrame([naive_summary, multi_summary])
    comparison_path = OUTPUT_DIR / "comparison_metrics.csv"
    comparison.to_csv(comparison_path, index=False)
    print(comparison.to_string(index=False))
    print(f"\nGuardado en: {comparison_path}")

    # Tambien guardar los dataframes por-pregunta con las metricas calculadas,
    # utiles para el reporte y para inspeccionar casos individuales.
    naive_df.to_csv(OUTPUT_DIR / "naive_rag_per_question.csv", index=False)
    multi_df.to_csv(OUTPUT_DIR / "multi_agent_react_per_question.csv", index=False)

    significance = paired_analysis(naive_df, multi_df)
    significance_path = OUTPUT_DIR / "significance_analysis.json"
    with open(significance_path, "w", encoding="utf-8") as f:
        json.dump(significance, f, ensure_ascii=False, indent=2)
    print("\n=== Analisis pareado (Token-F1 core, misma pregunta en ambos sistemas) ===")
    for k, v in significance.items():
        print(f"  {k}: {v}")
    print(f"\nGuardado en: {significance_path}")

    if args.with_ragas:
        ragas_df = run_ragas(naive_df, multi_df)
        ragas_path = OUTPUT_DIR / "ragas_metrics.csv"
        ragas_df.to_csv(ragas_path, index=False)
        print("\n=== RAGAS ===")
        print(ragas_df.to_string(index=False))
        print(f"Guardado en: {ragas_path}")


if __name__ == "__main__":
    main()
