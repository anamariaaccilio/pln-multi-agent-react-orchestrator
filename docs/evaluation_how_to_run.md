# Evaluation Layer — cómo correr la comparación Naive RAG vs Multi-Agent ReAct

Esto corre las 50 preguntas de `50_preguntas_wikiqa.csv` contra ambos
sistemas, calcula métricas y genera tablas/gráficos/conclusiones.

## 0. Requisitos

1. `.env` en la raíz con `GOOGLE_API_KEY` (ver `.env.example` y
   [`docs/how_to_run.md`](how_to_run.md)).
2. `config/settings.yaml` ya viene configurado con `provider: "gemini"` y
   retriever real (`use_fallback: false`) — no hay que tocar nada.
3. `pip install -r requirements.txt`

## 0.1 Si el free tier de una sola key no alcanza (20 requests/día)

El free tier de Gemini es muy chico para este batch (~250 llamadas entre
ambos sistemas). Si varias personas del equipo crean su propia key
(cuenta personal de Google, sin tarjeta), `run_batch.py` puede **rotar
automáticamente** entre todas ellas: cuando la key activa se queda sin
cuota (429), salta a la siguiente sin rehacer la pregunta desde cero (solo
reintenta esa llamada puntual al LLM).

En `.env`:
```
GOOGLE_API_KEYS=key_persona_a,key_persona_b,key_persona_c
```

Si se agotan **todas** las keys, el batch se detiene solo (no reintenta en
loop infinito) y deja guardado todo lo ya resuelto. Correr el mismo
comando más tarde (con más keys agregadas, o al otro día cuando resetee la
cuota) retoma exactamente donde quedó — no hace falta ningún flag especial.
Si sabés que la cuota ya reseteó y querés volver a empezar desde la
primera key: `--reset-keys`.

## 1. Prueba rápida (5 preguntas, valida que todo funciona)

```bash
python -m evaluation.run_batch --system both --n 5 --sleep 1
```

Revisa `outputs/evaluation_ready/naive_rag.jsonl` y
`outputs/evaluation_ready/multi_agent_react.jsonl`.

## 2. Correr las 50 preguntas completas

```bash
python -m evaluation.run_batch --system both --sleep 1
```

- Es **resumible**: si se corta por rate limit o error de red, correr el
  mismo comando de nuevo continúa donde quedó (no repite preguntas ya
  guardadas en el `.jsonl`).
- `--sleep` controla la pausa entre preguntas; súbelo (`--sleep 3`) si ves
  errores 429 (rate limit) de Gemini.
- Tiempo esperado: Naive RAG ~10-15 min, Multi-Agent ReAct ~60-100 min
  (varias llamadas al LLM por pregunta + posible búsqueda web).

## 3. Calcular métricas

```bash
python -m evaluation.metrics
```

Genera:
- `outputs/evaluation_ready/comparison_metrics.csv` — tabla resumen (1 fila
  por sistema).
- `outputs/evaluation_ready/{naive_rag,multi_agent_react}_per_question.csv`
  — detalle por pregunta.

Métricas incluidas (sin costo adicional de API): latencia, iteraciones,
`evidence_score` / `hallucination_risk` / `audit_passed` (guardrail propio
del multi-agente), longitud de respuesta, número de chunks recuperados, y
**token-F1 contra `expected_answer`** (ground truth de WikiQA — mide
correctness aproximado gratis).

Opcional, con costo adicional de API (faithfulness / answer_relevancy /
context_precision vía RAGAS + Gemini como juez):

```bash
pip install ragas langchain-google-genai
python -m evaluation.metrics --with-ragas
```

## 4. Generar tablas, gráficos y borrador de conclusiones

```bash
python -m evaluation.report
```

Genera en `outputs/evaluation_ready/report/`:
- `latency_comparison.png`
- `quality_metrics_comparison.png`
- `latency_distribution.png`
- `report.md` (tabla + gráficos embebidos + conclusiones con TODOs para
  completar el análisis cualitativo)
