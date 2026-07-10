# Reporte de evaluacion — Naive RAG vs Multi-Agent ReAct

50 preguntas de WikiQA (`50_preguntas_wikiqa.csv`). Metricas propias calculadas por `evaluation/metrics.py` (sin costo adicional de API).

## Tabla resumen

| system_type       |   n_questions |   n_errors |   avg_latency_seconds |   avg_iterations |   avg_evidence_score |   avg_hallucination_risk |   audit_pass_rate |   avg_answer_length_chars |   avg_core_answer_length_chars |   avg_num_contexts |   avg_token_f1_vs_expected |   avg_token_f1_core_vs_expected |
|:------------------|--------------:|-----------:|----------------------:|-----------------:|---------------------:|-------------------------:|------------------:|--------------------------:|-------------------------------:|-------------------:|---------------------------:|--------------------------------:|
| naive_rag         |            50 |          0 |                  3.17 |                1 |              nan     |                  nan     |               nan |                     187.2 |                          187.2 |               5    |                     0.4443 |                          0.4443 |
| multi_agent_react |            50 |          0 |                  6.32 |                1 |                0.956 |                    0.044 |                 1 |                    4818.9 |                          413.6 |               4.14 |                     0.1064 |                          0.4311 |

## Graficos

![Latencia](latency_comparison.png)

![Calidad](quality_metrics_comparison.png)

![Distribucion de latencia](latency_distribution.png)

## Conclusiones (borrador — completar con analisis cualitativo)

- **Latencia**: Multi-Agent ReAct tardo en promedio 6.32s/pregunta vs 3.17s/pregunta de Naive RAG (+99%).
- **Correctness (token-F1 vs expected_answer, solo texto redactado)**: Naive RAG = 0.4443, Multi-Agent ReAct = 0.4311 (-3%). [Nota: la version cruda que incluye el bloque "Evidencia usada" del Writer da 0.4443 vs 0.1064 — mucho mas baja para Multi-Agent porque ese bloque vuelca el contexto crudo completo, no texto redactado por el modelo; por eso se usa la version 'core' para comparar.]
- **Guardrail propio**: el auditor aprobo el 100% de los borradores en la primera(s) iteracion(es) (promedio 1.0 iteraciones/pregunta); hallucination_risk promedio = 0.044.
- **Errores**: Naive RAG tuvo 0 fallos, Multi-Agent ReAct 0 (de 50 preguntas cada uno).

> TODO (manual): agregar 2-3 ejemplos cualitativos de `naive_rag_per_question.csv` / `multi_agent_react_per_question.csv` donde el guardrail del auditor evito una alucinacion evidente, y 1-2 casos donde el costo extra de latencia del multi-agente no se tradujo en mejor respuesta.

> Si se corrio `evaluation/metrics.py --with-ragas`, agregar aqui la comparacion de `outputs/evaluation_ready/ragas_metrics.csv` (faithfulness, answer_relevancy, context_precision).