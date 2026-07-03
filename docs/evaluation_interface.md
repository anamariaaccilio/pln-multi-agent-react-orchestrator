# Guía de integración con el Evaluation Layer

El Evaluation Layer es responsable de evaluar y comparar este sistema
(Multi-Agent ReAct) contra el baseline Naive RAG del Data & Retrieval Layer,
usando RAGAS o LLM-as-a-Judge. El Multi-Agent Orchestration Layer entrega los
resultados en un formato plano y estable para que el Evaluation Layer no
dependa de la estructura interna del `AgentState`.

## 1. Formato de entrega

`src/pipeline/eval_format.py::convert_to_eval_format(result)` toma el
diccionario devuelto por `multi_agent_rag()` y produce:

```python
{
    "question": "...",
    "answer": "...",            # final_answer del Writer Agent
    "contexts": ["...", "..."], # lista de strings, uno por chunk recuperado
    "system_type": "multi_agent_react",
    "audit_passed": true,
    "evidence_score": 0.82,
    "hallucination_risk": 0.18,
}
```

- `question` / `answer` / `contexts`: son las tres columnas mínimas que
  RAGAS espera para métricas como `faithfulness` y `answer_relevancy`
  (agregar `ground_truths` aparte si el Evaluation Layer usa `context_recall`).
- `system_type` permite filtrar/agrupar resultados al comparar contra
  `"naive_rag"` en el mismo dataframe.
- `audit_passed`, `evidence_score`, `hallucination_risk` son las métricas
  propias del guardrail del Multi-Agent Orchestration Layer, útiles para
  correlacionar contra las métricas de RAGAS (por ejemplo, ver si
  `audit_passed=False` correlaciona con `faithfulness` bajo).

## 2. Cómo generar un dataset de evaluación en lote

```python
from src.pipeline.multi_agent_rag import multi_agent_rag
from src.pipeline.eval_format import append_eval_jsonl

preguntas = [...]  # subconjunto de WikiQA elegido para el Evaluation Layer (50 consultas)

for q in preguntas:
    result = multi_agent_rag(q)
    append_eval_jsonl(result, "outputs/evaluation_ready/multi_agent_react.jsonl")
```

Esto genera un `.jsonl` con un registro por pregunta, listo para cargar como
`datasets.Dataset` o `pandas.DataFrame`. Ver también
`examples/export_eval_format_demo.py` para un ejemplo ejecutable
(`python examples/export_eval_format_demo.py`).

## 3. Ejemplo de carga en el Evaluation Layer

```python
import json
import pandas as pd

records = [json.loads(line) for line in open("outputs/evaluation_ready/multi_agent_react.jsonl", encoding="utf-8")]
df = pd.DataFrame(records)

from datasets import Dataset
eval_dataset = Dataset.from_pandas(df[["question", "answer", "contexts"]])

# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_precision
# ragas_result = evaluate(eval_dataset, metrics=[faithfulness, answer_relevancy, context_precision])
```

## 4. Comparación contra el baseline Naive RAG

Para que la comparación sea justa, el Evaluation Layer debería ejecutar el
mismo set de preguntas (idealmente las 50 consultas de evaluación) contra:
1. El baseline Naive RAG del Data & Retrieval Layer.
2. `multi_agent_rag(question)` del Multi-Agent Orchestration Layer.

y comparar filas con el mismo `question` pero distinto `system_type`
(`"naive_rag"` vs `"multi_agent_react"`). Las columnas `evidence_score` /
`hallucination_risk` del Multi-Agent Orchestration Layer no tienen
equivalente directo en Naive RAG, pero pueden reportarse aparte como valor
agregado del guardrail.

## 5. Checklist de integración

- [ ] El Evaluation Layer confirma que puede cargar
      `outputs/evaluation_ready/*.jsonl` sin transformar campos.
- [ ] El Evaluation Layer confirma el mapeo de columnas esperado por RAGAS
      (`question`, `answer`, `contexts`, opcionalmente `ground_truths`).
- [ ] Se corrió al menos un batch pequeño (5-10 preguntas de WikiQA) de punta
      a punta: Data & Retrieval Layer (retriever real) -> Multi-Agent
      Orchestration Layer (`multi_agent_rag`) -> Evaluation Layer (RAGAS).
