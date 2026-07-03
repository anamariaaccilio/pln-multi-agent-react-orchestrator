# Guía de integración con el Data & Retrieval Layer

El Data & Retrieval Layer es responsable del dataset (WikiQA), los embeddings,
la base vectorial (FAISS o ChromaDB), el retriever real y el baseline Naive
RAG. El Multi-Agent Orchestration Layer (este módulo) consume esa pieza a
través de una interfaz fija, para poder desarrollarse en paralelo sin
bloquearse y sin modificar ningún agente al cambiar de retriever.

## 1. Contrato exacto esperado

El Data & Retrieval Layer debe entregar una función con esta firma exacta:

```python
def retrieve_context(question: str, top_k: int = 5) -> list[dict]:
    ...
```

Y debe devolver una lista de diccionarios con este formato:

```python
[
    {"content": "...", "source": "...", "score": 0.85},
    {"content": "...", "source": "...", "score": 0.78},
]
```

- `content`: texto del fragmento recuperado (chunk).
- `source`: identificador del documento/fila de origen (id de WikiQA, nombre
  de archivo, etc.). Se usa solo para trazabilidad, no se muestra al usuario
  final salvo en la evidencia detallada.
- `score`: score de similitud/relevancia normalizado idealmente en [0, 1].

Este contrato está declarado formalmente en `src/retriever/interface.py`
(`RetrievedChunk`, `RetrieverFn`).

## 2. Cómo registrar el retriever real

```python
from src.retriever.interface import register_retriever
from retrieval_pipeline import retrieve_context  # función real del Data & Retrieval Layer

register_retriever(retrieve_context)

from src.pipeline.multi_agent_rag import multi_agent_rag
result = multi_agent_rag("Cuando se construyo la Torre Eiffel?", retriever=retrieve_context)
```

Pasar `retriever=retrieve_context` directamente a `multi_agent_rag()` hace
dos cosas automáticamente:
1. Llama a `register_retriever()` internamente.
2. Fuerza `use_fallback_retriever=False` para esa llamada (no hace falta
   tocar `config/settings.yaml`).

Si prefieres dejarlo fijo para toda la sesión en vez de pasarlo en cada
llamada, alcanza con `register_retriever(retrieve_context)` una vez y poner
`retriever.use_fallback: false` en `config/settings.yaml`.

Ver también `examples/integration_example_retriever.py`, que ejecuta este
patrón con un retriever de ejemplo, sin tocar ningún archivo de `src/agents/`.

## 3. Qué pasa si el Data & Retrieval Layer no está listo

Mientras `USE_FALLBACK_RETRIEVER = True` (default), `resolve_retriever()` usa
automáticamente `src/retriever/fallback_retriever.py::fallback_retrieve_context`,
que devuelve contenido determinista sobre la Torre Eiffel (bueno) y un
escenario de baja evidencia activado con la palabra clave `"baja_evidencia"`
en la pregunta. Esto permite desarrollar y demostrar todo el flujo (aprobado
y rechazado) sin depender de FAISS/ChromaDB.

## 4. Checklist de integración

- [ ] La función del Data & Retrieval Layer se llama `retrieve_context` y
      acepta `(question: str, top_k: int = 5)`.
- [ ] Devuelve una lista (posiblemente vacía) de dicts con las claves
      `content`, `source`, `score`.
- [ ] `score` es un `float` (no `numpy.float32`, que puede romper el
      formateo `f"{score:.2f}"` en `format_context_block`; castear con
      `float(score)` si hace falta).
- [ ] Se probó `register_retriever(retrieve_context)` seguido de
      `multi_agent_rag("...pregunta real de WikiQA...")` y el resultado
      tiene `retrieved_context` no vacío.
- [ ] Se probó con `top_k` distinto de 5 (por ejemplo 3) para confirmar que
      el parámetro se respeta.

## 5. Errores comunes

Ver también `docs/troubleshooting.md`.

- **`RuntimeError: USE_FALLBACK_RETRIEVER=False pero no se registro ningun
  retriever real`**: falta llamar a `register_retriever()` antes de invocar
  `multi_agent_rag()`, o no se pasó `retriever=...` a la función.
- **`KeyError: 'content'` / `'source'` / `'score'`**: el retriever real del
  Data & Retrieval Layer no respeta el formato exacto del contrato; revisar
  el diccionario devuelto por cada chunk.
- **`evidence_score` siempre bajo con el retriever real**: puede ser que el
  contenido devuelto esté en inglés y el vocabulario no coincida con el
  `draft_answer` en español, o que los chunks sean demasiado largos/ruidosos.
  Ajustar el chunking del Data & Retrieval Layer o el umbral `MIN_EVIDENCE_SCORE`.
