# Cómo correr el proyecto — paso a paso

## 1. Crear entorno

```bash
cd PLN_P2
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

## 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

Esto instala LangGraph, dependencias de vector store (FAISS/ChromaDB, para
cuando se conecte el Data & Retrieval Layer), dependencias opcionales de LLM
local cuantizado (`torch`, `transformers`, `bitsandbytes`) y de Gemini
(`google-generativeai`), y RAGAS (para el Evaluation Layer).

Si solo quieres correr el módulo de orquestación en modo fallback (sin GPU,
sin API keys), basta con `langgraph`, `langchain-core` y `pyyaml`; el resto
son dependencias de las otras capas incluidas para que el repo instale
limpio en un solo paso.

## 3. Correr la demo (modo fallback, sin dependencias externas)

```bash
python examples/run_multi_agent_demo.py
```

Debe imprimir: el diagrama Mermaid del grafo, un ejemplo aprobado
(`audit_passed=True`, confianza Alta) y un ejemplo rechazado que cicla y
termina con advertencia. Además guarda las trazas en `outputs/traces/`.

## 4. Correr los demás examples

```bash
python examples/integration_example_retriever.py
python examples/export_eval_format_demo.py
```

El primero demuestra cómo se conecta un retriever real sin tocar los
agentes; el segundo exporta resultados al formato que consume el Evaluation
Layer, dejándolos en `outputs/evaluation_ready/`.

## 5. Correr el notebook de entrega

Abrir `notebooks/02_multi_agent_langgraph_pipeline.ipynb` en Jupyter, VS
Code o Google Colab, y ejecutar las celdas en orden. El notebook solo
importa desde `src/`; no duplica lógica.

En Colab, la primera celda de código clona el repo (opcional, si no lo
tienes ya montado) e instala `requirements.txt`.

## 6. Integrar el retriever real del Data & Retrieval Layer

```python
from src.retriever.interface import register_retriever
from retrieval_pipeline import retrieve_context  # función real, ver docs/integration_with_retriever.md

register_retriever(retrieve_context)

from src.pipeline.multi_agent_rag import multi_agent_rag
result = multi_agent_rag("...", retriever=retrieve_context)
```

Ningún archivo de `src/agents/` cambia al hacer esto. Detalle completo en
`docs/integration_with_retriever.md`.

## 7. Cambiar de LLM fallback a LLM local o Gemini

En `config/settings.yaml`:

```yaml
llm:
  use_fallback: false
  provider: "local"   # o "gemini"
```

O explícitamente en código:

```python
from src.llm.local_llm import generate_llm_response
result = multi_agent_rag("...", llm=generate_llm_response)
```

## 8. Entregar resultados al Evaluation Layer

```python
from src.pipeline.eval_format import convert_to_eval_format
eval_item = convert_to_eval_format(result)
```

Detalle completo en `docs/evaluation_interface.md`.
