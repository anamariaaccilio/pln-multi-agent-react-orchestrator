# Troubleshooting - Multi-Agent Orchestration Layer

## Errores de entorno / instalacion

**`ModuleNotFoundError: No module named 'langgraph'`**
Instalar dependencias: `pip install -r requirements.txt`. En Colab, correr
la celda de instalacion al inicio de `notebooks/02_multi_agent_langgraph_pipeline.ipynb`
antes que cualquier otra.

**`ModuleNotFoundError: No module named 'src'`**
Los scripts de `examples/` ya incluyen un bootstrap de `sys.path` al inicio
(`sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`), por lo que
`python examples/run_multi_agent_demo.py` funciona ejecutado directamente. Si
el error aparece en un notebook o script propio, agregar ese mismo bootstrap
o correrlo con `python -m examples.nombre_del_script` desde la raíz del repo.

## Errores de ejecucion del grafo

**`GraphRecursionError` (o `RecursionError`) al invocar el grafo**
El ciclo `researcher -> auditor` supero el `recursion_limit` pasado a
`compiled_graph.invoke()`. Esto no deberia pasar con `MAX_ITERATIONS` bajo
(2-3), porque `multi_agent_rag()` ya calcula un limite generoso
`(max_iterations + 2) * 4`. Si se sube `MAX_ITERATIONS` manualmente a un
numero alto, subir tambien el `recursion_limit` en
`src/pipeline/multi_agent_rag.py`.

**El grafo nunca llega a `writer` (loop aparente)**
Revisar que `auditor_node` siempre setea `route_decision` a uno de los tres
valores validos (`"writer"`, `"researcher"`, `"writer_with_warning"") y que
`iterations` efectivamente se incrementa en `researcher_node` (si no se
incrementa, `iterations >= MAX_ITERATIONS` nunca se cumple).

**`RuntimeError: USE_FALLBACK_RETRIEVER=False pero no se registro ningun
retriever real`**
Falta `register_retriever(retrieve_context)` o pasar `retriever=...` a
`multi_agent_rag()`. Ver `docs/integration_with_retriever.md`.

**`RuntimeError: USE_FALLBACK_LLM=False pero no hay LLM real disponible`**
Falta pasar `llm=generate_llm_response` (de `local_llm.py` o `gemini_llm.py`)
a `multi_agent_rag()`, llamar a `register_llm()`, o configurar
`llm.provider: local` / `llm.provider: gemini` en `config/settings.yaml`.

## Errores de guardrails / calidad de respuesta

**`evidence_score` siempre da 0.0 con el retriever real**
Verificar que `retrieved_context` no este vacio y que el `draft_answer` no
este vacio. Si el LLM real devuelve una respuesta en un idioma distinto al
contexto, la heuristica lexica (interseccion de palabras) puede fallar;
normalizar idioma o mejorar el tokenizador en `src/agents/auditor.py`.

**El sistema siempre responde "No hay evidencia suficiente..."**
Puede ser: (a) el retriever esta devolviendo listas vacias -- revisar
`top_k` y la base vectorial del Data & Retrieval Layer; o (b)
`MIN_EVIDENCE_SCORE` está configurado demasiado alto para el dominio de las
preguntas.

**Loop infinito de rechazos sin mejora**
Es esperado que, sin un LLM real capaz de incorporar `audit_feedback`, el
fallback no "aprenda" de una iteracion a otra. Con un LLM real (local o
Gemini), el `audit_feedback` se inyecta en el prompt del Researcher Agent via
`build_researcher_prompt(..., prior_feedback=...)` para guiar la correccion.

## Errores de visualizacion

**`draw_mermaid_png()` lanza excepcion o no genera PNG**
Requiere dependencias adicionales no incluidas por defecto (graphviz /
mermaid-cli) y a veces acceso a internet en Colab. No es critico: el
diagrama Mermaid en texto (`MERMAID_DIAGRAM` / `outputs/graph/graph_diagram.mmd`)
siempre se genera igual y puede pegarse en cualquier renderer de Mermaid
(incluido GitHub, que lo renderiza nativo en Markdown).

## Errores de tipos / formato

**`TypeError` al formatear `score` con `:.2f`**
Si el retriever real del Data & Retrieval Layer devuelve `score` como
`numpy.float32` u otro tipo no nativo, castear a `float()` en el retriever o
en `format_context_block` (`src/utils/formatting.py`).

**`KeyError` en `convert_to_eval_format`**
El resultado pasado no viene de `multi_agent_rag()` (por ejemplo, es un
`AgentState` parcial de un nodo individual). Usar siempre el diccionario
completo devuelto por `multi_agent_rag()`.
