# Guion de defensa - Multi-Agent Orchestration Layer (Multi-Agent ReAct + LangGraph + Guardrails)

Guion breve (3-4 minutos) para sustentar este módulo ante el profesor.
Ajustar el tono, no es necesario leerlo literal.

## 1. Qué problema resuelve este módulo (30s)

"Este módulo implementa el motor de razonamiento multi-agente del sistema:
en vez de responder con un solo paso de RAG, el sistema investiga, audita su
propia respuesta contra la evidencia, y solo después redacta la respuesta
final. Esto reduce el riesgo de alucinaciones frente a un Naive RAG de un
solo paso."

## 2. Arquitectura en una frase (30s)

"El flujo está implementado con LangGraph: tres agentes -- Researcher, Fact
Auditor y Writer -- conectados por un grafo de estados con una arista
condicional. Si el Fact Auditor rechaza el borrador, el flujo vuelve al
Researcher hasta un máximo de iteraciones; si aprueba, pasa al Writer."

Mostrar el diagrama Mermaid (`docs/architecture.md` o
`outputs/graph/graph_diagram.mmd`).

## 3. Guardrails (45s)

"El punto crítico es que el Fact Auditor no le pregunta al mismo LLM si su
respuesta está bien. Calcula `evidence_score` y `hallucination_risk` con una
heurística independiente de superposición léxica entre la respuesta y el
contexto recuperado. Si `evidence_score` cae debajo de 0.70 o el riesgo de
alucinación supera 0.30, se rechaza. Si se agotan las iteraciones, igual se
entrega una respuesta, pero con una advertencia explícita y baja el nivel de
confianza."

## 4. Trazabilidad ReAct (30s)

"Cada agente registra una traza corta: Thought, Action, Observation. No es
una cadena de razonamiento larga -- es deliberadamente breve para que sea
auditable: se puede ver exactamente por qué el sistema rechazó o aprobó una
respuesta, en `outputs/traces/*.json`."

Mostrar `print_trace(result)` en vivo con `python examples/run_multi_agent_demo.py`.

## 5. Demo en vivo (60s)

Correr `python examples/run_multi_agent_demo.py`, que muestra en una sola
ejecución:
1. El diagrama del grafo.
2. Un ejemplo **aprobado**: buen contexto -> `audit_passed=True`, confianza Alta.
3. Un ejemplo **rechazado**: contexto pobre -> rechazo, ciclo de vuelta al
   Researcher, respuesta final con advertencia.

## 6. Analogía con Aprendizaje por Refuerzo (30s)

"Como en el curso vimos agente, estado, acción, observación, política y
recompensa, se puede leer este grafo con esa misma lente: el `AgentState` es
el estado, `route_after_audit` es la política de control, y la mejora de
faithfulness / reducción de alucinaciones sería la recompensa proxy. Aclaro
que no hay entrenamiento por refuerzo real -- es una analogía para explicar
el ciclo de control secuencial del grafo, no un agente entrenado con RL."

## 7. Integración con el resto del sistema (30s)

"El módulo está desacoplado por interfaz: `retrieve_context(question,
top_k)` para conectar con la base vectorial del Data & Retrieval Layer, y
`convert_to_eval_format(result)` para entregarle al Evaluation Layer
exactamente las columnas que RAGAS necesita. Todo el módulo se desarrolló y
validó en modo fallback, sin depender de esas dos piezas para poder
avanzar."

## 8. LLM: no depende de una sola API (20s)

"El módulo no está atado a Gemini ni a ninguna API externa como única
opción. La interfaz `generate_llm_response(prompt)` se resuelve según
`provider`: fallback para pruebas rápidas, LLM local cuantizado (Mistral-7B
o Llama-3-8B en 4 bits) como modo principal de arquitectura -- tal como pide
el enunciado --, y Gemini como atajo opcional de desarrollo."

## 9. Cierre (15s)

"En resumen: tres agentes, un grafo con ciclo de corrección, guardrails
cuantitativos contra alucinaciones, y trazabilidad completa de cada
decisión -- listo para conectarse al retriever real y al Evaluation Layer."

## Preguntas esperadas del profesor (preparar respuesta corta)

- **"Por qué no dejar que el mismo LLM se autoevalúe?"** -> Porque tiende a
  confirmar su propia salida; se usó una heurística independiente y
  reproducible como guardrail mínimo, documentado como punto de mejora hacia
  un juez basado en NLI/LLM.
- **"Qué pasa si nunca hay buena evidencia?"** -> Se agotan las
  `MAX_ITERATIONS`, el Writer Agent entrega la respuesta con advertencia y
  confianza Baja, o el mensaje estándar de "no hay evidencia suficiente" si
  no hay contexto en absoluto.
- **"Es RL de verdad?"** -> No, es control secuencial con una política
  determinista (`route_after_audit`); la analogía es solo pedagógica.
- **"Por qué no usar solo Gemini?"** -> El enunciado pide priorizar
  inferencia local; Gemini queda como proveedor opcional para desarrollo sin
  GPU, no como dependencia obligatoria.
