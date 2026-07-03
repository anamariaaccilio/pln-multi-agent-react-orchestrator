# Arquitectura del Sistema Multi-Agent ReAct

## 1. Vision general

Este modulo implementa el **Sistema Operativo de Agentes Cognitivos** exigido
por la Opcion 2 del enunciado: un flujo multi-agente autonomo basado en
**ReAct** (Thought -> Action -> Observation), orquestado con **LangGraph**,
con **guardrails** contra alucinaciones y un **ciclo de rechazo y correccion**.

El modulo es agnostico a quien provee el contexto (retriever) y el modelo de
lenguaje (LLM): ambos se inyectan por interfaz, lo que permite desarrollarlo
y probarlo de forma aislada (modo fallback) y luego conectarlo a los
componentes reales del Data & Retrieval Layer y del Evaluation Layer sin
cambiar el grafo.

## 2. Los tres agentes

```
START
  |
  v
researcher  --------------------------------------------------+
  |                                                            |
  v                                                            |
auditor --route_after_audit-->  "writer"              --> writer --> END
                            |--> "researcher" (ciclo) --------^
                            '--> "writer_with_warning" --> writer --> END
```

### Researcher Agent (`src/agents/researcher.py`)
- Llama a `retrieve_context(question, top_k)`.
- Genera un `draft_answer` apoyado unicamente en el contexto recuperado.
- Si hay `audit_feedback` de una iteracion previa, lo incluye en el prompt
  para corregir el borrador (ciclo de correccion).
- Reporta `limitations` si el contexto es escaso o de baja relevancia.
- Resuelve el LLM activo (fallback, local o Gemini) via `src/llm/interface.py::resolve_llm`.

### Fact Auditor Agent (`src/agents/auditor.py`)
- No confia en que el LLM se autoevalue: calcula `evidence_score` y
  `hallucination_risk` con una **heuristica lexica independiente**
  (interseccion de palabras clave entre `draft_answer` y `retrieved_context`).
- Aplica las reglas de aprobacion (ver seccion 4, Guardrails).
- Decide `route_decision`: `"writer"`, `"researcher"` o `"writer_with_warning"`.

### Writer Agent (`src/agents/writer.py`)
- Construye la respuesta final con el formato exigido: Respuesta final,
  Evidencia usada, Nivel de confianza, Advertencia.
- Si no hay evidencia suficiente, devuelve el mensaje estandar de
  insuficiencia en vez de inventar una respuesta.

## 3. AgentState

`src/agents/state.py` define el estado compartido (`TypedDict`) que LangGraph
pasa de nodo a nodo. Cada nodo devuelve solo las claves que actualiza; el
resto del estado persiste. Ver el archivo para el listado completo de claves
(`question`, `retrieved_context`, `draft_answer`, `audit_passed`,
`evidence_score`, `hallucination_risk`, `trace`, `iterations`, etc.).

## 4. Guardrails contra alucinaciones

Umbrales configurables en `src/config.py` / `config/settings.yaml`:

| Variable                  | Default | Efecto                                            |
|----------------------------|---------|---------------------------------------------------|
| `MIN_EVIDENCE_SCORE`       | 0.70    | Por debajo de esto, se rechaza el borrador.        |
| `MAX_HALLUCINATION_RISK`   | 0.30    | Por encima de esto, se rechaza el borrador.        |
| `MAX_ITERATIONS`           | 2       | Limite del ciclo researcher -> auditor.            |

Reglas exactas en `auditor_node`:
1. Sin contexto -> rechazar.
2. `evidence_score < MIN_EVIDENCE_SCORE` -> rechazar.
3. `hallucination_risk > MAX_HALLUCINATION_RISK` -> rechazar.
4. En otro caso -> aprobar.
5. Si se alcanzo `MAX_ITERATIONS` sin aprobar -> se permite pasar al Writer
   Agent pero con `route_decision = "writer_with_warning"`, lo que fuerza una
   advertencia visible en la respuesta final y baja el `confidence_level`.

## 5. Trazabilidad ReAct

Cada agente registra eventos `{agent, thought, action, observation}` via
`src/utils/trace.py::add_trace`. El `thought` es deliberadamente breve y
operativo (una frase auditable), no una cadena de razonamiento larga: el
objetivo es poder inspeccionar **por que** el sistema tomo cada decision,
no simular un monologo interno del LLM.

## 6. Analogia con Aprendizaje por Refuerzo (conceptual, no entrenamiento)

El curso trabaja los conceptos de agente, estado, accion, observacion,
politica, recompensa y episodio. El grafo LangGraph puede leerse con esa
misma lente, como un **problema de control secuencial**, aunque **no hay
entrenamiento por refuerzo real** (no hay funcion de valor ni actualizacion
de politica por gradiente):

| Concepto RL          | Equivalente en este sistema                                             |
|-----------------------|--------------------------------------------------------------------------|
| **Estado**            | `AgentState`: pregunta, contexto, borrador, auditoria, iteraciones.       |
| **Accion**            | `retrieve_context`, generar respuesta, auditar, aprobar/rechazar, redactar. |
| **Observacion**       | Contexto recuperado, `audit_feedback`, `evidence_score`.                  |
| **Politica de control** | `route_after_audit(state)` (funcion determinista, no aprendida).        |
| **Recompensa proxy**  | Mejora de faithfulness / answer relevance y reducción de `hallucination_risk` entre iteraciones (evaluada externamente por el Evaluation Layer, no optimizada online). |
| **Episodio**          | Una ejecucion completa de `multi_agent_rag(question)`, de START a END.    |

Es importante remarcar esto en la defensa (ver `docs/defense_script.md`):
es una **analogia pedagogica** sobre el ciclo de control del grafo, no una
implementacion de RL.

## 7. Por que una heuristica lexica y no solo el LLM

Pedirle al mismo LLM que genero la respuesta que tambien diga "si, esta
bien sustentada" es un guardrail debil (el modelo tiende a confirmar su
propia salida). Por eso `evidence_score` y `hallucination_risk` se calculan
con una funcion independiente basada en superposicion de vocabulario entre
`draft_answer` y `retrieved_context`. Es un proxy simple pero auditable y
100% reproducible en modo fallback. Una mejora natural (ver README, "Trabajo
futuro") es sustituirlo por un juez basado en NLI o en un segundo LLM.

## 8. Proveedores de LLM: fallback, local y Gemini

`src/llm/interface.py` define el contrato único `generate_llm_response(prompt)`.
Ningún agente importa un backend concreto directamente; todos resuelven el
proveedor activo via `resolve_llm(use_fallback, provider)`:

| provider   | Módulo                     | Uso previsto                                                        |
|------------|----------------------------|----------------------------------------------------------------------|
| `fallback` | `src/llm/fallback_llm.py`  | Pruebas locales sin GPU ni API keys (default).                       |
| `local`    | `src/llm/local_llm.py`     | Modo principal de arquitectura: Mistral-7B/Llama-3-8B cuantizado 4-bit. |
| `gemini`   | `src/llm/gemini_llm.py`    | Desarrollo rápido opcional, sin GPU (requiere `GOOGLE_API_KEY`).      |

Esto evita que el módulo de orquestación dependa exclusivamente de una API
externa: el diseño está preparado para inferencia local, como pide el
enunciado, y Gemini queda como atajo opcional de desarrollo.
