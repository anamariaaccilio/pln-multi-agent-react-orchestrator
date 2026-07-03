# Enunciado oficial del proyecto

> **Placeholder.** Pega aquí el texto oficial completo del enunciado
> ("Sistema Operativo de Agentes Cognitivos para Inteligencia Competitiva -
> Multi-Agent ReAct") tal como lo entregó la cátedra, para que quede
> versionado junto con el código y sirva de referencia de cumplimiento.

Mientras tanto, el resumen funcional que guía la implementación de este
repositorio es:

- Sistema multi-agente basado en ReAct (Thought -> Action -> Observation).
- Orquestación con LangGraph.
- Al menos 3 agentes: Researcher Agent, Fact Auditor Agent, Writer Agent.
- Base vectorial local con FAISS o ChromaDB sobre WikiQA (Data & Retrieval Layer).
- Guardrails para mitigar alucinaciones, con ciclo de rechazo y revisión.
- Evaluación posterior con RAGAS o LLM-as-a-Judge (Evaluation Layer),
  comparando contra un baseline Naive RAG.

Ver `docs/architecture.md` para el detalle de cómo este repositorio cumple
cada uno de estos puntos en el Multi-Agent Orchestration Layer.
