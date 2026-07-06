"""
Naive RAG baseline: retrieve context + single LLM call.

This is the baseline system to compare against the multi-agent ReAct pipeline.
No agents, no auditor, no guardrails — just retrieve and generate.

Usage:
    from src.pipeline.naive_rag import naive_rag
    result = naive_rag("When was the Eiffel Tower built?")

    # With explicit retriever:
    from src.retriever.retrieval_pipeline import retrieve_context
    result = naive_rag("When was the Eiffel Tower built?", retriever=retrieve_context)
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from src.config import AgentConfig, DEFAULT_CONFIG
from src.llm.interface import resolve_llm
from src.retriever.interface import register_retriever, resolve_retriever


NAIVE_RAG_PROMPT = (
    "Answer the following question based ONLY on the provided context. "
    "If the context does not contain enough information to answer, say so clearly. "
    "Do not make up information.\n\n"
    "CONTEXT:\n{context}\n\n"
    "QUESTION: {question}\n\n"
    "ANSWER:"
)


def _format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a single context string."""
    if not chunks:
        return "(no context available)"
    lines = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "unknown")
        content = chunk.get("content", "")
        score = chunk.get("score", 0.0)
        lines.append(f"[{i}] (source: {source}, score: {score:.2f})\n{content}")
    return "\n\n".join(lines)


def naive_rag(
    question: str,
    retriever: Optional[Callable[..., Any]] = None,
    llm: Optional[Callable[[str], str]] = None,
    config: Optional[AgentConfig] = None,
) -> Dict[str, Any]:
    """
    Naive RAG baseline: retrieve + generate in a single pass.

    Args:
        question: User question in natural language.
        retriever: retrieve_context(question, top_k) function. If provided,
            it is registered and used instead of the fallback.
        llm: generate_llm_response(prompt) function. If provided, used
            instead of the configured provider.
        config: AgentConfig for settings (top_k, provider, etc.).
            Defaults to DEFAULT_CONFIG from settings.yaml.

    Returns:
        Dict with: question, retrieved_context, final_answer, system_type,
        and metadata for evaluation comparison.
    """
    cfg = config or DEFAULT_CONFIG

    # Resolve retriever
    if retriever is not None:
        register_retriever(retriever)
        use_fallback_retriever = False
    else:
        use_fallback_retriever = cfg.use_fallback_retriever

    retrieve_fn = resolve_retriever(use_fallback_retriever)

    # Resolve LLM
    if llm is not None:
        generate = llm
    else:
        generate = resolve_llm(cfg.use_fallback_llm, cfg.llm_provider)

    # 1. Retrieve
    retrieved_context = retrieve_fn(question, top_k=cfg.top_k)

    # 2. Build prompt
    context_text = _format_context(retrieved_context)
    prompt = NAIVE_RAG_PROMPT.format(context=context_text, question=question)

    # 3. Generate
    try:
        final_answer = generate(prompt)
    except Exception as e:
        final_answer = f"Error generating response: {e}"

    if not final_answer:
        final_answer = "No answer could be generated."

    # 4. Compute basic metrics for comparison
    max_score = max((c.get("score", 0.0) for c in retrieved_context), default=0.0)

    return {
        "question": question,
        "retrieved_context": retrieved_context,
        "final_answer": final_answer.strip(),
        "system_type": "naive_rag",
        # Metadata for evaluation
        "top_k": cfg.top_k,
        "max_retrieval_score": max_score,
        "num_chunks_retrieved": len(retrieved_context),
        "iterations": 1,  # Always 1 for naive RAG
        "audit_passed": None,  # No auditor in naive RAG
        "evidence_score": None,
        "confidence_level": None,
        "warnings": [],
        "trace": [
            {
                "agent": "naive_rag",
                "thought": "Single-pass RAG: retrieve context and generate answer.",
                "action": f"retrieve_context(question, top_k={cfg.top_k}) + generate(prompt)",
                "observation": f"Retrieved {len(retrieved_context)} chunks, max_score={max_score:.2f}.",
            }
        ],
    }
