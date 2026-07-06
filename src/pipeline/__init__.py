"""Pipeline end-to-end del módulo de orquestación: multi_agent_rag(), naive_rag() y conversión a formato de evaluación."""
from src.pipeline.eval_format import convert_to_eval_format
from src.pipeline.multi_agent_rag import multi_agent_rag
from src.pipeline.naive_rag import naive_rag

__all__ = ["multi_agent_rag", "naive_rag", "convert_to_eval_format"]
