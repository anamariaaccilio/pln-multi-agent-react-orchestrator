"""Pipeline end-to-end del módulo de orquestación: multi_agent_rag() y conversión a formato de evaluación."""
from src.pipeline.eval_format import convert_to_eval_format
from src.pipeline.multi_agent_rag import multi_agent_rag

__all__ = ["multi_agent_rag", "convert_to_eval_format"]
