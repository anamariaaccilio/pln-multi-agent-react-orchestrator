"""Configuración global del módulo de orquestación (Multi-Agent ReAct + LangGraph + Guardrails).

Expone tanto un objeto de configuración tipado (AgentConfig) como variables
sueltas a nivel de módulo, para que coincidan literalmente con el contrato
del proyecto:

    USE_FALLBACK_RETRIEVER = True
    USE_FALLBACK_LLM = True
    MAX_ITERATIONS = 2
    TOP_K = 5
    MIN_EVIDENCE_SCORE = 0.70
    MAX_HALLUCINATION_RISK = 0.30
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS_PATH = ROOT_DIR / "config" / "settings.yaml"


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass(frozen=True)
class AgentConfig:
    """Parámetros de control del ciclo Researcher -> Fact Auditor -> Writer."""

    use_fallback_retriever: bool = True
    use_fallback_llm: bool = True
    top_k: int = 5
    max_iterations: int = 2
    min_evidence_score: float = 0.70
    max_hallucination_risk: float = 0.30

    # provider: "fallback" | "local" | "gemini"
    llm_provider: str = "fallback"
    llm_model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    llm_temperature: float = 0.2
    llm_max_new_tokens: int = 512

    traces_dir: str = "outputs/traces"
    graph_dir: str = "outputs/graph"
    eval_dir: str = "outputs/evaluation_ready"
    system_type: str = "multi_agent_react"

    def with_overrides(self, **kwargs) -> "AgentConfig":
        """Devuelve una copia con los campos indicados sobreescritos (config inmutable)."""
        return replace(self, **kwargs)

    @classmethod
    def from_yaml(cls, path: Optional[Path] = None) -> "AgentConfig":
        data = _load_yaml(path or DEFAULT_SETTINGS_PATH)
        retriever = data.get("retriever", {})
        llm = data.get("llm", {})
        agents = data.get("agents", {})
        paths = data.get("paths", {})
        system = data.get("system", {})
        return cls(
            use_fallback_retriever=retriever.get("use_fallback", True),
            use_fallback_llm=llm.get("use_fallback", True),
            top_k=retriever.get("top_k", 5),
            max_iterations=agents.get("max_iterations", 2),
            min_evidence_score=agents.get("min_evidence_score", 0.70),
            max_hallucination_risk=agents.get("max_hallucination_risk", 0.30),
            llm_provider=llm.get("provider", "fallback"),
            llm_model_name=llm.get("model_name", "mistralai/Mistral-7B-Instruct-v0.3"),
            llm_temperature=llm.get("temperature", 0.2),
            llm_max_new_tokens=llm.get("max_new_tokens", 512),
            traces_dir=paths.get("traces_dir", "outputs/traces"),
            graph_dir=paths.get("graph_dir", "outputs/graph"),
            eval_dir=paths.get("eval_dir", "outputs/evaluation_ready"),
            system_type=system.get("system_type", "multi_agent_react"),
        )


# Instancia por defecto, cargada una vez desde config/settings.yaml.
DEFAULT_CONFIG = AgentConfig.from_yaml()

# Variables sueltas (contrato del proyecto). Si cambias settings.yaml y
# quieres que estas variables reflejen el cambio sin reiniciar el kernel,
# vuelve a llamar a AgentConfig.from_yaml().
USE_FALLBACK_RETRIEVER = DEFAULT_CONFIG.use_fallback_retriever
USE_FALLBACK_LLM = DEFAULT_CONFIG.use_fallback_llm
MAX_ITERATIONS = DEFAULT_CONFIG.max_iterations
TOP_K = DEFAULT_CONFIG.top_k
MIN_EVIDENCE_SCORE = DEFAULT_CONFIG.min_evidence_score
MAX_HALLUCINATION_RISK = DEFAULT_CONFIG.max_hallucination_risk
