"""
Tool execution node for the LangGraph agent graph.

This node receives tool calls from the researcher agent (stored in state),
executes the corresponding tool functions, and stores the observations
back in the state so the researcher can use them.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from src.agents.state import AgentState
from src.graph.tools import get_tool_by_name
from src.utils.trace import add_trace


def tool_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node: executes pending tool calls from the researcher.

    Reads `tool_calls` from state, executes each one, and returns
    `tool_results` with the observations. Also passes structured
    retrieved_context from KB calls for downstream use.
    """
    tool_calls: List[Dict[str, Any]] = state.get("tool_calls", [])
    trace = list(state.get("trace", []))
    tool_results: List[Dict[str, str]] = []
    retrieved_context = []

    for call in tool_calls:
        tool_name = call.get("tool_name", "")
        tool_args = call.get("args", {})

        tool = get_tool_by_name(tool_name)
        if tool is None:
            observation = f"ERROR: Herramienta '{tool_name}' no encontrada."
        else:
            try:
                observation = tool.fn(**tool_args)
            except Exception as e:
                observation = f"ERROR ejecutando '{tool_name}': {e}"

        tool_results.append({
            "tool_name": tool_name,
            "observation": observation,
        })

        # Capture structured KB results for the auditor
        if tool_name == "knowledge_base_search":
            from src.graph.tools import get_last_kb_results
            retrieved_context = get_last_kb_results()

        trace = add_trace(
            trace,
            agent="tool_node",
            thought=f"Ejecutando herramienta: {tool_name}",
            action=f"{tool_name}({tool_args})",
            observation=observation[:200] + ("..." if len(observation) > 200 else ""),
        )

    result = {
        "tool_results": tool_results,
        "tool_calls": [],  # Clear pending calls
        "trace": trace,
    }
    if retrieved_context:
        result["retrieved_context"] = retrieved_context

    return result


def parse_tool_calls(llm_output: str) -> List[Dict[str, Any]]:
    """
    Parse tool calls from the LLM's ReAct-style output.

    Handles multiple formats that LLMs might produce:
      1. ACTION: {"tool_name": "...", "args": {...}}
      2. ACTION: tool_name(arg1="value1", ...)
      3. ```json blocks containing tool call JSON
      4. Inline JSON objects with "tool_name" key

    Returns a list of parsed tool call dicts.
    """
    calls = []

    # Remove markdown code fences if present (Gemini likes to wrap in ```json ... ```)
    cleaned = re.sub(r'```(?:json)?\s*', '', llm_output)
    cleaned = re.sub(r'```', '', cleaned)

    for line in cleaned.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Match lines starting with ACTION:
        if line.upper().startswith("ACTION:"):
            json_str = line[line.index(":") + 1:].strip()
            parsed = _try_parse_json_tool(json_str)
            if parsed:
                calls.append(parsed)
                continue

            # Try function-call format: tool_name(key="value", ...)
            func_match = re.match(r'(\w+)\(([^)]*)\)', json_str)
            if func_match:
                calls.append(_parse_func_call(func_match))
                continue

        # Also try to find JSON objects with "tool_name" anywhere in the line
        # (handles cases where LLM doesn't prefix with ACTION:)
        json_matches = re.findall(r'\{[^{}]*"tool_name"[^{}]*\{[^{}]*\}[^{}]*\}', line)
        for match in json_matches:
            parsed = _try_parse_json_tool(match)
            if parsed and parsed not in calls:
                calls.append(parsed)

    # Last resort: find any JSON with tool_name in the full text
    if not calls:
        json_pattern = r'\{\s*"tool_name"\s*:\s*"(\w+)"\s*,\s*"args"\s*:\s*(\{[^}]*\})\s*\}'
        for m in re.finditer(json_pattern, cleaned):
            tool_name = m.group(1)
            try:
                args = json.loads(m.group(2))
            except json.JSONDecodeError:
                args = {}
            calls.append({"tool_name": tool_name, "args": args})

    return calls


def _try_parse_json_tool(text: str) -> Dict[str, Any] | None:
    """Try to parse a JSON string as a tool call."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "tool_name" in parsed:
            return {
                "tool_name": parsed["tool_name"],
                "args": parsed.get("args", {}),
            }
    except json.JSONDecodeError:
        pass
    return None


def _parse_func_call(match: re.Match) -> Dict[str, Any]:
    """Parse a function-call style tool invocation."""
    tool_name = match.group(1)
    args_str = match.group(2)
    args = {}
    if args_str.strip():
        arg_pattern = r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|(\d+))'
        for m in re.finditer(arg_pattern, args_str):
            key = m.group(1)
            value = m.group(2) or m.group(3) or m.group(4)
            try:
                value = int(value)
            except (ValueError, TypeError):
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    pass
            args[key] = value
    return {"tool_name": tool_name, "args": args}
