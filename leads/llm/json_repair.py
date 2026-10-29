"""Recovering valid JSON from imperfect model output.

Reasoning models on the free tier routinely: wrap JSON in prose, fence it in
markdown, emit trailing commas, or get cut off mid-string when they hit the
token ceiling. A production agent must survive all four, so this module returns
parsed data or a precise error — never a silent guess.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..errors import JSONParseError


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:json|JSON)?\s*(.*?)```", text, re.S)
    return m.group(1).strip() if m else text


def _first_json_start(text: str) -> int:
    positions = [i for i in (text.find("{"), text.find("[")) if i != -1]
    return min(positions) if positions else -1


def _repair_truncated(text: str) -> str:
    """Close unterminated strings/brackets after a mid-token truncation."""
    out: list[str] = []
    stack: list[str] = []
    in_string = False
    escaped = False

    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            out.append(ch)
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
            out.append(ch)
            continue
        if ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
            out.append(ch)
            continue
        out.append(ch)

    repaired = "".join(out)

    if in_string:
        # Truncated inside a string: close it, then drop an incomplete key.
        repaired += '"'
        repaired = re.sub(r',\s*"[^"]*"\s*$', "", repaired)

    repaired = re.sub(r",\s*$", "", repaired)
    repaired = re.sub(r":\s*$", ": null", repaired)
    repaired += "".join(reversed(stack))
    return repaired


def _try_loads(text: str) -> Any | None:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _close_only(text: str) -> str:
    """Append only the missing closing brackets for an otherwise valid prefix."""
    stack: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack and stack[-1] == ch:
            stack.pop()
    return text + "".join(reversed(stack))


def parse_json(text: str) -> Any:
    """Best-effort parse of model output into Python data.

    Raises ``JSONParseError`` when nothing usable can be recovered.
    """
    if not text or not text.strip():
        raise JSONParseError("empty model output")

    cleaned = _strip_fences(text).strip()
    start = _first_json_start(cleaned)
    if start == -1:
        raise JSONParseError(f"no JSON object in output: {text[:120]!r}")
    cleaned = cleaned[start:]

    # 1. as-is
    loaded = _try_loads(cleaned)
    if loaded is not None:
        return loaded

    # 2. trim trailing prose after the closing brace
    trimmed = cleaned[: cleaned.rfind("}") + 1] if "}" in cleaned else cleaned
    trimmed = trimmed if trimmed else cleaned
    loaded = _try_loads(trimmed)
    if loaded is not None:
        return loaded

    # 3. repair truncation
    repaired = _repair_truncated(cleaned)
    loaded = _try_loads(repaired)
    if loaded is not None:
        return loaded

    # 4. progressive truncation: keep the longest valid prefix of a structure
    for cut in sorted(
        {m.start() for m in re.finditer(r"[}\]]", cleaned)}, reverse=True
    )[:200]:
        candidate = _close_only(cleaned[: cut + 1])
        loaded = _try_loads(candidate)
        if loaded is not None:
            return loaded

    raise JSONParseError(f"unrecoverable JSON: {text[:200]!r}")


def parse_json_object(text: str) -> dict[str, Any]:
    """Like :func:`parse_json` but guarantees a dict result."""
    data = parse_json(text)
    if isinstance(data, list):
        return {"items": data}
    if not isinstance(data, dict):
        raise JSONParseError(f"expected a JSON object, got {type(data).__name__}")
    return data
