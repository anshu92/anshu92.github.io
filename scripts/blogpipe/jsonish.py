from __future__ import annotations

import ast
import json
import re
from typing import Any


def loads_object(text: str) -> dict[str, Any]:
    payload = extract_object(text)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        data = ast.literal_eval(_pythonish_payload(payload))
    if not isinstance(data, dict):
        raise ValueError("structured output is not an object")
    return data


def extract_object(text: str) -> str:
    raw = _strip_fence(text or "")
    start = raw.find("{")
    if start < 0:
        return raw.strip()
    fragment = raw[start:].strip()
    complete = _first_balanced_object(fragment)
    if complete:
        return complete
    return _complete_fragment(fragment)


def _strip_fence(text: str) -> str:
    raw = text.strip()
    fenced = re.match(r"^```(?:json|python)?\s*\n([\s\S]*?)(?:\n```)?$", raw, re.I)
    if fenced:
        return fenced.group(1).strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw.lstrip("`")
    if raw.endswith("```"):
        raw = raw[:-3]
    return raw.strip()


def _first_balanced_object(text: str) -> str:
    stack: list[str] = []
    in_string = False
    quote = ""
    escaped = False
    for idx, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                in_string = False
            continue
        if char in {'"', "'"}:
            in_string = True
            quote = char
            continue
        if char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack:
                continue
            expected = stack.pop()
            if char != expected:
                return ""
            if not stack:
                return text[: idx + 1]
    return ""


def _complete_fragment(text: str) -> str:
    stack: list[str] = []
    in_string = False
    quote = ""
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                in_string = False
            continue
        if char in {'"', "'"}:
            in_string = True
            quote = char
            continue
        if char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if stack and char == stack[-1]:
                stack.pop()
    if in_string:
        text += quote
    return text.rstrip().rstrip(",") + "".join(reversed(stack))


def _pythonish_payload(text: str) -> str:
    payload = re.sub(r"\btrue\b", "True", text, flags=re.I)
    payload = re.sub(r"\bfalse\b", "False", payload, flags=re.I)
    payload = re.sub(r"\bnull\b", "None", payload, flags=re.I)
    return payload
