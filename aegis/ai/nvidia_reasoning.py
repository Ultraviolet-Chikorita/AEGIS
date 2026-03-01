from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import requests


logger = logging.getLogger(__name__)


class NvidiaReasoningClient:
    def __init__(self, api_key: str, base_url: str, model_id: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id

    async def run_with_graph_access(
        self,
        system_prompt: str,
        task_prompt: str,
        seed_context: dict[str, Any],
        output_schema: dict[str, Any],
        graph_tools,
        max_rounds: int = 8,
    ) -> dict[str, Any]:
        if not self.api_key.strip():
            raise ValueError("NVIDIA_API_KEY is required when REASONING_PROVIDER=nvidia")

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    f"{system_prompt}\n\n"
                    "You can access graph tools. Reply with JSON only using one of two shapes:\n"
                    "1) Tool call: {\"tool_call\": {\"name\": <tool_name>, \"input\": <object>}}\n"
                    "2) Final output: {\"final\": <object matching required schema>}\n"
                    "Do not include markdown fences."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task: {task_prompt}\n\n"
                    f"Seed context: {json.dumps(seed_context)}\n\n"
                    f"Required output schema: {json.dumps(output_schema)}"
                ),
            },
        ]

        parse_error: Exception | None = None
        for _ in range(max_rounds):
            content = await asyncio.to_thread(self._chat, messages)
            try:
                obj = _extract_json(content)
            except Exception as exc:
                parse_error = exc
                logger.warning("NVIDIA response was not valid JSON object; requesting strict JSON retry: %s", exc)
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply was not parseable JSON. "
                            "Return exactly one JSON object only, with no markdown fences and no extra text."
                        ),
                    }
                )
                continue

            tool_call = obj.get("tool_call") if isinstance(obj, dict) else None
            final = obj.get("final") if isinstance(obj, dict) else None

            if isinstance(tool_call, dict):
                name = str(tool_call.get("name", "")).strip()
                tool_input = tool_call.get("input", {})
                if not name:
                    raise ValueError("NVIDIA reasoning returned tool_call without a tool name")
                if not isinstance(tool_input, dict):
                    tool_input = {}

                tool_result = await graph_tools.execute(name, tool_input)
                messages.append({"role": "assistant", "content": json.dumps(obj)})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Tool result:\n"
                            f"{json.dumps({'tool_name': name, 'tool_result': tool_result})}\n\n"
                            "Continue. Return either the next tool_call JSON or final JSON."
                        ),
                    }
                )
                continue

            if isinstance(final, dict):
                return final

            if isinstance(obj, dict):
                return obj

            raise ValueError("NVIDIA reasoning returned unexpected JSON response shape")

        if parse_error is not None:
            raise RuntimeError(f"Max NVIDIA reasoning rounds exceeded after parse failures: {parse_error}")
        raise RuntimeError("Max NVIDIA reasoning rounds exceeded")

    def _chat(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": 0.2,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()

        body = response.json()
        return body["choices"][0]["message"]["content"]


def _extract_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj

    raise ValueError("NVIDIA reasoning response did not contain a valid JSON object")
