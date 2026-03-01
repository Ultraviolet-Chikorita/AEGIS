from __future__ import annotations

import asyncio
import json
from typing import Any

import boto3

from aegis.graph.primitives import GRAPH_PRIMITIVES


class BedrockReasoningClient:
    def __init__(self, model_id: str, region: str = "us-east-1") -> None:
        self.client = boto3.client("bedrock-runtime", region_name=region)
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
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {
                        "text": (
                            f"{task_prompt}\n\n"
                            f"Context: {json.dumps(seed_context)}\n\n"
                            f"Required output schema: {json.dumps(output_schema)}"
                        )
                    }
                ],
            }
        ]

        tool_config = {"tools": self._format_tools(GRAPH_PRIMITIVES)}

        for _ in range(max_rounds):
            response = await asyncio.to_thread(
                self.client.converse,
                modelId=self.model_id,
                messages=messages,
                system=[{"text": system_prompt}],
                toolConfig=tool_config,
            )

            if response["stopReason"] == "tool_use":
                assistant_message = response["output"]["message"]
                messages.append(assistant_message)

                tool_results: list[dict[str, Any]] = []
                for block in assistant_message.get("content", []):
                    tool_use = block.get("toolUse")
                    if tool_use is None:
                        continue

                    result = await graph_tools.execute(tool_use["name"], tool_use.get("input", {}))
                    tool_results.append(
                        {
                            "toolResult": {
                                "toolUseId": tool_use["toolUseId"],
                                "content": [{"text": json.dumps(result)}],
                            }
                        }
                    )

                messages.append({"role": "user", "content": tool_results})
                continue

            text = response["output"]["message"]["content"][0]["text"]
            return json.loads(text)

        raise RuntimeError("Max Bedrock tool rounds exceeded")

    def _format_tools(self, primitives: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for primitive in primitives:
            parameters = primitive["parameters"]
            required = [name for name, value in parameters.items() if value.get("required")]
            properties = {
                name: {k: v for k, v in definition.items() if k != "required"}
                for name, definition in parameters.items()
            }
            tools.append(
                {
                    "toolSpec": {
                        "name": primitive["name"],
                        "description": primitive["description"],
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": properties,
                                "required": required,
                            }
                        },
                    }
                }
            )
        return tools
