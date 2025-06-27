# TODO: WIP
__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
from typing import List, Dict, Any

# Third-party imports
from openai import OpenAI
from mcp import ClientSession

# Custom imports
from .base import BaseEngine


class OpenAIEngine(BaseEngine):
    def __init__(self):
        self.model = OpenAI()
        self.model_name = "gpt-4.1"

    def _build_user_prompt(self, prompt: str) -> List[Dict[str, str]]:
        return [{"role": "user", "content": prompt}]

    def _wrap_mcp_tools(self, mcp_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool.get("input_schema", {})
                }
            }
            for tool in mcp_tools
        ]

    async def generate_without_tools(self, prompt: str) -> str:
        messages = self._build_user_prompt(prompt)
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
        content = response.choices[0].message.content or ""
        return content.strip()

    async def generate_with_tools(
        self,
        prompt: str,
        session: ClientSession,
        tools: List[Dict[str, Any]]
    ) -> str:

        messages = self._build_user_prompt(prompt)
        result = []
        wrapped_tools = self._wrap_mcp_tools(tools)

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=wrapped_tools
        )

        content = response.choices[0].message.content or ""
        if content:
            result.append(content.strip())

        tool_calls = getattr(response.choices[0].message, "tool_calls", None)
        if tool_calls:
            for tool in tool_calls:
                tool_result = await session.call_tool(tool.function.name, tool.function.arguments)
                messages += [
                    {"role": "assistant", "tool_calls": [tool]},
                    {"role": "user", "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool.id,
                        "content": tool_result.content
                    }]}
                ]

                followup = self.model.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=wrapped_tools
                )
                fcontent = followup.choices[0].message.content or ""
                if fcontent:
                    result.append(fcontent.strip())

        return "\n".join(result)
