__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
from typing import List

# Third-party imports
from google import genai
from google.genai import types
from mcp import ClientSession

# Custom imports
from .base import BaseEngine


class GeminiEngine(BaseEngine):
    def __init__(self):
        self.model = genai.Client()
        self.model_name = "gemini-2.5-flash"

    def _wrap_mcp_tools(self, mcp_tool_specs: List) -> List[types.Tool]:
        wrapped_tools = []
        for spec in mcp_tool_specs:
            # spec이 ('tools', [Tool, Tool, ...]) 형태일 경우
            if isinstance(spec, tuple) and spec[0] == "tools" and isinstance(spec[1], list):
                for tool in spec[1]:
                    wrapped_tools.append(
                        types.Tool(function_declarations=[
                            types.FunctionDeclaration(
                                name=tool.name,
                                description=tool.description,
                                parameters=getattr(tool, "inputSchema", {}) or getattr(tool, "input_schema", {})
                            )
                        ])
                    )
        return wrapped_tools

    def _parse_response_parts(self, parts: List) -> List[str]:
        result = []
        for part in parts:
            if hasattr(part, "text") and part.text:
                result.append(part.text)
        return result

    def _build_config(self, tools: List[dict] = None) -> types.GenerateContentConfig:
        if tools:
            return types.GenerateContentConfig(tools=self._wrap_mcp_tools(tools))
        return types.GenerateContentConfig()

    async def generate(self, prompt: str) -> str:
        config = self._build_config()
        response = self.model.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )
        candidate = response.candidates[0]
        result = self._parse_response_parts(candidate.content.parts)
        return "\n".join(result)

    async def generate_with_tools(
        self,
        prompt: str,
        session: ClientSession,
        tools: List[dict]
    ) -> str:

        config = self._build_config(tools)
        response = self.model.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )

        result = []
        candidate = response.candidates[0]

        for part in candidate.content.parts:
            if hasattr(part, "text") and part.text:
                result.append(part.text)

            elif hasattr(part, "function_call"):
                func_call = part.function_call
                tool_result = await session.call_tool(func_call.name, func_call.args)
                result.append(f"[Function Call: {func_call.name}]")
                result.append(str(tool_result.content))

        return "\n".join(result)
