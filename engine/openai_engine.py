__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
from typing import List, Dict, Any

# Third-party imports
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition
from mcp import ClientSession, ClientSessionGroup

# Custom imports
from .base import BaseEngine


class OpenAIEngine(BaseEngine):
    def __init__(self):
        self.model = OpenAI()
        self.model_name = "gpt-4.1"

    def _build_user_prompt(self, prompt: str) -> List[ChatCompletionUserMessageParam]:
        return [{"role": "user", "content": prompt}]

    def _wrap_mcp_tools(self, mcp_tools: List[Dict[str, Any]]) -> List[ChatCompletionToolParam]:
        return [
            ChatCompletionToolParam(
                type="function",
                function=FunctionDefinition(
                    name=tool["name"],
                    description=tool["description"],
                    parameters=tool.get("input_schema", {})
                )
            )
            for tool in mcp_tools
        ]

    async def generate(self, prompt: str) -> str:
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
        session: ClientSession | ClientSessionGroup,
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
                # 실제 함수 실행
                tool_result = await session.call_tool(tool.function.name, tool.function.arguments)
                print(f"[Tool Call] {tool.function.name}({tool.function.arguments})")
                print(f"[Tool Result] {tool_result.content}")

                # 메시지에 tool 호출과 결과 추가
                messages += [
                    {"role": "assistant", "tool_calls": [tool]},
                    {"role": "user", "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool.id,
                        "content": tool_result.content
                    }]}
                ]

                # 후속 응답 요청
                followup = self.model.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=wrapped_tools
                )

                # 후속 응답에서 텍스트만 정제하여 추가
                fcontent = followup.choices[0].message.content or ""
                if fcontent:
                    result.append(fcontent.strip())

        return "\n".join(result)
