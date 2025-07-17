__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
from typing import List

# Third-party imports
from anthropic import Anthropic
from anthropic.types import MessageParam, ContentBlock
from mcp import ClientSession, ClientSessionGroup

# Custom imports
from .base import BaseEngine


class ClaudeEngine(BaseEngine):
    def __init__(self):
        self.model = Anthropic()
        self.model_name = "claude-3-5-sonnet-20241022"
        self.max_tokens = 1000

    def _build_user_prompt(self, prompt: str) -> List[MessageParam]:
        """Claude 전용 사용자 프롬프트 생성"""
        return [
            MessageParam(
                role="user",
                content=[ContentBlock(type="text", text=prompt)]
            )
        ]

    def _parse_response(self, content: List) -> List[str]:
        results = []
        for item in content:
            if item.type == "text":
                results.append(item.text)
        return results

    async def generate(self, prompt: str) -> str:
        messages = self._build_user_prompt(prompt)
        response = self.model.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            messages=messages
        )
        result = self._parse_response(response.content)
        return "\n".join(result)

    async def generate_with_tools(
        self,
        prompt: str,
        session: ClientSession | ClientSessionGroup,
        tool_specs: List
    ) -> str:

        messages = self._build_user_prompt(prompt)
        result = []

        response = self.model.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            messages=messages,
            tools=tool_specs
        )

        for item in response.content:
            if item.type == "text":
                result.append(item.text)

            elif item.type == "tool_use":
                input_data = item.input
                if not isinstance(input_data, dict):
                    input_data = input_data.to_dict() if hasattr(input_data, "to_dict") else {}
                tool_result = await session.call_tool(item.name, input_data)
                print(f"[Tool Call] {item.name}({item.input}) -> {tool_result.content}")
                print(f"[Tool Result] {tool_result.content}")

                # messages에 tool 호출 및 결과 추가
                messages += [
                    {"role": "assistant", "content": [item]},
                    {
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": item.id,
                            "content": tool_result.content
                        }]
                    }
                ]
                # 후속 응답 요청
                followup = self.model.messages.create(
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    messages=messages,
                    tools=tool_specs
                )
                # 후속 응답의 텍스트 부분만 파싱해서 결과에 추가
                fresult = self._parse_response(followup.content)
                result.extend(fresult)

        return "\n".join(result)
