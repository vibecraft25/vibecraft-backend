__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Third-party imports
from openai import OpenAI
from mcp import ClientSession

# Custom imports
from .base import BaseEngine
from utils.flags import parse_flags_from_text


class OpenAIEngine(BaseEngine):
    def __init__(self):
        self.openai = OpenAI()

    async def generate_with_tools(self, prompt, tools, session: ClientSession):
        messages = [{"role": "user", "content": prompt}]
        result = []
        redo = go_back = False

        # 첫 호출
        response = self.openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=1000,
        )

        while True:
            choice = response.choices[0].message
            if choice.content:
                result.append(choice.content)
                redo, go_back = parse_flags_from_text(choice.content)
                break

            if choice.tool_calls:
                for tool_call in choice.tool_calls:
                    tool_result = await session.call_tool(tool_call.function.name, tool_call.function.arguments)
                    messages.extend([
                        {"role": "assistant", "tool_calls": [tool_call]},
                        {"role": "tool", "tool_call_id": tool_call.id, "content": tool_result.content}
                    ])

                # 도구 호출 후 후속 메시지 생성
                response = self.openai.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=1000,
                )
            else:
                break

        return "\n".join(result), redo, go_back
