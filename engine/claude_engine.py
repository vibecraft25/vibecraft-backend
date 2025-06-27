__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Third-party imports
from anthropic import Anthropic
from mcp import ClientSession

# Custom imports
from .base import BaseEngine
from utils.flags import parse_flags_from_text


class ClaudeEngine(BaseEngine):
    def __init__(self):
        self.anthropic = Anthropic()

    async def generate_with_tools(self, prompt, tools, session: ClientSession):
        messages = [{"role": "user", "content": prompt}]
        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages,
            tools=tools
        )

        result = []
        redo = go_back = False

        for item in response.content:
            if item.type == "text":
                result.append(item.text)
                redo, go_back = parse_flags_from_text(item.text)

            elif item.type == "tool_use":
                tool_result = await session.call_tool(item.name, item.input)
                messages.extend([
                    {"role": "assistant", "content": [item]},
                    {"role": "user", "content": [{
                        "type": "tool_result",
                        "tool_use_id": item.id,
                        "content": tool_result.content
                    }]}
                ])
                followup = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=messages,
                    tools=tools
                )
                for fitem in followup.content:
                    if fitem.type == "text":
                        result.append(fitem.text)
                        redo, go_back = parse_flags_from_text(fitem.text)

        return "\n".join(result), redo, go_back
