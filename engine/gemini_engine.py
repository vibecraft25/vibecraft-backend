__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Third-party imports
from google import genai
from mcp import ClientSession

# Custom imports
from .base import BaseEngine
from utils.flags import parse_flags_from_text


class GeminiEngine(BaseEngine):
    def __init__(self):
        # google-genai는 일반적으로 'gemini-1.5-pro-latest'와 같은 모델 이름을 사용합니다.
        # 또는 특정 버전을 명시할 수도 있습니다 (예: 'gemini-1.5-pro-001').
        self.model = genai.Client()

    async def generate_with_tools(self, prompt, tools, session: ClientSession):
        messages = [{"role": "user", "content": prompt}]
        tools = await session.list_tools()
        available_tools = [{
            "name": t.name,
            "description": t.description,
            "input_schema": t.inputSchema
        } for t in tools.tools]

        response = await self.model.aio.models.generate_content(
            model="gemini-2.0-flash",
            contents="Roll 3 dice!",
            config=genai.types.GenerateContentConfig(
                temperature=0,
                tools=[session],  # Pass the FastMCP client session
            ),
        )
        print(response.text)


        chat_session = self.model.start_chat()

        # tools를 google-genai 모델 형식에 맞게 변환
        # google-genai는 anthropic과 유사하게 'tools' 리스트에 'function_declarations' 딕셔너리를 포함합니다.
        # 각 도구는 'function_declarations'의 요소로 'name', 'description', 'parameters'를 가집니다.
        gemini_tools = [
            genai.types.FunctionDeclaration(  # FunctionDeclaration 객체 사용
                name=tool["function"]["name"],
                description=tool["function"]["description"],
                parameters=tool["function"]["parameters"]
            ) for tool in tools
        ]

        result = []
        redo = go_back = False

        # 첫 호출
        response = chat_session.send_message(prompt, tools=gemini_tools)

        while True:
            # google-genai의 응답 처리 방식은 다소 다릅니다.
            # response.candidates[0].content.parts를 통해 콘텐츠 파트를 순회합니다.
            parts = response.candidates[0].content.parts

            tool_calls_made_in_this_turn = False
            for part in parts:
                if part.text:
                    result.append(part.text)
                    redo, go_back = parse_flags_from_text(part.text)
                    # 텍스트가 있으면 결과 반환 및 종료 (Gemini는 도구 호출 후 최종 텍스트 응답을 바로 제공할 수 있음)
                    return "\n".join(result), redo, go_back

                elif part.function_call:
                    tool_calls_made_in_this_turn = True
                    tool_name = part.function_call.name
                    tool_args = part.function_call.args

                    tool_result_content = await session.call_tool(tool_name, tool_args)

                    # 도구 호출 결과를 메시지에 추가하고 다시 모델 호출
                    # google-genai에서는 ToolCodePart 객체를 사용하여 도구 결과를 전달합니다.
                    response = chat_session.send_message(
                        genai.types.ToolCodePart(  # ToolCodePart 객체 사용
                            name=tool_name,
                            result=tool_result_content.content
                        ),
                        tools=gemini_tools  # 재호출 시에도 도구 정보를 함께 전달합니다.
                    )

            # 현재 턴에서 도구 호출이 없었다면 루프 종료
            if not tool_calls_made_in_this_turn:
                break

        return "\n".join(result), redo, go_back
