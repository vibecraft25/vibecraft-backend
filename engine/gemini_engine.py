__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
from typing import List

# Third-party imports
from google import genai
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.base import ConversationChain
from mcp import ClientSession, ClientSessionGroup


# Custom imports
from .base import BaseEngine


class GeminiEngine(BaseEngine):
    def __init__(self):
        self.model = genai.Client()
        self.model_name = "gemini-2.5-flash"
        self.llm = ChatGoogleGenerativeAI(model=self.model_name, temperature=0)
        self.memory = ConversationBufferMemory(return_messages=True)

        # TODO: system prompt 세팅 방식 추가 WIP
        self.system = ChatPromptTemplate([
            ('system', '100자 이내로 한국어로 답변해주세요.'),
            ('user', 'temp')
        ])

        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True
        )

    def _wrap_mcp_tools(self, mcp_tool_specs: List[dict]) -> List[types.Tool]:
        return [
            types.Tool(function_declarations=[
                types.FunctionDeclaration(
                    name=tool["name"],
                    description=tool["description"],
                    parameters=tool.get("input_schema", {})
                )
            ])
            for tool in mcp_tool_specs
        ]

    def _build_config(self, tools: List[dict] = None) -> types.GenerateContentConfig:
        if tools:
            return types.GenerateContentConfig(tools=self._wrap_mcp_tools(tools))
        return types.GenerateContentConfig()

    def _parse_response_parts(self, parts: List) -> List[str]:
        result = []
        for part in parts:
            if hasattr(part, "text") and part.text:
                result.append(part.text)
        return result

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

    async def generate_langchain(self, prompt: str) -> str:
        response = self.conversation.invoke(input=prompt)
        return response['response']

    async def generate_langchain_with_tools(
            self,
            prompt: str,
            session: ClientSession | ClientSessionGroup,
            tools: List[dict]
    ) -> str:

        config = self._build_config(tools)
        # 초기 응답 요청
        response = self.model.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )

        result = []
        result_str = None
        candidate = response.candidates[0]

        # 각 파트 처리
        for part in candidate.content.parts:
            if hasattr(part, "text") and part.text:
                pass
            elif hasattr(part, "function_call"):
                func_call = part.function_call
                tool_result = await session.call_tool(func_call.name, func_call.args)
                result.append(f"[{func_call.name} Result]: {tool_result.content}, "
                              f"args: ({func_call.args})")
                print(f"[Function Call]: {func_call.name}({func_call.args})")
                print(f"[Function Result]: {tool_result.content}")

        if result:
            # 5. 후속 프롬프트 구성
            result_str = '\n'.join(result)
            prompt = (
                f"{result_str}\n"
                f"---\n"
                f"이 결과를 바탕으로 다음 응답을 생성해 주세요.\n"
                f"---\n"
                f"{prompt}"
            ).strip()

        # LLM 호출 (langchain invoke)
        response = self.conversation.invoke(prompt)
        llm_output = response['response']
        if result_str:
            return f"{result_str}\n{llm_output}"
        return llm_output
