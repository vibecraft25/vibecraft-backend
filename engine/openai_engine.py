__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
from typing import List, Dict, Any

# Third-party imports
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.base import ConversationChain
from mcp import ClientSession, ClientSessionGroup

# Custom imports
from .base import BaseEngine


class OpenAIEngine(BaseEngine):
    def __init__(self):
        self.model = OpenAI()
        self.model_name = "gpt-4.1"
        self.llm = ChatOpenAI(model=self.model_name, temperature=0)
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

    async def generate_langchain(self, prompt: str) -> str:
        response = self.conversation.invoke(input=prompt)
        return response['response']

    async def generate_with_tools(
        self,
        prompt: str,
        session: ClientSession | ClientSessionGroup,
        tools: List[Dict[str, Any]]
    ) -> str:

        messages = self._build_user_prompt(prompt)
        wrapped_tools = self._wrap_mcp_tools(tools)
        # 초기 응답 요청
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=wrapped_tools
        )

        result = []
        result_str = None

        tool_calls = getattr(response.choices[0].message, "tool_calls", None)
        if tool_calls:
            for tool in tool_calls:
                func_call = tool.function
                tool_result = await session.call_tool(func_call.name, func_call.arguments)
                result.append(f"[{func_call.name} Result]: {tool_result.content}, "
                              f"args: ({func_call.arguments})")
                print(f"[Tool Call] {func_call.name}({func_call.arguments})")
                print(f"[Tool Result] {tool_result.content}")

                # # 메시지에 tool 호출과 결과 추가
                # messages += [
                #     {"role": "assistant", "tool_calls": [tool]},
                #     {"role": "user", "content": [{
                #         "type": "tool_result",
                #         "tool_use_id": tool.id,
                #         "content": tool_result.content
                #     }]}
                # ]
                # # 후속 응답 요청
                # followup = self.model.chat.completions.create(
                #     model=self.model_name,
                #     messages=messages,
                #     tools=wrapped_tools
                # )
                #
                # # 후속 응답에서 텍스트만 정제하여 추가
                # fcontent = followup.choices[0].message.content or ""
                # if fcontent:
                #     result.append(fcontent.strip())

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
