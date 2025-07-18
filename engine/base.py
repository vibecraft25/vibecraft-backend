__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
from typing import List

# Third-party imports
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.base import ConversationChain
from langgraph.prebuilt import create_react_agent


class BaseEngine:
    def __init__(self, model_cls, model_name: str, model_kwargs: dict):
        self.model_name = model_name
        self.llm = model_cls(model=model_name, **model_kwargs)
        self.memory = ConversationBufferMemory(return_messages=True)

        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True
        )

    """ Basic generation method without LangChain and tools """
    async def generate(self, prompt: str) -> str:
        response = await self.llm.ainvoke(prompt)
        return response.content

    """ Generation using LangChain without tool integration """
    async def generate_langchain(self, prompt: str) -> str:
        response = await self.conversation.ainvoke(input=prompt)
        return response['response']

    """ Generation using LangChain with external tools integration (via MCP) """
    async def generate_langchain_with_tools(
        self, prompt: str, tools: List[BaseTool]
    ) -> str:
        agent = create_react_agent(self.llm, tools)
        response = await agent.ainvoke({"messages": prompt})
        ai_messages = self.parse_ai_messages(response['messages'])

        if ai_messages and isinstance(ai_messages[-1], dict):
            self.memory.save_context(
                {"input": prompt},
                {"output": ai_messages}
            )
            return ai_messages[-1].get("content", "")
        return ""

    @staticmethod
    def parse_ai_messages(messages: List) -> List[dict]:
        return [msg.__dict__ for msg in messages if isinstance(msg, AIMessage)]
