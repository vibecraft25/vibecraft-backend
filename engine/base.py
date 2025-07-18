__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
import uuid
from typing import List

# Third-party imports
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.graph import START, END, MessagesState, StateGraph


class BaseEngine:
    def __init__(self, model_cls, model_name: str, model_kwargs: dict):
        # 추후 user id로 변경 가능
        self.thread_id = uuid.uuid4()
        self.config = {"configurable": {"thread_id": self.thread_id}}

        self.workflow = StateGraph(state_schema=MessagesState)

        self.model_name = model_name
        self.llm = model_cls(model=model_name, **model_kwargs)
        self.memory = MemorySaver()

        # 1. Set START
        self.workflow.add_edge(START, "model")
        # 2. Set Node
        self.workflow.add_node("model", self.call_model)
        # 3. Set END
        self.workflow.add_edge("model", END)

        self.app = self.workflow.compile(checkpointer=self.memory)
        print("--- LangGraph Flow ---")
        print(self.app.get_graph().draw_ascii())
        print("----------------------")

    def call_model(self, state: MessagesState) -> dict:
        response = self.llm.invoke(state["messages"])
        # We return a list, because this will get added to the existing list
        return {"messages": response}

    async def generate(self, prompt: str) -> str:
        """ Basic generation method without LangChain and tools """

        response = await self.llm.ainvoke(prompt)
        return response.content

    async def generate_langchain(self, prompt: str) -> str:
        """ Generation using LangChain without tool integration """

        try:
            input_message = HumanMessage(content=prompt)
            response = await self.app.ainvoke({"messages": [input_message]}, self.config)
            ai_messages = self.parse_ai_messages(response['messages'])

            if ai_messages and isinstance(ai_messages[-1], dict):
                return ai_messages[-1].get("content", "")
            return ""
        except Exception as e:
            return str(e)

    async def generate_langchain_with_tools(
            self, prompt: str, tools: List[BaseTool]
    ) -> str:
        """ Generation using LangChain with external tools integration (via MCP) """

        try:
            input_message = HumanMessage(content=prompt)
            agent = create_react_agent(self.llm, tools=tools, checkpointer=self.memory)
            response = await agent.ainvoke({"messages": [input_message]}, self.config)
            ai_messages = self.parse_ai_messages(response['messages'])

            if ai_messages and isinstance(ai_messages[-1], dict):
                return ai_messages[-1].get("content", "")
            return ""
        except Exception as e:
            return str(e)

    @staticmethod
    def parse_ai_messages(messages: List) -> List[dict]:
        return [msg.__dict__ for msg in messages if isinstance(msg, AIMessage)]
