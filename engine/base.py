__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
import uuid
from typing import List, Literal, Optional

# Third-party imports
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, END, MessagesState, StateGraph


class State(MessagesState):
    summary: str


class BaseEngine:
    def __init__(
            self,
            model_cls, model_name: str, model_kwargs: dict,
            tools: Optional[List[BaseTool]] = None,
    ):
        # 추후 user id로 변경 가능
        self.thread_id = uuid.uuid4()
        self.config = {"configurable": {"thread_id": self.thread_id}}

        self.workflow = StateGraph(state_schema=MessagesState)
        self.model_name = model_name
        if tools:
            self.llm = model_cls(model=model_name, **model_kwargs).bind_tools(tools)
        else:
            tools = []
            self.llm = model_cls(model=model_name, **model_kwargs)
        tool_node = ToolNode(tools)

        self.memory = MemorySaver()

        # Set Node
        self.workflow.add_node("agent", self.call_model)
        self.workflow.add_node("summarize_conversation", self.summarize_conversation)
        self.workflow.add_node("tools", tool_node)
        # Set Edge
        self.workflow.add_edge(START, "agent")
        self.workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            ["summarize_conversation", "tools", END]
        )
        self.workflow.add_edge("summarize_conversation", END)
        self.workflow.add_edge("tools", "agent")

        self.app = self.workflow.compile(checkpointer=self.memory)
        print("--- LangGraph Flow ---")
        print(self.app.get_graph().draw_ascii())
        print("----------------------")

    # LanGraph Logic Start
    def call_model(self, state: State):
        # 요약이 있다면, 시스템 메시지로 추가한다.
        summary = state.get("summary", "")

        if summary:
            system_message = f"Summary of conversation earlier: {summary}"
            messages = [SystemMessage(content=system_message)] + state["messages"]
        else:
            messages = state["messages"]
        response = self.llm.invoke(messages)

        return {"messages": [response]}

    def should_continue(
            self, state: State
    ) -> Literal["summarize_conversation", "tools", END]:
        """Return the next node to execute."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        elif len(messages) > 10:
            return "summarize_conversation"
        return END

    def summarize_conversation(self, state: State):
        # 우선, 대화를 요약한다.
        summary = state.get("summary", "")
        if summary:
            # 이미 요약이 있다면, 요약하기 위해 다른 시스템 프롬프트를 사용한다.
            summary_message = (
                f"This is summary of the conversation to date: {summary}\n\n"
                "Extend the summary by taking into account the new messages above:"
            )
        else:
            summary_message = "Create a summary of the conversation above in Korean:"

        messages = state["messages"] + [HumanMessage(content=summary_message)]
        response = self.llm.invoke(messages)
        # 더 이상 표시하고 싶지 않은 메시지를 삭제해야 한다.
        # 지난 두 메시지를 제외한 모든 메시지를 삭제할 것이다. 하지만 변경할 수도 있다.
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
        return {"summary": response.content, "messages": delete_messages}
    # LanGraph Logic End

    def update_tools(self, tools: List[BaseTool]):
        if "tools" in self.workflow.nodes:
            del self.workflow.nodes["tools"]

        # LLM에 새 툴 바인딩
        self.llm = self.llm.bind_tools(tools)

        # tools 노드 재등록
        tool_node = ToolNode(tools)
        self.workflow.add_node("tools", tool_node)

        # 재컴파일 필요 (엣지는 유지됨)
        self.app = self.workflow.compile(checkpointer=self.memory)
        print("[*] Tools updated and LangGraph recompiled.")
        print(self.app.get_graph().draw_ascii())

    async def generate(self, prompt: str) -> str:
        """ Basic generation method without LangChain and tools """

        response = await self.llm.ainvoke(prompt)
        return response.content

    async def generate_langchain(self, prompt: str) -> str:
        """ Generation using LangChain without tool integration """

        try:
            input_message = HumanMessage(content=prompt)
            response = await self.app.ainvoke(
                {"messages": [input_message]},
                self.config
            )
            ai_messages = self.parse_ai_messages(response['messages'])

            if ai_messages and isinstance(ai_messages[-1], dict):
                return ai_messages[-1].get("content", "")
            return ""
        except Exception as e:
            return str(e)

    # TODO: WIP
    def save_conversation(self):
        print()

    # TODO: WIP
    def clear_memory(self):
        checkpoints = list(self.app.get_state_history(self.config))
        if len(checkpoints) > 1:
            temp1 = self.app.get_state_history(self.config)
            previous_state = checkpoints[1].values  # 두 번째로 최근 상태
            self.app.update_state(self.config, previous_state)
            temp2 = self.app.get_state_history(self.config)
            print()

    @staticmethod
    def parse_ai_messages(messages: List) -> List[dict]:
        return [msg.__dict__ for msg in messages if isinstance(msg, AIMessage)]
