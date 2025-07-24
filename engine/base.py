__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
import os
import uuid
import json
from typing import List, Literal, Optional
from pathlib import Path

# Third-party imports
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import START, END, MessagesState, StateGraph

# Custom imports
from schemas import ChatHistory


class State(MessagesState):
    summary: str


class BaseEngine:
    def __init__(
        self,
        model_cls, model_name: str, model_kwargs: dict,
        tools: Optional[List[BaseTool]] = None,
    ):
        self.thread_id = uuid.uuid4()
        self.config = {"configurable": {"thread_id": self.thread_id}}
        self.model_name = model_name

        self.workflow = None
        if tools:
            self.llm = model_cls(model=model_name, **model_kwargs).bind_tools(tools)
        else:
            tools = []
            self.llm = model_cls(model=model_name, **model_kwargs)
        self.memory = MemorySaver()
        self.app = self.build_graph(tools)

    def build_graph(self, tools: Optional[List[BaseTool]] = None) -> CompiledStateGraph:
        self.workflow = StateGraph(state_schema=State)
        tool_node = ToolNode(tools)

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

        app = self.workflow.compile(checkpointer=self.memory)
        print("--- LangGraph Flow ---")
        print(app.get_graph().draw_ascii())
        print("----------------------")

        return app

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
            self.save_chat_history()

            ai_messages = self.parse_ai_messages(response['messages'])
            if ai_messages and isinstance(ai_messages[-1], dict):
                return ai_messages[-1].get("content", "")
            return ""
        except Exception as e:
            return str(e)

    async def stream_generate(self, prompt: str):
        """ Streaming generation method without LangChain and tools """
        async for chunk in self.llm.astream(prompt):
            yield None, chunk.content

        self.save_chat_history()

    async def stream_generate_langchain(self, prompt: str):
        input_message = HumanMessage(content=prompt)

        async for chunk in self.app.astream(
                {"messages": [input_message]},
                self.config,
                stream_mode="messages"
        ):
            content_type = chunk[0].type
            content = chunk[0].content
            yield content_type, content

        self.save_chat_history()

    def save_chat_history(self, save_dir: str = "chat-data"):
        os.makedirs(save_dir, exist_ok=True)

        # 현재 스레드 기준 상태 이력 조회
        snapshot = self.app.get_state(self.config)
        if not snapshot:
            print("[!] 저장할 대화 기록이 없습니다.")
            return

        chat_entry = ChatHistory(
            thread_id=str(self.thread_id),
            values=snapshot.values,
            next=snapshot.next,
            config=self.config,
            metadata=snapshot.metadata,
            created_at=snapshot.created_at,
            parent_config=snapshot.parent_config,
        )

        filepath = Path(save_dir) / f"chat_{self.thread_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(chat_entry.model_dump_json(indent=2))

        print(f"[✔] 대화 기록이 '{filepath}' 에 저장되었습니다.")

    def merge_chat_history(self, thread_id: str, save_dir: str = "chat-data"):
        """
        Load a saved chat history and merge its messages with the current session.

        This method retrieves a chat record from a JSON file, extracts the message history,
        and appends it in front of the current session's messages. It keeps the current
        configuration and application state intact, updating only the message history.

        Parameters:
            thread_id (str): Unique identifier of the chat history to load.
            save_dir (str): Directory path where chat history JSON files are stored. Default is "chat-data".
        """
        filepath = Path(save_dir) / f"chat_{thread_id}.json"
        if not filepath.exists():
            print(f"[!] Chat history file does not exist: {filepath}")
            return

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                record = ChatHistory(**data)
                print(f"[✔] Chat history successfully loaded: {filepath}")

                loaded_messages = record.values.get("messages", [])
                current_messages = self.app.get_state(self.config).values.get("messages", [])
                merged_messages = loaded_messages + current_messages

                self.memory = MemorySaver()
                self.app = self.workflow.compile(checkpointer=self.memory)
                self.app.update_state(self.config, {"messages": merged_messages})

        except Exception as e:
            print(f"[!] Failed to load chat history: {e}")

    def load_chat_history(self, thread_id: str, save_dir: str = "chat-data"):
        """
        Load a saved chat history and completely replace the current session state.

        This method loads a full chat session including the thread ID, configuration, and
        conversation messages from a JSON file, and resets the current application state
        to match the loaded session.

        Parameters:
            thread_id (str): Unique identifier of the chat history to load.
            save_dir (str): Directory path where chat history JSON files are stored. Default is "chat-data".
        """
        filepath = Path(save_dir) / f"chat_{thread_id}.json"
        if not filepath.exists():
            print(f"[!] Chat history file does not exist: {filepath}")
            return

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                record = ChatHistory(**data)
                print(f"[✔] Chat history successfully loaded: {filepath}")

                self.thread_id = record.thread_id
                self.config = record.config

                self.memory = MemorySaver()
                self.app = self.workflow.compile(checkpointer=self.memory)
                self.app.update_state(self.config, record.values)

        except Exception as e:
            print(f"[!] Failed to load chat history: {e}")

    # TODO: implement HERE (go-back-to-previous-step)
    def clear_memory(self):
        checkpoints = list(self.app.get_state_history(self.config))
        if len(checkpoints) > 1:
            temp1 = list(self.app.get_state_history(self.config))
            previous_state = checkpoints[1].values
            self.app.update_state(self.config, previous_state)
            temp2 = list(self.app.get_state_history(self.config))

    @staticmethod
    def parse_ai_messages(messages: List) -> List[dict]:
        return [msg.__dict__ for msg in messages if isinstance(msg, AIMessage)]
