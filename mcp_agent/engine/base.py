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
from mcp_agent.schemas import ChatHistory
from config import settings


class State(MessagesState):
    summary: str
    should_summarize: bool = False  # 요약 트리거 플래그 추가


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

    """LanGraph Logic"""
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

        # 1. 수동 요약 트리거 확인 (최우선)
        # 2. 메세지 특정 개수 이상일 때도 요약 진행
        if (state.get("should_summarize", False)
                or self.check_should_summarize(10)):
            return "summarize_conversation"

        # 도구 호출이 있는 경우
        if last_message.tool_calls:
            return "tools"

        return END

    def summarize_conversation(self, state: State):
        # 다른 에이전트에게 전달할 목적의 요약 생성
        summary = state.get("summary", "")

        if summary:
            # 이미 요약이 있다면, 새로운 대화 내용을 포함하여 업데이트
            summary_message = f"""
                기존 요약: {summary}

                위 요약에 최근 대화 내용을 추가하여 다른 AI 에이전트에게 전달할 요약을 업데이트해주세요:

                업데이트된 요약에 포함할 내용:
                1. 사용자의 전체적인 요청사항과 목표
                2. 모든 핵심 주제와 결정사항
                3. 제공된 중요한 데이터나 정보
                4. 현재까지의 진행 상황과 다음 단계
                5. 기술적 요구사항이나 제약사항

                한국어로 명확하고 구조화된 형태로 작성해주세요.
                """
        else:
            # 처음 요약하는 경우
            summary_message = """
                    다음 대화 내용을 다른 AI 에이전트에게 전달하기 위해 요약해주세요:

                    요약에 포함해야 할 내용:
                    1. 사용자의 주요 요청사항과 목표
                    2. 논의된 핵심 주제와 결정사항  
                    3. 제공된 중요한 데이터나 정보
                    4. 현재 진행 상황과 다음 단계
                    5. 기술적 요구사항이나 제약사항 (있는 경우)

                    요약 형식:
                    - 한국어로 명확하고 구조화된 형태로 작성
                    - 다른 에이전트가 맥락을 완전히 이해할 수 있도록 충분한 세부사항 포함
                    - 단순한 대화 흐름보다는 실질적이고 actionable한 내용에 집중
                    - 중요한 데이터나 파일 경로, 설정값 등은 정확히 포함
                    """

        messages = state["messages"] + [HumanMessage(content=summary_message)]
        response = self.llm.invoke(messages)

        # 더 이상 표시하고 싶지 않은 메시지를 삭제해야 한다.
        # 지난 두 메시지를 제외한 모든 메시지를 삭제할 것이다. 하지만 변경할 수도 있다.
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]

        return {
            "summary": response.content,
            "messages": delete_messages,
            "should_summarize": False  # 요약 완료 후 플래그 리셋
        }

    """수동 요약 트리거 메서드들"""
    def trigger_summarize(self):
        """
        대화 요약을 수동으로 트리거합니다.
        """
        prompt = """
            다음 대화 내용을 다른 AI 에이전트에게 전달하기 위해 요약해주세요:

            요약 시 포함해야 할 내용:
            1. 사용자의 주요 요청사항과 목표
            2. 논의된 핵심 주제와 결정사항
            3. 제공된 중요한 데이터나 정보
            4. 현재 진행 상황과 다음 단계
            5. 기술적 요구사항이나 제약사항 (있는 경우)

            요약 형식:
            - 명확하고 구조화된 형태로 작성
            - 다른 에이전트가 맥락을 이해할 수 있도록 충분한 세부사항 포함
            - 불필요한 대화의 흐름보다는 실질적인 내용에 집중
        """

        input_message = HumanMessage(content=prompt)
        # 메시지 추가와 함께 요약 트리거
        response = self.app.invoke(
            {"messages": [input_message], "should_summarize": True},
            self.config
        )

        self.save_chat_history()
        return response

    async def trigger_summarize_async(self):
        """
        대화 요약을 비동기로 수동 트리거합니다.
        요약 프롬프트는 chat_history에 저장되지 않고 summary에만 반영됩니다.
        """
        # 현재 상태에서 요약 트리거 (메시지 히스토리에 추가하지 않음)
        current_state = self.app.get_state(self.config)
        self.app.update_state(self.config, {"should_summarize": True})

        response = await self.app.ainvoke({"messages": []}, self.config)

        self.save_chat_history()
        return response

    def check_should_summarize(self, message_count_threshold: int = 10) -> bool:
        """
        요약이 필요한지 확인합니다.

        Args:
            message_count_threshold (int): 요약을 권장하는 메시지 개수 임계값

        Returns:
            bool: 요약이 권장되는지 여부
        """
        current_state = self.app.get_state(self.config)
        if not current_state:
            return False

        messages = current_state.values.get("messages", [])
        return len(messages) >= message_count_threshold

    def get_conversation_stats(self) -> dict:
        """
        현재 대화의 통계 정보를 반환합니다.

        Returns:
            dict: 메시지 수, 요약 존재 여부 등의 정보
        """
        current_state = self.app.get_state(self.config)
        if not current_state:
            return {"message_count": 0, "has_summary": False}

        messages = current_state.values.get("messages", [])
        summary = current_state.values.get("summary", "")

        return {
            "message_count": len(messages),
            "has_summary": bool(summary),
            "summary_preview": summary[:50] + "..." if len(summary) > 50 else summary,
            "summary": summary,
            "should_summarize_recommended": self.check_should_summarize()
        }

    """Tool method"""
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

    """LLM Response methods"""
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

    """Chat history methods"""
    def get_chat_history(self) -> Optional[ChatHistory]:
        snapshot = self.app.get_state(self.config)
        if not snapshot:
            print("[!] 저장할 대화 기록이 없습니다.")
            return None

        return ChatHistory(
            thread_id=str(self.thread_id),
            values=snapshot.values,
            next=snapshot.next,
            config=self.config,
            metadata=snapshot.metadata,
            created_at=snapshot.created_at,
            parent_config=snapshot.parent_config,
        )

    def save_chat_history(self):
        os.makedirs(settings.chat_path, exist_ok=True)
        chat_entry = self.get_chat_history()

        if chat_entry:
            filepath = Path(settings.chat_path) / f"chat_{self.thread_id}.json"
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(chat_entry.model_dump_json(indent=2))

            print(f"[✔] 대화 기록이 '{filepath}' 에 저장되었습니다.")
        else:
            print(f"[!] 저장할 대화 기록이 없습니다.")

    @staticmethod
    def load_chat_history_file(thread_id: str) -> ChatHistory | None:
        """
        Load chat history from a JSON file.

        Parameters:
            thread_id (str): Unique identifier of the chat history to load.

        Returns:
            ChatHistory | None: Loaded chat history object or None if loading fails.
        """
        filepath = Path(settings.chat_path) / f"chat_{thread_id}.json"

        if not filepath.exists():
            print(f"[!] Chat history file does not exist: {filepath}")
            return None

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                record = ChatHistory(**data)
                print(f"[✔] Chat history successfully loaded: {filepath}")
                return record
        except Exception as e:
            print(f"[!] Failed to load chat history: {e}")
            return None

    def merge_chat_history(self, thread_id: str):
        """
        Load a saved chat history and merge its messages with the current session.

        This method retrieves a chat record from a JSON file, extracts the message history,
        and appends it in front of the current session's messages. It keeps the current
        configuration and application state intact, updating only the message history.

        Parameters:
            thread_id (str): Unique identifier of the chat history to load.
        """
        record = self.load_chat_history_file(thread_id)
        if record is None:
            return

        loaded_messages = record.values.get("messages", [])
        current_messages = self.app.get_state(self.config).values.get("messages", [])
        merged_messages = loaded_messages + current_messages

        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)
        self.app.update_state(self.config, {"messages": merged_messages})

    def load_chat_history(self, thread_id: str):
        """
        Load a saved chat history and completely replace the current session state.

        This method loads a full chat session including the thread ID, configuration, and
        conversation messages from a JSON file, and resets the current application state
        to match the loaded session.

        Parameters:
            thread_id (str): Unique identifier of the chat history to load.
        """
        record = self.load_chat_history_file(thread_id)
        if record is None:
            return

        self.thread_id = record.thread_id
        self.config = record.config

        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)
        self.app.update_state(self.config, record.values)

    def clear_memory(self):
        checkpoints = list(self.app.get_state_history(self.config))
        if len(checkpoints) > 1:
            previous_state = checkpoints[1].values
            self.app.update_state(self.config, previous_state)

    """Utils"""
    @staticmethod
    def parse_ai_messages(messages: List) -> List[dict]:
        return [msg.__dict__ for msg in messages if isinstance(msg, AIMessage)]
