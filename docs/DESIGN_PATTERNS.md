# VibeCraft 디자인 패턴 문서

VibeCraft 프로젝트에서 사용된 디자인 패턴과 아키텍처 패턴을 정리한 문서입니다.

---

## 목차

1. [아키텍처 패턴](#아키텍처-패턴)
2. [생성 패턴](#생성-패턴)
3. [구조 패턴](#구조-패턴)
4. [행위 패턴](#행위-패턴)
5. [패턴 조합 예시](#패턴-조합-예시)
6. [디자인 원칙](#디자인-원칙)

---

## 아키텍처 패턴

### 1. Layered Architecture (계층화 아키텍처)

**목적**: 관심사의 분리 및 계층 간 독립성 확보

**구조**:
```
routers/        (Presentation Layer)   - API 엔드포인트
    ↓
services/       (Business Logic Layer) - 비즈니스 로직
    ↓
mcp_agent/      (Domain Layer)         - 핵심 도메인 로직
    ↓
core/, utils/   (Infrastructure Layer) - 공통 유틸리티
```

**구현 위치**:
- `routers/chat.py:23` - API 라우터 정의
- `services/business/chat_service.py:17` - 비즈니스 로직
- `mcp_agent/client/vibe_craft_client.py:28` - 도메인 로직
- `core/cors.py:4` - 인프라 레이어

**장점**:
- 각 계층의 독립적인 테스트 가능
- 변경 사항의 영향 범위 최소화
- 명확한 책임 분리

**예시**:
```python
# routers/chat.py (Presentation Layer)
@router.get("/new-chat")
async def new_chat(query: str):
    return await chat_service.execute_chat(query)

# services/business/chat_service.py (Business Logic Layer)
async def execute_chat(self, query: str):
    client = await self._create_client()
    return await client.execute_step(query)

# mcp_agent/client/vibe_craft_client.py (Domain Layer)
async def execute_step(self, prompt: str):
    return await self.engine.generate_langchain(prompt)
```

---

## 생성 패턴

### 1. Singleton Pattern (싱글톤 패턴)

**목적**: 클래스의 인스턴스가 하나만 존재하도록 보장

**구현 위치**:
- `services/business/chat_service.py:96`
- `services/business/workflow_service.py`

**코드 예시**:
```python
# services/business/chat_service.py
class ChatService(BaseStreamService):
    """채팅 관련 비즈니스 로직을 처리하는 서비스 클래스"""
    # ... 클래스 정의 ...

# 싱글톤 인스턴스 생성
chat_service = ChatService()
```

**장점**:
- 전역 접근점 제공
- 리소스 공유 및 메모리 절약
- 일관된 상태 유지

---

### 2. Factory Method Pattern (팩토리 메서드 패턴)

**목적**: 객체 생성 로직을 캡슐화

**구현 위치**:
- `mcp_agent/client/vibe_craft_client.py:29`

**코드 예시**:
```python
class VibeCraftClient:
    def __init__(self, engine: str):
        # 팩토리 메서드: engine 타입에 따라 다른 객체 생성
        if engine == "claude":
            self.engine = ClaudeEngine()
        elif engine == "gemini":
            self.engine = GeminiEngine()
        elif engine == "gpt":
            self.engine = OpenAIEngine()
        else:
            raise ValueError("Not Supported Engine")
```

**장점**:
- 객체 생성 로직의 중앙화
- 새로운 엔진 추가 시 기존 코드 수정 최소화
- 느슨한 결합

---

### 3. Builder Pattern (빌더 패턴)

**목적**: 복잡한 객체의 생성 과정을 단순화

**구현 위치**:
- `schemas/sse_response_schemas.py`

**코드 예시**:
```python
# SSE 이벤트 생성을 위한 빌더
class SSEEventBuilder:
    @staticmethod
    def create_ai_message_chunk(message: str):
        return ServerSentEvent(
            event="ai",
            data=json.dumps({"content": message})
        )

    @staticmethod
    def create_complete_event(thread_id: str):
        return ServerSentEvent(
            event="complete",
            data=json.dumps({"thread_id": thread_id})
        )

    @staticmethod
    def create_error_event(message: str):
        return ServerSentEvent(
            event="error",
            data=json.dumps({"error": message})
        )
```

**사용 예시**:
```python
# vibe_craft_client.py
yield SSEEventBuilder.create_info_event("Step 1 시작")
yield SSEEventBuilder.create_complete_event(thread_id)
```

**장점**:
- 일관된 객체 생성 방식
- 가독성 향상
- 생성 과정의 복잡성 숨김

---

## 구조 패턴

### 1. Facade Pattern (파사드 패턴)

**목적**: 복잡한 서브시스템에 대한 단순화된 인터페이스 제공

**구현 위치**:
- `mcp_agent/client/vibe_craft_client.py:392`

**코드 예시**:
```python
class VibeCraftClient:
    async def run_pipeline(self, topic_prompt: str, file_path: str):
        """
        간소화된 자동 파이프라인
        - 내부적으로 복잡한 6단계 프로세스를 숨김
        """
        # Step 1: 주제 설정
        await self.topic_selection(topic_prompt)

        # Step 2: 데이터 업로드 또는 생성
        await self.set_data(file_path)

        # Step 3: 데이터 전처리
        await self.auto_process_and_save_data()

        # Step 4: 인과관계 분석
        analysis_result = await self.execute_step(analysis_query)

        # Step 5: 시각화 타입 자동 결정
        v_type = await self.auto_recommend_visualization_type()

        # Step 6: 코드 자동 생성
        result = self.run_code_generator(self.get_thread_id(), v_type)

        return result
```

**장점**:
- 복잡한 프로세스를 단순한 인터페이스로 제공
- 사용자가 내부 구현을 알 필요 없음
- 결합도 감소

---

### 2. Adapter Pattern (어댑터 패턴)

**목적**: 호환되지 않는 인터페이스를 함께 동작하도록 변환

**구현 위치**:
- `mcp_agent/engine/base.py:54` - RAG 엔진을 LangChain 도구로 변환

**코드 예시**:
```python
class BaseEngine:
    def __init__(self, model_cls, model_name: str, model_kwargs: dict, tools):
        # ChromaDB Retriever를 LangChain Tool로 어댑터
        self.retriever = rag_engine.as_retriever()
        self.retriever_tool = create_retriever_tool(
            self.retriever,
            name="rag_analysis",
            description="""Expert tool for comprehensive data causal relationship analysis..."""
        )

        # MCP 도구와 RAG 도구를 통합
        if tools:
            all_tools = tools + [self.retriever_tool]
            self.llm = model_cls(model=model_name, **model_kwargs).bind_tools(all_tools)
```

**장점**:
- 기존 코드 수정 없이 새로운 기능 통합
- 재사용성 향상

---

### 3. Proxy Pattern (프록시 패턴)

**목적**: 객체에 대한 접근 제어

**구현 위치**:
- `mcp_agent/client/vibe_craft_client.py:58` - MCP 도구 로딩

**코드 예시**:
```python
class VibeCraftClient:
    async def load_tools(self, mcp_servers: Optional[List[MCPServerConfig]] = None):
        """MCP 서버 연결을 프록시 역할로 관리"""
        mcp_servers = mcp_servers or self.mcp_tools
        if mcp_servers:
            try:
                # MultiServerMCPClient가 실제 MCP 서버의 프록시 역할
                self.client = MultiServerMCPClient({
                    tool.name: {
                        "command": tool.command,
                        "args": tool.args,
                        "transport": tool.transport
                    }
                    for tool in mcp_servers
                })
                self.tools = await self.client.get_tools()
                self.engine.update_tools(self.tools)
            except Exception as e:
                print(f"⚠️ 서버 연결 실패: {e}")
```

**장점**:
- 지연 로딩 (Lazy Loading)
- 접근 제어
- 부가 기능 추가 용이

---

## 행위 패턴

### 1. Strategy Pattern (전략 패턴)

**목적**: 알고리즘 군을 정의하고 캡슐화하여 교체 가능하게 만듦

**구현 위치**:
- `mcp_agent/client/vibe_craft_client.py:29`
- `mcp_agent/engine/base.py:47`

**구조**:
```
BaseEngine (추상 전략)
    ├── ClaudeEngine (구체적 전략 1)
    ├── OpenAIEngine (구체적 전략 2)
    └── GeminiEngine (구체적 전략 3)
```

**코드 예시**:
```python
# 전략 인터페이스
class BaseEngine:
    def build_graph(self, tools):
        """모든 엔진이 구현해야 하는 공통 인터페이스"""
        pass

    async def generate_langchain(self, prompt: str):
        """LangChain 기반 생성"""
        pass

# 구체적 전략 1
class ClaudeEngine(BaseEngine):
    def __init__(self):
        super().__init__(
            model_cls=ChatAnthropic,
            model_name="claude-3-5-sonnet-20241022",
            model_kwargs={"max_tokens": 8096, "temperature": 0}
        )

# 구체적 전략 2
class GeminiEngine(BaseEngine):
    def __init__(self):
        super().__init__(
            model_cls=ChatGoogleGenerativeAI,
            model_name="gemini-2.0-flash-exp",
            model_kwargs={"temperature": 0}
        )

# 전략 선택
class VibeCraftClient:
    def __init__(self, engine: str):
        if engine == "claude":
            self.engine = ClaudeEngine()
        elif engine == "gemini":
            self.engine = GeminiEngine()
```

**장점**:
- 런타임에 알고리즘 교체 가능
- Open/Closed 원칙 준수
- 조건문 제거

---

### 2. Template Method Pattern (템플릿 메서드 패턴)

**목적**: 알고리즘의 골격을 정의하고 일부 단계를 서브클래스에서 구현

**구현 위치**:
- `mcp_agent/engine/base.py:104`

**코드 예시**:
```python
class BaseEngine:
    def build_graph(self, tools: Optional[List[BaseTool]] = None):
        """
        LangGraph 워크플로우의 템플릿 메서드
        - 모든 엔진이 동일한 그래프 구조를 사용
        """
        self.workflow = StateGraph(state_schema=State)

        all_tools = tools + [self.retriever_tool] if tools else [self.retriever_tool]
        tool_node = ToolNode(all_tools)

        # 노드 구성 (템플릿의 골격)
        self.workflow.add_node("agent", self.call_agent)
        self.workflow.add_node("tools", tool_node)
        self.workflow.add_node("rag_analysis", self.perform_rag_analysis)
        self.workflow.add_node("final_synthesis", self.synthesize_final_analysis)
        self.workflow.add_node("summarize_conversation", self.summarize_conversation)

        # 엣지 구성 (워크플로우 정의)
        self.workflow.add_edge(START, "agent")
        self.workflow.add_conditional_edges(
            "agent",
            self.route_agent_decision,
            ["tools", "summarize_conversation", END]
        )

        return self.workflow.compile(checkpointer=self.memory)

    def call_agent(self, state: State):
        """서브클래스에서 오버라이드 가능한 훅 메서드"""
        # 기본 구현
        pass
```

**장점**:
- 코드 재사용
- 일관된 알고리즘 구조
- 확장 가능한 프레임워크

---

### 3. Chain of Responsibility (책임 연쇄 패턴)

**목적**: 요청을 처리할 수 있는 객체가 나타날 때까지 체인을 따라 전달

**구현 위치**:
- `mcp_agent/engine/base.py:120`

**코드 예시**:
```python
class BaseEngine:
    def build_graph(self):
        # 조건부 엣지를 통한 책임 연쇄
        self.workflow.add_conditional_edges(
            "agent",
            self.route_agent_decision,
            ["tools", "summarize_conversation", END]
        )

        self.workflow.add_conditional_edges(
            "tools",
            self.route_after_tools,
            ["agent", "rag_analysis"]
        )

    def route_agent_decision(self, state: State) -> str:
        """에이전트 결정에 따라 다음 노드로 라우팅"""
        messages = state["messages"]
        last_message = messages[-1]

        # 요약이 필요한가?
        if state.get("should_summarize", False):
            return "summarize_conversation"

        # 도구 호출이 필요한가?
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"

        # 모든 처리 완료
        return END

    def route_after_tools(self, state: State) -> str:
        """도구 실행 후 다음 처리 결정"""
        complexity_score = self._calculate_query_complexity(query_text)

        # 복잡도에 따라 다음 핸들러 선택
        if complexity_score >= complexity_threshold:
            return "rag_analysis"  # RAG 분석으로 전달
        else:
            return "agent"  # 에이전트로 돌아감
```

**장점**:
- 동적 체인 구성
- 각 핸들러의 독립성
- 유연한 처리 흐름

---

### 4. State Pattern (상태 패턴)

**목적**: 객체의 상태에 따라 동작을 변경

**구현 위치**:
- `mcp_agent/engine/base.py:38`

**코드 예시**:
```python
class State(MessagesState):
    """LangGraph의 상태 관리"""
    summary: str                          # 대화 요약
    title: str = ""                       # 대화 제목
    should_summarize: bool = False        # 요약 필요 여부
    should_search_rag: bool = False       # RAG 검색 필요 여부
    analysis_stage: str = "initial"       # 분석 단계 추적

# 상태에 따른 동작 변경
def route_after_tools(self, state: State) -> str:
    analysis_stage = state.get("analysis_stage", "initial")

    # 분석 단계에 따라 다른 임계값 사용
    complexity_threshold = 35 if analysis_stage == "initial" else 25

    if complexity_score >= complexity_threshold:
        return "rag_analysis"
    else:
        return "agent"
```

**장점**:
- 상태별 동작 명확화
- 상태 전환 추적 용이
- 확장 가능한 상태 관리

---

### 5. Observer Pattern (옵저버 패턴)

**목적**: 객체의 상태 변화를 관찰자에게 자동 통지

**구현 위치**:
- SSE 스트리밍 구현 (`mcp_agent/client/vibe_craft_client.py:681`)

**코드 예시**:
```python
class VibeCraftClient:
    async def stream_generate_langchain(self, prompt: str, system: Optional[str] = None):
        """스트림 생성 - 옵저버들에게 변화 통지"""
        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))

        # 상태 변화를 구독자(클라이언트)에게 실시간 전달
        async for chunk in self.app.astream(
            {"messages": messages}, self.config, stream_mode="messages"
        ):
            content_type = chunk[0].type
            content = chunk[0].content

            # 옵저버에게 통지
            yield content_type, content

# routers/chat.py - 옵저버 (구독자)
@router.get("/stream/new-chat")
async def stream_new_chat(query: str):
    return EventSourceResponse(
        chat_service.execute_stream_chat(query)  # 스트림 구독
    )
```

**장점**:
- 실시간 업데이트
- 느슨한 결합
- 다수의 구독자 지원

---

### 6. Command Pattern (커맨드 패턴)

**목적**: 요청을 객체로 캡슐화

**구현 위치**:
- LangGraph의 메시지 시스템

**코드 예시**:
```python
# 각 메시지가 커맨드 객체
messages = []
messages.append(SystemMessage(content=system_prompt))  # System 커맨드
messages.append(HumanMessage(content=user_query))      # Human 커맨드

# 커맨드 실행
response = self.llm.invoke(messages)

# 커맨드 이력 저장 (Undo 가능)
def save_chat_history(self):
    chat_entry = self.get_chat_history()
    filepath = Path(settings.chat_path) / f"chat_{self.thread_id}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(chat_entry.model_dump_json(indent=2))
```

---

## 패턴 조합 예시

실제 코드에서 여러 패턴이 함께 사용되는 예시입니다.

### 조합 1: Facade + Strategy + Builder

**위치**: `mcp_agent/client/vibe_craft_client.py:427`

```python
class VibeCraftClient:
    async def stream_run_pipeline(self, topic_prompt: str, file_path: str):
        """
        Facade: 복잡한 파이프라인을 간단한 메서드로 제공
        """
        # Step 1: 주제 설정
        yield SSEEventBuilder.create_info_event("Step 1: 주제 설정")  # Builder
        async for event in self.stream_topic_selection(topic_prompt):
            yield event

        # Step 2: 데이터 로드
        self.stream_set_data(file_path)

        # Strategy: 선택된 엔진 사용
        async for event, chunk in self.execute_stream_step(query):  # Strategy
            yield ServerSentEvent(event=event, data=chunk)  # Builder

        # Step 6: 코드 생성
        async for event in self.stream_run_code_generator(...):
            yield event
```

**패턴 역할**:
- **Facade**: `stream_run_pipeline`이 복잡한 6단계를 숨김
- **Strategy**: `self.engine`이 선택된 AI 모델 사용
- **Builder**: `SSEEventBuilder`가 일관된 이벤트 생성

---

### 조합 2: Layered Architecture + Repository + Dependency Injection

**위치**: 전체 아키텍처

```python
# 1. Dependency Injection
# config.py
settings = Settings.load_from_yaml(env="development")

# 2. Repository Pattern
# mcp_agent/engine/base.py
class BaseEngine:
    def save_chat_history(self):
        """데이터 저장 로직 캡슐화"""
        os.makedirs(settings.chat_path, exist_ok=True)  # DI
        filepath = Path(settings.chat_path) / f"chat_{self.thread_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(chat_entry.model_dump_json(indent=2))

    @staticmethod
    def load_chat_history_file(thread_id: str):
        """데이터 로드 로직 캡슐화"""
        filepath = Path(settings.chat_path) / f"chat_{thread_id}.json"  # DI
        with open(filepath, "r", encoding="utf-8") as f:
            return ChatHistory(**json.load(f))

# 3. Layered Architecture
# routers/chat.py → services/chat_service.py → BaseEngine
@router.get("/history")
async def get_chat_history(thread_id: str):
    return chat_service.get_chat_history(thread_id)  # Layer 1 → 2

class ChatService:
    def get_chat_history(self, thread_id: str):
        return BaseEngine.load_chat_history_file(thread_id)  # Layer 2 → 3
```

**패턴 역할**:
- **Layered Architecture**: Router → Service → Repository
- **Repository Pattern**: 데이터 접근 로직 캡슐화
- **Dependency Injection**: `settings` 객체 주입

---

### 조합 3: Template Method + Chain of Responsibility + State

**위치**: `mcp_agent/engine/base.py:104`

```python
class BaseEngine:
    def build_graph(self, tools):
        """Template Method: 그래프 구조의 템플릿 정의"""
        self.workflow = StateGraph(state_schema=State)  # State Pattern

        # 노드 추가
        self.workflow.add_node("agent", self.call_agent)
        self.workflow.add_node("tools", tool_node)
        self.workflow.add_node("rag_analysis", self.perform_rag_analysis)

        # Chain of Responsibility: 조건부 라우팅
        self.workflow.add_conditional_edges(
            "agent",
            self.route_agent_decision,  # 책임 연쇄의 라우터
            ["tools", "summarize_conversation", END]
        )

        return self.workflow.compile(checkpointer=self.memory)

    def route_agent_decision(self, state: State) -> str:
        """Chain of Responsibility + State Pattern"""
        # State Pattern: 상태에 따른 라우팅
        if state.get("should_summarize", False):
            return "summarize_conversation"

        # Chain of Responsibility: 다음 핸들러로 전달
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"

        return END
```

**패턴 역할**:
- **Template Method**: `build_graph`가 워크플로우 골격 정의
- **Chain of Responsibility**: 조건부 라우팅으로 책임 전달
- **State Pattern**: `State` 객체로 상태 관리

---

### 조합 4: Singleton + Factory + Facade

**위치**: `services/business/chat_service.py`

```python
# Factory Pattern: 클라이언트 생성
class ChatService(BaseStreamService):
    async def _create_client(self, thread_id: Optional[str] = None):
        """Factory Method: 클라이언트 생성"""
        client = super()._create_client()  # VibeCraftClient 생성
        await client.load_tools()

        if thread_id:
            client.load_chat_history(thread_id)

        return client

    # Facade Pattern: 복잡한 채팅 로직을 간단한 메서드로
    async def execute_chat(self, query: str, system: Optional[str] = None,
                          thread_id: Optional[str] = None):
        """Facade: 채팅 실행의 복잡성 숨김"""
        client = await self._create_client(thread_id)  # Factory
        response = await client.execute_step(query, system)

        return JSONResponse(
            content=ChatResponse(
                data=response,
                thread_id=client.get_thread_id()
            ).model_dump()
        )

# Singleton Pattern: 서비스 인스턴스
chat_service = ChatService()  # 싱글톤 인스턴스
```

**패턴 역할**:
- **Singleton**: `chat_service` 전역 인스턴스 하나만 존재
- **Factory**: `_create_client`가 클라이언트 생성 로직 캡슐화
- **Facade**: `execute_chat`이 복잡한 프로세스를 단순화

---

## 디자인 원칙

VibeCraft는 다음과 같은 SOLID 원칙을 따릅니다:

### 1. Single Responsibility Principle (단일 책임 원칙)

각 클래스는 하나의 책임만 가집니다:

- `ChatService`: 채팅 관련 비즈니스 로직만 처리
- `WorkflowService`: 워크플로우 관련 로직만 처리
- `BaseEngine`: LLM 엔진 관리만 담당
- `VibeCraftClient`: 파이프라인 오케스트레이션만 담당

```python
# 좋은 예: 단일 책임
class ChatService:
    """채팅 관련 비즈니스 로직만 처리"""
    async def execute_chat(self, query: str):
        pass

    async def get_chat_history(self, thread_id: str):
        pass
```

---

### 2. Open/Closed Principle (개방-폐쇄 원칙)

확장에는 열려 있고 수정에는 닫혀 있습니다:

```python
# BaseEngine은 수정 없이 확장 가능
class BaseEngine:
    """기본 엔진 - 수정하지 않음"""
    pass

# 새로운 엔진 추가 (확장)
class ClaudeEngine(BaseEngine):
    pass

class GeminiEngine(BaseEngine):
    pass

class OpenAIEngine(BaseEngine):
    pass

# 미래에 추가 가능
class MistralEngine(BaseEngine):  # 기존 코드 수정 없이 확장
    pass
```

---

### 3. Liskov Substitution Principle (리스코프 치환 원칙)

하위 타입은 상위 타입을 대체할 수 있습니다:

```python
# 모든 엔진이 BaseEngine을 대체 가능
def process_with_engine(engine: BaseEngine, prompt: str):
    # ClaudeEngine, GeminiEngine, OpenAIEngine 모두 사용 가능
    return engine.generate_langchain(prompt)

# 어떤 엔진이든 동일하게 동작
claude = ClaudeEngine()
gemini = GeminiEngine()

result1 = process_with_engine(claude, "Hello")   # OK
result2 = process_with_engine(gemini, "Hello")   # OK
```

---

### 4. Interface Segregation Principle (인터페이스 분리 원칙)

클라이언트는 사용하지 않는 인터페이스에 의존하지 않습니다:

```python
# 좋은 예: 분리된 서비스
class ChatService:
    """채팅 전용 메서드"""
    async def execute_chat(self):
        pass

class WorkflowService:
    """워크플로우 전용 메서드"""
    async def execute_topic_selection(self):
        pass

    async def execute_run_workflow(self):
        pass

# 나쁜 예: 모든 기능이 하나의 서비스에
# class UniversalService:
#     async def execute_chat(self):
#         pass
#     async def execute_workflow(self):
#         pass
#     async def upload_file(self):
#         pass
```

---

### 5. Dependency Inversion Principle (의존성 역전 원칙)

고수준 모듈은 저수준 모듈에 의존하지 않으며, 둘 다 추상화에 의존합니다:

```python
# 추상화 (BaseEngine)
class BaseEngine(ABC):
    @abstractmethod
    async def generate_langchain(self, prompt: str):
        pass

# 고수준 모듈 (VibeCraftClient)
class VibeCraftClient:
    def __init__(self, engine: str):
        # 구체적인 엔진이 아닌 추상화(BaseEngine)에 의존
        if engine == "claude":
            self.engine: BaseEngine = ClaudeEngine()
        elif engine == "gemini":
            self.engine: BaseEngine = GeminiEngine()

    async def execute_step(self, prompt: str):
        # BaseEngine 인터페이스에만 의존
        return await self.engine.generate_langchain(prompt)
```

---

## 추가 아키텍처 패턴

### 1. Repository Pattern (저장소 패턴)

**목적**: 데이터 접근 로직을 캡슐화

**구현 위치**: `mcp_agent/engine/base.py:714`

```python
class BaseEngine:
    def save_chat_history(self):
        """채팅 기록을 파일 시스템에 저장"""
        os.makedirs(settings.chat_path, exist_ok=True)
        chat_entry = self.get_chat_history()
        filepath = Path(settings.chat_path) / f"chat_{self.thread_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(chat_entry.model_dump_json(indent=2))

    @staticmethod
    def load_chat_history_file(thread_id: str):
        """파일 시스템에서 채팅 기록 로드"""
        filepath = Path(settings.chat_path) / f"chat_{thread_id}.json"
        if not filepath.exists():
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            return ChatHistory(**json.load(f))
```

**장점**:
- 데이터 저장 방식 변경 용이 (파일 → DB)
- 비즈니스 로직과 데이터 접근 로직 분리
- 테스트 용이성

---

### 2. Dependency Injection (의존성 주입)

**목적**: 객체 간 결합도 낮추기

**구현 위치**: `config.py:11`

```python
# config.py
class Settings(BaseSettings):
    @classmethod
    def load_from_yaml(cls, env: str = "development"):
        config_file = Path(__file__).parent / f"config-{env}.yml"
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls(
            version=config["version"]["server"],
            host=config["host"],
            port=config["port"],
            data_path=config["resource"]["data"],
            # ...
        )

# 싱글톤 설정 객체
settings = Settings.load_from_yaml(env="development")

# 다른 모듈에서 주입받아 사용
# mcp_agent/engine/base.py
from config import settings  # 의존성 주입

class BaseEngine:
    def save_chat_history(self):
        os.makedirs(settings.chat_path, exist_ok=True)  # 주입된 설정 사용
```

**장점**:
- 설정 변경 시 코드 수정 불필요
- 테스트 시 Mock 객체 주입 가능
- 환경별 설정 관리 용이

---

## 패턴 적용 가이드

### 언제 각 패턴을 사용할까?

| 패턴 | 사용 시점 | VibeCraft 예시 |
|------|-----------|----------------|
| **Singleton** | 전역적으로 하나의 인스턴스만 필요할 때 | `chat_service`, `workflow_service` |
| **Factory** | 객체 생성 로직이 복잡하거나 조건부일 때 | AI 엔진 선택 (`ClaudeEngine`, `GeminiEngine`) |
| **Strategy** | 알고리즘을 런타임에 선택해야 할 때 | AI 모델 교체 |
| **Template Method** | 알고리즘 골격은 같고 일부만 다를 때 | LangGraph 워크플로우 구조 |
| **Facade** | 복잡한 서브시스템을 단순화할 때 | `run_pipeline` 메서드 |
| **Chain of Responsibility** | 여러 핸들러가 순차적으로 처리할 때 | LangGraph 조건부 라우팅 |
| **State** | 상태에 따라 동작이 달라질 때 | 분석 단계별 처리 (`analysis_stage`) |
| **Observer** | 상태 변화를 여러 객체에 통지할 때 | SSE 스트리밍 |
| **Repository** | 데이터 접근 로직을 캡슐화할 때 | 채팅 기록 저장/로드 |

---

## 안티패턴 피하기

VibeCraft에서 피한 안티패턴들:

### 1. God Object (신 객체)

**나쁜 예**:
```python
# 모든 기능이 하나의 클래스에 집중
class MegaService:
    def chat(self): pass
    def workflow(self): pass
    def upload_file(self): pass
    def process_data(self): pass
    def generate_code(self): pass
    # ... 수십 개의 메서드
```

**VibeCraft의 해결**:
```python
# 책임별로 분리
class ChatService:
    def execute_chat(self): pass

class WorkflowService:
    def execute_workflow(self): pass

class ContentService:
    def upload_file(self): pass
```

---

### 2. Tight Coupling (강한 결합)

**나쁜 예**:
```python
class ChatService:
    def execute_chat(self):
        # 구체적인 클래스에 직접 의존
        engine = ClaudeEngine()  # 강한 결합!
        return engine.generate(prompt)
```

**VibeCraft의 해결**:
```python
class ChatService:
    async def _create_client(self):
        # 추상화된 인터페이스 사용
        client = VibeCraftClient(engine="claude")  # 느슨한 결합
        return client
```

---

### 3. Magic Numbers/Strings (매직 넘버/문자열)

**나쁜 예**:
```python
if complexity_score >= 35:  # 35는 무엇?
    return "rag_analysis"
```

**VibeCraft의 해결**:
```python
# 의미 있는 변수로 추출
complexity_threshold = 35 if analysis_stage == "initial" else 25
if complexity_score >= complexity_threshold:
    return "rag_analysis"
```

---

## 참고 자료

- [Design Patterns: Elements of Reusable Object-Oriented Software](https://en.wikipedia.org/wiki/Design_Patterns) - Gang of Four
- [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html) - Robert C. Martin
- [SOLID Principles](https://en.wikipedia.org/wiki/SOLID)
- [LangChain Design Patterns](https://python.langchain.com/docs/introduction/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

---

## 마치며

VibeCraft는 **10개 이상의 디자인 패턴**을 조합하여 구축된 복잡한 시스템입니다. 이러한 패턴들은:

1. **코드 재사용성** 향상
2. **유지보수성** 개선
3. **확장성** 확보
4. **테스트 용이성** 증가

각 패턴은 독립적으로 동작하면서도 서로 협력하여 **견고하고 유연한 아키텍처**를 만듭니다.

새로운 기능을 추가하거나 기존 기능을 수정할 때는 이 문서를 참고하여 일관된 패턴을 유지하는 것을 권장합니다.

---

**Last Updated**: 2025-10-15
**Author**: Se Hoon Kim (sehoon787@korea.ac.kr)
