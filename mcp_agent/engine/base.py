__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
import os
import uuid
import json
from typing import List, Optional
from pathlib import Path

# Third-party imports
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools.retriever import create_retriever_tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import START, END, MessagesState, StateGraph

# Custom imports
from mcp_agent.schemas import ChatHistory
from services.data_processing import rag_engine
from config import settings
from utils.prompts import (
    TITLE_PROMPT,
    BASE_SYSTEM_PROMPT,
    SUMMARY_PROMPT,
    INITIAL_SUMMARY_PROMPT,
    RAG_PROMPT,
    RAG_ANALYSIS_PROMPT,
    FINAL_SYNTHESIS_PROMPT
)


class State(MessagesState):
    """State class for data analysis"""
    summary: str
    title: str = ""  # Conversation title based on first message
    should_summarize: bool = False
    should_search_rag: bool = False
    analysis_stage: str = "initial"  # Analysis stage tracking


class BaseEngine:
    def __init__(
            self,
            model_cls, model_name: str, model_kwargs: dict,
            tools: Optional[List[BaseTool]] = None,
    ):
        # Set Rag tool
        self.retriever = rag_engine.as_retriever()
        self.retriever_tool = create_retriever_tool(
            self.retriever,
            name="rag_analysis",
            description="""
            Expert tool for comprehensive wildfire risk index and causal relationship analysis.

            Specifically searches for:
            - Forest Fire Diagnostic Model development (Section 2.4)
            - SHapley Additive exPlanations (SHAP) methodology (Section 2.5)  
            - Variable Impact Analysis Based on SHAP Value (Section 3.3)
            - Forest Fire Forecast for Republic of Korea (Section 3.4)
            - Causal relationships between meteorological variables and fire risk
            - Model validation and performance metrics

            Use this tool to find academic explanations for how environmental data 
            translates into wildfire risk predictions through established models.
            """
        )

        # Essential settings
        self.thread_id = uuid.uuid4()
        self.config = RunnableConfig(
            recursion_limit=20,
            configurable={"thread_id": self.thread_id}
        )
        self.model_name = model_name

        # Set tools
        if tools:
            all_tools = tools + [self.retriever_tool]
            self.llm = model_cls(model=model_name, **model_kwargs).bind_tools(all_tools)
        else:
            self.llm = model_cls(model=model_name, **model_kwargs).bind_tools([self.retriever_tool])

        # Create rag chain
        self.rag_chain = self.create_rag_chain()

        # Compile workflow
        self.workflow = None
        self.memory = MemorySaver()
        self.app = self.build_graph(tools if tools else [])

    """Initialize Logic"""

    def build_graph(self, tools: Optional[List[BaseTool]] = None) -> CompiledStateGraph:
        """Build optimized data analysis graph"""
        self.workflow = StateGraph(state_schema=State)

        all_tools = tools + [self.retriever_tool] if tools else [self.retriever_tool]
        tool_node = ToolNode(all_tools)

        # Node configuration
        self.workflow.add_node("agent", self.call_agent)
        self.workflow.add_node("tools", tool_node)
        self.workflow.add_node("rag_analysis", self.perform_rag_analysis)
        self.workflow.add_node("final_synthesis", self.synthesize_final_analysis)
        self.workflow.add_node("summarize_conversation", self.summarize_conversation)

        # Edge configuration - optimized flow
        self.workflow.add_edge(START, "agent")
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
        self.workflow.add_edge("rag_analysis", "final_synthesis")
        self.workflow.add_edge("final_synthesis", END)
        self.workflow.add_edge("summarize_conversation", END)

        app = self.workflow.compile(checkpointer=self.memory)
        print("--- Optimized Data Analysis LangGraph ---")
        print(app.get_graph().draw_ascii())
        print("------------------------------------------")

        # from langchain_core.runnables.graph_mermaid import draw_mermaid_png
        # graph = app.get_graph()
        # mermaid_syntax = graph.draw_mermaid()  # Get the Mermaid syntax
        # # Draw and save the PNG
        # draw_mermaid_png(mermaid_syntax, output_file_path="langgraph_visualization.png")

        return app

    def create_rag_chain(self):
        rag_chain = (
                {"context": lambda x: x["context"], "question": lambda x: x["question"]}
                | ChatPromptTemplate.from_template(RAG_PROMPT)
                | self.llm
                | StrOutputParser()
        )
        return rag_chain

    def update_tools(self, tools: List[BaseTool]):
        if "tools" in self.workflow.nodes:
            del self.workflow.nodes["tools"]

        all_tools = tools + [self.retriever_tool]
        self.llm = self.llm.bind_tools(all_tools)

        tool_node = ToolNode(all_tools)
        self.workflow.add_node("tools", tool_node)

        self.app = self.workflow.compile(checkpointer=self.memory)
        print("[*] Tools updated and Data Analysis LangGraph recompiled.")

    """LangGraph Logic"""

    def call_agent(self, state: State):
        """Initial agent call - analysis planning"""
        summary = state.get("summary", "")
        title = state.get("title", "")
        messages = state["messages"]

        # Filter only actual Human-AI conversations (exclude Tool messages)
        conversation_messages = self.get_conversation_messages(messages)

        # Generate title if it's the first message
        if title == "" and conversation_messages:
            first_human_message = conversation_messages[0]
            title = self._generate_title(first_human_message.content)

        # Use provided system prompt if available, otherwise use default prompt
        existing_system_messages = self.get_system_messages(messages)
        if existing_system_messages:
            # Use the most recent provided system prompt
            system_message = existing_system_messages[-1]
            # 첫 대화는 자유도를 위해 사전 정의된 system prompt만 사용
            # 첫 대화 이후의 system message는 base system message의 제어를 받게 수정
            if 2 < len(conversation_messages):
                base_system_message = SystemMessage(content=BASE_SYSTEM_PROMPT)
                system_message = f"{base_system_message.content}\n\n{system_message.content}"
        else:
            # Use default base system prompt if none provided
            system_message = SystemMessage(content=BASE_SYSTEM_PROMPT)

        if summary:
            # Combine system prompt and summary into single SystemMessage for Gemini compatibility
            combined_system_content = f"{system_message.content}\n\nConversation summary: {summary}"
            combined_system_message = SystemMessage(content=combined_system_content)
            inference_messages = [combined_system_message] + conversation_messages
        else:
            inference_messages = [system_message] + conversation_messages
        response = self.llm.invoke(inference_messages)

        result = {
            "title": title,
            "messages": [response],
            "analysis_stage": "tool_planning"
        }

        return result

    def route_agent_decision(self, state: State) -> str:
        """Route based on agent decisions"""
        messages = state["messages"]
        last_message = messages[-1]

        # Check summarization trigger - update summary every 10 messages
        if (state.get("should_summarize", False)
                or self.check_should_summarize(10)):
            return "summarize_conversation"

        # When tool calls are needed
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"

        return END

    def route_after_tools(self, state: State) -> str:
        """
        Advanced query complexity classification using linguistic features and information-theoretic metrics.
        Based on recent NLP research on syntactic complexity measurement and question classification.
        """
        messages = state["messages"]
        analysis_stage = state.get("analysis_stage", "initial")

        # Extract recent human messages for analysis
        recent_human_messages = []
        human_messages = self.get_human_messages(messages)
        # Remove first prewritten human prompt
        if 1 < len(human_messages):
            human_messages = human_messages[1:]
        # Focus on last 3 messages for better context
        for msg in human_messages[-3:]:
            content = ""
            if hasattr(msg, 'content'):
                content = msg.content
            elif isinstance(msg, dict):
                content = msg.get('content', '')
            if content:
                recent_human_messages.append(content)

        if not recent_human_messages:
            return END

        # Combine recent queries for comprehensive analysis
        query_text = ' '.join(recent_human_messages)

        # Calculate multi-dimensional complexity score (0-100)
        complexity_score = self._calculate_query_complexity(query_text)

        # Dynamic threshold based on analysis stage
        complexity_threshold = 35 if analysis_stage == "initial" else 25

        # Route based on complexity score
        if complexity_score >= complexity_threshold:
            return "rag_analysis"
        else:
            return "agent"

    def _calculate_query_complexity(self, query: str) -> float:
        """
        Multi-dimensional query complexity scoring based on linguistic research.
        Combines syntactic, semantic, and information-theoretic measures.
        """
        if not query.strip():
            return 0.0

        query_lower = query.lower().strip()

        # 1. LENGTH-BASED COMPLEXITY (Syntactic Dimension)
        # Research: Mean length strongly correlates with complexity
        sentences = [s.strip() for s in query.split('.') if s.strip()]
        words = query.split()

        sentence_length_score = min(len(words) / 3.0, 20)  # Normalized to 0-20
        sentence_count_score = min(len(sentences) * 5, 15)  # Multi-sentence queries

        # 2. SUBORDINATION COMPLEXITY (Syntactic Sophistication)
        # Research: Subordinate clauses indicate higher complexity
        subordination_indicators = [
            '때문에', 'because', '그러므로', 'therefore', '만약', 'if', '비록', 'although',
            '~하면서', 'while', '~때', 'when', '~면', 'if', '~지만', 'but', '~거나', 'or',
            'since', 'whereas', 'unless', 'provided', 'assuming'
        ]
        subordination_score = sum(2 for indicator in subordination_indicators if indicator in query_lower)
        subordination_score = min(subordination_score, 15)

        # 3. SEMANTIC DOMAIN COMPLEXITY (Content Analysis)
        # Multi-weighted semantic categories based on domain complexity
        semantic_weights = {
            # High complexity (academic/analytical)
            'academic': ['연구', 'research', '논문', 'paper', '학술', 'academic', '이론', 'theory'],
            'technical': ['모델', 'model', '알고리즘', 'algorithm', '시뮬레이션', 'simulation', '분석', 'analysis'],
            'causal': ['인과관계', 'causal', '원인', 'cause', '결과', 'result', '영향', 'impact', '상관관계', 'correlation'],
            'predictive': ['예측', 'prediction', '전망', 'forecast', '추정', 'estimation', '예상', 'expectation'],
            'evaluative': ['평가', 'evaluation', '비교', 'comparison', '검토', 'review', '판단', 'assessment'],

            # Medium complexity (informational)
            'quantitative': ['수치', 'number', '통계', 'statistics', '데이터', 'data', '지수', 'index'],
            'temporal': ['변화', 'change', '추세', 'trend', '경향', 'tendency', '패턴', 'pattern'],
            'spatial': ['지역', 'region', '위치', 'location', '분포', 'distribution', '범위', 'range'],

            # Low complexity (factual/simple)
            'descriptive': ['현재', 'current', '지금', 'now', '오늘', 'today', '어제', 'yesterday'],
            'basic': ['무엇', 'what', '어디', 'where', '언제', 'when', '어떻게', 'how', '누구', 'who']
        }

        complexity_multipliers = {
            'academic': 4.0, 'technical': 3.5, 'causal': 3.0, 'predictive': 2.8, 'evaluative': 2.5,
            'quantitative': 2.0, 'temporal': 1.8, 'spatial': 1.5,
            'descriptive': 0.8, 'basic': 0.5
        }

        semantic_score = 0.0
        for category, keywords in semantic_weights.items():
            category_matches = sum(1 for keyword in keywords if keyword in query_lower)
            if category_matches > 0:
                semantic_score += category_matches * complexity_multipliers[category]

        semantic_score = min(semantic_score, 25)

        # 4. INTERROGATIVE COMPLEXITY (Question Type Analysis)
        # Research: Wh-question complexity hierarchy
        question_complexity = {
            'what': 1.0, '무엇': 1.0, '뭐': 1.0,
            'where': 1.2, '어디': 1.2,
            'when': 1.2, '언제': 1.2,
            'who': 1.3, '누구': 1.3,
            'how': 2.0, '어떻게': 2.0, '방법': 2.0,
            'why': 2.5, '왜': 2.5, '이유': 2.5,
            'which': 2.0, '어느': 2.0, '어떤': 2.0
        }

        interrogative_score = 0.0
        for question_word, weight in question_complexity.items():
            if question_word in query_lower:
                interrogative_score = max(interrogative_score, weight * 3)

        # 5. INFORMATION-THEORETIC COMPLEXITY
        # Approximation of Kolmogorov complexity using text entropy
        char_freq = {}
        for char in query_lower:
            char_freq[char] = char_freq.get(char, 0) + 1

        if len(query) > 0:
            entropy = -sum((freq / len(query)) * __import__('math').log2(freq / len(query))
                           for freq in char_freq.values())
            entropy_score = min(entropy * 2, 10)
        else:
            entropy_score = 0

        # 6. NEGATION AND MODAL COMPLEXITY
        # Research: Negations and modals increase processing complexity
        negation_modals = [
            '안', 'not', '없', 'no', '아니', '못', "can't", "won't", "shouldn't",
            '할 수 있', 'can', '해야', 'should', '필요', 'need', '가능', 'possible'
        ]
        modal_score = sum(1.5 for modal in negation_modals if modal in query_lower)
        modal_score = min(modal_score, 10)

        # FINAL COMPLEXITY CALCULATION
        total_score = (
                sentence_length_score +  # 0-20: Length-based complexity
                sentence_count_score +  # 0-15: Multi-sentence complexity
                subordination_score +  # 0-15: Syntactic subordination
                semantic_score +  # 0-25: Domain-specific complexity
                interrogative_score +  # 0-7.5: Question type complexity
                entropy_score +  # 0-10: Information-theoretic complexity
                modal_score  # 0-10: Modal/negation complexity
        )

        # Normalize to 0-100 scale with sigmoid-like curve for better discrimination
        normalized_score = min(100, total_score * 1.1)

        # Apply sigmoid transformation for better threshold discrimination
        sigmoid_score = 100 / (1 + __import__('math').exp(-(normalized_score - 50) / 15))

        return round(sigmoid_score, 1)

    def perform_rag_analysis(self, state: State):
        """RAG analysis based on collected data"""
        messages = state["messages"]
        data_summary = self._extract_data_summary(messages)

        rag_queries = [
            f"Diagnostic model development data analysis {data_summary}",
            f"SHAP analysis prediction variables {data_summary}",
            f"Variable Impact Analysis risk factors {data_summary}",
            f"Predictive model validation methodology {data_summary}"
        ]

        combined_context = ""
        for query in rag_queries:
            try:
                context = self.retriever.invoke(query)
                if context:
                    combined_context += f"\n\n=== Query: {query} ===\n{context}"
            except Exception as e:
                print(f"RAG search failed for query '{query}': {e}")

        analysis_prompt = RAG_ANALYSIS_PROMPT.format(
            collected_data=data_summary,
            rag_context=combined_context
        )

        analysis_message = HumanMessage(content=analysis_prompt)
        updated_messages = messages + [analysis_message]

        response = self.llm.invoke(updated_messages)

        return {
            "messages": messages + [response],
            "analysis_stage": "rag_complete"
        }

    def synthesize_final_analysis(self, state: State):
        """Final comprehensive analysis - integrate data and RAG results"""
        messages = state["messages"]

        synthesis_prompt = FINAL_SYNTHESIS_PROMPT
        synthesis_message = HumanMessage(content=synthesis_prompt)

        updated_messages = messages + [synthesis_message]
        final_response = self.llm.invoke(updated_messages)

        return {
            "messages": messages + [final_response],
            "analysis_stage": "complete"
        }

    def _extract_data_summary(self, messages: List) -> str:
        """Extract data summary from messages"""
        data_elements = []

        for message in messages:
            if hasattr(message, 'content') and message.content:
                content = message.content.lower()

                if any(keyword in content for keyword in ['temperature', '온도', 'temp']):
                    data_elements.append('temperature')
                if any(keyword in content for keyword in ['humidity', '습도', 'moisture']):
                    data_elements.append('humidity')
                if any(keyword in content for keyword in ['wind', '바람', '풍속']):
                    data_elements.append('wind')
                if any(keyword in content for keyword in ['precipitation', '강수', 'rainfall']):
                    data_elements.append('precipitation')
                if any(keyword in content for keyword in ['vegetation', '식생', 'ndvi']):
                    data_elements.append('vegetation')
                if any(keyword in content for keyword in ['risk', '위험지수', 'index']):
                    data_elements.append('risk_index')

        return " ".join(set(data_elements)) if data_elements else "general data analysis"

    def summarize_conversation(self, state: State):
        """Data analysis specialized summary"""
        summary = state.get("summary", "")

        if summary:
            summary_message_content = f"Previous summary: {summary}\n\n{SUMMARY_PROMPT}"
        else:
            summary_message_content = INITIAL_SUMMARY_PROMPT

        summary_message = HumanMessage(content=summary_message_content)

        messages = state["messages"] + [summary_message]
        response = self.llm.invoke(messages)

        return {
            "summary": response.content,
            "messages": state["messages"],
            "should_summarize": False
        }

    """Util methods"""

    def _generate_title(self, first_message_content: str) -> str:
        title_prompt = TITLE_PROMPT.format(first_message_content=first_message_content)

        try:
            response = self.llm.invoke([HumanMessage(content=title_prompt)])
            title = response.content.strip()
            # Remove unnecessary quotes or symbols
            title = title.replace('"', '').replace("'", '').replace('제목:', '').strip()
            return title if title else "New conversation"
        except Exception as e:
            print(f"Title generation failed: {e}")
            return "New conversation"

    @staticmethod
    def get_system_messages(messages: List, is_json: bool = False) -> List:
        """Filter System Messages only

        Args:
            messages: Message list
            is_json: True for dictionary format, False for LangChain objects (default)
        """

        if is_json:
            return [msg for msg in messages if isinstance(msg, dict) and
                    msg.get('type', '') == 'system']
        return [msg for msg in messages if isinstance(msg, SystemMessage)]

    @staticmethod
    def get_human_messages(messages: List, is_json: bool = False) -> List:
        """Filter Human Messages only

        Args:
            messages: Message list
            is_json: True for dictionary format, False for LangChain objects (default)
        """

        if is_json:
            return [msg for msg in messages if isinstance(msg, dict) and
                    msg.get('type', '') == 'human']
        return [msg for msg in messages if isinstance(msg, HumanMessage)]

    @staticmethod
    def get_ai_messages(messages: List, is_json: bool = False) -> List:
        """Filter AI Messages only

        Args:
            messages: Message list
            is_json: True for dictionary format, False for LangChain objects (default)
        """

        if is_json:
            return [msg for msg in messages if isinstance(msg, dict) and
                    msg.get('type', '') == 'ai']
        return [msg for msg in messages if isinstance(msg, AIMessage)]

    @staticmethod
    def get_conversation_messages(messages: List, is_json: bool = False) -> List:
        """Filter Human-AI conversation messages only (exclude Tool messages)

        Args:
            messages: Message list
            is_json: True for dictionary format, False for LangChain objects (default)
        """

        if is_json:
            return [msg for msg in messages if isinstance(msg, dict)
                    and msg.get('type', '') in ['human', 'ai']]
        return [msg for msg in messages if isinstance(msg, (HumanMessage, AIMessage))]

    @staticmethod
    def filter_system_message(messages: List, is_json: bool = False) -> List:
        """Return messages excluding System Messages (including Human, AI, Tool messages)

        Args:
            messages: Message list
            is_json: True for dictionary format, False for LangChain objects (default)
        """

        if is_json:
            return [msg for msg in messages if isinstance(msg, dict) and
                    msg.get('type', '') != 'system']
        return [msg for msg in messages if not isinstance(msg, SystemMessage)]

    def get_conversation_stats(self) -> dict:
        current_state = self.app.get_state(self.config)
        if not current_state:
            return {"message_count": 0, "has_summary": False}

        messages = current_state.values.get("messages", [])
        summary = current_state.values.get("summary", "")
        title = current_state.values.get("title", "")

        conversation_messages = self.get_conversation_messages(messages)

        return {
            "message_count": len(conversation_messages),
            "has_summary": bool(summary),
            "summary_preview": summary[:50] + "..." if len(summary) > 50 else summary,
            "summary": summary,
            "title": title,
            "should_summarize_recommended": self.check_should_summarize(),
            "analysis_stage": current_state.values.get("analysis_stage", "initial")
        }

    """RAG Util methods"""

    def search_with_rag(self, prompt: str):
        context = self.retriever.invoke(prompt)
        response = self.process_with_rag(prompt, context)
        return response

    def process_with_rag(self, question, context):
        return self.rag_chain.invoke({"question": question, "context": context})

    """Summary Util methods"""

    def trigger_summarize(self):
        input_message = HumanMessage(content=SUMMARY_PROMPT)
        # ID can be added during summary trigger, but avoid duplication as it's handled in summarize_conversation
        response = self.app.invoke(
            {"messages": [input_message], "should_summarize": True},
            self.config
        )
        self.save_chat_history()
        return response

    def check_should_summarize(self, message_count_threshold: int = 10) -> bool:
        current_state = self.app.get_state(self.config)
        if not current_state:
            return False
        messages = current_state.values.get("messages", [])
        conversation_messages = self.get_conversation_messages(messages)

        # Execute summary when total conversation count is a multiple of message_count_threshold
        # and is greater than or equal to message_count_threshold
        message_count = len(conversation_messages)
        return ((message_count >= message_count_threshold)
                and (message_count % message_count_threshold == 0))

    """LLM Response methods"""

    async def generate(self, prompt: str) -> str:
        response = await self.llm.ainvoke(prompt)
        return response.content

    async def generate_langchain(self, prompt: str, system: Optional[str] = None) -> str:
        try:
            messages = []
            if system:
                messages.append(SystemMessage(content=system))
            messages.append(HumanMessage(content=prompt))
            response = await self.app.ainvoke({"messages": messages}, self.config)

            self.save_chat_history()

            last_message = response['messages'][-1]
            # For LangChain objects
            if hasattr(last_message, 'content'):
                return last_message.content
            # For dictionary objects
            elif isinstance(last_message, dict):
                return last_message.get("content", "")
            return ""
        except Exception as e:
            return str(e)

    async def stream_generate(self, prompt: str):
        async for chunk in self.llm.astream(prompt):
            yield None, chunk.content
        self.save_chat_history()

    async def stream_generate_langchain(self, prompt: str, system: Optional[str] = None):
        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))

        async for chunk in self.app.astream(
                {"messages": messages}, self.config, stream_mode="messages",
        ):
            content_type = chunk[0].type
            content = chunk[0].content
            yield content_type, content

        self.save_chat_history()

    """Chat history methods"""

    def get_chat_history(self) -> Optional[ChatHistory]:
        snapshot = self.app.get_state(self.config)
        if not snapshot:
            return None

        history_values = snapshot.values.copy()
        history_values["messages"] = [msg for msg in history_values["messages"]]

        return ChatHistory(
            thread_id=str(self.thread_id),
            values=history_values,
            next=snapshot.next,
            config=dict(self.config),
            metadata=snapshot.metadata,
            created_at=snapshot.created_at,
            parent_config=snapshot.parent_config,
        )

    def save_chat_history(self):
        os.makedirs(settings.chat_path, exist_ok=True)
        chat_entry = self.get_chat_history()
        if not chat_entry:
            return

        filtered_messages = self.filter_system_message(chat_entry.values["messages"])
        if filtered_messages:
            chat_entry.values["messages"] = filtered_messages
            filepath = Path(settings.chat_path) / f"chat_{self.thread_id}.json"
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(chat_entry.model_dump_json(indent=2))

    @staticmethod
    def load_chat_history_file(thread_id: str) -> ChatHistory | None:
        """
        Load chat history from file.
        """
        filepath = Path(settings.chat_path) / f"chat_{thread_id}.json"

        if not filepath.exists():
            return None

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                return ChatHistory(**data)
        except Exception as e:
            return None

    def merge_chat_history(self, thread_id: str):
        record = self.load_chat_history_file(thread_id)
        if record is None:
            return

        loaded_messages = record.values.get("messages", [])
        current_messages = self.app.get_state(self.config).values.get("messages", [])
        filtered_loaded_messages = self.filter_system_message(loaded_messages, is_json=True)
        filtered_current_messages = self.filter_system_message(current_messages)

        merged_messages = filtered_loaded_messages + filtered_current_messages

        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)
        self.app.update_state(self.config, {"messages": merged_messages})

    def load_chat_history(self, thread_id: str):
        record = self.load_chat_history_file(thread_id)
        if record is None:
            return

        # Update thread_id and config to match current engine instance during load
        self.thread_id = uuid.UUID(record.thread_id)
        self.config['configurable']['thread_id'] = str(self.thread_id)

        # Refresh memory and update state
        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)
        self.app.update_state(self.config, record.values)

    def clear_memory(self):
        checkpoints = list(self.app.get_state_history(self.config))
        if len(checkpoints) > 1:
            previous_state = checkpoints[1].values
            self.app.update_state(self.config, previous_state)
