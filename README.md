# VibeCraft

**VibeCraft** is an automated pipeline for generating data-driven web pages based on user-defined topics. It integrates large language models (LLMs) like **Claude**, **OpenAI GPT**, and **Gemini** with the **MCP (Modular Control Pipeline)** ecosystem to streamline the entire workflow—from topic selection to web page code generation.

---

## 🚀 Overview

This project consists of four main stages:

1. **Topic Definition**
   - Receives a user prompt and uses an AI model (Claude/GPT/Gemini) to generate and formalize a topic.
   - The topic is passed to downstream modules via MCP tools.

2. **Data Collection or Upload**
   - If the user provides data, it is saved as CSV or SQLite format.
   - If no data is uploaded, the system automatically searches and scrapes topic-relevant data from the web, cleans it, and stores it locally.

3. **Code Generation**
   - Uses the collected data to generate a complete web page with visualization, layout structure, and UI components.

4. **Auto Deployment (WIP)**
   - The generated web page is automatically deployed to the **Vercel** platform using the `deploy_client`.
   - Once deployment is complete, the user receives the URL to access the published web page.
---

## 🧰 MCP & Environment Setup

This project is built on the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction), which enables modular communication between clients and tools via structured protocols.

### 🔌 MCP Components

- **MCP Server**: Provides specific functionality (e.g., file I/O, HTTP calls, database operations) via tools.  
- **MCP Client**: Interacts with MCP servers by sending requests and receiving structured responses.

### 🛠 Environment Setup
#### 1. Clone the repository
```bash
git clone https://github.com/vibecraft25/vibecraft-mcp.git
cd vibecraft-mcp
```
#### 2. Install [`uv`](https://github.com/astral-sh/uv) (Python project manager)
```bash
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# MacOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```
#### 3. Create and activate the virtual environment
```bash
uv venv --python=python3.12
# Windows
.venv\Scripts\activate
# MacOS/Linux
source .venv/bin/activate

uv init
```
#### 4. Install dependencies
```bash
# Essential packages
uv add mcp[cli]   # Windows
uv add "mcp[cli]" # MacOS/Linux
uv add langchain langchain-google-genai google-generativeai langchain-anthropic
uv add langchain_community
uv add langchain-mcp-adapters langgraph
uv add langchain_mcp_adapters
uv add grandalf   # Optional

# Additional packages
uv add pandas
uv add chardet
```
#### 5. Install required npm packages
```bash
# Download and install Node.js from the official website:
#👉 https://nodejs.org
npm -v
npm install -g @aakarsh-sasi/memory-bank-mcp
```
#### 6. Add .env for your API keys
```bash
touch .env
```
### .env File Format
⚠️Do not share or commit your .env file. It contains sensitive credentials.⚠️
```text
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
export GOOGLE_API_KEY=...
```

## 🧠 Engine Architecture

Each engine implements a common interface via `BaseEngine`:

- `ClaudeEngine` – Uses Anthropic Claude - [claude-3-5-sonnet-20241022].
- `OpenAIEngine` – Uses OpenAI GPT - [gpt-4.1].
- `GeminiEngine` – Uses Google Gemini - [gemini-2.5-flash].

Each engine supports:
- Multi-turn conversation
- Dynamic tool invocation via MCP
- Text and function response handling

---

## ⚙️ How It Works

1. Choose a model: `claude`, `gpt`, or `gemini`
2. Enter a prompt to define the topic
3. The pipeline will:
   - Connect to each server (topic, data, code)
   - Call relevant MCP tools
   - Proceed through 3 stages unless "redo" or "go back" flags are detected

### Example

```bash
$ python main.py
✅ Choose a model: claude / gemini / gpt (default: claude)
🎤 Enter a topic prompt:
```

```plaintext
.
├── engine/
│   ├── base.py               # Abstract base engine
│   ├── claude_engine.py      # Claude model integration
│   ├── openai_engine.py      # OpenAI GPT integration
│   └── gemini_engine.py      # Gemini model integration
│
├── client/
│   └── vibe_craft_client.py  # Main pipeline client using MCP stdio
│
├── schemas/
│   └── pipeline_schemas.py   # Pipeline schemas
│
├── utils/
│   ├── data_loader_utils.py  # Data load and File utils
│   ├── menus.py              # User menu options
│   └── prompts.py            # LLM prompts
│
├── main.py                   # Entry point for running the pipeline
├── .env                      # Environment variables (optional)
└── README.md
```

## ✅ Features
- 🔧 Pluggable model engines (Claude, GPT, Gemini)
- 🧠 Intelligent prompt-to-topic generation
- 🌐 Web scraping fallback for missing user data
- 💻 Code generation with charting and visualization
- 🔁 Stage navigation via redo / go back keywords

### References
- https://python.langchain.com/docs/introduction/
- https://python.langchain.com/docs/versions/migrating_memory/conversation_buffer_memory/
- https://digitalbourgeois.tistory.com/1017
- https://velog.io/@exoluse/AI-langchain-mcp-%EA%B5%AC%ED%98%84-%EB%8F%84%EC%A4%91-%EB%AC%B8%EC%A0%9C
- https://markbyun.tistory.com/entry/How-to-use-Gemini-API-via-LangChain
- https://sean-j.tistory.com/entry/LangGraph-Delete-Messages
- https://langchain-ai.github.io/langgraph/how-tos/memory/add-memory/
- https://rudaks.tistory.com/entry/langgraph-%EB%8C%80%ED%99%94-%EC%9D%B4%EB%A0%A5%EC%9D%84-%EC%9A%94%EC%95%BD%ED%95%98%EB%8A%94-%EB%B0%A9%EB%B2%95
- https://rudaks.tistory.com/entry/langgraph-FastAPI%EC%99%80-LangGraph%EB%A1%9C-%EA%B8%B0%EB%B3%B8-RAG%EB%A5%BC-Streaming%EC%9C%BC%EB%A1%9C-%EC%B6%9C%EB%A0%A5%ED%95%98%EA%B8%B0