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
uv venv
# Windows
.venv\Scripts\activate
# MacOS/Linux
source .venv/bin/activate
```
#### 4. Install dependencies
```bash
# Essential packages
uv add mcp[cli]
uv add anthropic
uv add google-genai
uv add openai

# Additional packages
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
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
```

## 🧠 Engine Architecture

Each engine implements a common interface via `BaseEngine`:

- `ClaudeEngine` – Uses Anthropic Claude - [version].
- `OpenAIEngine` – Uses OpenAI GPT - [version].
- `GeminiEngine` – Uses Google Gemini - [version].

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
│   └── tools.py              # MCP tool spec extractor
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