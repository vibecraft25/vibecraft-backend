# mcp (fastmcp < 2.0.0)
### https://modelcontextprotocol.io/introduction
### https://github.com/modelcontextprotocol/python-sdk?tab=readme-ov-file#overview


### MCP Server
- MCP 서버: 특정 기능(도구)을 제공하고, MCP 메시지를 받아 처리.  
    예: 파일 시스템 접근, 데이터 쿼리, HTTP 호출 등.
- MCP 클라이언트: MCP 서버에 요청을 보내고 응답을 받아 작업을 수행.

## Set Environment
```bash
# On Windows:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# On Unix or MacOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create project directory
uv init vibecraft
cd vibecraft

# Create virtual environment
uv venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Unix or MacOS:
source .venv/bin/activate

# Install required packages
uv add "mcp[cli]"
# (Optional)
uv add mcp anthropic python-dotenv

# Remove boilerplate files
# On Windows:
del main.py
# On Unix or MacOS:
rm main.py

# Create our main file
touch client.py
```

### run mcp server
- stdio(Standard Input/Output): 호스트에서 직접 실행하는 방식
- sse(Server-Sent Events): 웹서버를 실행하고 나서 호출하는 방식
- Streamable HTTP: 일반 HTTP 요청/응답을 사용하지만, response body를 chunk 단위로 스트리밍
```bash
mcp run servers/samples/server_sample_v1.py
or
mcp dev servers/samples/server_sample_v1.py
or
mcp install servers/samples/server_sample_v1.py # with Claude Desktop
```

- mcp install topic_server.py
- mcp install data_upload_server.py
