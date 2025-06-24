### MCP Server
- MCP 서버: 특정 기능(도구)을 제공하고, MCP 메시지를 받아 처리.  
    예: 파일 시스템 접근, 데이터 쿼리, HTTP 호출 등.
- MCP 클라이언트: MCP 서버에 요청을 보내고 응답을 받아 작업을 수행.

## Set Environment
### 1. Create Python venv
python -m venv [가상환경이름]
```bash
python -m venv vibecraft
```
- Window
```bash
vibecraft\Scripts\activate
```
- Mac/Linux
```bash
source vibecraft/bin/activate
```

### 2. install uv & project init
```bash
pip install uv
uv init
uv venv
```
### 3. install packages
```bash
uv add "mcp[cli]"
```
### 4. run mcp server
- stdio(Standard Input/Output): 호스트에서 직접 실행하는 방식
- sse(Server-Sent Events): 웹서버를 실행하고 나서 호출하는 방식
- Streamable HTTP: 일반 HTTP 요청/응답을 사용하지만, response body를 chunk 단위로 스트리밍
```bash
mcp dev server.py
or
mcp install server.py # with Claude Desktop
```

# WIP
mcp install topic.py
mcp install data_collector.py
mcp install code_generator.py
mcp install deploy.py
