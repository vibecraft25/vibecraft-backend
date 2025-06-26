# fastmcp (fastmcp >= 2.0.0)
### https://gofastmcp.com/getting-started/welcome
### https://github.com/jlowin/fastmcp

## Set Environment
```bash
mkdir vibecraft
cd vibecraft

# Create Python venv
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Unix or MacOS:
source .venv/bin/activate

# install uv & project init
pip install uv
uv init
uv add fastmcp
# (Optional)
uv add mcp anthropic

# Remove boilerplate files
# On Windows:
del main.py
# On Unix or MacOS:
rm main.py
```

### run mcp server
- stdio(Standard Input/Output): 호스트에서 직접 실행하는 방식
- sse(Server-Sent Events): 웹서버를 실행하고 나서 호출하는 방식
- Streamable HTTP: 일반 HTTP 요청/응답을 사용하지만, response body를 chunk 단위로 스트리밍
```bash
fastmcp run servers/samples/server_sample_v2.py
or
fastmcp dev servers/samples/server_sample_v2.py
or
fastmcp install servers/samples/server_sample_v2.py
```

- Install Custom Servers to Claude
```bash
fastmcp install servers/topic_server.py
fastmcp install servers/data_upload_server.py --with pandas
```
