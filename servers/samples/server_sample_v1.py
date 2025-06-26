""" Sample """
from mcp.server.fastmcp import FastMCP


# Create an MCP servers
mcp = FastMCP("name=vibecraft_v1")


# Add an additional tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location."""

    # 항상 맑음으로 리턴한다.
    # 실제로는 api를 조회하여 결과를 가져오게 하면 된다.

    return f"{location}은 항상 맑아요~~"


# Add a dynamic greeing resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"


def run_server():
    """MCP 서버를 실행합니다."""

    mcp.run()


if __name__ == "__main__":
    mcp.run()
