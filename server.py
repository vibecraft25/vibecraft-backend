""" Sample """
from mcp.server.fastmcp import FastMCP


# Create an MCP clients
mcp = FastMCP("vivecraft")


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


def run_server(transport: str = "stdio"):
    """MCP 서버를 실행합니다.

    Args:
        transport: 통신 방식 ("stdio" 또는 "sse")
    """
    mcp.run(transport=transport)


if __name__ == "__main__":
    mcp.run()
