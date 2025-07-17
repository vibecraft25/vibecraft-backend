__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
from typing import List, Dict

# Third-party imports
from mcp import ClientSession, ClientSessionGroup


class BaseEngine:
    """ Basic generation method without LangChain or tools """
    async def generate(self, prompt: str) -> str:
        raise NotImplementedError

    """ Generation using LangChain without tool integration """
    async def generate_langchain(self, prompt: str) -> str:
        raise NotImplementedError

    """ Generation using LangChain with external tool integration (via MCP) """
    async def generate_langchain_with_tools(
        self, prompt: str, tools: List[Dict], session: ClientSession | ClientSessionGroup
    ) -> str:
        raise NotImplementedError
