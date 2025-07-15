__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
from typing import List, Dict

# Third-party imports
from mcp import ClientSession, ClientSessionGroup


class BaseEngine:
    async def generate(self, prompt: str) -> str:
        raise NotImplementedError

    async def generate_with_tools(
        self, prompt: str, tools: List[Dict], session: ClientSession | ClientSessionGroup
    ) -> str:
        raise NotImplementedError
