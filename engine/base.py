__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
from typing import List, Dict, Tuple

# Third-party imports
from mcp import ClientSession

class BaseEngine:
    async def generate_with_tools(
        self, prompt: str, tools: List[Dict], session: ClientSession
    ) -> Tuple[str, bool, bool]:
        raise NotImplementedError
