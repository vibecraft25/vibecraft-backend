__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"


def extract_tool_specs(tools) -> list[dict]:
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.inputSchema
        } for t in tools.tools
    ]
