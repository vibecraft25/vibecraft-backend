# https://smithery.ai/search?q=web%20search
# TODO: WIP
"""
MCP Claude Client: Source Code Automation Workflow

- Features:
  1. Generate source code (generate_source_code)
  2. (Optional) Build source code (build_generated_code)
  3. Generate GitHub workflow code (generate_github_workflow)
  4. Push to GitHub merged source code (push_to_github)

Backend server assumed to expose these tools via FastMCP and Claude MCP.
"""

import asyncio
from fastmcp import Client

# Server endpoint (stdio or HTTP/SSE)
# Option 1: stdio transport
# client = Client("my_server.py")
# Option 2: HTTP/SSE transport
SERVER_URL = "http://localhost:8090/mcp"


async def main():
    # Initialize client
    client = Client(SERVER_URL)
    print(f"Connecting to MCP server at {client.target}")

    try:
        async with client:
            print("--- Connected to MCP server ---")

            # 1. Generate source code
            prompt = "Create a Python project for data processing"
            gen_resp = await client.call_tool("generate_source_code", {"prompt": prompt})
            print(f"Source generation status: {gen_resp.get('status')}")
            print(f"Generated files: {gen_resp.get('generated_files')}")

            # 2. Build generated code (optional)
            build_resp = await client.call_tool("build_generated_code", {})
            print(f"Build status: {build_resp.get('status')}")
            if build_resp.get('stdout'):
                print(f"Build output:\n{build_resp.get('stdout')}")

            # 3. Generate GitHub Actions workflow
            wf_prompt = "Setup CI for Python tests and linting"
            wf_resp = await client.call_tool("generate_github_workflow", {"prompt": wf_prompt})
            print(f"Workflow file created at: {wf_resp.get('workflow_file')}")

            # 4. Push to GitHub
            repo_url = "git@github.com:user/generated-project.git"
            commit_msg = "Initial commit: auto-generated project"
            push_resp = await client.call_tool("push_to_github", {"repo_url": repo_url, "commit_msg": commit_msg})
            print(f"Push status: {push_resp.get('status')}")

    except Exception as e:
        print(f"Error during MCP client interaction: {e}")
    finally:
        print("--- Client session ended ---")

if __name__ == "__main__":
    asyncio.run(main())
