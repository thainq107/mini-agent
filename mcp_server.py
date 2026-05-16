import contextlib
import os
import sys

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from agent import ReActAgent
from llm import OpenAIClient

load_dotenv()

mcp = FastMCP("mini-research-agent")


@mcp.tool()
def ask_agent(question: str) -> str:
  """Run a ReAct agent that can answer multi-step questions using
  calculator, current time, and web search tools. Returns final answer
  as a string."""
  try:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
      return "Error: OPENAI_API_KEY not configured on server."

    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    client = OpenAIClient(api_key=api_key, model=model)
    agent = ReActAgent(client, max_iterations=10)

    # Redirect agent's print() logs to stderr so they don't corrupt
    # the MCP JSON-RPC stream on stdout.
    with contextlib.redirect_stdout(sys.stderr):
      return agent.run(question)
  except Exception as e:
    return f"Error running agent: {type(e).__name__}: {e}"


if __name__ == "__main__":
  mcp.run()
