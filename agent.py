import json
from typing import Any

from llm import OpenAIClient
from tools import TOOL_REGISTRY, get_openai_tools


class ReActAgent:
  """ReAct agent using OpenAI function calling."""

  def __init__(self, client: OpenAIClient, max_iterations: int = 10) -> None:
    """Initialize ReAct agent.

    Args:
        client: OpenAIClient instance.
        max_iterations: Max iterations before stopping (default: 10).
    """
    self.client = client
    self.max_iterations = max_iterations

  def run(self, question: str) -> str:
    """Run agent loop to answer a question.

    Args:
        question: The question to answer.

    Returns:
        Final answer from the agent.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": self._build_system_prompt()},
        {"role": "user", "content": question},
    ]

    for iteration in range(self.max_iterations):
      response = self.client.chat(messages, tools=get_openai_tools())

      choice = response.choices[0]
      finish_reason = choice.finish_reason

      if finish_reason == "tool_calls":
        thought = choice.message.content or ""
        if thought:
          print(f"[Iter {iteration + 1}] Thought: {thought}")

        assistant_msg = {"role": "assistant", "content": choice.message.content or ""}
        if choice.message.tool_calls:
          assistant_msg["tool_calls"] = choice.message.tool_calls
        messages.append(assistant_msg)

        if choice.message.tool_calls:
          for tool_call in choice.message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"[Iter {iteration + 1}] Action: {tool_name}({tool_args})")

            observation = self._call_tool(tool_name, tool_args)
            print(f"[Iter {iteration + 1}] Observation: {observation}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": observation,
            })
      else:
        final_answer = choice.message.content or ""
        print(f"[Final] Answer: {final_answer}")
        return final_answer

    print("[Max iterations reached]")
    return "Max iterations reached without final answer."

  def _call_tool(self, name: str, args: dict[str, Any]) -> str:
    """Call a tool from TOOL_REGISTRY.

    Args:
        name: Tool name.
        args: Tool arguments dict.

    Returns:
        Tool result as string.
    """
    if name not in TOOL_REGISTRY:
      return f"Error: Unknown tool '{name}'"

    tool_fn: Any = TOOL_REGISTRY[name]["function"]
    try:
      result = tool_fn(**args)
      return str(result)
    except Exception as e:
      return f"Error: {str(e)}"

  def _build_system_prompt(self) -> str:
    """Build system prompt for ReAct agent.

    Returns:
        System prompt string.
    """
    return """You are a helpful ReAct agent. Think step by step.

When you need to use tools, use them to gather information.
When you have enough information to answer, provide your final answer.

Use the available tools to:
- Perform calculations (calculator)
- Get the current year (get_current_year)
- Search information (search_web)
- Get weather (get_weather)

Think carefully about what tools you need, then use them."""
