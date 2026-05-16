import ast
import operator
import os
import time
from datetime import datetime
from typing import Any, Callable

import httpx

TAVILY_ENDPOINT = "https://api.tavily.com/search"
TAVILY_MAX_RETRIES = 2


def calculator(expression: str) -> str:
  """Evaluate a mathematical expression safely.

  Args:
      expression: A mathematical expression string (e.g., "2026 * 3").

  Returns:
      String result of the calculation.
  """
  try:
    node = ast.parse(expression, mode="eval")
    allowed_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    def evaluate(node: Any) -> Any:
      if isinstance(node, ast.Expression):
        return evaluate(node.body)
      elif isinstance(node, ast.Constant):
        return node.value
      elif isinstance(node, ast.BinOp):
        left = evaluate(node.left)
        right = evaluate(node.right)
        op = allowed_ops.get(type(node.op))
        if op is None:
          raise ValueError(f"Unsupported operator: {type(node.op)}")
        return op(left, right)
      elif isinstance(node, ast.UnaryOp):
        operand = evaluate(node.operand)
        if isinstance(node.op, ast.USub):
          return -operand
        elif isinstance(node.op, ast.UAdd):
          return operand
        else:
          raise ValueError(f"Unsupported unary operator: {type(node.op)}")
      else:
        raise ValueError(f"Unsupported node type: {type(node)}")

    result = evaluate(node)
    return str(result)
  except Exception as e:
    return f"Error: {str(e)}"


def get_current_year() -> str:
  """Get the current year.

  Returns:
      Current year as a string.
  """
  return str(datetime.now().year)


def web_search(query: str) -> dict[str, str | None]:
  """Search the web via Tavily API.

  Args:
      query: Search query string.

  Returns:
      Dict with keys "result" (concatenated top results) and "error"
      (None on success, error message otherwise).
  """
  api_key = os.getenv("TAVILY_API_KEY")
  if not api_key:
    return {"result": "", "error": "TAVILY_API_KEY not set"}

  payload = {
      "api_key": api_key,
      "query": query,
      "max_results": 3,
      "search_depth": "basic",
  }

  last_error: str = ""
  for attempt in range(TAVILY_MAX_RETRIES + 1):
    try:
      response = httpx.post(TAVILY_ENDPOINT, json=payload, timeout=10.0)
      status = response.status_code
      if status == 429:
        return {"result": "", "error": "rate limit (429)"}
      if 500 <= status < 600:
        last_error = f"server error {status}"
        if attempt < TAVILY_MAX_RETRIES:
          time.sleep(2 ** attempt)
          continue
        return {"result": "", "error": last_error}
      response.raise_for_status()
      data = response.json()
      results = data.get("results", []) or []
      snippets = []
      for item in results[:3]:
        title = item.get("title", "")
        content = item.get("content", "")
        url = item.get("url", "")
        snippets.append(f"- {title}: {content} ({url})")
      return {"result": "\n".join(snippets) if snippets else "No results.", "error": None}
    except httpx.TimeoutException as e:
      last_error = f"timeout: {e}"
      if attempt < TAVILY_MAX_RETRIES:
        time.sleep(2 ** attempt)
        continue
      return {"result": "", "error": last_error}
    except httpx.HTTPStatusError as e:
      return {"result": "", "error": f"http error: {e}"}
    except Exception as e:
      return {"result": "", "error": f"{type(e).__name__}: {e}"}

  return {"result": "", "error": last_error or "unknown error"}


def get_weather(city: str) -> str:
  """Get weather for a city (stub for lab).

  Args:
      city: City name.

  Returns:
      Fake weather result.
  """
  return f"Mock weather for {city}: 25°C, sunny."


TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "calculator": {
        "function": calculator,
        "schema": {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate a mathematical expression (e.g., '2026 * 3').",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate.",
                        }
                    },
                    "required": ["expression"],
                },
            },
        },
    },
    "get_current_year": {
        "function": get_current_year,
        "schema": {
            "type": "function",
            "function": {
                "name": "get_current_year",
                "description": "Get the current year.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
    },
    "web_search": {
        "function": web_search,
        "schema": {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web via Tavily for up-to-date information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query.",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
    },
    "get_weather": {
        "function": get_weather,
        "schema": {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name.",
                        }
                    },
                    "required": ["city"],
                },
            },
        },
    },
}


def get_openai_tools() -> list[dict[str, Any]]:
  """Get tool schemas formatted for OpenAI API.

  Returns:
      List of tool schema dicts for OpenAI function calling.
  """
  return [tool["schema"] for tool in TOOL_REGISTRY.values()]
