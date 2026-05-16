import time
from typing import Any, Optional

from openai import OpenAI, APIError


class OpenAIClient:
  """Wrapper for OpenAI API with retry logic and exponential backoff."""

  def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
    """Initialize OpenAI client.

    Args:
        api_key: OpenAI API key.
        model: Model name (default: gpt-4o).
    """
    self.client = OpenAI(api_key=api_key)
    self.model = model

  def chat(
      self,
      messages: list[dict[str, Any]],
      tools: Optional[list[dict[str, Any]]] = None,
      max_retries: int = 3,
  ) -> Any:
    """Call OpenAI chat API with retry logic.

    Args:
        messages: List of message dicts (role, content).
        tools: Optional list of tool definitions (OpenAI function format).
        max_retries: Max retry attempts (default: 3).

    Returns:
        ChatCompletion response object.

    Raises:
        APIError: If max retries exceeded.
    """
    backoff_times = [1, 2, 4]

    for attempt in range(max_retries):
      try:
        kwargs = {"model": self.model, "messages": messages}
        if tools:
          kwargs["tools"] = tools
          kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)
        return response
      except APIError as e:
        if attempt < max_retries - 1:
          wait_time = backoff_times[attempt]
          print(f"API error (attempt {attempt + 1}), retrying in {wait_time}s...")
          time.sleep(wait_time)
        else:
          raise
