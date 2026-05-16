from unittest.mock import MagicMock, patch

import pytest

from llm import OpenAIClient
from openai import APIError


@pytest.fixture
def mock_openai_api() -> MagicMock:
  """Mock OpenAI API."""
  with patch("llm.OpenAI") as mock:
    yield mock


class TestOpenAIClient:
  """Test OpenAIClient."""

  def test_initialization(self, mock_openai_api: MagicMock) -> None:
    client = OpenAIClient(api_key="test-key", model="gpt-4o")
    assert client.model == "gpt-4o"
    mock_openai_api.assert_called_once_with(api_key="test-key")

  def test_chat_without_tools(self, mock_openai_api: MagicMock) -> None:
    mock_response = MagicMock()
    mock_openai_api.return_value.chat.completions.create.return_value = mock_response

    client = OpenAIClient(api_key="test-key", model="gpt-4o")
    messages = [{"role": "user", "content": "Hello"}]

    response = client.chat(messages)

    assert response == mock_response
    mock_openai_api.return_value.chat.completions.create.assert_called_once()

  def test_chat_with_tools(self, mock_openai_api: MagicMock) -> None:
    mock_response = MagicMock()
    mock_openai_api.return_value.chat.completions.create.return_value = mock_response

    client = OpenAIClient(api_key="test-key", model="gpt-4o")
    messages = [{"role": "user", "content": "Hello"}]
    tools = [{"type": "function", "function": {"name": "test_tool"}}]

    response = client.chat(messages, tools=tools)

    assert response == mock_response
    call_kwargs = mock_openai_api.return_value.chat.completions.create.call_args[1]
    assert call_kwargs["tools"] == tools
    assert call_kwargs["tool_choice"] == "auto"

