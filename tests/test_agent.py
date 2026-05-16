from unittest.mock import MagicMock, patch

import pytest

from agent import ReActAgent
from llm import OpenAIClient


@pytest.fixture
def mock_client() -> MagicMock:
  """Mock OpenAI client."""
  return MagicMock(spec=OpenAIClient)


class TestReActAgent:
  """Test ReActAgent."""

  def test_agent_initialization(self, mock_client: MagicMock) -> None:
    agent = ReActAgent(mock_client, max_iterations=5)
    assert agent.max_iterations == 5
    assert agent.client == mock_client

  def test_agent_final_answer(self, mock_client: MagicMock) -> None:
    """Test agent returns final answer when LLM stops without tool calls."""
    mock_response = MagicMock()
    mock_response.choices[0].finish_reason = "stop"
    mock_response.choices[0].message.content = "The answer is 42"
    mock_client.chat.return_value = mock_response

    agent = ReActAgent(mock_client, max_iterations=10)
    result = agent.run("What is the answer?")

    assert "42" in result
    mock_client.chat.assert_called_once()

  def test_agent_tool_call(self, mock_client: MagicMock) -> None:
    """Test agent handles tool calls correctly."""
    tool_call = MagicMock()
    tool_call.id = "call_1"
    tool_call.function.name = "get_current_year"
    tool_call.function.arguments = "{}"

    first_response = MagicMock()
    first_response.choices[0].finish_reason = "tool_calls"
    first_response.choices[0].message.content = "Let me get the current year."
    first_response.choices[0].message.tool_calls = [tool_call]

    second_response = MagicMock()
    second_response.choices[0].finish_reason = "stop"
    second_response.choices[0].message.content = "The current year is 2026."

    mock_client.chat.side_effect = [first_response, second_response]

    agent = ReActAgent(mock_client, max_iterations=10)
    result = agent.run("What year is it?")

    assert "2026" in result
    assert mock_client.chat.call_count == 2

  def test_agent_max_iterations(self, mock_client: MagicMock) -> None:
    """Test agent stops at max iterations."""
    mock_response = MagicMock()
    mock_response.choices[0].finish_reason = "tool_calls"
    mock_response.choices[0].message.content = "Thinking..."
    mock_response.choices[0].message.tool_calls = []

    mock_client.chat.return_value = mock_response

    agent = ReActAgent(mock_client, max_iterations=2)
    result = agent.run("Some question")

    assert "Max iterations" in result
    assert mock_client.chat.call_count == 2

  def test_agent_unknown_tool(self, mock_client: MagicMock) -> None:
    """Test agent handles unknown tool gracefully."""
    tool_call = MagicMock()
    tool_call.id = "call_1"
    tool_call.function.name = "unknown_tool"
    tool_call.function.arguments = "{}"

    first_response = MagicMock()
    first_response.choices[0].finish_reason = "tool_calls"
    first_response.choices[0].message.content = "Using unknown tool."
    first_response.choices[0].message.tool_calls = [tool_call]

    second_response = MagicMock()
    second_response.choices[0].finish_reason = "stop"
    second_response.choices[0].message.content = "Tool not found."

    mock_client.chat.side_effect = [first_response, second_response]

    agent = ReActAgent(mock_client, max_iterations=10)
    result = agent.run("Unknown tool question")

    assert mock_client.chat.call_count == 2

  def test_agent_calculator_tool(self, mock_client: MagicMock) -> None:
    """Test agent uses calculator tool correctly."""
    tool_call = MagicMock()
    tool_call.id = "call_1"
    tool_call.function.name = "calculator"
    tool_call.function.arguments = '{"expression": "2026 * 3"}'

    first_response = MagicMock()
    first_response.choices[0].finish_reason = "tool_calls"
    first_response.choices[0].message.content = "I'll calculate 2026 times 3."
    first_response.choices[0].message.tool_calls = [tool_call]

    second_response = MagicMock()
    second_response.choices[0].finish_reason = "stop"
    second_response.choices[0].message.content = "The result is 6078."

    mock_client.chat.side_effect = [first_response, second_response]

    agent = ReActAgent(mock_client, max_iterations=10)
    result = agent.run("What is 2026 * 3?")

    assert "6078" in result
