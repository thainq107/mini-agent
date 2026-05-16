from unittest.mock import MagicMock, patch

import httpx
import pytest

from tools import calculator, get_current_year, get_weather, web_search


class TestCalculator:
  """Test calculator tool."""

  def test_basic_multiplication(self) -> None:
    result = calculator("2026 * 3")
    assert result == "6078"

  def test_basic_addition(self) -> None:
    result = calculator("10 + 5")
    assert result == "15"

  def test_basic_subtraction(self) -> None:
    result = calculator("10 - 3")
    assert result == "7"

  def test_division(self) -> None:
    result = calculator("10 / 2")
    assert result == "5.0"

  def test_floor_division(self) -> None:
    result = calculator("10 // 3")
    assert result == "3"

  def test_modulo(self) -> None:
    result = calculator("10 % 3")
    assert result == "1"

  def test_power(self) -> None:
    result = calculator("2 ** 3")
    assert result == "8"

  def test_negative_number(self) -> None:
    result = calculator("-5 + 3")
    assert result == "-2"

  def test_complex_expression(self) -> None:
    result = calculator("(2 + 3) * 4")
    assert result == "20"

  def test_division_by_zero(self) -> None:
    result = calculator("10 / 0")
    assert "Error" in result

  def test_invalid_expression(self) -> None:
    result = calculator("invalid_expr")
    assert "Error" in result


class TestGetCurrentYear:
  """Test get_current_year tool."""

  def test_returns_year(self) -> None:
    result = get_current_year()
    assert result.isdigit()
    assert len(result) == 4


class TestWebSearch:
  """Test web_search tool (Tavily)."""

  def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "results": [
            {"title": "Hanoi", "content": "Capital of Vietnam", "url": "https://x"},
        ]
    }
    with patch("tools.httpx.post", return_value=mock_resp) as mock_post:
      result = web_search("capital of Vietnam")
    assert result["error"] is None
    assert "Hanoi" in result["result"]
    mock_post.assert_called_once()

  def test_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
    monkeypatch.setattr("tools.time.sleep", lambda _: None)
    with patch("tools.httpx.post", side_effect=httpx.TimeoutException("slow")):
      result = web_search("anything")
    assert result["result"] == ""
    assert "timeout" in result["error"].lower()

  def test_rate_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    with patch("tools.httpx.post", return_value=mock_resp):
      result = web_search("anything")
    assert result["result"] == ""
    assert "429" in result["error"] or "rate" in result["error"].lower()

  def test_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    result = web_search("anything")
    assert result["result"] == ""
    assert "TAVILY_API_KEY" in result["error"]


class TestGetWeather:
  """Test get_weather tool."""

  def test_get_weather_returns_string(self) -> None:
    result = get_weather("London")
    assert isinstance(result, str)
    assert "Mock weather" in result or len(result) > 0
