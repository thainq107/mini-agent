import pytest

from tools import calculator, get_current_year, search_web, get_weather


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


class TestSearchWeb:
  """Test search_web tool."""

  def test_search_web_returns_string(self) -> None:
    result = search_web("test query")
    assert isinstance(result, str)
    assert "Mock search result" in result or len(result) > 0


class TestGetWeather:
  """Test get_weather tool."""

  def test_get_weather_returns_string(self) -> None:
    result = get_weather("London")
    assert isinstance(result, str)
    assert "Mock weather" in result or len(result) > 0
