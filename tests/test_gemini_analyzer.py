"""Unit tests for GeminiAnalyzer utility functions and model fallback resolution."""

from src.core.gemini_analyzer import GeminiAnalyzer
from src.core.schemas import PlatformPreset, ProcessingOptions


def test_resolve_model_name():
    # Deprecated model mapping
    assert GeminiAnalyzer._resolve_model_name("gemini-2.5-flash") == "gemini-3.6-flash"
    assert GeminiAnalyzer._resolve_model_name("models/gemini-2.5-flash") == "gemini-3.6-flash"

    # Current model names
    assert GeminiAnalyzer._resolve_model_name("gemini-3.6-flash") == "gemini-3.6-flash"
    assert GeminiAnalyzer._resolve_model_name("gemini-3.5-flash") == "gemini-3.5-flash"


def test_extract_retry_delay():
    e1 = Exception("Resource has been exhausted (e.g. check quota). Please retry in 15.5s.")
    assert GeminiAnalyzer._extract_retry_delay(e1) == 16.0

    e2 = Exception("Rate limit exceeded. Retry after 5s.")
    assert GeminiAnalyzer._extract_retry_delay(e2) == 5.5

    e3 = Exception("Unknown server error")
    assert GeminiAnalyzer._extract_retry_delay(e3, default_delay=10.0) == 10.0


def test_is_transient_model_error():
    assert GeminiAnalyzer._is_transient_model_error(Exception("The model is overloaded due to high demand"))
    assert GeminiAnalyzer._is_transient_model_error(Exception("HTTP 429 Too Many Requests"))
    assert GeminiAnalyzer._is_transient_model_error(Exception("RateLimitError: quota exceeded"))
    assert not GeminiAnalyzer._is_transient_model_error(Exception("Invalid API key provided"))


def test_build_prompt():
    # Analyzer prompt building
    analyzer = GeminiAnalyzer(api_key="test_key_dummy")
    options = ProcessingOptions(
        target_platform=PlatformPreset.TIKTOK,
        max_clips=3,
        min_duration_seconds=15,
        max_duration_seconds=45,
        custom_prompt="Foque nos momentos mais engracados",
    )
    prompt = analyzer._build_prompt(options)

    assert "até 3 trechos" in prompt
    assert "entre 15 e 45 segundos" in prompt
    assert "estilo TikTok" in prompt
    assert "Foque nos momentos mais engracados" in prompt
