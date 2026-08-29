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


def test_extract_lightweight_media_proxy(tmp_path):
    """Verifies that heavy video files are converted to lightweight audio proxies for Gemini."""
    import os
    import subprocess

    # 1. Create a dummy video file with video + audio
    video_path = str(tmp_path / "heavy_test_video.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=2:size=1280x720:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=1000:duration=2",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac",
        video_path,
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    # 2. Extract proxy
    proxy_path, is_temp, orig_size, proxy_size = GeminiAnalyzer.extract_lightweight_media_proxy(video_path)

    assert is_temp is True
    assert proxy_path.endswith("_proxy.mp3")
    assert os.path.exists(proxy_path)
    assert proxy_size > 0
    assert orig_size > 0
    # Proxy MP3 should be smaller than video container
    assert proxy_size < orig_size

    # Clean up proxy
    if os.path.exists(proxy_path):
        os.remove(proxy_path)

    # 3. Audio file bypass check
    audio_path = str(tmp_path / "direct_audio.mp3")
    cmd_audio = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "sine=frequency=1000:duration=1",
        "-c:a", "libmp3lame",
        audio_path,
    ]
    subprocess.run(cmd_audio, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    bypass_path, bypass_is_temp, _, _ = GeminiAnalyzer.extract_lightweight_media_proxy(audio_path)
    assert bypass_is_temp is False
    assert bypass_path == audio_path

