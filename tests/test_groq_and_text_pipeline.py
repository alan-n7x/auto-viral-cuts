"""Tests for Groq integration, text-based transcript LLM pipeline, and audio/subtitle separation."""

import io
import os
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from src.core.groq_analyzer import GroqAnalyzer
from src.core.schemas import (
    AiProvider,
    ClipMetadata,
    ProcessingOptions,
    SubtitleCue,
    SubtitleLanguage,
    ViralAnalysisResponse,
)
from src.core.transcriber import AudioTranscriber, WordTimestamp
from src.infrastructure.adapters.local_processor_adapter import LocalProcessorAdapter
from src.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_format_transcript_with_timestamps():
    """Validates that words are formatted into readable timestamped sentences."""
    words = [
        WordTimestamp(word="Keep", start=10.0, end=10.5),
        WordTimestamp(word="your", start=10.6, end=10.9),
        WordTimestamp(word="crew", start=11.0, end=11.4),
        WordTimestamp(word="together.", start=11.5, end=12.2),
        WordTimestamp(word="The", start=14.0, end=14.3),
        WordTimestamp(word="only", start=14.4, end=14.7),
        WordTimestamp(word="way.", start=14.8, end=15.2),
    ]

    formatted = AudioTranscriber.format_transcript_with_timestamps(words, max_pause_sec=0.8)
    lines = formatted.strip().split("\n")

    assert len(lines) == 2
    assert "00:00:10.000 -> 00:00:12.200" in lines[0]
    assert "Keep your crew together." in lines[0]
    assert "00:00:14.000 -> 00:00:15.200" in lines[1]
    assert "The only way." in lines[1]


def test_groq_analyzer_build_prompt_pt_br():
    """Validates that Groq prompt builder generates PT-BR audiovisual translator directives."""
    analyzer = GroqAnalyzer(api_key="gsk_dummy")
    options = ProcessingOptions(
        max_clips=3,
        min_duration_seconds=20,
        max_duration_seconds=60,
        ai_provider=AiProvider.GROQ,
        subtitle_language=SubtitleLanguage.PT_BR,
    )
    formatted_transcript = "[00:00:10.000 -> 00:00:15.000] This is a viral test statement."
    prompt = analyzer._build_prompt(formatted_transcript, options)

    assert "editor sênior especializado em cortes virais" in prompt
    assert "tradutor audiovisual para Português do Brasil (PT-BR)" in prompt
    assert "TRANSCRIÇÃO COM TIMESTAMPS:" in prompt
    assert formatted_transcript in prompt


def test_groq_analyzer_analyze_transcript_mock():
    """Validates that GroqAnalyzer parses JSON response from Groq LPU."""
    analyzer = GroqAnalyzer(api_key="gsk_dummy")
    mock_json = """{
        "video_summary": "Resumo dos melhores momentos",
        "key_themes": ["ação", "estratégia"],
        "clips": [
            {
                "title": "A Virada Épica",
                "title_pt": "A Virada Épica",
                "start_time": "00:00:10",
                "end_time": "00:00:40",
                "virality_score": 98,
                "virality_reason": "Momento de alta tensão",
                "reason_pt": "Momento de alta tensão",
                "hook_summary": "Você não vai acreditar nisso",
                "hook_pt": "Você não vai acreditar nisso",
                "suggested_caption": "Olha o que aconteceu!",

                "hashtags": ["#viral", "#cortes"],
                "subtitles_pt": [
                    {
                        "start": 10.5,
                        "end": 14.0,
                        "text": "Nós precisamos agir agora"
                    }
                ]
            }
        ]
    }"""

    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = mock_json
    mock_completion.choices = [mock_choice]
    mock_completion.usage = MagicMock(total_tokens=150)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completion
    analyzer._client = mock_client

    options = ProcessingOptions(subtitle_language=SubtitleLanguage.PT_BR)
    result = analyzer.analyze_transcript("Sample transcript", options)

    assert isinstance(result, ViralAnalysisResponse)
    assert len(result.clips) == 1
    assert result.clips[0].title_pt == "A Virada Épica"
    assert result.clips[0].virality_score == 98
    assert len(result.clips[0].subtitles_pt) == 1
    assert result.clips[0].subtitles_pt[0].text == "Nós precisamos agir agora"



def test_local_processor_adapter_delegates_to_groq(tmp_path):
    """Validates that LocalProcessorAdapter delegates transcript analysis to GroqAnalyzer."""
    fake_video = str(tmp_path / "sample.mp4")
    with open(fake_video, "wb") as f:
        f.write(b"fake mp4 content")

    mock_words = [WordTimestamp(word="Ola", start=1.0, end=2.0)]
    mock_transcriber = MagicMock(spec=AudioTranscriber)
    mock_transcriber.is_available.return_value = True
    mock_transcriber.transcribe.return_value = mock_words

    mock_analysis = ViralAnalysisResponse(
        video_summary="Teste Groq",
        key_themes=["teste"],
        clips=[
            ClipMetadata(
                title="Corte 1",
                start_time="00:00:01",
                end_time="00:00:02",
                virality_score=90,
                virality_reason="Forte",
            )
        ],
    )

    mock_groq = MagicMock(spec=GroqAnalyzer)
    mock_groq.analyze_transcript.return_value = mock_analysis

    adapter = LocalProcessorAdapter(transcriber=mock_transcriber, groq_analyzer=mock_groq)
    options = ProcessingOptions(ai_provider=AiProvider.GROQ, groq_api_key="gsk_test")

    manifests = adapter.generate_manifest(fake_video, options)

    assert len(manifests) == 1
    assert manifests[0].title == "Corte 1"
    mock_groq.analyze_transcript.assert_called_once()


def test_generate_manifest_endpoint_with_groq(client):
    """Validates /generate-manifest endpoint with ai_provider=groq and subtitle_language=pt_br."""
    fake_audio = b"RIFF" + b"\x00" * 1500
    files = {"file": ("audio_sample.wav", io.BytesIO(fake_audio), "audio/wav")}
    data = {
        "max_clips": 1,
        "ai_provider": "groq",
        "subtitle_language": "pt_br",
        "groq_api_key": "gsk_test_key",
    }

    mock_words = [WordTimestamp(word="hello", start=5.0, end=7.0)]
    mock_analysis = ViralAnalysisResponse(
        video_summary="Groq summary",
        key_themes=["groq"],
        clips=[
            ClipMetadata(
                title="English Original",
                title_pt="Título em Português pelo Groq",
                start_time="00:00:05",
                end_time="00:00:20",
                virality_score=96,
                virality_reason="Groq LPU retention",
                reason_pt="Retenção alta pelo Groq",
                hook_summary="English hook",
                hook_pt="Gancho em Português",
                subtitles_pt=[
                    SubtitleCue(start=5.0, end=7.0, text="Olá mundo traduzido pelo Groq")
                ],
            )
        ],
    )

    with patch("src.core.transcriber.AudioTranscriber.is_available", return_value=True), \
         patch("src.core.transcriber.AudioTranscriber.transcribe", return_value=mock_words), \
         patch("src.core.groq_analyzer.GroqAnalyzer.analyze_transcript", return_value=mock_analysis):

        response = client.post("/api/v1/generate-manifest", files=files, data=data)

        assert response.status_code == 200
        manifests = response.json()
        assert len(manifests) == 1
        cut = manifests[0]
        assert cut["title"] == "Título em Português pelo Groq"
        assert cut["subtitle_language"] == "pt_br"
        assert len(cut["subtitles_pt"]) == 1
        assert cut["subtitles_pt"][0]["text"] == "Olá mundo traduzido pelo Groq"


def test_groq_audio_compression(tmp_path):
    """Validates that GroqAnalyzer compresses audio to stay under 25MB limit."""
    import subprocess
    analyzer = GroqAnalyzer(api_key="gsk_dummy")

    # Generate a dummy wav with ffmpeg
    sample_wav = str(tmp_path / "sample.wav")
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "sine=frequency=1000:duration=1",
        "-ar", "44100",
        "-ac", "2",
        sample_wav,
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    compressed = analyzer._compress_audio_for_groq(sample_wav)
    assert os.path.exists(compressed)
    assert compressed.endswith(".mp3")
    assert os.path.getsize(compressed) < os.path.getsize(sample_wav)
    if compressed != sample_wav and os.path.exists(compressed):
        os.remove(compressed)


def test_groq_transcribe_audio_fast_mock(tmp_path):
    """Validates that transcribe_audio_fast creates TranscriberWord objects with start/end in seconds."""
    analyzer = GroqAnalyzer(api_key="gsk_dummy")

    fake_audio = str(tmp_path / "test.mp3")
    with open(fake_audio, "wb") as f:
        f.write(b"fake audio data")

    mock_raw_transcription = MagicMock()
    mock_raw_transcription.words = [
        {"word": "Yo,", "start": 0.0, "end": 5.46},
        {"word": "welcome", "start": 5.5, "end": 6.2},
    ]
    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.return_value = mock_raw_transcription
    analyzer._client = mock_client

    words = analyzer.transcribe_audio_fast(fake_audio)
    assert len(words) == 2
    assert words[0].word == "Yo,"
    assert words[0].start == 0.0
    assert words[0].end == 5.46
    assert words[1].word == "welcome"
    assert words[1].start == 5.5
    assert words[1].end == 6.2


