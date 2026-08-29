"""Unit and integration tests for ClientCutManifest, WordTimestamp, and /generate-manifest endpoint."""

import io
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from src.core.schemas import (
    ClientCutManifest,
    ClipMetadata,
    ViralAnalysisResponse,
    WordTimestamp,
)
from src.core.transcriber import WordTimestamp as TranscriberWord
from src.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_word_timestamp_schema():
    """Validates WordTimestamp schema creation and field validation."""
    wt = WordTimestamp(word="inteligência", start_ms=250, end_ms=800)
    assert wt.word == "inteligência"
    assert wt.start_ms == 250
    assert wt.end_ms == 800

    data = wt.model_dump()
    assert data == {"word": "inteligência", "start_ms": 250, "end_ms": 800}


def test_client_cut_manifest_schema():
    """Validates ClientCutManifest schema structure and serialization."""
    words = [
        WordTimestamp(word="este", start_ms=0, end_ms=200),
        WordTimestamp(word="momento", start_ms=220, end_ms=500),
        WordTimestamp(word="viral", start_ms=520, end_ms=900),
    ]

    manifest = ClientCutManifest(
        cut_id="cut_1_a1b2c3",
        title="O Segredo do Sucesso",
        start_sec=15.0,
        end_sec=45.0,
        viral_score=95,
        hook="Você não vai acreditar nisso...",
        crop_mode="center_crop",
        words=words,
    )

    assert manifest.cut_id == "cut_1_a1b2c3"
    assert manifest.title == "O Segredo do Sucesso"
    assert manifest.start_sec == 15.0
    assert manifest.end_sec == 45.0
    assert manifest.viral_score == 95
    assert manifest.crop_mode == "center_crop"
    assert len(manifest.words) == 3
    assert manifest.words[2].word == "viral"


def test_word_relative_alignment_logic():
    """Tests that word timestamps are correctly converted to milliseconds relative to clip start."""
    start_sec = 10.0
    end_sec = 20.0

    raw_words = [
        TranscriberWord(word="fora_antes", start=5.0, end=8.0),
        TranscriberWord(word="primeira", start=10.2, end=10.8),
        TranscriberWord(word="segunda", start=11.0, end=11.5),
        TranscriberWord(word="ultima", start=19.2, end=19.8),
        TranscriberWord(word="fora_depois", start=21.0, end=23.0),
    ]

    aligned_words = []
    for w in raw_words:
        if w.end > start_sec and w.start < end_sec:
            start_ms = max(0, int(round((w.start - start_sec) * 1000)))
            end_ms = max(start_ms + 80, int(round((w.end - start_sec) * 1000)))
            aligned_words.append(WordTimestamp(word=w.word, start_ms=start_ms, end_ms=end_ms))

    assert len(aligned_words) == 3
    assert aligned_words[0].word == "primeira"
    assert aligned_words[0].start_ms == 200  # (10.2 - 10.0) * 1000 = 200ms
    assert aligned_words[0].end_ms == 800    # (10.8 - 10.0) * 1000 = 800ms

    assert aligned_words[1].word == "segunda"
    assert aligned_words[1].start_ms == 1000
    assert aligned_words[1].end_ms == 1500

    assert aligned_words[2].word == "ultima"
    assert aligned_words[2].start_ms == 9200
    assert aligned_words[2].end_ms == 9800


@patch("src.api.routes.GeminiAnalyzer")
@patch("src.api.routes.AudioTranscriber")
def test_generate_manifest_endpoint(mock_transcriber_cls, mock_analyzer_cls, client):
    """Tests the /api/v1/generate-manifest endpoint with mocked Gemini and Whisper services."""
    # Mock transcriber
    mock_transcriber_cls.is_available.return_value = True
    mock_transcriber_inst = MagicMock()
    mock_transcriber_inst.transcribe.return_value = [
        TranscriberWord(word="bem-vindos", start=10.5, end=11.2),
        TranscriberWord(word="ao", start=11.3, end=11.5),
        TranscriberWord(word="futuro", start=11.6, end=12.5),
    ]
    mock_transcriber_cls.return_value = mock_transcriber_inst

    # Mock analyzer
    mock_analyzer_inst = MagicMock()
    mock_analyzer_inst.analyze_video.return_value = ViralAnalysisResponse(
        video_summary="Vídeo sobre inovação e tecnologia",
        key_themes=["tecnologia", "futuro"],
        clips=[
            ClipMetadata(
                title="Momento Inovador",
                start_time="00:00:10",
                end_time="00:00:25",
                virality_score=92,
                virality_reason="Gancho extremamente forte e visual",
                hook_summary="Veja o que está acontecendo",
            )
        ],
    )
    mock_analyzer_cls.return_value = mock_analyzer_inst

    # Send dummy audio file
    fake_wav_content = b"RIFF....WAVEfmt ....data...."
    files = {"file": ("test_audio.wav", io.BytesIO(fake_wav_content), "audio/wav")}
    data = {
        "max_clips": 3,
        "crop_mode": "center_crop",
        "custom_prompt": "Foco em inovação",
    }

    response = client.post("/api/v1/generate-manifest", files=files, data=data)

    assert response.status_code == 200
    manifests = response.json()
    assert isinstance(manifests, list)
    assert len(manifests) == 1

    cut = manifests[0]
    assert cut["title"] == "Momento Inovador"
    assert cut["start_sec"] == 10.0
    assert cut["end_sec"] == 25.0
    assert cut["viral_score"] == 92
    assert cut["hook"] == "Veja o que está acontecendo"
    assert cut["crop_mode"] == "center_crop"
    assert len(cut["words"]) == 3

    # Check first word relative timing
    first_word = cut["words"][0]
    assert first_word["word"] == "bem-vindos"
    assert first_word["start_ms"] == 500  # (10.5 - 10.0) * 1000
    assert first_word["end_ms"] == 1200   # (11.2 - 10.0) * 1000


def test_root_endpoint_includes_client(client):
    """Verifies that the root redirect includes /client documentation."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["client"] == "/client"
    assert data["ui"] == "/ui"
    assert data["docs"] == "/docs"


def test_client_studio_static_mount(client):
    """Verifies that the WebCodecs Client Studio HTML interface is served at /client/."""
    response = client.get("/client/")
    assert response.status_code == 200
    assert "Auto Viral Cuts" in response.text
    assert "previewCanvas" in response.text
    assert "WebCodecs" in response.text

