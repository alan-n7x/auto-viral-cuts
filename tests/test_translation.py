"""Tests for PT-BR audiovisual translation and original language preservation."""

import io
import os
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from src.core.gemini_analyzer import GeminiAnalyzer
from src.core.schemas import (
    ClipMetadata,
    CropMode,
    ProcessingOptions,
    SubtitleCue,
    SubtitleStyle,
    ViralAnalysisResponse,
)
from src.core.subtitle_generator import SubtitleGenerator
from src.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_subtitle_cue_and_clip_metadata_translation_schema():
    """Validates SubtitleCue and ClipMetadata with translation fields."""
    cue = SubtitleCue(start=14.5, end=17.2, text="Primeira frase traduzida sincronizada")
    assert cue.start == 14.5
    assert cue.end == 17.2
    assert cue.text == "Primeira frase traduzida sincronizada"

    clip = ClipMetadata(
        title="Original English Title",
        title_pt="Título chamativo e instigante",
        start_time="00:00:14",
        end_time="00:00:52",
        virality_score=95,
        virality_reason="High adrenaline robbery scene",
        reason_pt="Cena de ação com alta retenção",
        hook_summary="Original hook",
        hook_pt="Frase de impacto inicial do corte",
        subtitles_pt=[cue],
    )

    assert clip.title_pt == "Título chamativo e instigante"
    assert clip.hook_pt == "Frase de impacto inicial do corte"
    assert clip.reason_pt == "Cena de ação com alta retenção"
    assert len(clip.subtitles_pt) == 1
    assert clip.subtitles_pt[0].text == "Primeira frase traduzida sincronizada"


def test_gemini_analyzer_build_prompt_translation_mode():
    """Validates that _build_prompt injects the audiovisual translator prompt when translate_to_pt is True."""
    analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
    options = ProcessingOptions(
        max_clips=5,
        min_duration_seconds=20,
        max_duration_seconds=60,
        translate_to_pt=True,
    )
    prompt = analyzer._build_prompt(options)

    assert "editor sênior especializado em cortes virais" in prompt
    assert "tradutor audiovisual para Português do Brasil (PT-BR)" in prompt
    assert "title_pt e hook_pt" in prompt
    assert "subtitles_pt" in prompt
    assert "Tradução dinâmica (PT-BR)" in prompt


def test_gemini_analyzer_build_prompt_original_mode_preserves_original():
    """Validates that _build_prompt instructs preserving original language when translate_to_pt is False."""
    analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
    options = ProcessingOptions(
        max_clips=3,
        min_duration_seconds=15,
        max_duration_seconds=45,
        translate_to_pt=False,
    )
    prompt = analyzer._build_prompt(options)

    assert "Mantenha os títulos, ganchos e falas no idioma original" in prompt
    assert "tradutor audiovisual para Português do Brasil" not in prompt


def test_subtitle_generator_generate_ass_from_cues(tmp_path):
    """Validates ASS generation from translated phrase-level SubtitleCue objects."""
    gen = SubtitleGenerator()
    cues = [
        SubtitleCue(start=10.0, end=13.5, text="Mantenha sua equipe unida"),
        SubtitleCue(start=14.0, end=18.0, text="Nós vamos conseguir sair daqui"),
    ]
    out_ass = str(tmp_path / "translated.ass")
    gen.generate_ass_from_cues(cues, out_ass, style=SubtitleStyle.HORMOZI, offset_start_sec=10.0)

    assert os.path.exists(out_ass)
    with open(out_ass, "r", encoding="utf-8") as f:
        content = f.read()

    assert "[Script Info]" in content
    assert "PlayResX: 1080" in content
    assert "PlayResY: 1920" in content
    assert "ViralStyle" in content
    # First cue offset by 10s is 0:00:00.00
    assert "0:00:00.00" in content
    assert "MANTENHA" in content
    assert "SUA EQUIPE UNIDA" in content



def test_generate_manifest_endpoint_with_translation(client):
    """Validates /generate-manifest endpoint with translate_to_pt=true."""
    fake_audio = b"RIFF" + b"\x00" * 2000
    files = {"file": ("english_clip.wav", io.BytesIO(fake_audio), "audio/wav")}
    data = {
        "max_clips": 1,
        "translate_to_pt": "true",
    }

    mock_analysis = ViralAnalysisResponse(
        video_summary="English action scene",
        key_themes=["action", "gaming"],
        clips=[
            ClipMetadata(
                title="Original English Title",
                title_pt="Título em Português",
                start_time="00:00:05",
                end_time="00:00:25",
                virality_score=92,
                virality_reason="High action",
                reason_pt="Ação frenética",
                hook_summary="English Hook",
                hook_pt="Gancho em Português",
                subtitles_pt=[
                    SubtitleCue(start=5.5, end=8.0, text="Primeira fala traduzida"),
                    SubtitleCue(start=8.2, end=12.0, text="Segunda fala traduzida"),
                ],
            )
        ],
    )

    with patch("src.core.gemini_analyzer.GeminiAnalyzer.analyze_video", return_value=mock_analysis), \
         patch("src.core.transcriber.AudioTranscriber.is_available", return_value=True), \
         patch("src.core.transcriber.AudioTranscriber.transcribe", return_value=[]):

        response = client.post("/api/v1/generate-manifest", files=files, data=data)

        assert response.status_code == 200
        manifests = response.json()
        assert len(manifests) == 1
        cut = manifests[0]
        assert cut["title"] == "Título em Português"
        assert cut["title_pt"] == "Título em Português"
        assert cut["hook"] == "Gancho em Português"
        assert cut["hook_pt"] == "Gancho em Português"
        assert len(cut["subtitles_pt"]) == 2
        assert cut["subtitles_pt"][0]["text"] == "Primeira fala traduzida"
