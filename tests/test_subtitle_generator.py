"""Unit tests for SubtitleGenerator and AudioTranscriber interval clipping."""

import os
import subprocess
from src.core.schemas import SubtitleStyle
from src.core.subtitle_generator import SubtitleGenerator
from src.core.transcriber import AudioTranscriber, WordTimestamp


def test_format_ass_time():
    sg = SubtitleGenerator()
    assert sg.format_ass_time(0.0) == "0:00:00.00"
    assert sg.format_ass_time(1.5) == "0:00:01.50"
    assert sg.format_ass_time(65.25) == "0:01:05.25"
    assert sg.format_ass_time(3661.12) == "1:01:01.12"


def test_style_config():
    sg = SubtitleGenerator()
    hormozi = sg.get_style_config(SubtitleStyle.HORMOZI)
    assert hormozi["bold"] == 1
    assert hormozi["highlight_color"] == "&H0000FFFF"  # Yellow
    assert hormozi["uppercase"] is True

    neon = sg.get_style_config(SubtitleStyle.NEON)
    assert neon["highlight_color"] == "&H00FFFF00"  # Cyan

    minimal = sg.get_style_config(SubtitleStyle.MINIMAL)
    assert minimal["uppercase"] is False


def test_generate_ass_file(tmp_path):
    sg = SubtitleGenerator()
    words = [
        WordTimestamp("ISSO", 0.1, 0.4),
        WordTimestamp("VAI", 0.45, 0.7),
        WordTimestamp("VIRALIZAR", 0.75, 1.2),
    ]
    output_ass = str(tmp_path / "subtitles.ass")
    res_path = sg.generate_ass(words, output_ass, style=SubtitleStyle.HORMOZI)

    assert os.path.exists(res_path)
    with open(res_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "[Script Info]" in content
    assert "PlayResX: 1080" in content
    assert "PlayResY: 1920" in content
    assert "Style: ViralStyle" in content
    assert "Dialogue:" in content
    assert "VIRALIZAR" in content


def test_audio_transcriber_get_words_for_interval():
    words = [
        WordTimestamp("Antes", 1.0, 2.0),
        WordTimestamp("Dentro1", 5.2, 5.8),
        WordTimestamp("Dentro2", 6.0, 6.7),
        WordTimestamp("Depois", 12.0, 13.0),
    ]
    interval_words = AudioTranscriber.get_words_for_interval(words, start_sec=5.0, end_sec=8.0)

    assert len(interval_words) == 2
    assert interval_words[0].word == "Dentro1"
    # Adjusted relative to start_sec=5.0
    assert abs(interval_words[0].start - 0.2) < 0.05
    assert abs(interval_words[0].end - 0.8) < 0.05
    assert interval_words[1].word == "Dentro2"


def test_ffmpeg_burn_in_subtitles(tmp_path):
    """Verifies that FFmpeg with libass successfully burns subtitles into a video frame."""
    sg = SubtitleGenerator()
    words = [
        WordTimestamp("MOMENTO", 0.0, 0.5),
        WordTimestamp("EPICO", 0.5, 1.0),
    ]
    output_ass = str(tmp_path / "test.ass")
    sg.generate_ass(words, output_ass, style=SubtitleStyle.HORMOZI)

    output_video = str(tmp_path / "burned.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=1:size=1080x1920:rate=30",
        "-vf", f"subtitles={output_ass}",
        "-c:v", "libx264", "-preset", "ultrafast",
        output_video,
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode == 0
    assert os.path.exists(output_video)
    assert os.path.getsize(output_video) > 0
