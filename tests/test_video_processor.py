"""Unit tests for VideoProcessor with AMD GPU / VAAPI acceleration."""

import os
import subprocess
import pytest
from src.core.schemas import ClipMetadata, CropMode, HwAccelMode, ProcessingOptions
from src.core.video_processor import VideoProcessor


def test_parse_timestamp():
    vp = VideoProcessor()
    assert vp.parse_timestamp("00:01:30") == 90.0
    assert vp.parse_timestamp("01:30") == 90.0
    assert vp.parse_timestamp("45") == 45.0
    assert vp.parse_timestamp("45.5") == 45.5
    assert vp.parse_timestamp("01:00:00") == 3600.0

    with pytest.raises(ValueError):
        vp.parse_timestamp("invalid:time:stamp:too:long")


def test_detect_hw_accel():
    mode, device, gpu_name = VideoProcessor.detect_hw_accel()
    assert isinstance(mode, HwAccelMode)
    # On this AMD RX 570 Linux system, it should detect VAAPI
    if os.path.exists("/dev/dri/renderD128") and "h264_vaapi" in VideoProcessor.get_supported_encoders():
        assert mode == HwAccelMode.VAAPI
        assert device == "/dev/dri/renderD128"


def test_build_video_filter():
    vp = VideoProcessor()

    # Center crop
    vf_center = vp._build_video_filter(1920, 1080, CropMode.CENTER_CROP, is_vaapi=False)
    assert "crop=" in vf_center
    assert "1080:1920" in vf_center
    assert "hwupload" not in vf_center

    # Center crop with VAAPI
    vf_center_vaapi = vp._build_video_filter(1920, 1080, CropMode.CENTER_CROP, is_vaapi=True)
    assert "hwupload" in vf_center_vaapi
    assert "format=nv12" in vf_center_vaapi
    assert "scale_vaapi=w=1080:h=1920" in vf_center_vaapi

    # Blurred background
    vf_blur = vp._build_video_filter(1920, 1080, CropMode.BLURRED_BACKGROUND, is_vaapi=False)
    assert "split" in vf_blur
    assert "boxblur" in vf_blur
    assert "overlay" in vf_blur

    # Blurred background with VAAPI
    vf_blur_vaapi = vp._build_video_filter(1920, 1080, CropMode.BLURRED_BACKGROUND, is_vaapi=True)
    assert "split" in vf_blur_vaapi
    assert "boxblur" in vf_blur_vaapi
    assert "format=nv12,hwupload" in vf_blur_vaapi
    assert "scale_vaapi=w=1080:h=1920" in vf_blur_vaapi



def test_build_ffmpeg_cmd_vaapi():
    vp = VideoProcessor()
    options = ProcessingOptions(hw_accel=HwAccelMode.VAAPI)
    cmd, desc = vp._build_ffmpeg_cmd(
        video_path="input.mp4",
        output_path="output.mp4",
        start_sec=10.0,
        duration=15.0,
        options=options,
        in_w=1920,
        in_h=1080,
        resolved_accel=HwAccelMode.VAAPI,
        vaapi_device="/dev/dri/renderD128",
    )
    assert "-init_hw_device" in cmd
    assert "vaapi=va:/dev/dri/renderD128" in cmd
    assert "-c:v" in cmd
    assert "h264_vaapi" in cmd
    assert "AMD GPU" in desc


def test_build_ffmpeg_cmd_cpu():
    vp = VideoProcessor()
    options = ProcessingOptions(hw_accel=HwAccelMode.CPU)
    cmd, desc = vp._build_ffmpeg_cmd(
        video_path="input.mp4",
        output_path="output.mp4",
        start_sec=10.0,
        duration=15.0,
        options=options,
        in_w=1920,
        in_h=1080,
        resolved_accel=HwAccelMode.CPU,
    )
    assert "-init_hw_device" not in cmd
    assert "-c:v" in cmd
    assert "libx264" in cmd
    assert "cpu" in desc


def test_end_to_end_clip_rendering(tmp_path):
    """Generates a real 1-second synthetic video and extracts a clip with VideoProcessor."""
    # 1. Create a dummy 2-second mp4 video
    test_video = str(tmp_path / "test_input.mp4")
    gen_cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=2:size=1920x1080:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=1000:duration=2",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac",
        test_video,
    ]
    subprocess.run(gen_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    # 2. Process clip with Auto hardware acceleration
    output_dir = str(tmp_path / "cuts")
    vp = VideoProcessor(output_dir=output_dir)
    clip_meta = ClipMetadata(
        title="Hook Incrivel",
        start_time="00:00:00",
        end_time="00:00:01",
        virality_score=95,
        virality_reason="Gancho impactante",
    )
    options = ProcessingOptions(
        hw_accel=HwAccelMode.AUTO,
        crop_mode=CropMode.BLURRED_BACKGROUND,
        burn_subtitles=False,
    )
    processed = vp.process_clip(test_video, clip_meta, 1, options)

    assert processed.status == "completed"
    assert os.path.exists(processed.file_path)
    assert processed.file_size_bytes > 0
    assert processed.hw_accel_used is not None


def test_end_to_end_clip_rendering_with_subtitles(tmp_path):
    """Verifies that a clip is rendered with burned-in subtitles on GPU/CPU."""
    from src.core.transcriber import WordTimestamp

    test_video = str(tmp_path / "test_input2.mp4")
    gen_cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=2:size=1920x1080:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=1000:duration=2",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac",
        test_video,
    ]
    subprocess.run(gen_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    output_dir = str(tmp_path / "cuts2")
    vp = VideoProcessor(output_dir=output_dir)
    clip_meta = ClipMetadata(
        title="Hook Legendado",
        start_time="00:00:00",
        end_time="00:00:01",
        virality_score=90,
        virality_reason="Com fala",
    )
    options = ProcessingOptions(
        hw_accel=HwAccelMode.AUTO,
        crop_mode=CropMode.CENTER_CROP,
        burn_subtitles=True,
    )
    dummy_words = [
        WordTimestamp("TESTE", 0.1, 0.5),
        WordTimestamp("LEGENDA", 0.5, 0.9),
    ]
    processed = vp.process_clip(test_video, clip_meta, 1, options, all_words=dummy_words)

    assert processed.status == "completed"
    assert processed.has_subtitles is True
    assert processed.subtitle_path is not None
    assert os.path.exists(processed.subtitle_path)
    assert os.path.exists(processed.file_path)
    assert vp.get_video_dimensions(processed.file_path) == (1080, 1920)


def test_4k_downscale_to_1080x1920_resolution(tmp_path):
    """Verifies that 4K UHD (3840x2160) input is accurately cropped and downscaled to 1080x1920."""
    test_video_4k = str(tmp_path / "test_input_4k.mp4")
    gen_cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=1:size=3840x2160:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=1000:duration=1",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac",
        test_video_4k,
    ]
    subprocess.run(gen_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    output_dir = str(tmp_path / "cuts_4k")
    vp = VideoProcessor(output_dir=output_dir)
    clip_meta = ClipMetadata(
        title="Corte 4K Downscaled",
        start_time="00:00:00",
        end_time="00:00:01",
        virality_score=98,
        virality_reason="4K Test Downscale",
    )
    options = ProcessingOptions(
        hw_accel=HwAccelMode.AUTO,
        crop_mode=CropMode.CENTER_CROP,
        burn_subtitles=False,
    )
    processed = vp.process_clip(test_video_4k, clip_meta, 1, options)

    assert processed.status == "completed"
    assert os.path.exists(processed.file_path)
    # Output must be standardized to exactly 1080x1920 vertical format
    out_w, out_h = vp.get_video_dimensions(processed.file_path)
    assert (out_w, out_h) == (1080, 1920)

