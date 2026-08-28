"""FastAPI REST routes for Auto Viral Cuts SaaS backend."""

import os
import shutil
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, status
from typing import Optional

from src.core.gemini_analyzer import GeminiAnalyzer
from src.core.schemas import HealthStatus, ProcessingOptions, ProcessingResult, ViralAnalysisResponse
from src.core.video_processor import VideoProcessor

from src.core.transcriber import AudioTranscriber

router = APIRouter(prefix="/api/v1", tags=["Auto Viral Cuts API"])

TEMP_DIR = os.getenv("TEMP_DIR", "temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)


@router.get("/health", response_model=HealthStatus)
def health_check() -> HealthStatus:
    """Checks system health, FFmpeg availability, Gemini configuration, GPU acceleration, and Whisper."""
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    gemini_key = os.getenv("GEMINI_API_KEY")
    gemini_ok = bool(gemini_key and len(gemini_key) > 5)

    hw_mode, _, gpu_name = VideoProcessor.detect_hw_accel()
    hw_ok = hw_mode.value != "cpu"
    whisper_ok = AudioTranscriber.is_available()

    return HealthStatus(
        status="healthy" if (ffmpeg_ok and gemini_ok) else "degraded",
        ffmpeg_available=ffmpeg_ok,
        gemini_configured=gemini_ok,
        hw_accel_available=hw_ok,
        hw_accel_backend=hw_mode.value,
        gpu_detected=gpu_name,
        whisper_available=whisper_ok,
        version="0.1.0",
    )


@router.post("/analyze", response_model=ViralAnalysisResponse)
async def analyze_video_endpoint(
    file: UploadFile = File(...),
    custom_prompt: Optional[str] = None,
    max_clips: int = 5,
) -> ViralAnalysisResponse:
    """Uploads a video and returns viral clip suggestions from Gemini AI."""
    temp_file_path = os.path.join(TEMP_DIR, file.filename or "uploaded_video.mp4")
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        analyzer = GeminiAnalyzer()
        options = ProcessingOptions(max_clips=max_clips, custom_prompt=custom_prompt)
        analysis = analyzer.analyze_video(temp_file_path, options)
        return analysis
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro na análise do vídeo: {str(e)}",
        )
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass
