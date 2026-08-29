import os
import shutil
import uuid
from typing import List, Optional
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, status

from src.core.gemini_analyzer import GeminiAnalyzer
from src.core.schemas import (
    ClientCutManifest,
    HealthStatus,
    PlatformPreset,
    ProcessingOptions,
    ProcessingResult,
    ViralAnalysisResponse,
    WordTimestamp,
)
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


@router.post("/generate-manifest", response_model=List[ClientCutManifest])
async def generate_manifest_endpoint(
    file: UploadFile = File(...),
    custom_prompt: Optional[str] = None,
    max_clips: int = 5,
    min_duration_seconds: int = 15,
    max_duration_seconds: int = 60,
    target_platform: PlatformPreset = PlatformPreset.GENERAL,
    crop_mode: str = "center_crop",
    whisper_model: str = "base",
) -> List[ClientCutManifest]:
    """
    Receives an audio or video file, runs Gemini intelligence + Whisper transcription,
    and returns a structured client cut manifest for browser-side WebCodecs rendering.
    """
    ext = os.path.splitext(file.filename or "media.wav")[1].lower()
    if not ext:
        ext = ".wav"
    temp_file_name = f"manifest_input_{uuid.uuid4().hex[:8]}{ext}"
    temp_file_path = os.path.join(TEMP_DIR, temp_file_name)

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Transcribe audio with word-level timestamps using faster-whisper
        if not AudioTranscriber.is_available():
            raise RuntimeError("faster-whisper não está disponível no servidor.")

        transcriber = AudioTranscriber()
        all_words = transcriber.transcribe(temp_file_path, model_size=whisper_model)

        # 2. Analyze video/audio with Gemini to extract viral clips
        analyzer = GeminiAnalyzer()
        options = ProcessingOptions(
            max_clips=max_clips,
            min_duration_seconds=min_duration_seconds,
            max_duration_seconds=max_duration_seconds,
            target_platform=target_platform,
            custom_prompt=custom_prompt,
        )
        analysis = analyzer.analyze_video(temp_file_path, options)

        # 3. Map clips to word timestamps
        vp = VideoProcessor()
        manifests: List[ClientCutManifest] = []

        for idx, clip in enumerate(analysis.clips):
            start_sec = vp.parse_timestamp(clip.start_time)
            end_sec = vp.parse_timestamp(clip.end_time)

            if end_sec <= start_sec:
                continue

            # Filter and re-align words to relative milliseconds from clip start
            clip_words: List[WordTimestamp] = []
            for w in all_words:
                if w.end > start_sec and w.start < end_sec:
                    start_ms = max(0, int(round((w.start - start_sec) * 1000)))
                    end_ms = max(start_ms + 80, int(round((w.end - start_sec) * 1000)))
                    clip_words.append(
                        WordTimestamp(
                            word=w.word,
                            start_ms=start_ms,
                            end_ms=end_ms,
                        )
                    )

            manifests.append(
                ClientCutManifest(
                    cut_id=f"cut_{idx + 1}_{uuid.uuid4().hex[:6]}",
                    title=clip.title,
                    start_sec=round(start_sec, 2),
                    end_sec=round(end_sec, 2),
                    viral_score=clip.virality_score,
                    hook=clip.hook_summary or clip.title,
                    crop_mode=crop_mode,
                    words=clip_words,
                )
            )

        return manifests

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao gerar manifesto para cliente: {str(e)}",
        )
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass

