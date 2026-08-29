import os
import shutil
import uuid
from typing import List, Optional

import aiofiles
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)

from src.api.dependencies import get_process_video_use_case, get_task_manager
from src.application.task_manager import TaskManager
from src.application.use_cases.process_video_use_case import ProcessVideoUseCase
from src.core.gemini_analyzer import GeminiAnalyzer
from src.core.groq_analyzer import GroqAnalyzer
from src.core.schemas import (
    AiProvider,
    AsyncTaskResponse,
    ClientCutManifest,
    CropMode,
    HealthStatus,
    HwAccelMode,
    PlatformPreset,
    ProcessingOptions,
    ProcessingResult,
    SubtitleCue,
    SubtitleLanguage,
    SubtitleStyle,
    TaskState,
    TaskStatusResponse,
    ViralAnalysisResponse,
    WordTimestamp,
)

from src.core.transcriber import AudioTranscriber
from src.core.video_processor import VideoProcessor

router = APIRouter(prefix="/api/v1", tags=["Auto Viral Cuts API"])

TEMP_DIR = os.getenv("TEMP_DIR", "temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)

CHUNK_SIZE = 64 * 1024  # 64 KB chunks for NVMe high performance and low RAM consumption


async def save_upload_file_stream(upload_file: UploadFile, destination_path: str) -> int:
    """Streams an UploadFile to disk in 64KB chunks using aiofiles.

    Avoids RAM memory spikes and leverages NVMe sequential write speed.
    """
    total_bytes = 0
    async with aiofiles.open(destination_path, "wb") as out_file:
        while chunk := await upload_file.read(CHUNK_SIZE):
            await out_file.write(chunk)
            total_bytes += len(chunk)
    return total_bytes


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


@router.post(
    "/process-video-async",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=AsyncTaskResponse,
)
async def process_video_async_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    custom_prompt: Optional[str] = Form(None),
    max_clips: int = Form(5),
    min_duration_seconds: int = Form(15),
    max_duration_seconds: int = Form(60),
    crop_mode: CropMode = Form(CropMode.CENTER_CROP),
    hw_accel: HwAccelMode = Form(HwAccelMode.AUTO),
    burn_subtitles: bool = Form(True),
    subtitle_style: SubtitleStyle = Form(SubtitleStyle.HORMOZI),
    target_platform: PlatformPreset = Form(PlatformPreset.GENERAL),
    translate_to_pt: bool = Form(False),
    ai_provider: AiProvider = Form(AiProvider.GROQ),
    subtitle_language: SubtitleLanguage = Form(SubtitleLanguage.ORIGINAL),
    groq_api_key: Optional[str] = Form(None),
    use_case: ProcessVideoUseCase = Depends(get_process_video_use_case),
    manager: TaskManager = Depends(get_task_manager),
) -> AsyncTaskResponse:

    """Receives a video file, streams it asynchronously to disk in 64KB chunks,

    and enqueues background processing, returning HTTP 202 Accepted immediately.
    """
    task_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename or "video.mp4")[1].lower()
    if not ext:
        ext = ".mp4"
    dest_path = os.path.join(TEMP_DIR, f"task_{task_id}{ext}")

    # 1. Stream file to disk in 64KB chunks with aiofiles (non-blocking, NVMe optimized)
    await save_upload_file_stream(file, dest_path)

    # 2. Register task in TaskManager
    task = manager.create_task(
        task_id=task_id, file_name=file.filename or "uploaded_video.mp4"
    )

    # 3. Build processing options
    options = ProcessingOptions(
        max_clips=max_clips,
        min_duration_seconds=min_duration_seconds,
        max_duration_seconds=max_duration_seconds,
        crop_mode=crop_mode,
        hw_accel=hw_accel,
        burn_subtitles=burn_subtitles,
        subtitle_style=subtitle_style,
        target_platform=target_platform,
        translate_to_pt=(subtitle_language == SubtitleLanguage.PT_BR or translate_to_pt),
        ai_provider=ai_provider,
        subtitle_language=subtitle_language,
        groq_api_key=groq_api_key,
        custom_prompt=custom_prompt,
    )



    # 4. Dispatch Use Case execution in background
    background_tasks.add_task(use_case.execute, task_id, dest_path, options)

    return AsyncTaskResponse(
        task_id=task_id,
        status=task.status,
        message="Vídeo recebido e enfileirado com sucesso para processamento em background.",
        file_name=task.file_name,
        created_at=task.created_at,
    )


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
def get_task_status_endpoint(
    task_id: str,
    manager: TaskManager = Depends(get_task_manager),
) -> TaskStatusResponse:
    """Returns the current execution status and generated cuts for a background task."""
    task = manager.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tarefa com ID '{task_id}' não encontrada.",
        )
    return task


@router.post("/analyze", response_model=ViralAnalysisResponse)
async def analyze_video_endpoint(
    file: UploadFile = File(...),
    custom_prompt: Optional[str] = None,
    max_clips: int = 5,
) -> ViralAnalysisResponse:
    """Uploads a video and returns viral clip suggestions from Gemini AI."""
    temp_file_path = os.path.join(TEMP_DIR, file.filename or "uploaded_video.mp4")
    try:
        await save_upload_file_stream(file, temp_file_path)

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
    custom_prompt: Optional[str] = Form(None),
    max_clips: int = Form(5),
    min_duration_seconds: int = Form(15),
    max_duration_seconds: int = Form(60),
    target_platform: PlatformPreset = Form(PlatformPreset.GENERAL),
    crop_mode: str = Form("center_crop"),
    whisper_model: str = Form("base"),
    translate_to_pt: bool = Form(False),
    ai_provider: AiProvider = Form(AiProvider.GROQ),
    subtitle_language: SubtitleLanguage = Form(SubtitleLanguage.ORIGINAL),
    groq_api_key: Optional[str] = Form(None),
) -> List[ClientCutManifest]:

    """Receives an audio or video file, runs speech transcription, formats text with timestamps,

    and queries Groq LLaMA 3.3 70B or Gemini to return a structured client cut manifest.
    """
    ext = os.path.splitext(file.filename or "media.wav")[1].lower()
    if not ext:
        ext = ".wav"
    temp_file_name = f"manifest_input_{uuid.uuid4().hex[:8]}{ext}"
    temp_file_path = os.path.join(TEMP_DIR, temp_file_name)

    try:
        await save_upload_file_stream(file, temp_file_path)

        # 1. Speech-to-text transcription (Groq Whisper cloud or local faster-whisper)
        all_words: List[WordTimestamp] = []
        if ai_provider == AiProvider.GROQ and (groq_api_key or os.getenv("GROQ_API_KEY")):
            try:
                groq_inst = GroqAnalyzer(api_key=groq_api_key)
                all_words = groq_inst.transcribe_audio_fast(temp_file_path)
            except Exception as e:
                print(f"Aviso: Transcrição via Groq Cloud falhou ({e}), usando faster-whisper local...")

        if not all_words:
            if not AudioTranscriber.is_available():
                raise RuntimeError("faster-whisper não está disponível no servidor.")
            transcriber = AudioTranscriber()
            all_words = transcriber.transcribe(temp_file_path, model_size=whisper_model)

        formatted_transcript = AudioTranscriber.format_transcript_with_timestamps(all_words)

        # 2. Processing configuration
        should_translate = (
            subtitle_language == SubtitleLanguage.PT_BR
            or translate_to_pt
        )
        options = ProcessingOptions(
            max_clips=max_clips,
            min_duration_seconds=min_duration_seconds,
            max_duration_seconds=max_duration_seconds,
            target_platform=target_platform,
            translate_to_pt=should_translate,
            ai_provider=ai_provider,
            subtitle_language=subtitle_language,
            groq_api_key=groq_api_key,
            custom_prompt=custom_prompt,
        )

        # 3. Analyze formatted transcript via selected LLM (Groq LPU or Gemini)
        if ai_provider == AiProvider.GROQ and (groq_api_key or os.getenv("GROQ_API_KEY")):
            groq_inst = GroqAnalyzer(api_key=groq_api_key)
            if formatted_transcript:
                analysis = groq_inst.analyze_transcript(formatted_transcript, options)
            else:
                analyzer = GeminiAnalyzer()
                analysis = analyzer.analyze_video(temp_file_path, options)
        else:
            analyzer = GeminiAnalyzer()
            analysis = analyzer.analyze_video(temp_file_path, options)


        # 4. Map clips to word timestamps
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
                    title=clip.title_pt if (should_translate and clip.title_pt) else clip.title,
                    title_pt=clip.title_pt,
                    start_sec=round(start_sec, 2),
                    end_sec=round(end_sec, 2),
                    viral_score=clip.virality_score,
                    hook=clip.hook_pt if (should_translate and clip.hook_pt) else (clip.hook_summary or clip.title),
                    hook_pt=clip.hook_pt,
                    crop_mode=crop_mode,
                    words=clip_words,
                    subtitles_pt=clip.subtitles_pt,
                    subtitle_language=subtitle_language.value,
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


