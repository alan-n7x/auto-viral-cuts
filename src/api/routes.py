import os
import shutil
import subprocess
import tempfile
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
from fastapi.responses import FileResponse

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

from src.core.transcriber import AudioTranscriber, WordTimestamp as TranscriberWord
from src.core.video_processor import VideoProcessor
from src.core.local_scene_analyzer import LocalSceneAnalyzer, SceneSegment


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
    ai_provider: AiProvider = Form(AiProvider.GEMINI),
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
    ai_provider: AiProvider = Form(AiProvider.GEMINI),
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

        # 1a. If the received file is a video (not WAV/MP3/audio), extract audio via FFmpeg first.
        #     This handles the case where the browser sent the raw video (>200 MB) directly.
        audio_path_for_transcription = temp_file_path
        _tmp_extracted_audio: Optional[str] = None

        uploaded_mime = (file.content_type or "").lower()
        uploaded_ext = os.path.splitext(temp_file_path)[1].lower()
        is_video_file = uploaded_mime.startswith("video/") or uploaded_ext in (
            ".mp4", ".mkv", ".mov", ".avi", ".webm", ".ts", ".m2ts", ".flv", ".wmv",
        )

        if is_video_file:
            print(
                f"[{__import__('time').strftime('%H:%M:%S')}] Video detectado ({uploaded_ext}), "
                f"extraindo audio via FFmpeg para transcricao..."
            )
            _tmp_extracted_audio = tempfile.NamedTemporaryFile(
                suffix="_audio.wav", delete=False, dir=TEMP_DIR
            ).name
            ffmpeg_extract_cmd = [
                "ffmpeg", "-y",
                "-i", temp_file_path,
                "-vn",  # no video
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                _tmp_extracted_audio,
            ]
            try:
                subprocess.run(
                    ffmpeg_extract_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True,
                )
                audio_path_for_transcription = _tmp_extracted_audio
                print(
                    f"[{__import__('time').strftime('%H:%M:%S')}] Audio extraido: "
                    f"{os.path.getsize(_tmp_extracted_audio) / (1024**2):.1f} MB"
                )
            except Exception as ffmpeg_err:
                print(f"Aviso: Extracao de audio via FFmpeg falhou ({ffmpeg_err}), tentando com arquivo original.")
                audio_path_for_transcription = temp_file_path

        # 1b. Speech-to-text transcription (Groq Whisper cloud or local faster-whisper)
        all_words: List[TranscriberWord] = []
        if ai_provider == AiProvider.GROQ and (groq_api_key or os.getenv("GROQ_API_KEY")):

            try:
                groq_inst = GroqAnalyzer(api_key=groq_api_key)
                all_words = groq_inst.transcribe_audio_fast(audio_path_for_transcription)
            except Exception as e:
                print(f"Aviso: Transcricao via Groq Cloud falhou ({e}), usando faster-whisper local...")

        if not all_words:
            if not AudioTranscriber.is_available():
                raise RuntimeError("faster-whisper nao esta disponivel no servidor.")
            transcriber = AudioTranscriber()
            all_words = transcriber.transcribe(audio_path_for_transcription, model_size=whisper_model)

        # Clean up extracted audio temp file if it was created
        if _tmp_extracted_audio and _tmp_extracted_audio != temp_file_path:
            try:
                os.remove(_tmp_extracted_audio)
            except Exception:
                pass

        if not all_words:
            raise RuntimeError(
                "A transcricao retornou vazia. Verifique se o video possui faixa de audio "
                "audivel. Codec suportados: AAC, MP3, Opus, PCM. Se o video e muito grande, "
                "tente um clipe menor ou o Modo Servidor (Gradio) para processamento completo."
            )

        print(
            f"[{__import__('time').strftime('%H:%M:%S')}] Transcricao concluida: "
            f"{len(all_words)} palavras detectadas."
        )

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


@router.post("/local-scene-manifest", response_model=List[ClientCutManifest])
async def local_scene_manifest_endpoint(
    file: UploadFile = File(...),
    max_clips: int = Form(8),
    min_duration_seconds: int = Form(25),
    max_duration_seconds: int = Form(65),
    scene_threshold: float = Form(0.4),
    crop_mode: str = Form("face_crop"),
) -> List[ClientCutManifest]:
    """
    Detects viral clip candidates from a video file using ONLY local tools:
      - FFmpeg scene change detection (scdet)
      - FFmpeg audio energy analysis (volumedetect, silencedetect)
      - MediaPipe face detection for smart 9:16 crop positioning

    No AI API (Groq / Gemini) is used. No internet connection required.
    """
    ext = os.path.splitext(file.filename or "video.mp4")[1].lower() or ".mp4"
    temp_file_path = os.path.join(TEMP_DIR, f"local_scene_{uuid.uuid4().hex[:8]}{ext}")

    try:
        await save_upload_file_stream(file, temp_file_path)

        analyzer = LocalSceneAnalyzer()
        segments = analyzer.analyze_video(
            video_path=temp_file_path,
            max_clips=max_clips,
            min_duration=float(min_duration_seconds),
            max_duration=float(max_duration_seconds),
            scene_threshold=scene_threshold,
        )

        if not segments:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Nenhum segmento detectado. Tente reduzir o scene_threshold ou verificar se o video possui audio e video.",
            )

        manifests: List[ClientCutManifest] = []
        for idx, seg in enumerate(segments):
            virality_score = min(100, max(0, int(seg.score)))
            manifests.append(
                ClientCutManifest(
                    cut_id=f"local_{idx + 1}_{uuid.uuid4().hex[:6]}",
                    title=seg.title,
                    start_sec=seg.start_sec,
                    end_sec=seg.end_sec,
                    viral_score=virality_score,
                    hook=f"Cena detectada localmente - {seg.duration:.0f}s - {'Com fala' if seg.has_speech else 'Sem fala'}",
                    crop_mode=crop_mode,
                    words=[],
                    subtitles_pt=[],
                    subtitle_language="original",
                )
            )

        return manifests

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro na analise local de cenas: {str(e)}",
        )
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass


@router.post("/render-single-clip")
async def render_single_clip_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form("corte"),
    start_sec: float = Form(...),
    end_sec: float = Form(...),
    crop_mode: CropMode = Form(CropMode.CENTER_CROP),
    burn_subtitles: bool = Form(False),
    subtitle_style: SubtitleStyle = Form(SubtitleStyle.HORMOZI),
    subtitles_json: Optional[str] = Form(None),
) -> FileResponse:
    """
    Renders a single 9:16 vertical clip with FFmpeg on the server:
      - 100% perfect stereo audio synchronization (AAC 192kbps)
      - Hardware acceleration (VAAPI / NVENC / CPU libx264)
      - Precision crop (center_crop / face_crop)
      - Dynamic subtitle burn-in (if provided)

    Streams the rendered MP4 file directly for download, zero browser RAM freezing.
    """
    ext = os.path.splitext(file.filename or "video.mp4")[1].lower() or ".mp4"
    temp_input = os.path.join(TEMP_DIR, f"render_in_{uuid.uuid4().hex[:8]}{ext}")
    safe_title = "".join(c if c.isalnum() else "_" for c in title[:35]).strip("_") or "corte"
    temp_output = os.path.join(TEMP_DIR, f"{safe_title}_{uuid.uuid4().hex[:6]}.mp4")

    try:
        await save_upload_file_stream(file, temp_input)

        vp = VideoProcessor()
        in_w, in_h = vp.get_video_dimensions(temp_input)
        hw_mode, _, _ = VideoProcessor.detect_hw_accel()

        duration = max(1.0, end_sec - start_sec)
        options = ProcessingOptions(
            crop_mode=crop_mode,
            hw_accel=hw_mode,
            burn_subtitles=burn_subtitles,
            subtitle_style=subtitle_style,
        )

        subtitle_file = None
        if burn_subtitles and subtitles_json:
            try:
                import json
                cues_raw = json.loads(subtitles_json)
                cues = [SubtitleCue(**c) for c in cues_raw]
                if cues:
                    sg = SubtitleGenerator()
                    subtitle_file = sg.generate_ass_from_cues(
                        cues=cues,
                        clip_start=start_sec,
                        clip_duration=duration,
                        style=subtitle_style,
                    )
            except Exception as sub_err:
                print(f"[RenderSingleClip] Aviso nas legendas: {sub_err}")

        cmd, _ = vp._build_ffmpeg_cmd(
            video_path=temp_input,
            output_path=temp_output,
            start_sec=start_sec,
            duration=duration,
            options=options,
            in_w=in_w,
            in_h=in_h,
            resolved_accel=hw_mode,
            subtitle_file=subtitle_file,
        )

        print(f"[{__import__('time').strftime('%H:%M:%S')}] Renderizando clipe via FFmpeg: {start_sec:.1f}s a {end_sec:.1f}s...")
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            # Fallback to pure CPU libx264
            print("[RenderSingleClip] Falha na aceleracao por hardware, tentando fallback CPU libx264...")
            cmd_cpu, _ = vp._build_ffmpeg_cmd(
                video_path=temp_input,
                output_path=temp_output,
                start_sec=start_sec,
                duration=duration,
                options=options,
                in_w=in_w,
                in_h=in_h,
                resolved_accel=HwAccelMode.CPU,
                subtitle_file=subtitle_file,
            )
            res_cpu = subprocess.run(cmd_cpu, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res_cpu.returncode != 0:
                raise RuntimeError(f"FFmpeg falhou: {res_cpu.stderr[-500:]}")

        # Schedule temp cleanup after file is streamed
        def cleanup_files():
            for p in (temp_input, temp_output, subtitle_file):
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

        background_tasks.add_task(cleanup_files)

        return FileResponse(
            temp_output,
            media_type="video/mp4",
            filename=f"{safe_title}_9x16.mp4",
        )

    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(temp_input):
            try:
                os.remove(temp_input)
            except Exception:
                pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao renderizar clipe no servidor: {str(e)}",
        )


@router.post("/extract-clip-audio")
async def extract_clip_audio_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    start_sec: float = Form(...),
    end_sec: float = Form(...),
) -> FileResponse:
    """
    Extracts high quality 48kHz stereo WAV audio slice for a specific clip interval.
    Used by the client browser to avoid reading multi-gigabyte video files in RAM.
    """
    ext = os.path.splitext(file.filename or "video.mp4")[1].lower() or ".mp4"
    temp_input = os.path.join(TEMP_DIR, f"audio_in_{uuid.uuid4().hex[:8]}{ext}")
    temp_wav = os.path.join(TEMP_DIR, f"clip_audio_{uuid.uuid4().hex[:8]}.wav")

    try:
        await save_upload_file_stream(file, temp_input)
        duration = max(0.5, end_sec - start_sec)

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(max(0.0, start_sec)),
            "-i", temp_input,
            "-t", str(duration),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "48000",
            "-ac", "2",
            temp_wav,
        ]

        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg audio extraction failed: {res.stderr[-300:]}")

        def cleanup():
            for p in (temp_input, temp_wav):
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

        background_tasks.add_task(cleanup)

        return FileResponse(
            temp_wav,
            media_type="audio/wav",
            filename="clip_audio.wav",
        )
    except Exception as e:
        if os.path.exists(temp_input):
            try:
                os.remove(temp_input)
            except Exception:
                pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao extrair áudio do clipe: {str(e)}",
        )

