"""Pydantic schemas and data transfer objects for Auto Viral Cuts."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class CropMode(str, Enum):
    """Supported aspect ratio and cropping modes."""

    CENTER_CROP = "center_crop"
    BLURRED_BACKGROUND = "blurred_background"
    FIT_BLACK_BARS = "fit_black_bars"
    NO_CROP = "no_crop"


class PlatformPreset(str, Enum):
    """Target social platform presets."""

    TIKTOK = "tiktok"
    REELS = "reels"
    SHORTS = "shorts"
    GENERAL = "general"


class HwAccelMode(str, Enum):
    """Hardware acceleration modes."""

    AUTO = "auto"
    VAAPI = "vaapi"  # AMD / Intel on Linux (e.g. Radeon RX 570 via /dev/dri/renderD128)
    NVENC = "nvenc"  # NVIDIA NVENC
    VIDEOTOOLBOX = "videotoolbox"  # Apple Silicon
    CPU = "cpu"  # Pure CPU (libx264)


class SubtitleStyle(str, Enum):
    """Visual presets for dynamic viral subtitles."""

    HORMOZI = "hormozi"  # Bold yellow active highlight, white secondary, thick black outline
    NEON = "neon"  # Vibrant electric cyan with drop shadow
    MINIMAL = "minimal"  # Clean white text with subtle shadow


class ClipMetadata(BaseModel):
    """Metadata for a single viral clip candidate."""

    title: str = Field(
        ...,
        description="Título chamativo ou hook intrigante para atrair atenção imediata.",
    )
    start_time: str = Field(
        ...,
        description="Timestamp de início no formato 'HH:MM:SS' ou 'MM:SS' (ex: 00:01:23 ou 01:23).",
    )
    end_time: str = Field(
        ...,
        description="Timestamp de fim no formato 'HH:MM:SS' ou 'MM:SS' (ex: 00:02:10 ou 02:10).",
    )
    duration_seconds: Optional[float] = Field(
        default=None,
        description="Duração calculada do corte em segundos.",
    )
    virality_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Pontuação estimada de potencial de viralização (0 a 100).",
    )
    virality_reason: str = Field(
        ...,
        description="Explicação detalhada do porquê o trecho tem alto potencial de retenção.",
    )
    hook_summary: Optional[str] = Field(
        default=None,
        description="Resumo do gancho inicial utilizado nos primeiros 3 a 5 segundos.",
    )
    suggested_caption: Optional[str] = Field(
        default=None,
        description="Sugestão de legenda pronta com emojis e chamada para ação (CTA).",
    )
    hashtags: List[str] = Field(
        default_factory=list,
        description="Lista de hashtags estratégicas recomendadas para o corte.",
    )


class ViralAnalysisResponse(BaseModel):
    """Structured response containing all analyzed viral clips."""

    video_summary: str = Field(
        ...,
        description="Resumo geral do conteúdo do vídeo longo.",
    )
    key_themes: List[str] = Field(
        default_factory=list,
        description="Principais temas e tópicos detectados na mídia.",
    )
    clips: List[ClipMetadata] = Field(
        default_factory=list,
        description="Lista de cortes virais identificados e ranqueados.",
    )


class ProcessingOptions(BaseModel):
    """Configuration options for analysis and video rendering."""

    crop_mode: CropMode = Field(
        default=CropMode.CENTER_CROP,
        description="Modo de enquadramento vertical.",
    )
    target_aspect_ratio: str = Field(
        default="9:16",
        description="Proporção de aspecto de saída (padrão: 9:16 para verticais).",
    )
    video_codec: str = Field(
        default="libx264",
        description="Codec de vídeo FFmpeg.",
    )
    audio_codec: str = Field(
        default="aac",
        description="Codec de áudio FFmpeg.",
    )
    crf: int = Field(
        default=20,
        ge=0,
        le=51,
        description="Constant Rate Factor (qualidade visual, menor é melhor qualidade).",
    )
    preset: str = Field(
        default="fast",
        description="Preset de velocidade de compressão do FFmpeg.",
    )
    max_clips: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Número máximo de cortes virais a serem extraídos.",
    )
    min_duration_seconds: int = Field(
        default=15,
        ge=5,
        le=300,
        description="Duração mínima de cada corte em segundos.",
    )
    max_duration_seconds: int = Field(
        default=60,
        ge=10,
        le=600,
        description="Duração máxima de cada corte em segundos.",
    )
    custom_prompt: Optional[str] = Field(
        default=None,
        description="Instruções adicionais de nicho ou direcionamento de conteúdo para a IA.",
    )
    target_platform: PlatformPreset = Field(
        default=PlatformPreset.GENERAL,
        description="Plataforma alvo para otimização de hashtags e ganchos.",
    )
    hw_accel: HwAccelMode = Field(
        default=HwAccelMode.AUTO,
        description="Modo de aceleração por hardware (auto, vaapi para AMD/Intel, nvenc, videotoolbox, cpu).",
    )
    burn_subtitles: bool = Field(
        default=True,
        description="Embutir legendas animadas palavra por palavra no vídeo via FFmpeg.",
    )
    subtitle_style: SubtitleStyle = Field(
        default=SubtitleStyle.HORMOZI,
        description="Estilo visual das legendas dinâmicas.",
    )
    whisper_model: str = Field(
        default="base",
        description="Tamanho do modelo faster-whisper para transcrição (tiny, base, small).",
    )


class ProcessedClip(BaseModel):
    """Details of an exported video cut."""

    clip_index: int = Field(
        ...,
        description="Índice sequencial do corte gerado.",
    )
    metadata: ClipMetadata = Field(
        ...,
        description="Metadados analíticos associados ao corte.",
    )
    file_path: str = Field(
        ...,
        description="Caminho absoluto do arquivo de vídeo gerado.",
    )
    file_name: str = Field(
        ...,
        description="Nome do arquivo salvo no disco.",
    )
    file_size_bytes: int = Field(
        ...,
        description="Tamanho do arquivo em bytes.",
    )
    duration_seconds: float = Field(
        ...,
        description="Duração real do vídeo exportado em segundos.",
    )
    status: str = Field(
        default="completed",
        description="Status do processamento ('completed', 'failed', etc.).",
    )
    hw_accel_used: Optional[str] = Field(
        default=None,
        description="Backend de aceleração de hardware utilizado (ex: 'vaapi', 'cpu').",
    )
    has_subtitles: bool = Field(
        default=False,
        description="Indica se o corte possui legendas embutidas.",
    )
    subtitle_path: Optional[str] = Field(
        default=None,
        description="Caminho do arquivo .ass gerado para a legenda.",
    )


class ProcessingResult(BaseModel):
    """Overall result of a full video analysis and cut pipeline."""

    source_video: str = Field(
        ...,
        description="Caminho do vídeo original processado.",
    )
    clips: List[ProcessedClip] = Field(
        default_factory=list,
        description="Lista de cortes gerados com sucesso.",
    )
    total_clips: int = Field(
        ...,
        description="Total de cortes extraídos.",
    )
    execution_time_seconds: float = Field(
        ...,
        description="Tempo total de execução em segundos.",
    )
    status: str = Field(
        default="success",
        description="Status geral do pipeline.",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Mensagem de erro em caso de falha.",
    )


class HealthStatus(BaseModel):
    """Service health and capability status."""

    status: str
    ffmpeg_available: bool
    gemini_configured: bool
    hw_accel_available: bool = False
    hw_accel_backend: Optional[str] = None
    gpu_detected: Optional[str] = None
    whisper_available: bool = False
    version: str


class WordTimestamp(BaseModel):
    """Word timestamp representation in milliseconds for client-side rendering."""

    word: str = Field(..., description="Texto da palavra.")
    start_ms: int = Field(
        ...,
        description="Timestamp de início em milissegundos relativos ao início do corte.",
    )
    end_ms: int = Field(
        ...,
        description="Timestamp de término em milissegundos relativos ao início do corte.",
    )


class ClientCutManifest(BaseModel):
    """Structured cut manifest for client-side WebCodecs / Canvas video rendering."""

    cut_id: str = Field(..., description="Identificador único do corte.")
    title: str = Field(..., description="Título chamativo do corte.")
    start_sec: float = Field(
        ..., description="Timestamp de início em segundos no vídeo original."
    )
    end_sec: float = Field(
        ..., description="Timestamp de término em segundos no vídeo original."
    )
    viral_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Pontuação estimada de potencial de viralização (0 a 100).",
    )
    hook: str = Field(..., description="Gancho inicial ou resumo de atenção.")
    crop_mode: str = Field(
        default="center_crop",
        description="Modo de enquadramento (ex: center_crop, fit_black_bars).",
    )
    words: List[WordTimestamp] = Field(
        default_factory=list,
        description="Lista de palavras com timestamps para renderização de legendas.",
    )


class TaskState(str, Enum):
    """Execution state for background video processing tasks."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AsyncTaskResponse(BaseModel):
    """Immediate response for asynchronous background job submission (202 Accepted)."""

    task_id: str = Field(..., description="ID único para rastreamento da tarefa.")
    status: TaskState = Field(default=TaskState.QUEUED, description="Estado inicial da tarefa.")
    message: str = Field(..., description="Mensagem de confirmação de enfileiramento.")
    file_name: str = Field(..., description="Nome do arquivo enviado.")
    created_at: str = Field(..., description="Timestamp de criação ISO 8601.")


class TaskStatusResponse(BaseModel):
    """Detailed status response for background job polling."""

    task_id: str = Field(..., description="Identificador da tarefa.")
    status: TaskState = Field(..., description="Estado atual da execução.")
    file_name: str = Field(..., description="Nome do arquivo em processamento.")
    created_at: str = Field(..., description="Timestamp de criação ISO 8601.")
    updated_at: str = Field(..., description="Timestamp da última atualização ISO 8601.")
    progress: int = Field(default=0, ge=0, le=100, description="Progresso estimado da tarefa (0 a 100).")
    result: Optional[ProcessingResult] = Field(default=None, description="Resultado do processamento com a lista de cortes.")
    error: Optional[str] = Field(default=None, description="Mensagem de erro caso o status seja 'failed'.")


