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


class ClipMetadata(BaseModel):
    """Metadata for a single viral clip candidate."""

    title: str = Field(
        ...,
        description="Título chamativo ou hook intrigante para atrair atenção imediata.",
    )
    start_time: str = Field(
        ...,
        description="Timestamp de início no formato 'HH:MM:SS' ou 'MM:SS'.",
        examples=["00:01:23", "01:23"],
    )
    end_time: str = Field(
        ...,
        description="Timestamp de fim no formato 'HH:MM:SS' ou 'MM:SS'.",
        examples=["00:02:10", "02:10"],
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
    version: str
