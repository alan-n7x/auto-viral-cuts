"""Gemini AI Analyzer module for Auto Viral Cuts using google-genai."""

import json
import os
import re
import subprocess
import tempfile
import time
from typing import Optional, Tuple
from google import genai
from dotenv import load_dotenv

from src.core.schemas import ProcessingOptions, ViralAnalysisResponse

load_dotenv()


class GeminiAnalyzer:
    """Handles video upload, state polling via File API, and structured viral analysis using google-genai."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None) -> None:
        """Initialize Gemini client with API key and model selection."""
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY não encontrada. Defina no ambiente ou passe no construtor."
            )
        self.client = genai.Client(api_key=self.api_key)
        configured_model = model_name or os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        self.model_name = self._resolve_model_name(configured_model)
        self.fallback_model_name = self._resolve_model_name(
            os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash")
        )

    @staticmethod
    def extract_lightweight_media_proxy(
        media_path: str, target_bitrate: str = "96k"
    ) -> Tuple[str, bool, int, int]:
        """Extracts a lightweight audio proxy (e.g. 96kbps MP3) from heavy video files

        to minimize bandwidth and upload times to Gemini File API.

        Returns:
            Tuple of (upload_path, is_temporary, original_size_bytes, upload_size_bytes)
        """
        if not os.path.exists(media_path):
            raise FileNotFoundError(f"Arquivo de mídia não encontrado: {media_path}")

        original_size = os.path.getsize(media_path)
        ext = os.path.splitext(media_path)[1].lower()

        # If already an audio file under 50MB, no need to re-encode
        audio_extensions = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}
        if ext in audio_extensions and original_size < 50 * 1024 * 1024:
            return media_path, False, original_size, original_size

        fd, temp_proxy_path = tempfile.mkstemp(suffix="_proxy.mp3")
        os.close(fd)

        # Extract 96kbps MP3 audio proxy using FFmpeg
        cmd = [
            "ffmpeg", "-y",
            "-i", media_path,
            "-vn",
            "-c:a", "libmp3lame",
            "-b:a", target_bitrate,
            "-ar", "24000",
            temp_proxy_path,
        ]

        try:
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            proxy_size = os.path.getsize(temp_proxy_path)
            return temp_proxy_path, True, original_size, proxy_size
        except Exception:
            # Fallback to AAC if libmp3lame has issues
            try:
                cmd_aac = [
                    "ffmpeg", "-y",
                    "-i", media_path,
                    "-vn",
                    "-c:a", "aac",
                    "-b:a", target_bitrate,
                    "-ar", "24000",
                    temp_proxy_path,
                ]
                subprocess.run(
                    cmd_aac,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True,
                )
                proxy_size = os.path.getsize(temp_proxy_path)
                return temp_proxy_path, True, original_size, proxy_size
            except Exception as e:
                if os.path.exists(temp_proxy_path):
                    try:
                        os.remove(temp_proxy_path)
                    except Exception:
                        pass
                print(f"Aviso: Falha ao extrair proxy com FFmpeg ({e}). Usando arquivo original.")
                return media_path, False, original_size, original_size

    def analyze_video(
        self, video_path: str, options: Optional[ProcessingOptions] = None
    ) -> ViralAnalysisResponse:
        """Uploads video to Gemini File API, polls until active, and extracts viral clips."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Vídeo não encontrado no caminho: {video_path}")

        options = options or ProcessingOptions()

        # Extract lightweight audio proxy for heavy or 4K video files
        upload_path, is_temp, orig_size, upload_size = self.extract_lightweight_media_proxy(
            video_path
        )

        if is_temp and orig_size > 0:
            savings_pct = (1.0 - upload_size / orig_size) * 100
            print(
                f"[{time.strftime('%H:%M:%S')}] Otimização de Mídia (4K/HD): "
                f"Original: {orig_size / (1024 * 1024):.2f} MB -> Proxy IA: {upload_size / (1024 * 1024):.2f} MB "
                f"({savings_pct:.1f}% de economia de banda e upload)."
            )

        print(f"[{time.strftime('%H:%M:%S')}] Enviando mídia otimizada para Gemini File API ({upload_path})...")
        video_file = self.client.files.upload(file=upload_path)
        print(f"[{time.strftime('%H:%M:%S')}] Mídia enviada. Nome/URI: {video_file.name}")


        try:
            print(f"[{time.strftime('%H:%M:%S')}] Aguardando processamento do arquivo no Gemini...")
            file_info = self.client.files.get(name=video_file.name)
            while file_info.state.name == "PROCESSING":
                time.sleep(5)
                file_info = self.client.files.get(name=video_file.name)

            if file_info.state.name != "ACTIVE":
                raise RuntimeError(
                    f"Falha no processamento do vídeo no Gemini. Estado atual: {file_info.state.name}"
                )
            mime_type = file_info.mime_type or video_file.mime_type or "video/mp4"
            input_type = "audio" if mime_type.startswith("audio/") else "video"

            prompt = self._build_prompt(options)

            model_names = [self.model_name]
            if self.fallback_model_name != self.model_name:
                model_names.append(self.fallback_model_name)

            interaction = None
            for index, current_model_name in enumerate(model_names):
                print(
                    f"[{time.strftime('%H:%M:%S')}] Solicitando análise estruturada "
                    f"com o modelo {current_model_name}..."
                )
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        interaction = self.client.interactions.create(
                            model=current_model_name,
                            input=[
                                {
                                    "type": input_type,
                                    "uri": video_file.uri,
                                    "mime_type": mime_type,
                                },
                                {"type": "text", "text": prompt},
                            ],
                            generation_config={"temperature": 0.4},
                            response_format={
                                "type": "text",
                                "mime_type": "application/json",
                                "schema": ViralAnalysisResponse.model_json_schema(),
                            },
                        )
                        break
                    except Exception as error:
                        is_transient = self._is_transient_model_error(error)
                        if is_transient and attempt < max_retries - 1:
                            delay = self._extract_retry_delay(error, default_delay=10.0)
                            print(
                                f"[{time.strftime('%H:%M:%S')}] Limite de taxa/resposta da API "
                                f"({current_model_name}). Aguardando {delay:.1f}s antes da tentativa {attempt + 2}/{max_retries}..."
                            )
                            time.sleep(delay)
                            continue

                        has_fallback = index < len(model_names) - 1
                        if not has_fallback or not is_transient:
                            raise
                        print(
                            f"Modelo {current_model_name} indisponível após erro/limite. "
                            f"Tentando fallback {model_names[index + 1]}..."
                        )
                        break
                if interaction is not None:
                    break

            if interaction is None:
                raise RuntimeError("Não foi possível iniciar a análise com os modelos configurados.")

            if interaction.status != "completed" or not interaction.output_text:
                errors = "; ".join(str(error) for error in interaction.errors or [])
                raise RuntimeError(
                    "O Gemini não concluiu a análise do vídeo "
                    f"(status: {interaction.status}). {errors}"
                )

            response_text = interaction.output_text

            print(f"[{time.strftime('%H:%M:%S')}] Análise concluída com sucesso.")
            parsed_data = json.loads(response_text)
            return ViralAnalysisResponse(**parsed_data)

        finally:
            # Cleanup remote file from Gemini storage
            try:
                print(f"[{time.strftime('%H:%M:%S')}] Removendo arquivo temporário dos servidores Gemini...")
                self.client.files.delete(name=video_file.name)
            except Exception as e:
                print(f"Aviso: Não foi possível deletar o arquivo remoto do Gemini: {e}")

            # Cleanup local temporary proxy file
            if is_temp and os.path.exists(upload_path):
                try:
                    os.remove(upload_path)
                except Exception as e:
                    print(f"Aviso: Não foi possível deletar o arquivo proxy temporário local: {e}")


    @staticmethod
    def _resolve_model_name(model_name: str) -> str:
        """Maps retired model names to a supported default model."""
        retired_models = {"gemini-2.5-flash"}
        normalized_name = model_name.strip().removeprefix("models/")

        if normalized_name in retired_models:
            print(
                "Aviso: GEMINI_MODEL=gemini-2.5-flash foi descontinuado. "
                "Usando gemini-3.6-flash. Atualize seu arquivo .env."
            )
            return "gemini-3.6-flash"

        return normalized_name

    @staticmethod
    def _extract_retry_delay(error: Exception, default_delay: float = 10.0) -> float:
        """Parses exception message to extract requested retry delay in seconds, if present."""
        error_str = str(error)
        match = re.search(r"retry (?:in|after) ([0-9\.]+)s?", error_str, re.IGNORECASE)
        if match:
            try:
                delay = float(match.group(1))
                return max(1.0, delay + 0.5)
            except ValueError:
                pass
        return default_delay

    @staticmethod
    def _is_transient_model_error(error: Exception) -> bool:
        """Returns whether a Gemini error justifies an automatic retry or model fallback."""
        transient_markers = (
            "high demand",
            "overloaded",
            "temporarily unavailable",
            "server_error",
            "quota exceeded",
            "too_many_requests",
            "429",
            "rate limit",
            "ratelimiterror",
            "resource_exhausted",
            "createinteractionclienterror",
        )
        error_message = str(error).lower()
        return any(marker in error_message for marker in transient_markers)

    def _build_prompt(self, options: ProcessingOptions) -> str:
        """Constructs the prompt guiding Gemini to find the best viral clips."""
        platform_instructions = {
            "tiktok": "Foque em ganchos absurdamente fortes nos primeiros 2 segundos, cortes dinâmicos e estilo TikTok.",
            "reels": "Foque em apelo estético, storytelling marcante e conteúdo inspirador ou polêmico para Instagram Reels.",
            "shorts": "Foque em ritmo acelerado, curiosidade imediata e alta retenção para YouTube Shorts.",
            "general": "Foque em momentos de forte impacto emocional, revelações marcantes e debates profundos.",
        }

        platform_hint = platform_instructions.get(
            options.target_platform.value, platform_instructions["general"]
        )

        custom_instructions = (
            f"\nInstruções adicionais do usuário: {options.custom_prompt}"
            if options.custom_prompt
            else ""
        )

        if options.translate_to_pt:
            prompt = f"""
Você é um editor sênior especializado em cortes virais (TikTok, Reels e YouTube Shorts) e tradutor audiovisual para Português do Brasil (PT-BR).

Analise o conteúdo (áudio ou vídeo) fornecido e execute as três etapas:
1. SELEÇÃO: Identifique até {options.max_clips} melhores trechos com alto potencial de retenção.
2. ENGAJAMENTO: Crie títulos e ganchos chamativos em PT-BR para cada corte (preenchendo title_pt e hook_pt).
3. TRADUÇÃO & SINCRONIA: Traduza todas as falas internas para PT-BR (preenchendo subtitles_pt), adaptando expressões idiomáticas de forma natural e mantendo os timestamps numéricos exatos de início e fim de cada frase.

DIRETRIZES DE EDIÇÃO:
- Duração por corte: Entre {options.min_duration_seconds} e {options.max_duration_seconds} segundos.
- Gancho inicial forte: O trecho deve prender a atenção nos primeiros 3 a 5 segundos.
- Narrativa fechada: Cada corte precisa de começo, meio e fim coerentes (sem falas cortadas ao meio).
- Tradução dinâmica (PT-BR): Linguagem natural e concisa para leitura rápida em tela, adaptando gírias e expressões idiomáticas.
- Fidelidade temporal: Os valores numéricos de "start" e "end" em subtitles_pt devem corresponder exatamente aos segundos da mídia.
- {platform_hint}
{custom_instructions}

Retorne estritamente um JSON válido correspondente ao schema solicitado contendo o resumo geral do vídeo, temas principais e a lista de clips com title, title_pt, hook_pt, reason_pt e subtitles_pt.
"""
            return prompt.strip()

        prompt = f"""
Você é um especialista mundial em edição de vídeo viral e retenção de audiência para redes sociais.
Analise o conteúdo (áudio ou vídeo) fornecido e identifique exatamente até {options.max_clips} trechos com maior potencial de viralização.

Diretrizes de Extração:
1. Duração de cada corte: entre {options.min_duration_seconds} e {options.max_duration_seconds} segundos.
2. Mantenha os títulos, ganchos e falas no idioma original da mídia.
3. Timestamps precisos no formato 'HH:MM:SS' ou 'MM:SS'.
4. Pontuação de viraildade (virality_score) de 0 a 100 baseada no gancho inicial, clareza e emoção.
5. {platform_hint}
{custom_instructions}

Retorne estritamente um JSON válido correspondente ao schema solicitado contendo o resumo geral do vídeo, temas principais e a lista de clips no idioma original.
"""
        return prompt.strip()

    def analyze_transcript(
        self, formatted_transcript: str, options: ProcessingOptions
    ) -> ViralAnalysisResponse:
        """Analyzes a text-only formatted transcript with Gemini without uploading any media files."""
        options_copy = options.model_copy()
        custom_instructions = (
            f"\nTRANSCRIÇÃO COM TIMESTAMPS:\n\"\"\"\n{formatted_transcript}\n\"\"\""
        )
        if options_copy.custom_prompt:
            options_copy.custom_prompt += custom_instructions
        else:
            options_copy.custom_prompt = custom_instructions

        prompt = self._build_prompt(options_copy)

        model_names = [self.model_name]
        if self.fallback_model_name != self.model_name:
            model_names.append(self.fallback_model_name)

        last_error = None
        for current_model_name in model_names:
            try:
                print(
                    f"[{time.strftime('%H:%M:%S')}] Enviando texto transcrito para Gemini "
                    f"({current_model_name})..."
                )
                interaction = self.client.interactions.create(
                    model=current_model_name,
                    input=[
                        {"type": "text", "text": prompt},
                    ],
                    generation_config={"temperature": 0.3},
                    response_format={
                        "type": "text",
                        "mime_type": "application/json",
                        "schema": ViralAnalysisResponse.model_json_schema(),
                    },
                )
                response_text = interaction.outputs[-1].text
                return ViralAnalysisResponse.model_validate_json(response_text)
            except Exception as e:
                last_error = e
                print(f"Aviso: Erro ao consultar Gemini com texto no modelo {current_model_name}: {e}")

        raise RuntimeError(f"Falha ao analisar transcrição com Gemini: {last_error}")


