"""Gemini AI Analyzer module for Auto Viral Cuts using google-genai."""

import json
import os
import time
from typing import Optional
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

    def analyze_video(
        self, video_path: str, options: Optional[ProcessingOptions] = None
    ) -> ViralAnalysisResponse:
        """Uploads video to Gemini File API, polls until active, and extracts viral clips."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Vídeo não encontrado no caminho: {video_path}")

        options = options or ProcessingOptions()

        print(f"[{time.strftime('%H:%M:%S')}] Enviando vídeo para a API do Gemini ({video_path})...")
        video_file = self.client.files.upload(file=video_path)
        print(f"[{time.strftime('%H:%M:%S')}] Vídeo enviado. Nome/URI: {video_file.name}")

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
            print(f"[{time.strftime('%H:%M:%S')}] Arquivo ativo e pronto para análise.")

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
                try:
                    interaction = self.client.interactions.create(
                        model=current_model_name,
                        input=[
                            {
                                "type": "video",
                                "uri": video_file.uri,
                                "mime_type": video_file.mime_type,
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
                    has_fallback = index < len(model_names) - 1
                    if not has_fallback or not self._is_transient_model_error(error):
                        raise
                    print(
                        f"Modelo {current_model_name} indisponível temporariamente. "
                        f"Tentando {model_names[index + 1]}..."
                    )

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
    def _is_transient_model_error(error: Exception) -> bool:
        """Returns whether a Gemini error justifies an automatic model fallback."""
        transient_markers = (
            "high demand",
            "overloaded",
            "temporarily unavailable",
            "server_error",
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

        prompt = f"""
Você é um especialista mundial em edição de vídeo viral e retenção de audiência para redes sociais.
Analise o vídeo fornecido e identifique exatamente até {options.max_clips} trechos com maior potencial de viralização.

Diretrizes de Extração:
1. Duração de cada corte: entre {options.min_duration_seconds} e {options.max_duration_seconds} segundos.
2. Timestamps precisos no formato 'HH:MM:SS' ou 'MM:SS'.
3. Pontuação de viraildade (virality_score) de 0 a 100 baseada no gancho inicial, clareza e emoção.
4. {platform_hint}
{custom_instructions}

Retorne estritamente um JSON válido correspondente ao schema solicitado contendo o resumo geral do vídeo, temas principais e a lista de clips.
"""
        return prompt.strip()
