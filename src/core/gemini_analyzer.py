"""Gemini AI Analyzer module for Auto Viral Cuts."""

import json
import os
import time
from typing import Optional
import google.generativeai as genai
from dotenv import load_dotenv

from src.core.schemas import ProcessingOptions, ViralAnalysisResponse

load_dotenv()


class GeminiAnalyzer:
    """Handles video upload, state polling via File API, and structured viral analysis."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None) -> None:
        """Initialize Gemini client with API key and model selection."""
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY não encontrada. Defina no ambiente ou passe no construtor."
            )
        genai.configure(api_key=self.api_key)
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def analyze_video(
        self, video_path: str, options: Optional[ProcessingOptions] = None
    ) -> ViralAnalysisResponse:
        """Uploads video to Gemini File API, polls until active, and extracts viral clips."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Vídeo não encontrado no caminho: {video_path}")

        options = options or ProcessingOptions()

        print(f"[{time.strftime('%H:%M:%S')}] Enviando vídeo para a API do Gemini ({video_path})...")
        video_file = genai.upload_file(path=video_path)
        print(f"[{time.strftime('%H:%M:%S')}] Vídeo enviado. ID/URI: {video_file.name}")

        try:
            print(f"[{time.strftime('%H:%M:%S')}] Aguardando processamento do arquivo no Gemini...")
            while video_file.state.name == "PROCESSING":
                time.sleep(5)
                video_file = genai.get_file(video_file.name)

            if video_file.state.name != "ACTIVE":
                raise RuntimeError(
                    f"Falha no processamento do vídeo no Gemini. Estado atual: {video_file.state.name}"
                )
            print(f"[{time.strftime('%H:%M:%S')}] Arquivo ativo e pronto para análise.")

            prompt = self._build_prompt(options)

            print(f"[{time.strftime('%H:%M:%S')}] Solicitando análise estruturada com o modelo {self.model_name}...")
            
            # Using structured outputs via JSON schema matching ViralAnalysisResponse
            generation_config = {
                "temperature": 0.4,
                "response_mime_type": "application/json",
                "response_schema": ViralAnalysisResponse,
            }

            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
            )

            response = model.generate_content([video_file, prompt])
            response_text = response.text

            print(f"[{time.strftime('%H:%M:%S')}] Análise concluída com sucesso.")
            parsed_data = json.loads(response_text)
            return ViralAnalysisResponse(**parsed_data)

        finally:
            # Cleanup remote file from Gemini storage
            try:
                print(f"[{time.strftime('%H:%M:%S')}] Removendo arquivo temporário dos servidores Gemini...")
                genai.delete_file(video_file.name)
            except Exception as e:
                print(f"Aviso: Não foi possível deletar o arquivo remoto do Gemini: {e}")

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
