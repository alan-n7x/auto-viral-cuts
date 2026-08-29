"""Groq LPU integration module for ultra-fast transcript analysis and translation (LLaMA 3.3 70B & Whisper)."""

import json
import os
import time
from typing import List, Optional

from groq import Groq

from src.core.schemas import (
    ClipMetadata,
    ProcessingOptions,
    SubtitleCue,
    SubtitleLanguage,
    ViralAnalysisResponse,
    WordTimestamp,
)


class GroqAnalyzer:
    """Performs ultra-fast viral cut analysis and translation on timestamped transcripts using Groq LPUs."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self._client: Optional[Groq] = None
        self.model_name = "llama-3.3-70b-versatile"
        self.fallback_model_name = "llama-3.1-8b-instant"

    @property
    def client(self) -> Groq:
        """Lazy initialization of the Groq client."""
        if not self.api_key:
            self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Chave de API do Groq não encontrada. Configure a variável GROQ_API_KEY "
                "no arquivo .env ou informe sua chave na interface."
            )
        if self._client is None:
            self._client = Groq(api_key=self.api_key)
        return self._client

    def is_configured(self) -> bool:
        """Checks whether the Groq API key is set."""
        key = self.api_key or os.getenv("GROQ_API_KEY")
        return bool(key and len(key) > 5)

    def _build_prompt(self, formatted_transcript: str, options: ProcessingOptions) -> str:
        """Constructs the structured prompt for Groq LLaMA 3.3."""
        custom_instructions = (
            f"\nInstruções adicionais do usuário: {options.custom_prompt}"
            if options.custom_prompt
            else ""
        )

        should_translate = (
            options.translate_to_pt
            or options.subtitle_language == SubtitleLanguage.PT_BR
        )

        if should_translate:
            prompt = f"""Você é um editor sênior especializado em cortes virais (TikTok, Reels e YouTube Shorts) e tradutor audiovisual para Português do Brasil (PT-BR).

Analise a transcrição com marcações de tempo fornecida e execute as três etapas:
1. SELEÇÃO: Identifique até {options.max_clips} melhores trechos com alto potencial de retenção.
2. ENGAJAMENTO: Crie títulos e ganchos chamativos em PT-BR para cada corte (preenchendo title_pt e hook_pt).
3. TRADUÇÃO & SINCRONIA: Traduza todas as falas internas para PT-BR (preenchendo subtitles_pt com start, end e text), adaptando expressões idiomáticas de forma natural e mantendo os timestamps exatos de início e fim.

DIRETRIZES DE EDIÇÃO:
- Duração por corte: Entre {options.min_duration_seconds} e {options.max_duration_seconds} segundos.
- Gancho inicial forte: O trecho deve prender a atenção nos primeiros 3 a 5 segundos.
- Narrativa fechada: Cada corte precisa de começo, meio e fim coerentes (sem falas cortadas ao meio).
- Tradução dinâmica (PT-BR): Linguagem natural e concisa para leitura rápida em tela.
- Fidelidade temporal: Os valores numéricos de "start" e "end" em subtitles_pt devem corresponder exatamente aos segundos da transcrição.
{custom_instructions}

SAÍDA OBRIGATÓRIA (JSON PURO):
Retorne estritamente um objeto JSON com a seguinte estrutura:
{{
  "video_summary": "Resumo geral do conteúdo em Português",
  "key_themes": ["tema1", "tema2"],
  "clips": [
    {{
      "title": "Título em português",
      "title_pt": "Título chamativo em PT-BR",
      "start_time": "00:00:12",
      "end_time": "00:00:45",
      "virality_score": 95,
      "virality_reason": "Explicação do engajamento",
      "reason_pt": "Explicação em PT-BR",
      "hook_summary": "Frase de impacto inicial",
      "hook_pt": "Gancho de impacto em PT-BR",
      "suggested_caption": "Legenda pronta com emojis",
      "hashtags": ["#viral", "#cortes"],
      "subtitles_pt": [
        {{
          "start": 12.4,
          "end": 15.1,
          "text": "Frase traduzida sincronizada"
        }}
      ]
    }}
  ]
}}

TRANSCRIÇÃO COM TIMESTAMPS:
\"\"\"
{formatted_transcript}
\"\"\"
"""
        else:
            prompt = f"""Você é um especialista mundial em edição de vídeo viral e retenção de audiência para redes sociais.
Analise a transcrição com marcações de tempo fornecida e identifique até {options.max_clips} trechos com maior potencial de viralização.

DIRETRIZES DE EXTRAÇÃO:
- Duração de cada corte: Entre {options.min_duration_seconds} e {options.max_duration_seconds} segundos.
- Mantenha os títulos, ganchos e falas no idioma original da mídia.
- Timestamps precisos no formato 'HH:MM:SS' ou 'MM:SS'.
- Pontuação de viralidade (0 a 100) baseada no gancho inicial e emoção.
{custom_instructions}

SAÍDA OBRIGATÓRIA (JSON PURO):
Retorne estritamente um objeto JSON com a seguinte estrutura:
{{
  "video_summary": "Resumo geral do conteúdo",
  "key_themes": ["tema1", "tema2"],
  "clips": [
    {{
      "title": "Título chamativo",
      "start_time": "00:00:12",
      "end_time": "00:00:45",
      "virality_score": 90,
      "virality_reason": "Explicação do engajamento",
      "hook_summary": "Gancho inicial",
      "suggested_caption": "Legenda com hashtags",
      "hashtags": ["#viral", "#podcast"]
    }}
  ]
}}

TRANSCRIÇÃO COM TIMESTAMPS:
\"\"\"
{formatted_transcript}
\"\"\"
"""
        return prompt.strip()

    def analyze_transcript(
        self, formatted_transcript: str, options: ProcessingOptions
    ) -> ViralAnalysisResponse:
        """Sends formatted timestamped transcript to Groq LLaMA 3.3 with instant JSON response."""
        prompt = self._build_prompt(formatted_transcript, options)

        models_to_try = [self.model_name, self.fallback_model_name]
        last_error = None

        for model in models_to_try:
            try:
                start_t = time.time()
                print(
                    f"[{time.strftime('%H:%M:%S')}] Enviando texto transcrito para Groq LPU "
                    f"({model})..."
                )
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Você é um assistente especialista em análise de vídeo viral e tradução audiovisual. "
                                "Responda estritamente com JSON válido."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                )
                elapsed = time.time() - start_t
                content = response.choices[0].message.content or "{}"
                print(
                    f"[{time.strftime('%H:%M:%S')}] Groq respondeu em {elapsed:.2f}s "
                    f"({response.usage.total_tokens if hasattr(response, 'usage') and response.usage else '?'} tokens)!"
                )
                return ViralAnalysisResponse.model_validate_json(content)

            except Exception as e:
                last_error = e
                print(f"Aviso: Erro ao consultar Groq com o modelo {model}: {e}")

        raise RuntimeError(f"Falha ao analisar transcrição no Groq: {last_error}")

    def transcribe_audio_fast(self, audio_path: str) -> List[WordTimestamp]:
        """Transcribes audio in seconds using whisper-large-v3 on Groq Cloud."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Arquivo de áudio não encontrado: {audio_path}")

        print(f"[{time.strftime('%H:%M:%S')}] Enviando áudio para Groq Whisper (whisper-large-v3)...")
        with open(audio_path, "rb") as f:
            transcription = self.client.audio.transcriptions.create(
                file=f,
                model="whisper-large-v3",
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )

        words_list: List[WordTimestamp] = []
        raw_words = getattr(transcription, "words", None)
        if raw_words:
            for item in raw_words:
                w_text = item.get("word") if isinstance(item, dict) else getattr(item, "word", "")
                w_start = item.get("start") if isinstance(item, dict) else getattr(item, "start", 0.0)
                w_end = item.get("end") if isinstance(item, dict) else getattr(item, "end", 0.0)
                words_list.append(
                    WordTimestamp(
                        word=w_text.strip(),
                        start=float(w_start),
                        end=float(w_end),
                    )
                )

        print(
            f"[{time.strftime('%H:%M:%S')}] Groq Whisper concluiu: {len(words_list)} palavras detectadas!"
        )
        return words_list
