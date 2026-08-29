"""Groq LPU integration module for ultra-fast transcript analysis and translation (LLaMA 3.3 70B & Whisper)."""

import json
import os
import subprocess
import tempfile
import time
from typing import List, Optional

from groq import Groq


from src.core.schemas import (
    ClipMetadata,
    ProcessingOptions,
    SubtitleCue,
    SubtitleLanguage,
    ViralAnalysisResponse,
)
from src.core.transcriber import WordTimestamp as TranscriberWord



class GroqAnalyzer:
    """Performs ultra-fast viral cut analysis and translation on timestamped transcripts using Groq LPUs."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self._client: Optional[Groq] = None
        self.model_name = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        self.fallback_model_name = "openai/gpt-oss-120b"


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
            prompt = f"""Você é um editor sênior de cortes virais (Shorts, Reels, TikTok) e tradutor audiovisual especializado em localização para Português do Brasil (PT-BR).

Analise a transcrição com marcações de tempo (timestamps) fornecida e gere o manifesto estruturado para os cortes virais.

DIRETRIZES DE SELEÇÃO:
1. Identifique os {options.max_clips} melhores momentos com alta retenção e potencial de engajamento.
2. Duração por corte: Entre {options.min_duration_seconds}s e {options.max_duration_seconds}s (máximo absoluto: 75s).
3. Cada corte deve ter um gancho de impacto nos primeiros 3 a 5 segundos e uma narrativa completa (começo, meio e fim, sem falas truncadas).
4. O valor numérico de "start" e "end" de cada corte e legenda deve coincidir estritamente com os segundos reais da transcrição.

DIRETRIZES DE TRADUÇÃO (PT-BR):
1. Títulos e ganchos devem ser chamativos, instigantes e localizados para o público brasileiro.
2. Todas as falas dentro de "subtitles_pt" devem ser traduzidas para o Português do Brasil mantendo naturalidade, fluidez e concisão para facilitar a leitura rápida na tela.
3. Preserve a correspondência exata de tempo ("start" e "end") para cada frase legendada.
{custom_instructions}

SAÍDA OBRIGATÓRIA (JSON ESTREITO):
Responda EXCLUSIVAMENTE com o objeto JSON contendo a lista de cortes na chave "clips", sem textos antes ou depois:

{{
  "clips": [
    {{
      "corte_id": 1,
      "title_pt": "Título chamativo em português",
      "hook_pt": "Frase de impacto inicial do vídeo",
      "start": 14.5,
      "end": 52.0,
      "virality_score": 95,
      "reason_pt": "Gatilho de curiosidade ou valor entregue no trecho",
      "subtitles_pt": [
        {{
          "start": 14.5,
          "end": 17.2,
          "text": "Primeira frase traduzida sincronizada"
        }},
        {{
          "start": 17.3,
          "end": 20.8,
          "text": "Segunda frase traduzida sincronizada"
        }}
      ]
    }}
  ]
}}

TRANSCRIÇÃO ORIGINAL COM TIMESTAMPS:
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

    def get_candidate_models(self) -> List[str]:
        """Dynamically identifies available chat models on the user's Groq account."""
        candidates = []
        env_model = os.getenv("GROQ_MODEL")
        if env_model:
            candidates.append(env_model)
        if self.model_name and self.model_name not in candidates:
            candidates.append(self.model_name)
        if self.fallback_model_name and self.fallback_model_name not in candidates:
            candidates.append(self.fallback_model_name)

        candidates.extend([
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.6-27b",
            "groq/compound",
        ])

        try:
            active_data = self.client.models.list()
            active_ids = {m.id for m in active_data.data if getattr(m, "active", True)}
            valid = [m for m in candidates if m in active_ids]
            if valid:
                return valid
        except Exception:
            pass

        seen = set()
        unique_candidates = []
        for m in candidates:
            if m not in seen:
                seen.add(m)
                unique_candidates.append(m)
        return unique_candidates

    def analyze_transcript(
        self, formatted_transcript: str, options: ProcessingOptions
    ) -> ViralAnalysisResponse:
        """Sends formatted timestamped transcript to Groq with instant JSON response."""
        prompt = self._build_prompt(formatted_transcript, options)

        models_to_try = self.get_candidate_models()
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
                return self._parse_response(content)

            except Exception as e:
                last_error = e
                print(f"Aviso: Erro ao consultar Groq com o modelo {model}: {e}")

        raise RuntimeError(f"Falha ao analisar transcrição no Groq: {last_error}")

    def _parse_response(self, content: str) -> ViralAnalysisResponse:
        """Parses JSON content returned by Groq, handling arrays or objects flexibly."""
        clean_content = content.strip()
        if clean_content.startswith("```json"):
            clean_content = clean_content[7:]
        elif clean_content.startswith("```"):
            clean_content = clean_content[3:]
        if clean_content.endswith("```"):
            clean_content = clean_content[:-3]
        clean_content = clean_content.strip()

        data = json.loads(clean_content)

        if isinstance(data, list):
            clips = [ClipMetadata.model_validate(item) for item in data]
            return ViralAnalysisResponse(
                video_summary="Cortes virais selecionados com alta retenção",
                key_themes=[],
                clips=clips,
            )

        if isinstance(data, dict):
            raw_clips = data.get("clips") or data.get("cortes")
            if raw_clips is not None:
                clips = [ClipMetadata.model_validate(item) for item in raw_clips]
                return ViralAnalysisResponse(
                    video_summary=data.get("video_summary") or data.get("summary") or "Cortes virais selecionados",
                    key_themes=data.get("key_themes", []),
                    clips=clips,
                )
            return ViralAnalysisResponse.model_validate(data)

        raise ValueError(f"Formato JSON inesperado retornado pelo Groq: {type(data)}")


    def _compress_audio_for_groq(self, audio_path: str) -> str:
        """Compresses audio to lightweight 16kHz mono MP3 (64kbps) to stay well under Groq's 25MB limit."""
        size_bytes = os.path.getsize(audio_path)
        ext = os.path.splitext(audio_path)[1].lower()

        # If already compressed MP3 and under 20MB, reuse directly
        if ext in [".mp3", ".m4a", ".aac"] and size_bytes < 20 * 1024 * 1024:
            return audio_path

        temp_mp3 = tempfile.NamedTemporaryFile(suffix="_groq.mp3", delete=False).name
        cmd = [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-vn",
            "-acodec", "libmp3lame",
            "-b:a", "64k",
            "-ar", "16000",
            "-ac", "1",
            temp_mp3,
        ]
        try:
            subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
            )
            return temp_mp3
        except Exception as e:
            print(f"Aviso: Compressão para Groq falhou ({e}), tentando com áudio original.")
            if os.path.exists(temp_mp3):
                try:
                    os.remove(temp_mp3)
                except Exception:
                    pass
            return audio_path

    def transcribe_audio_fast(self, audio_path: str) -> List[TranscriberWord]:

        """Transcribes audio in seconds using whisper-large-v3 on Groq Cloud."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Arquivo de áudio não encontrado: {audio_path}")

        # Compress to 64kbps MP3 if needed to prevent 413 Request Entity Too Large (>25MB)
        upload_path = self._compress_audio_for_groq(audio_path)
        is_temp = upload_path != audio_path

        try:
            print(
                f"[{time.strftime('%H:%M:%S')}] Enviando áudio comprimido "
                f"({os.path.getsize(upload_path) / (1024 * 1024):.2f} MB) para Groq Whisper (whisper-large-v3)..."
            )
            with open(upload_path, "rb") as f:
                transcription = self.client.audio.transcriptions.create(
                    file=f,
                    model="whisper-large-v3",
                    response_format="verbose_json",
                    timestamp_granularities=["word"],
                )

            words_list: List[TranscriberWord] = []
            raw_words = getattr(transcription, "words", None)
            if raw_words:
                for item in raw_words:
                    w_text = item.get("word") if isinstance(item, dict) else getattr(item, "word", "")
                    w_start = item.get("start") if isinstance(item, dict) else getattr(item, "start", 0.0)
                    w_end = item.get("end") if isinstance(item, dict) else getattr(item, "end", 0.0)
                    words_list.append(
                        TranscriberWord(
                            word=w_text.strip(),
                            start=float(w_start),
                            end=float(w_end),
                        )
                    )


            print(
                f"[{time.strftime('%H:%M:%S')}] Groq Whisper concluiu: {len(words_list)} palavras detectadas!"
            )
            return words_list

        finally:
            if is_temp and os.path.exists(upload_path):
                try:
                    os.remove(upload_path)
                except Exception:
                    pass

