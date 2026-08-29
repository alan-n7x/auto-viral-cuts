"""Local infrastructure adapter implementing VideoProcessorPort."""

import time
import uuid
from typing import List, Optional

from src.core.gemini_analyzer import GeminiAnalyzer
from src.core.groq_analyzer import GroqAnalyzer
from src.core.schemas import (
    AiProvider,
    ClientCutManifest,
    ProcessingOptions,
    ProcessingResult,
    WordTimestamp,
)
from src.core.transcriber import AudioTranscriber
from src.core.video_processor import VideoProcessor
from src.domain.ports.video_processor_port import VideoProcessorPort


class LocalProcessorAdapter(VideoProcessorPort):
    """Infrastructure adapter executing video processing on local hardware, Groq LPUs, and Gemini."""

    def __init__(
        self,
        video_processor: Optional[VideoProcessor] = None,
        analyzer: Optional[GeminiAnalyzer] = None,
        groq_analyzer: Optional[GroqAnalyzer] = None,
        transcriber: Optional[AudioTranscriber] = None,
    ) -> None:
        self.video_processor = video_processor or VideoProcessor()
        self._analyzer = analyzer
        self._groq_analyzer = groq_analyzer
        self.transcriber = transcriber or AudioTranscriber()

    @property
    def analyzer(self) -> GeminiAnalyzer:
        """Lazy initialization of GeminiAnalyzer to avoid errors if API key is not yet configured."""
        if self._analyzer is None:
            self._analyzer = GeminiAnalyzer()
        return self._analyzer

    @property
    def groq_analyzer(self) -> GroqAnalyzer:
        """Lazy initialization of GroqAnalyzer."""
        if self._groq_analyzer is None:
            self._groq_analyzer = GroqAnalyzer()
        return self._groq_analyzer

    def extract_audio(self, video_path: str, output_path: Optional[str] = None) -> str:
        """Extracts audio from video using local FFmpeg."""
        return self.transcriber.extract_audio(video_path, output_wav=output_path)

    def _analyze_media_or_transcript(
        self, media_path: str, all_words: List[WordTimestamp], options: ProcessingOptions
    ):
        """Analyzes media using either Groq (instant text analysis) or Gemini."""
        formatted_transcript = AudioTranscriber.format_transcript_with_timestamps(all_words)

        if options.ai_provider == AiProvider.GROQ:
            groq = self._groq_analyzer if self._groq_analyzer is not None else (
                GroqAnalyzer(api_key=options.groq_api_key) if options.groq_api_key else self.groq_analyzer
            )
            if formatted_transcript:
                return groq.analyze_transcript(formatted_transcript, options)
            else:
                # If no speech detected locally, try fast cloud transcription with Groq Whisper
                try:
                    cloud_words = groq.transcribe_audio_fast(media_path)
                    if cloud_words:
                        formatted = AudioTranscriber.format_transcript_with_timestamps(cloud_words)
                        all_words.extend(cloud_words)
                        return groq.analyze_transcript(formatted, options)
                except Exception as e:
                    print(f"Aviso: Transcrição via Groq Whisper falhou ({e}), tentando Gemini.")

        return self.analyzer.analyze_video(media_path, options)


    def process_cuts(self, video_path: str, options: ProcessingOptions) -> ProcessingResult:
        """Executes full video analysis (Groq/Gemini) and renders 9:16 cuts via VideoProcessor."""
        print(f"[{time.strftime('%H:%M:%S')}] LocalProcessorAdapter: Extraindo áudio e transcrevendo...")
        all_words = []
        if self.transcriber.is_available():
            try:
                all_words = self.transcriber.transcribe(video_path, model_size=options.whisper_model)
            except Exception as e:
                print(f"Aviso: Transcrição inicial falhou ({e}).")

        analysis = self._analyze_media_or_transcript(video_path, all_words, options)

        print(f"[{time.strftime('%H:%M:%S')}] LocalProcessorAdapter: Renderizando {len(analysis.clips)} cortes...")
        return self.video_processor.process_all_clips(video_path, analysis, options)

    def generate_manifest(
        self, media_path: str, options: ProcessingOptions
    ) -> List[ClientCutManifest]:
        """Transcribes media and queries Groq/Gemini to create a client-side cut manifest."""
        # 1. Transcribe with word-level timestamps
        all_words = []
        if self.transcriber.is_available():
            all_words = self.transcriber.transcribe(media_path, model_size=options.whisper_model)

        # 2. Analyze viral moments via Groq or Gemini using formatted text
        analysis = self._analyze_media_or_transcript(media_path, all_words, options)

        # 3. Re-align words relative to clip start in milliseconds
        manifests: List[ClientCutManifest] = []

        for idx, clip in enumerate(analysis.clips):
            start_sec = self.video_processor.parse_timestamp(clip.start_time)
            end_sec = self.video_processor.parse_timestamp(clip.end_time)

            if end_sec <= start_sec:
                continue

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
                    title=clip.title_pt if (options.translate_to_pt and clip.title_pt) else clip.title,
                    title_pt=clip.title_pt,
                    start_sec=round(start_sec, 2),
                    end_sec=round(end_sec, 2),
                    viral_score=clip.virality_score,
                    hook=clip.hook_pt if (options.translate_to_pt and clip.hook_pt) else (clip.hook_summary or clip.title),
                    hook_pt=clip.hook_pt,
                    crop_mode=options.crop_mode.value if hasattr(options.crop_mode, "value") else str(options.crop_mode),
                    words=clip_words,
                    subtitles_pt=clip.subtitles_pt,
                )
            )

        return manifests
