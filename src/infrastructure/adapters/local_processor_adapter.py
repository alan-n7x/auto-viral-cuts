"""Local infrastructure adapter implementing VideoProcessorPort."""

import time
import uuid
from typing import List, Optional

from src.core.gemini_analyzer import GeminiAnalyzer
from src.core.schemas import ClientCutManifest, ProcessingOptions, ProcessingResult, WordTimestamp
from src.core.transcriber import AudioTranscriber
from src.core.video_processor import VideoProcessor
from src.domain.ports.video_processor_port import VideoProcessorPort


class LocalProcessorAdapter(VideoProcessorPort):
    """Infrastructure adapter executing video processing on local hardware and Gemini AI."""

    def __init__(
        self,
        video_processor: Optional[VideoProcessor] = None,
        analyzer: Optional[GeminiAnalyzer] = None,
        transcriber: Optional[AudioTranscriber] = None,
    ) -> None:
        self.video_processor = video_processor or VideoProcessor()
        self._analyzer = analyzer
        self.transcriber = transcriber or AudioTranscriber()

    @property
    def analyzer(self) -> GeminiAnalyzer:
        """Lazy initialization of GeminiAnalyzer to avoid errors if API key is not yet configured."""
        if self._analyzer is None:
            self._analyzer = GeminiAnalyzer()
        return self._analyzer

    def extract_audio(self, video_path: str, output_path: Optional[str] = None) -> str:
        """Extracts audio from video using local FFmpeg."""
        return self.transcriber.extract_audio(video_path, output_wav=output_path)

    def process_cuts(self, video_path: str, options: ProcessingOptions) -> ProcessingResult:
        """Executes full video analysis with Gemini and renders 9:16 cuts via VideoProcessor."""
        print(f"[{time.strftime('%H:%M:%S')}] LocalProcessorAdapter: Iniciando análise multimodal via Gemini...")
        analysis = self.analyzer.analyze_video(video_path, options)

        print(f"[{time.strftime('%H:%M:%S')}] LocalProcessorAdapter: Renderizando {len(analysis.clips)} cortes...")
        return self.video_processor.process_all_clips(video_path, analysis, options)

    def generate_manifest(
        self, media_path: str, options: ProcessingOptions
    ) -> List[ClientCutManifest]:
        """Transcribes media and queries Gemini to create a client-side cut manifest."""
        # 1. Transcribe with word-level timestamps
        all_words = []
        if self.transcriber.is_available():
            all_words = self.transcriber.transcribe(media_path, model_size=options.whisper_model)

        # 2. Analyze viral moments
        analysis = self.analyzer.analyze_video(media_path, options)

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
                    title=clip.title,
                    start_sec=round(start_sec, 2),
                    end_sec=round(end_sec, 2),
                    viral_score=clip.virality_score,
                    hook=clip.hook_summary or clip.title,
                    crop_mode=options.crop_mode.value if hasattr(options.crop_mode, "value") else str(options.crop_mode),
                    words=clip_words,
                )
            )

        return manifests
