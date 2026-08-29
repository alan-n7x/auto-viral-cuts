"""Audio extraction and speech-to-text transcription module using faster-whisper."""

import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class WordTimestamp:
    """Represents a single word with precise start and end timestamps."""

    word: str
    start: float
    end: float
    probability: float = 1.0


class AudioTranscriber:
    """Extracts audio from video and generates word-level transcriptions using faster-whisper."""

    _models = {}

    @classmethod
    def is_available(cls) -> bool:
        """Checks if faster-whisper is installed and available."""
        try:
            import faster_whisper
            return True
        except ImportError:
            return False

    @classmethod
    def get_model(cls, model_size: str = "base"):
        """Loads and caches the faster-whisper model."""
        if model_size not in cls._models:
            from faster_whisper import WhisperModel
            print(f"[{time.strftime('%H:%M:%S')}] Carregando modelo faster-whisper '{model_size}' (CPU int8)...")
            cls._models[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
        return cls._models[model_size]

    def extract_audio(self, video_path: str, output_wav: Optional[str] = None) -> str:
        """Extracts 16kHz mono audio from video using FFmpeg for optimal Whisper processing."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Arquivo de vídeo não encontrado: {video_path}")

        if output_wav is None:
            fd, output_wav = tempfile.mkstemp(suffix="_extracted.wav")
            os.close(fd)

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            output_wav,
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            return output_wav
        except subprocess.CalledProcessError as e:
            if os.path.exists(output_wav):
                try:
                    os.remove(output_wav)
                except Exception:
                    pass
            raise RuntimeError(f"Erro ao extrair áudio com FFmpeg: {e.stderr}")

    def transcribe(
        self,
        video_or_audio_path: str,
        model_size: str = "base",
        language: Optional[str] = None,
    ) -> List[WordTimestamp]:
        """Transcribes the input file and returns word-level timestamps."""
        if not self.is_available():
            raise RuntimeError("faster-whisper não está instalado no ambiente.")

        wav_path = None
        is_temp_wav = False
        ext = os.path.splitext(video_or_audio_path)[1].lower()

        try:
            if ext in (".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv"):
                print(f"[{time.strftime('%H:%M:%S')}] Extraindo áudio para transcrição local com faster-whisper...")
                wav_path = self.extract_audio(video_or_audio_path)
                is_temp_wav = True
                audio_input = wav_path
            else:
                audio_input = video_or_audio_path

            model = self.get_model(model_size)
            print(f"[{time.strftime('%H:%M:%S')}] Transcrevendo áudio com word-level timestamps...")
            segments, info = model.transcribe(
                audio_input,
                word_timestamps=True,
                language=language,
                beam_size=5,
                vad_filter=True,
            )

            all_words: List[WordTimestamp] = []
            for segment in segments:
                if segment.words:
                    for w in segment.words:
                        cleaned_word = w.word.strip()
                        if cleaned_word:
                            all_words.append(
                                WordTimestamp(
                                    word=cleaned_word,
                                    start=round(w.start, 3),
                                    end=round(w.end, 3),
                                    probability=round(w.probability, 3),
                                )
                            )

            print(
                f"[{time.strftime('%H:%M:%S')}] Transcrição concluída: "
                f"{len(all_words)} palavras detectadas (idioma: {info.language})."
            )
            return all_words

        finally:
            if is_temp_wav and wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

    @staticmethod
    def get_words_for_interval(
        words: List[WordTimestamp],
        start_sec: float,
        end_sec: float,
    ) -> List[WordTimestamp]:
        """Filters words belonging to [start_sec, end_sec] and adjusts timing relative to clip start."""
        interval_words: List[WordTimestamp] = []
        for w in words:
            # Check overlap with interval
            if w.end > start_sec and w.start < end_sec:
                rel_start = max(0.0, round(w.start - start_sec, 3))
                rel_end = max(rel_start + 0.1, round(w.end - start_sec, 3))
                interval_words.append(
                    WordTimestamp(
                        word=w.word,
                        start=rel_start,
                        end=rel_end,
                        probability=w.probability,
                    )
                )
        return interval_words

    @staticmethod
    def format_transcript_with_timestamps(
        words: List[WordTimestamp],
        max_words_per_line: int = 8,
        max_pause_sec: float = 0.8,
    ) -> str:
        """Formats a list of WordTimestamp into readable sentence chunks with exact [start -> end] timestamps.

        Example:
            [00:00:12.400 -> 00:00:15.100] Keep your crew together
            [00:00:15.200 -> 00:00:18.000] We can do this
        """
        if not words:
            return ""

        def format_sec(sec: float) -> str:
            hours = int(sec // 3600)
            minutes = int((sec % 3600) // 60)
            seconds = sec % 60
            return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

        lines = []
        current_chunk: List[WordTimestamp] = []

        for w in words:
            current_chunk.append(w)
            if len(current_chunk) >= max_words_per_line:
                t_start = format_sec(current_chunk[0].start)
                t_end = format_sec(current_chunk[-1].end)
                text = " ".join(item.word.strip() for item in current_chunk)
                lines.append(f"[{t_start} -> {t_end}] {text}")
                current_chunk = []
            elif len(current_chunk) > 1 and (w.end - current_chunk[-2].end) > max_pause_sec:
                chunk_to_save = current_chunk[:-1]
                t_start = format_sec(chunk_to_save[0].start)
                t_end = format_sec(chunk_to_save[-1].end)
                text = " ".join(item.word.strip() for item in chunk_to_save)
                lines.append(f"[{t_start} -> {t_end}] {text}")
                current_chunk = [w]

        if current_chunk:
            t_start = format_sec(current_chunk[0].start)
            t_end = format_sec(current_chunk[-1].end)
            text = " ".join(item.word.strip() for item in current_chunk)
            lines.append(f"[{t_start} -> {t_end}] {text}")

        return "\n".join(lines)

