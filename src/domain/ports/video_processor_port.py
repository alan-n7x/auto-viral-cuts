"""Domain ports module for Auto Viral Cuts."""

from abc import ABC, abstractmethod
from typing import List, Optional

from src.core.schemas import ClientCutManifest, ProcessingOptions, ProcessingResult


class VideoProcessorPort(ABC):
    """Domain Port defining the contract for video processing operations."""

    @abstractmethod
    def extract_audio(self, video_path: str, output_path: Optional[str] = None) -> str:
        """Extracts an audio track from a video file.

        Args:
            video_path: Absolute path to source video file.
            output_path: Optional destination path for extracted audio.

        Returns:
            Path to the extracted audio file.
        """
        pass

    @abstractmethod
    def process_cuts(self, video_path: str, options: ProcessingOptions) -> ProcessingResult:
        """Analyzes video and generates viral 9:16 cuts with subtitles and hardware acceleration.

        Args:
            video_path: Path to the input video.
            options: Processing and rendering configuration.

        Returns:
            ProcessingResult with list of generated clips and execution metadata.
        """
        pass

    @abstractmethod
    def generate_manifest(
        self, media_path: str, options: ProcessingOptions
    ) -> List[ClientCutManifest]:
        """Generates a structured cut manifest with word-level timestamps for client-side rendering.

        Args:
            media_path: Path to audio or video media file.
            options: Processing options.

        Returns:
            List of ClientCutManifest items.
        """
        pass
