"""Video Processor module using FFmpeg for clipping and vertical cropping."""

import os
import subprocess
import time
from typing import List, Optional, Tuple
from dotenv import load_dotenv

from src.core.schemas import (
    ClipMetadata,
    CropMode,
    ProcessedClip,
    ProcessingOptions,
    ProcessingResult,
    ViralAnalysisResponse,
)

load_dotenv()


class VideoProcessor:
    """Handles precision video cutting and 9:16 vertical re-framing using FFmpeg."""

    def __init__(self, output_dir: Optional[str] = None) -> None:
        """Initialize video processor with output directory."""
        self.output_dir = output_dir or os.getenv("OUTPUT_DIR", "output_cuts")
        os.makedirs(self.output_dir, exist_ok=True)

    def parse_timestamp(self, ts_str: str) -> float:
        """Converts timestamp string ('HH:MM:SS', 'MM:SS', or seconds) to float seconds."""
        ts_str = ts_str.strip()
        try:
            if ":" not in ts_str:
                return float(ts_str)
            
            parts = list(map(float, ts_str.split(":")))
            if len(parts) == 3:
                return parts[0] * 3600 + parts[1] * 60 + parts[2]
            elif len(parts) == 2:
                return parts[0] * 60 + parts[1]
            else:
                return parts[0]
        except Exception as e:
            raise ValueError(f"Formato de timestamp inválido '{ts_str}': {e}")

    def get_video_dimensions(self, video_path: str) -> Tuple[int, int]:
        """Retrieves video width and height using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=s=x:p=0",
            video_path,
        ]
        try:
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
            )
            output = result.stdout.strip()
            if "x" in output:
                w, h = map(int, output.split("x"))
                return w, h
            # Fallback parsing if comma or space separated
            parts = output.replace(",", "x").split("x")
            return int(parts[0]), int(parts[1])
        except Exception:
            # Default fallback assumption (1080p horizontal)
            return 1920, 1080

    def _build_video_filter(self, width: int, height: int, crop_mode: CropMode) -> str:
        """Constructs FFmpeg filtergraph string for 9:16 vertical conversion."""
        # Target vertical aspect ratio: 9:16 (e.g. 1080x1920)
        # We assume output width 1080, height 1920
        if crop_mode == CropMode.CENTER_CROP:
            # Crop center 9:16 and scale to 1080x1920
            # If input is wider than 9:16: crop height = in_h, crop width = in_h * 9 / 16
            return (
                r"scale=iw:ih,"
                r"crop='if(gt(iw/ih\,9/16)\,ih*9/16\,iw)':'if(gt(iw/ih\,9/16)\,ih\,iw*16/9)',"
                r"scale=1080:1920:flags=lanczos"
            )
        elif crop_mode == CropMode.BLURRED_BACKGROUND:
            # Create blurred background + centered scaled foreground
            return (
                f"[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:10[bg];"
                f"[0:v]scale=1080:1920:force_original_aspect_ratio=decrease[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2"
            )
        elif crop_mode == CropMode.FIT_BLACK_BARS:
            # Add black letterboxing/pillarboxing to fit 9:16
            return (
                f"scale=1080:1920:force_original_aspect_ratio=decrease,"
                f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black"
            )
        else:
            # No crop / keep original or simple scale
            return f"scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black"

    def process_clip(
        self,
        video_path: str,
        clip_meta: ClipMetadata,
        index: int,
        options: ProcessingOptions,
    ) -> ProcessedClip:
        """Cuts a single clip from the video using FFmpeg with precision seeking."""
        start_sec = self.parse_timestamp(clip_meta.start_time)
        end_sec = self.parse_timestamp(clip_meta.end_time)
        duration = end_sec - start_sec

        if duration <= 0:
            duration = 30.0  # Safe fallback

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        safe_title = "".join(c if c.isalnum() else "_" for c in clip_meta.title[:30]).strip("_")
        output_filename = f"clip_{index:02d}_{safe_title}.mp4"
        output_path = os.path.abspath(os.path.join(self.output_dir, output_filename))

        in_w, in_h = self.get_video_dimensions(video_path)
        vf_filter = self._build_video_filter(in_w, in_h, options.crop_mode)

        # FFmpeg command with input seeking (-ss before -i for speed, -accurate_seek for precision)
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(max(0, start_sec - 0.5)), # slight lead-in for smooth start
            "-i", video_path,
            "-t", str(duration + 0.5),
            "-vf", vf_filter,
            "-c:v", options.video_codec,
            "-preset", options.preset,
            "-crf", str(options.crf),
            "-c:a", options.audio_codec,
            "-b:a", "192k",
            "-ar", "44100",
            "-pix_fmt", "yuv420p",
            output_path,
        ]

        print(f"[{time.strftime('%H:%M:%S')}] Gerando corte {index}: '{clip_meta.title}' ({clip_meta.start_time} - {clip_meta.end_time})...")
        
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Erro no FFmpeg ao gerar corte {index}: {e.stderr}")
            raise RuntimeError(f"Falha na renderização FFmpeg do corte {index}: {e.stderr}")

        file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0

        # Update metadata duration if not set
        clip_meta.duration_seconds = duration

        return ProcessedClip(
            clip_index=index,
            metadata=clip_meta,
            file_path=output_path,
            file_name=output_filename,
            file_size_bytes=file_size,
            duration_seconds=duration,
            status="completed",
        )

    def process_all_clips(
        self,
        video_path: str,
        analysis: ViralAnalysisResponse,
        options: ProcessingOptions,
    ) -> ProcessingResult:
        """Processes all viral clips identified in the analysis response."""
        start_time_total = time.time()
        processed_clips: List[ProcessedClip] = []

        for idx, clip_meta in enumerate(analysis.clips, start=1):
            try:
                processed = self.process_clip(video_path, clip_meta, idx, options)
                processed_clips.append(processed)
            except Exception as e:
                print(f"Erro ao processar corte {idx}: {e}")

        execution_time = time.time() - start_time_total

        return ProcessingResult(
            source_video=video_path,
            clips=processed_clips,
            total_clips=len(processed_clips),
            execution_time_seconds=round(execution_time, 2),
            status="success" if processed_clips else "failed",
            error_message=None if processed_clips else "Nenhum corte pôde ser gerado.",
        )
