"""Video Processor module using FFmpeg with AMD/VAAPI hardware acceleration and subtitle burn-in."""

import os
import subprocess
import time
from typing import List, Optional, Set, Tuple
from dotenv import load_dotenv

from src.core.schemas import (
    ClipMetadata,
    CropMode,
    HwAccelMode,
    ProcessedClip,
    ProcessingOptions,
    ProcessingResult,
    ViralAnalysisResponse,
)
from src.core.subtitle_generator import SubtitleGenerator
from src.core.transcriber import AudioTranscriber, WordTimestamp

load_dotenv()


class VideoProcessor:
    """Handles precision video cutting, 9:16 vertical re-framing, and subtitle burn-in."""

    def __init__(self, output_dir: Optional[str] = None) -> None:
        """Initialize video processor with output directory."""
        self.output_dir = output_dir or os.getenv("OUTPUT_DIR", "output_cuts")
        os.makedirs(self.output_dir, exist_ok=True)
        self.transcriber = AudioTranscriber()
        self.subtitle_generator = SubtitleGenerator()

    @classmethod
    def get_supported_encoders(cls) -> Set[str]:
        """Returns set of video encoders available in the system FFmpeg."""
        try:
            res = subprocess.run(
                ["ffmpeg", "-encoders"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            encoders: Set[str] = set()
            for line in res.stdout.splitlines():
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0].startswith("V"):
                    encoders.add(parts[1])
            return encoders
        except Exception:
            return {"libx264"}

    @classmethod
    def detect_gpu_name(cls) -> Optional[str]:
        """Detects the name of the installed GPU (e.g. AMD Radeon RX 570)."""
        # Try lspci first
        try:
            res = subprocess.run(
                ["lspci"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            for line in res.stdout.splitlines():
                if any(k in line for k in ("VGA", "3D", "Display")):
                    if ":" in line:
                        return line.split(":", 2)[-1].strip()
        except Exception:
            pass

        # Try sysfs uevent on Linux
        uevent_path = "/sys/class/drm/renderD128/device/uevent"
        if os.path.exists(uevent_path):
            try:
                with open(uevent_path, "r") as f:
                    content = f.read()
                if "DRIVER=amdgpu" in content:
                    return "AMD GPU (amdgpu)"
            except Exception:
                pass

        return None

    @classmethod
    def detect_hw_accel(cls) -> Tuple[HwAccelMode, Optional[str], Optional[str]]:
        """Detects the best available hardware acceleration backend.

        Returns:
            Tuple of (detected_mode, device_path, gpu_name)
        """
        gpu_name = cls.detect_gpu_name()
        encoders = cls.get_supported_encoders()

        # Check AMD / Intel VAAPI on Linux (priority for AMD Radeon RX 570)
        vaapi_device = os.getenv("VAAPI_DEVICE", "/dev/dri/renderD128")
        if (
            "h264_vaapi" in encoders
            and os.path.exists(vaapi_device)
            and os.access(vaapi_device, os.R_OK | os.W_OK)
        ):
            return HwAccelMode.VAAPI, vaapi_device, gpu_name

        # Check NVIDIA NVENC
        if "h264_nvenc" in encoders:
            return HwAccelMode.NVENC, None, gpu_name

        # Check Apple Silicon VideoToolbox
        if "h264_videotoolbox" in encoders:
            return HwAccelMode.VIDEOTOOLBOX, None, gpu_name

        return HwAccelMode.CPU, None, gpu_name

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
            parts = output.replace(",", "x").split("x")
            return int(parts[0]), int(parts[1])
        except Exception:
            return 1920, 1080

    def _build_video_filter(
        self,
        width: int,
        height: int,
        crop_mode: CropMode,
        is_vaapi: bool = False,
        subtitle_file: Optional[str] = None,
    ) -> str:
        """Constructs FFmpeg filtergraph string for 9:16 vertical conversion (1080x1920) and subtitle overlay."""
        # Standardized target vertical resolution: 1080x1920 with strict 60 FPS CFR
        fps_prefix = "fps=60,"
        if crop_mode == CropMode.CENTER_CROP:
            vf = (
                fps_prefix +
                r"crop='if(gt(iw/ih\,9/16)\,ih*9/16\,iw)':'if(gt(iw/ih\,9/16)\,ih\,iw*16/9)',"
                r"scale=1080:1920:flags=lanczos"
            )
        elif crop_mode == CropMode.BLURRED_BACKGROUND:
            vf = (
                fps_prefix +
                "split[bg_in][fg_in];"
                "[bg_in]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:10[bg];"
                "[fg_in]scale=1080:1920:force_original_aspect_ratio=decrease[fg];"
                "[bg][fg]overlay=(W-w)/2:(H-h)/2"
            )
        elif crop_mode == CropMode.FIT_BLACK_BARS:
            vf = (
                fps_prefix +
                "scale=1080:1920:force_original_aspect_ratio=decrease,"
                "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black"
            )
        else:
            vf = (
                fps_prefix +
                "scale=1080:1920:force_original_aspect_ratio=decrease,"
                "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black"
            )

        # Apply stylized subtitles in software before GPU upload
        if subtitle_file and os.path.exists(subtitle_file):
            escaped_sub = (
                subtitle_file.replace("\\", "/")
                .replace(":", "\\:")
                .replace("'", "'\\''")
            )
            vf += f",subtitles='{escaped_sub}'"

        if is_vaapi:
            vf += ",format=nv12,hwupload,scale_vaapi=w=1080:h=1920"

        return vf


    def _build_ffmpeg_cmd(
        self,
        video_path: str,
        output_path: str,
        start_sec: float,
        duration: float,
        options: ProcessingOptions,
        in_w: int,
        in_h: int,
        resolved_accel: HwAccelMode,
        vaapi_device: Optional[str] = None,
        subtitle_file: Optional[str] = None,
    ) -> Tuple[List[str], str]:
        """Builds FFmpeg command line according to selected hardware acceleration and subtitles."""
        is_vaapi = resolved_accel == HwAccelMode.VAAPI
        vf_filter = self._build_video_filter(
            in_w,
            in_h,
            options.crop_mode,
            is_vaapi=is_vaapi,
            subtitle_file=subtitle_file,
        )

        cmd: List[str] = ["ffmpeg", "-y"]

        # Hardware initialization parameters (must precede inputs for VAAPI)
        if is_vaapi:
            device = vaapi_device or os.getenv("VAAPI_DEVICE", "/dev/dri/renderD128")
            cmd.extend([
                "-init_hw_device", f"vaapi=va:{device}",
                "-filter_hw_device", "va",
            ])

        # Input parameters with exact seek, mapping and 60 FPS CFR output
        cmd.extend([
            "-ss", str(max(0.0, start_sec)),
            "-i", video_path,
            "-t", str(duration),
            "-map", "0:v:0",
            "-map", "0:a:0?",
            "-vf", vf_filter,
            "-r", "60",
            "-fps_mode", "cfr",
        ])

        # Video encoder and quality selection
        if is_vaapi:
            cmd.extend([
                "-c:v", "h264_vaapi",
                "-qp", str(options.crf if options.crf > 0 else 20),
            ])
            accel_desc = "vaapi (AMD GPU)"
        elif resolved_accel == HwAccelMode.NVENC:
            cmd.extend([
                "-c:v", "h264_nvenc",
                "-preset", "p4",
                "-cq", str(options.crf if options.crf > 0 else 20),
            ])
            accel_desc = "nvenc (NVIDIA GPU)"
        elif resolved_accel == HwAccelMode.VIDEOTOOLBOX:
            cmd.extend([
                "-c:v", "h264_videotoolbox",
                "-q:v", "65",
            ])
            accel_desc = "videotoolbox (Apple Silicon)"
        else:
            cmd.extend([
                "-c:v", options.video_codec or "libx264",
                "-preset", options.preset,
                "-crf", str(options.crf),
                "-pix_fmt", "yuv420p",
            ])
            accel_desc = "cpu (libx264)"

        # Audio and container parameters (48kHz stereo, sample-accurate sync, faststart moov header)
        cmd.extend([
            "-c:a", options.audio_codec or "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-ac", "2",
            "-af", "aresample=async=1000:first_pts=0",
            "-movflags", "+faststart",
            output_path,
        ])

        return cmd, accel_desc


    def process_clip(
        self,
        video_path: str,
        clip_meta: ClipMetadata,
        index: int,
        options: ProcessingOptions,
        all_words: Optional[List[WordTimestamp]] = None,
    ) -> ProcessedClip:
        """Cuts a single clip from the video with GPU acceleration, subtitles, and CPU fallback."""
        start_sec = self.parse_timestamp(clip_meta.start_time)
        end_sec = self.parse_timestamp(clip_meta.end_time)
        duration = end_sec - start_sec

        if duration <= 0:
            duration = 30.0

        safe_title = "".join(c if c.isalnum() else "_" for c in clip_meta.title[:30]).strip("_")
        output_filename = f"clip_{index:02d}_{safe_title}.mp4"
        output_path = os.path.abspath(os.path.join(self.output_dir, output_filename))

        in_w, in_h = self.get_video_dimensions(video_path)

        # 1. Handle subtitle generation if requested
        subtitle_file: Optional[str] = None
        has_subtitles = False

        if options.burn_subtitles:
            # If translation is requested and translated cues exist, burn translated PT-BR subtitles
            if options.translate_to_pt and clip_meta.subtitles_pt:
                ass_filename = f"clip_{index:02d}_{safe_title}.ass"
                ass_path = os.path.abspath(os.path.join(self.output_dir, ass_filename))
                try:
                    self.subtitle_generator.generate_ass_from_cues(
                        clip_meta.subtitles_pt,
                        ass_path,
                        style=options.subtitle_style,
                        offset_start_sec=start_sec,
                    )
                    subtitle_file = ass_path
                    has_subtitles = True
                    print(
                        f"[{time.strftime('%H:%M:%S')}] Legendas traduzidas (PT-BR) geradas para corte {index} "
                        f"({len(clip_meta.subtitles_pt)} frases, estilo '{options.subtitle_style.value}')."
                    )
                except Exception as e:
                    print(f"Aviso: Falha ao gerar arquivo de legenda traduzida .ass ({e}).")
            else:
                # If all_words not provided in batch, transcribe on demand for original language
                if all_words is None and self.transcriber.is_available():
                    try:
                        all_words = self.transcriber.transcribe(
                            video_path, model_size=options.whisper_model
                        )
                    except Exception as e:
                        print(f"Aviso: Erro na transcrição Whisper ({e}). Continuando sem legendas.")

                if all_words:
                    clip_words = self.transcriber.get_words_for_interval(
                        all_words, start_sec, end_sec
                    )
                    if clip_words:
                        ass_filename = f"clip_{index:02d}_{safe_title}.ass"
                        ass_path = os.path.abspath(os.path.join(self.output_dir, ass_filename))
                        try:
                            self.subtitle_generator.generate_ass(
                                clip_words, ass_path, style=options.subtitle_style
                            )
                            subtitle_file = ass_path
                            has_subtitles = True
                            print(
                                f"[{time.strftime('%H:%M:%S')}] Legendas originais geradas para corte {index} "
                                f"({len(clip_words)} palavras, estilo '{options.subtitle_style.value}')."
                            )
                        except Exception as e:
                            print(f"Aviso: Falha ao gerar arquivo de legenda .ass ({e}).")


        # 2. Resolve target hardware acceleration
        env_hw = os.getenv("HW_ACCEL", "auto").lower()
        requested_accel = options.hw_accel
        if requested_accel == HwAccelMode.AUTO and env_hw != "auto":
            try:
                requested_accel = HwAccelMode(env_hw)
            except ValueError:
                requested_accel = HwAccelMode.AUTO

        detected_accel, vaapi_dev, gpu_name = self.detect_hw_accel()
        chosen_accel = detected_accel if requested_accel == HwAccelMode.AUTO else requested_accel

        accel_used_str = "cpu"
        sub_info = " + Legendas" if has_subtitles else ""
        try:
            cmd, accel_desc = self._build_ffmpeg_cmd(
                video_path=video_path,
                output_path=output_path,
                start_sec=start_sec,
                duration=duration,
                options=options,
                in_w=in_w,
                in_h=in_h,
                resolved_accel=chosen_accel,
                vaapi_device=vaapi_dev,
                subtitle_file=subtitle_file,
            )
            gpu_info = f" ({gpu_name})" if gpu_name and chosen_accel != HwAccelMode.CPU else ""
            print(
                f"[{time.strftime('%H:%M:%S')}] Gerando corte {index}: '{clip_meta.title}' "
                f"({clip_meta.start_time} - {clip_meta.end_time}) [{accel_desc}{gpu_info}{sub_info}]..."
            )
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            accel_used_str = accel_desc
        except subprocess.CalledProcessError as e:
            # Fallback to CPU if hardware acceleration failed
            if chosen_accel != HwAccelMode.CPU:
                print(
                    f"Aviso: Falha na aceleração por hardware ({chosen_accel}): {e.stderr[:200]}... "
                    "Tentando fallback automático para CPU (libx264)..."
                )
                cmd_cpu, accel_desc_cpu = self._build_ffmpeg_cmd(
                    video_path=video_path,
                    output_path=output_path,
                    start_sec=start_sec,
                    duration=duration,
                    options=options,
                    in_w=in_w,
                    in_h=in_h,
                    resolved_accel=HwAccelMode.CPU,
                    subtitle_file=subtitle_file,
                )
                subprocess.run(
                    cmd_cpu,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True,
                )
                accel_used_str = f"{accel_desc_cpu} (fallback)"
            else:
                print(f"Erro no FFmpeg ao gerar corte {index}: {e.stderr}")
                raise RuntimeError(f"Falha na renderização FFmpeg do corte {index}: {e.stderr}")

        file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        clip_meta.duration_seconds = duration

        return ProcessedClip(
            clip_index=index,
            metadata=clip_meta,
            file_path=output_path,
            file_name=output_filename,
            file_size_bytes=file_size,
            duration_seconds=duration,
            status="completed",
            hw_accel_used=accel_used_str,
            has_subtitles=has_subtitles,
            subtitle_path=subtitle_file,
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

        # Run transcription once for the whole video if subtitles are enabled
        all_words: Optional[List[WordTimestamp]] = None
        if options.burn_subtitles and self.transcriber.is_available():
            try:
                all_words = self.transcriber.transcribe(
                    video_path, model_size=options.whisper_model
                )
            except Exception as e:
                print(f"Aviso: Falha na transcrição geral com Whisper ({e}). Cortes serão gerados sem legendas.")

        for idx, clip_meta in enumerate(analysis.clips, start=1):
            try:
                processed = self.process_clip(
                    video_path, clip_meta, idx, options, all_words=all_words
                )
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
