"""Local scene and face analyzer - generates viral clips WITHOUT any AI API.

Uses three heuristics entirely on the local machine:
  1. FFmpeg scdet (scene change detection) - finds natural cut points
  2. FFmpeg silencedetect / volumedetect - finds audio energy peaks
  3. MediaPipe / OpenCV Face Detection - tracks face position per segment for smart 9:16 crop

No internet connection required for analysis. No API key required.
"""

import math
import os
import re
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class SceneSegment:
    """A video segment detected by scene change or audio analysis."""
    start_sec: float
    end_sec: float
    duration: float = 0.0
    avg_volume_db: float = -60.0
    has_speech: bool = False
    face_x_ratio: float = 0.5   # 0.0=left edge, 1.0=right edge of source frame
    face_y_ratio: float = 0.3   # 0.0=top, 1.0=bottom
    score: float = 0.0           # final heuristic score
    title: str = ""

    def __post_init__(self):
        self.duration = round(self.end_sec - self.start_sec, 2)


def _fmt_time(sec: float) -> str:
    s = int(sec)
    return f"{s // 60:02d}:{s % 60:02d}"


class LocalSceneAnalyzer:
    """
    Generates clip manifests from a video file using only local tools.
    No Groq, Gemini, or any cloud API is used.
    """

    def __init__(self) -> None:
        self._face_detector = None
        self._detector_type = None  # 'tasks' or 'solutions'

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def analyze_video(
        self,
        video_path: str,
        max_clips: int = 8,
        min_duration: float = 25.0,
        max_duration: float = 65.0,
        scene_threshold: float = 0.4,
    ) -> List[SceneSegment]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video nao encontrado: {video_path}")

        total_duration = self._get_duration(video_path)
        print(f"[{time.strftime('%H:%M:%S')}] [LocalSceneAnalyzer] Duracao total: {total_duration:.1f}s - detectando cenas...")

        raw_cuts = self._detect_scene_changes(video_path, scene_threshold)
        segments = self._build_segments(raw_cuts, total_duration, min_duration, max_duration)

        if not segments:
            segments = self._split_equal_intervals(total_duration, min_duration, max_duration)
            print(f"[{time.strftime('%H:%M:%S')}] [LocalSceneAnalyzer] Sem cortes detectados - usando divisao uniforme em {len(segments)} segmentos.")

        print(f"[{time.strftime('%H:%M:%S')}] [LocalSceneAnalyzer] {len(segments)} segmentos. Analisando audio e faces...")

        segments = self._score_audio(video_path, segments)
        
        try:
            segments = self._detect_faces(video_path, segments)
        except Exception as e:
            print(f"[LocalSceneAnalyzer] Aviso na deteccao de faces (usando enquadramento padrao): {e}")

        for seg in segments:
            seg.score = self._compute_score(seg)

        segments.sort(key=lambda s: s.score, reverse=True)
        top = segments[:max_clips]
        top.sort(key=lambda s: s.start_sec)

        for i, seg in enumerate(top):
            seg.title = f"Cena {i + 1} ({_fmt_time(seg.start_sec)} - {_fmt_time(seg.end_sec)})"

        print(f"[{time.strftime('%H:%M:%S')}] [LocalSceneAnalyzer] Top {len(top)} cortes selecionados.")
        return top

    # ---------------------------------------------------------------
    # Step 1: Scene change detection (FFmpeg scdet)
    # ---------------------------------------------------------------

    def _detect_scene_changes(self, video_path: str, threshold: float) -> List[float]:
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", f"scdet=threshold={threshold}:sc_pass=1",
            "-an", "-f", "null", "-",
        ]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            cuts: List[float] = []
            for line in result.stderr.splitlines():
                if "lavfi.scd.time" in line:
                    m = re.search(r"lavfi\.scd\.time:\s*([\d.]+)", line)
                    if m:
                        cuts.append(float(m.group(1)))
            return sorted(set(cuts))
        except Exception as e:
            print(f"[LocalSceneAnalyzer] scdet falhou: {e}")
            return []

    # ---------------------------------------------------------------
    # Step 2: Build segments
    # ---------------------------------------------------------------

    def _build_segments(self, cuts: List[float], total: float, min_dur: float, max_dur: float) -> List[SceneSegment]:
        boundaries = [0.0] + cuts + [total]
        raw: List[Tuple[float, float]] = [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]

        merged: List[Tuple[float, float]] = []
        acc_start, acc_end = raw[0]
        for start, end in raw[1:]:
            if (acc_end - acc_start) < min_dur:
                acc_end = end
            else:
                merged.append((acc_start, acc_end))
                acc_start, acc_end = start, end
        merged.append((acc_start, acc_end))

        final: List[SceneSegment] = []
        for start, end in merged:
            dur = end - start
            if dur > max_dur:
                n_pieces = math.ceil(dur / max_dur)
                piece_dur = dur / n_pieces
                for k in range(n_pieces):
                    s = round(start + k * piece_dur, 2)
                    e = round(min(start + (k+1) * piece_dur, end), 2)
                    if e - s >= min_dur:
                        final.append(SceneSegment(start_sec=s, end_sec=e))
            elif dur >= min_dur:
                final.append(SceneSegment(start_sec=round(start, 2), end_sec=round(end, 2)))
        return final

    def _split_equal_intervals(self, total: float, min_dur: float, max_dur: float) -> List[SceneSegment]:
        target = min(max_dur, max(min_dur, 45.0))
        n = max(1, int(total / target))
        piece = total / n
        segs: List[SceneSegment] = []
        for k in range(n):
            s = round(k * piece, 2)
            e = round(min((k+1) * piece, total), 2)
            if e - s >= min_dur:
                segs.append(SceneSegment(start_sec=s, end_sec=e))
        return segs

    # ---------------------------------------------------------------
    # Step 3: Audio energy scoring
    # ---------------------------------------------------------------

    def _score_audio(self, video_path: str, segments: List[SceneSegment]) -> List[SceneSegment]:
        for seg in segments:
            dur = seg.end_sec - seg.start_sec
            cmd = [
                "ffmpeg",
                "-ss", str(seg.start_sec), "-t", str(dur),
                "-i", video_path,
                "-af", "volumedetect,silencedetect=n=-40dB:d=0.5",
                "-vn", "-f", "null", "-",
            ]
            try:
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
                output = result.stderr
                m = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", output)
                if m:
                    seg.avg_volume_db = float(m.group(1))
                total_silence = sum(
                    float(dm.group(1))
                    for line in output.splitlines()
                    for dm in [re.search(r"silence_duration:\s*([\d.]+)", line)]
                    if dm
                )
                speech_ratio = max(0.0, 1.0 - (total_silence / max(dur, 1.0)))
                seg.has_speech = speech_ratio > 0.3
            except Exception as e:
                print(f"[LocalSceneAnalyzer] volumedetect falhou ({seg.start_sec:.1f}s): {e}")
        return segments

    # ---------------------------------------------------------------
    # Step 4: Face detection via MediaPipe Tasks / Solutions
    # ---------------------------------------------------------------

    def _get_face_detector(self):
        if self._face_detector is not None:
            return self._face_detector

        # Strategy 1: MediaPipe Tasks (MediaPipe >= 1.0)
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            model_dir = os.path.expanduser("~/.cache/mediapipe")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "blaze_face_short_range.tflite")

            if not os.path.exists(model_path):
                url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
                try:
                    urllib.request.urlretrieve(url, model_path)
                except Exception as dl_err:
                    print(f"[LocalSceneAnalyzer] Download do modelo face detection falhou: {dl_err}")

            if os.path.exists(model_path):
                base_options = python.BaseOptions(model_asset_path=model_path)
                options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.4)
                self._face_detector = vision.FaceDetector.create_from_options(options)
                self._detector_type = "tasks"
                return self._face_detector
        except Exception as e:
            pass

        # Strategy 2: MediaPipe Solutions (Legacy MediaPipe < 1.0)
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
                self._face_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=0.4,
                )
                self._detector_type = "solutions"
                return self._face_detector
        except Exception:
            pass

        return None

    def _extract_frame(self, video_path: str, timestamp: float) -> Optional[object]:
        """Extracts a single RGB frame at timestamp using FFmpeg pipe -> numpy array."""
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", video_path],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=5,
            )
            wh = probe.stdout.strip()
            if "x" not in wh:
                return None
            w, h = map(int, wh.split("x"))

            cmd = ["ffmpeg", "-ss", str(timestamp), "-i", video_path, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10)
            if not result.stdout:
                return None
            import numpy as np
            return np.frombuffer(result.stdout, dtype=np.uint8).reshape((h, w, 3))
        except Exception:
            return None

    def _detect_faces(self, video_path: str, segments: List[SceneSegment]) -> List[SceneSegment]:
        detector = self._get_face_detector()
        if detector is None:
            return segments

        for seg in segments:
            duration = seg.end_sec - seg.start_sec
            sample_times = [seg.start_sec + duration * f for f in (0.25, 0.50, 0.75)]
            xs, ys = [], []
            for t in sample_times:
                frame = self._extract_frame(video_path, t)
                if frame is None:
                    continue
                try:
                    if self._detector_type == "tasks":
                        import mediapipe as mp
                        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                        res = detector.detect(mp_img)
                        if res and res.detections:
                            bbox = res.detections[0].bounding_box
                            fh, fw, _ = frame.shape
                            cx = (bbox.origin_x + bbox.width / 2.0) / max(1, fw)
                            cy = (bbox.origin_y + bbox.height / 2.0) / max(1, fh)
                            xs.append(max(0.0, min(1.0, cx)))
                            ys.append(max(0.0, min(1.0, cy)))
                    elif self._detector_type == "solutions":
                        detection = detector.process(frame)
                        if detection and detection.detections:
                            bbox = detection.detections[0].location_data.relative_bounding_box
                            cx = bbox.xmin + bbox.width / 2
                            cy = bbox.ymin + bbox.height / 2
                            xs.append(max(0.0, min(1.0, cx)))
                            ys.append(max(0.0, min(1.0, cy)))
                except Exception:
                    pass
            if xs:
                seg.face_x_ratio = sum(xs) / len(xs)
                seg.face_y_ratio = sum(ys) / len(ys)
        return segments

    # ---------------------------------------------------------------
    # Step 5: Heuristic score
    # ---------------------------------------------------------------

    def _compute_score(self, seg: SceneSegment) -> float:
        vol_norm = max(0.0, min(1.0, (seg.avg_volume_db + 50) / 40.0))
        vol_score = vol_norm * 40.0
        speech_score = 20.0 if seg.has_speech else 0.0
        dur = seg.duration
        if 30 <= dur <= 60:
            dur_score = 25.0
        elif 20 <= dur < 30 or 60 < dur <= 75:
            dur_score = 15.0
        else:
            dur_score = 5.0
        face_score = 15.0 if (seg.face_x_ratio != 0.5 or seg.face_y_ratio != 0.3) else 0.0
        return vol_score + speech_score + dur_score + face_score

    # ---------------------------------------------------------------
    # Utility
    # ---------------------------------------------------------------

    def _get_duration(self, video_path: str) -> float:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", video_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            return float(result.stdout.strip())
        except ValueError:
            return 0.0


def face_crop_offset(seg: SceneSegment, source_w: int, source_h: int, target_w: int = 1080, target_h: int = 1920) -> Tuple[int, int]:
    """Returns (x_offset, y_offset) FFmpeg crop coordinates centered on detected face."""
    crop_w = int(source_h * target_w / target_h)
    crop_w = min(crop_w, source_w)
    face_px = int(seg.face_x_ratio * source_w)
    x_off = max(0, min(source_w - crop_w, face_px - crop_w // 2))
    return x_off, 0
