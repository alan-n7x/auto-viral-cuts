"""Subtitle Generator module for creating stylized 9:16 ASS subtitles."""

import math
import os
from typing import List, Optional
from src.core.schemas import SubtitleStyle
from src.core.transcriber import WordTimestamp


class SubtitleGenerator:
    """Generates stylized Advanced SubStation Alpha (.ass) subtitles with word-level highlights."""

    @staticmethod
    def format_ass_time(seconds: float) -> str:
        """Converts float seconds to ASS timestamp format 'H:MM:SS.cc'."""
        seconds = max(0.0, seconds)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        sec_int = int(secs)
        centis = int(round((secs - sec_int) * 100))
        if centis >= 100:
            sec_int += 1
            centis -= 100
        return f"{hours}:{minutes:02d}:{sec_int:02d}.{centis:02d}"

    def get_style_config(self, style: SubtitleStyle) -> dict:
        """Returns visual styling parameters for the chosen SubtitleStyle."""
        # Note: In ASS format, colors are &H<Alpha><Blue><Green><Red>& in hexadecimal
        if style == SubtitleStyle.HORMOZI:
            return {
                "font_name": "DejaVu Sans",
                "font_size": 60,
                "primary_color": "&H00FFFFFF",     # White default
                "highlight_color": "&H0000FFFF",   # Bright Yellow active word
                "outline_color": "&H00000000",     # Solid black outline
                "back_color": "&H64000000",        # Subtle semi-transparent shadow
                "outline_width": 4.5,
                "shadow_offset": 2.0,
                "bold": 1,
                "uppercase": True,
            }
        elif style == SubtitleStyle.NEON:
            return {
                "font_name": "DejaVu Sans",
                "font_size": 58,
                "primary_color": "&H00FFFFFF",
                "highlight_color": "&H00FFFF00",   # Electric Cyan active word
                "outline_color": "&H00000000",
                "back_color": "&H80502000",
                "outline_width": 4.0,
                "shadow_offset": 3.0,
                "bold": 1,
                "uppercase": True,
            }
        else:  # MINIMAL
            return {
                "font_name": "Liberation Sans",
                "font_size": 52,
                "primary_color": "&H00FFFFFF",
                "highlight_color": "&H00C0C0C0",   # Light silver active
                "outline_color": "&H00000000",
                "back_color": "&H80000000",
                "outline_width": 2.5,
                "shadow_offset": 1.0,
                "bold": 1,
                "uppercase": False,
            }

    def generate_ass(
        self,
        words: List[WordTimestamp],
        output_ass_path: str,
        style: SubtitleStyle = SubtitleStyle.HORMOZI,
        max_words_per_line: int = 4,
    ) -> str:
        """Generates an .ass subtitle file with word-by-word active highlights."""
        cfg = self.get_style_config(style)

        # Header for 1080x1920 vertical video
        # Alignment=2 is Bottom-Center. MarginV=420 elevates above TikTok/Reels UI controls
        header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ViralStyle,{cfg['font_name']},{cfg['font_size']},{cfg['primary_color']},&H000000FF,{cfg['outline_color']},{cfg['back_color']},{cfg['bold']},0,0,0,100,100,0,0,1,{cfg['outline_width']},{cfg['shadow_offset']},2,60,60,420,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

        events = []

        if not words:
            # Write empty subtitle file
            with open(output_ass_path, "w", encoding="utf-8") as f:
                f.write(header)
            return output_ass_path

        # Group words into short lines (bursts of 3-5 words)
        chunks: List[List[WordTimestamp]] = []
        current_chunk: List[WordTimestamp] = []

        for w in words:
            current_chunk.append(w)
            # Break chunk if max words reached or there's a long pause (> 0.6s)
            if len(current_chunk) >= max_words_per_line:
                chunks.append(current_chunk)
                current_chunk = []
            elif len(current_chunk) > 1 and (w.end - current_chunk[-2].end) > 0.6:
                chunks.append(current_chunk[:-1])
                current_chunk = [w]

        if current_chunk:
            chunks.append(current_chunk)

        # For each chunk, produce word-by-word highlighted lines
        for chunk in chunks:
            for active_idx, active_word in enumerate(chunk):
                start_time_str = self.format_ass_time(active_word.start)
                end_time_str = self.format_ass_time(active_word.end)

                # Assemble line with active word highlighted
                parts = []
                for idx, w in enumerate(chunk):
                    text_disp = w.word.upper() if cfg["uppercase"] else w.word
                    if idx == active_idx:
                        # Highlighted word
                        parts.append(f"{{\\c{cfg['highlight_color']}&}}{text_disp}{{\\c{cfg['primary_color']}&}}")
                    else:
                        parts.append(text_disp)

                line_text = " ".join(parts)
                events.append(
                    f"Dialogue: 0,{start_time_str},{end_time_str},ViralStyle,,0,0,0,,{line_text}"
                )

        full_content = header + "\n".join(events) + "\n"

        os.makedirs(os.path.dirname(os.path.abspath(output_ass_path)), exist_ok=True)
        with open(output_ass_path, "w", encoding="utf-8") as f:
            f.write(full_content)

        return output_ass_path
