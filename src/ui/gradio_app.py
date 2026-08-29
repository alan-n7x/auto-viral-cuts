"""Gradio UI module providing a Drag & Drop web interface for Auto Viral Cuts."""

import os
from typing import Any, Tuple
import gradio as gr

from src.core.gemini_analyzer import GeminiAnalyzer
from src.core.schemas import CropMode, HwAccelMode, PlatformPreset, ProcessingOptions, SubtitleStyle
from src.core.video_processor import VideoProcessor


def run_viral_pipeline(
    video_file: Any,
    api_key_input: str,
    target_platform: str,
    crop_mode: str,
    hw_accel: str,
    burn_subtitles: bool,
    subtitle_style: str,
    whisper_model: str,
    translate_to_pt: bool,
    max_clips: int,
    min_duration: int,
    max_duration: int,
    custom_prompt: str,
) -> Tuple[str, str, Any]:
    """Executes the full Gemini analysis and FFmpeg clipping pipeline with subtitles from the Gradio UI."""
    if video_file is None:
        return "⚠️ Por favor, envie ou arraste um arquivo de vídeo válido.", "", None

    video_path = video_file.name if hasattr(video_file, "name") else str(video_file)

    if not os.path.exists(video_path):
        return f"❌ Erro: Arquivo de vídeo não encontrado em {video_path}.", "", None

    api_key = api_key_input.strip() or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return (
            "❌ Chave da API do Gemini não informada. Insira sua chave ou configure o arquivo .env.",
            "",
            None,
        )

    try:
        options = ProcessingOptions(
            target_platform=PlatformPreset(target_platform),
            crop_mode=CropMode(crop_mode),
            hw_accel=HwAccelMode(hw_accel),
            burn_subtitles=burn_subtitles,
            subtitle_style=SubtitleStyle(subtitle_style),
            whisper_model=whisper_model,
            translate_to_pt=translate_to_pt,
            max_clips=int(max_clips),
            min_duration_seconds=int(min_duration),
            max_duration_seconds=int(max_duration),
            custom_prompt=custom_prompt.strip() if custom_prompt else None,
        )


        # 1. Analyze with Gemini
        analyzer = GeminiAnalyzer(api_key=api_key)
        analysis_result = analyzer.analyze_video(video_path, options)

        # 2. Process clips with FFmpeg + faster-whisper
        processor = VideoProcessor()
        result = processor.process_all_clips(video_path, analysis_result, options)

        if not result.clips:
            return (
                "⚠️ O vídeo foi analisado, mas nenhum corte foi gerado com sucesso.",
                f"**Resumo do Vídeo:** {analysis_result.video_summary}",
                None,
            )

        # Build markdown summary report
        md_report = f"### 🎬 Resumo do Vídeo\n> {analysis_result.video_summary}\n\n"
        md_report += f"**Temas Principais:** {', '.join(analysis_result.key_themes)}\n\n"
        md_report += f"--- \n### ✂️ Cortes Gerados com Sucesso ({len(result.clips)}):\n\n"

        file_paths = []
        for clip in result.clips:
            file_paths.append(clip.file_path)
            meta = clip.metadata
            md_report += f"#### **Corte #{clip.clip_index}: {meta.title}**\n"
            md_report += f"- ⏱️ **Timestamp:** `{meta.start_time}` até `{meta.end_time}` ({clip.duration_seconds}s)\n"
            md_report += f"- 🔥 **Pontuação de Viralidade:** `{meta.virality_score}/100`\n"
            md_report += f"- 💡 **Motivo:** {meta.virality_reason}\n"
            if clip.hw_accel_used:
                md_report += f"- ⚡ **Aceleração:** `{clip.hw_accel_used}`\n"
            if clip.has_subtitles:
                md_report += f"- 💬 **Legendas:** `Estilo {options.subtitle_style.value.upper()} (Highlight Ativo)`\n"
            if meta.suggested_caption:
                md_report += f"- 📝 **Legenda Pronta:** {meta.suggested_caption}\n"
            if meta.hashtags:
                md_report += f"- 🏷️ **Hashtags:** {' '.join(meta.hashtags)}\n"
            md_report += f"- 📁 **Arquivo:** `{clip.file_name}`\n\n"

        first_video = file_paths[0] if file_paths else None
        status_msg = f"✅ Sucesso! {len(result.clips)} cortes gerados em {result.execution_time_seconds}s."

        return status_msg, md_report, first_video

    except Exception as e:
        import traceback
        traceback.print_exc()
        error_message = str(e)
        transient_markers = (
            "high demand",
            "overloaded",
            "temporarily unavailable",
            "server_error",
        )
        if any(marker in error_message.lower() for marker in transient_markers):
            return (
                "⚠️ O Gemini está temporariamente sobrecarregado. "
                "Tente novamente em alguns minutos.",
                "",
                None,
            )
        return f"❌ Erro durante o processamento: {error_message}", "", None


def create_demo() -> gr.Blocks:
    """Creates the Gradio UI blocks application."""
    detected_mode, _, gpu_name = VideoProcessor.detect_hw_accel()
    gpu_desc = f" ({gpu_name})" if gpu_name else ""

    with gr.Blocks(title="Auto Viral Cuts - IA & FFmpeg") as demo:
        gr.Markdown(
            """
            # 🚀 Auto Viral Cuts
            ### Transforme vídeos longos em cortes verticais virais (TikTok, Reels, Shorts) com IA (Gemini), legendas dinâmicas (faster-whisper) e FFmpeg acelerado por GPU.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Entrada de Vídeo e Configurações")
                video_input = gr.Video(
                    label="Arraste e solte seu vídeo longo aqui (MP4, MKV, MOV, AVI)",
                    sources=["upload", "webcam"],
                    interactive=True,
                )

                api_key_input = gr.Textbox(
                    label="Google Gemini API Key (Opcional se definido no .env)",
                    type="password",
                    placeholder="AIzaSy...",
                )

                with gr.Row():
                    target_platform = gr.Dropdown(
                        choices=["tiktok", "reels", "shorts", "general"],
                        value="tiktok",
                        label="Plataforma Alvo",
                    )
                    crop_mode = gr.Dropdown(
                        choices=["center_crop", "blurred_background", "fit_black_bars", "no_crop"],
                        value="center_crop",
                        label="Modo de Enquadramento 9:16",
                    )

                with gr.Row():
                    hw_accel = gr.Dropdown(
                        choices=["auto", "vaapi", "nvenc", "videotoolbox", "cpu"],
                        value="auto",
                        label=f"Aceleração de Hardware / GPU{gpu_desc}",
                    )
                    max_clips = gr.Slider(
                        minimum=1, maximum=10, value=3, step=1, label="Máximo de Cortes"
                    )

                with gr.Row():
                    burn_subtitles = gr.Checkbox(
                        value=True,
                        label="🔥 Embutir Legendas Dinâmicas (Word-level faster-whisper)",
                    )
                    subtitle_style = gr.Dropdown(
                        choices=["hormozi", "neon", "minimal"],
                        value="hormozi",
                        label="Estilo da Legenda",
                    )
                    whisper_model = gr.Dropdown(
                        choices=["tiny", "base", "small"],
                        value="base",
                        label="Precisão do Whisper",
                    )

                with gr.Row():
                    translate_to_pt = gr.Checkbox(
                        value=False,
                        label="🇧🇷 Traduzir para Português (PT-BR) [Títulos, Ganchos e Legendas]",
                    )

                with gr.Row():
                    min_duration = gr.Slider(
                        minimum=10, maximum=60, value=15, step=5, label="Duração Mínima (s)"
                    )
                    max_duration = gr.Slider(
                        minimum=30, maximum=180, value=60, step=5, label="Duração Máxima (s)"
                    )

                custom_prompt = gr.Textbox(
                    label="Instruções Personalizadas de Nicho (Opcional)",
                    placeholder="Ex: Foque nas dicas práticas de programação e nos momentos de maior humor...",
                    lines=2,
                )

                submit_btn = gr.Button("🔥 Gerar Cortes Virais com Legendas", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("### 2. Resultados e Pré-visualização")
                status_output = gr.Textbox(label="Status do Pipeline", interactive=False)
                preview_video = gr.Video(label="Pré-visualização do 1º Corte Gerado")
                report_output = gr.Markdown(label="Relatório Analítico dos Cortes")

        submit_btn.click(
            fn=run_viral_pipeline,
            inputs=[
                video_input,
                api_key_input,
                target_platform,
                crop_mode,
                hw_accel,
                burn_subtitles,
                subtitle_style,
                whisper_model,
                translate_to_pt,
                max_clips,
                min_duration,
                max_duration,
                custom_prompt,
            ],
            outputs=[status_output, report_output, preview_video],
        )


        gr.Markdown(
            """
            ---
            *Auto Viral Cuts v0.1.0 | Desenvolvido com Google Gemini 3.6 Flash, faster-whisper & FFmpeg (GPU AMD RX 570 VAAPI).*
            """
        )

    return demo


if __name__ == "__main__":
    app = create_demo()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
