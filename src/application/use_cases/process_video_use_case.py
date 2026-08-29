"""Use Case for processing video cuts in background."""

import os
import time
from typing import Optional

from src.application.task_manager import TaskManager, task_manager as default_task_manager
from src.core.schemas import ProcessingOptions, ProcessingResult, TaskState
from src.domain.ports.video_processor_port import VideoProcessorPort


class ProcessVideoUseCase:
    """Orchestrates video cut processing in background, updating task status."""

    def __init__(
        self,
        processor_port: VideoProcessorPort,
        task_manager: Optional[TaskManager] = None,
    ) -> None:
        self.processor_port = processor_port
        self.task_manager = task_manager or default_task_manager

    def execute(self, task_id: str, video_path: str, options: ProcessingOptions) -> ProcessingResult:
        """Executes the video processing pipeline, updates progress, and ensures cleanup.

        Args:
            task_id: Unique task identifier.
            video_path: Path to the uploaded temporary video on disk.
            options: Video processing configuration.

        Returns:
            ProcessingResult containing the exported clips.
        """
        print(f"[{time.strftime('%H:%M:%S')}] Iniciando processamento em background (task_id: {task_id})...")
        self.task_manager.update_task(task_id, status=TaskState.PROCESSING, progress=15)

        try:
            # Process video cuts via injected Port
            result = self.processor_port.process_cuts(video_path, options)

            # Update task to completed
            self.task_manager.update_task(
                task_id,
                status=TaskState.COMPLETED,
                progress=100,
                result=result,
            )
            print(
                f"[{time.strftime('%H:%M:%S')}] Task {task_id} concluída com sucesso! "
                f"{result.total_clips} cortes gerados em {result.execution_time_seconds:.1f}s."
            )
            return result

        except Exception as e:
            error_msg = f"Falha no processamento do vídeo: {str(e)}"
            print(f"[{time.strftime('%H:%M:%S')}] Erro na Task {task_id}: {error_msg}")
            self.task_manager.update_task(
                task_id,
                status=TaskState.FAILED,
                progress=0,
                error=error_msg,
            )
            raise

        finally:
            # Always clean up the uploaded temporary video file from disk
            if os.path.exists(video_path):
                try:
                    os.remove(video_path)
                    print(f"[{time.strftime('%H:%M:%S')}] Arquivo temporário de upload removido: {video_path}")
                except Exception as cleanup_err:
                    print(f"Aviso: Não foi possível remover {video_path}: {cleanup_err}")
