"""FastAPI dependencies providing manual Clean Architecture dependency injection."""

from functools import lru_cache
from fastapi import Depends

from src.application.task_manager import TaskManager, task_manager
from src.application.use_cases.process_video_use_case import ProcessVideoUseCase
from src.domain.ports.video_processor_port import VideoProcessorPort
from src.infrastructure.adapters.local_processor_adapter import LocalProcessorAdapter


@lru_cache()
def get_task_manager() -> TaskManager:
    """Provides the in-memory TaskManager singleton."""
    return task_manager


@lru_cache()
def get_video_processor_port() -> VideoProcessorPort:
    """Provides the infrastructure VideoProcessorPort adapter instance."""
    return LocalProcessorAdapter()


def get_process_video_use_case(
    processor_port: VideoProcessorPort = Depends(get_video_processor_port),
    manager: TaskManager = Depends(get_task_manager),
) -> ProcessVideoUseCase:
    """Provides the ProcessVideoUseCase with injected dependencies."""
    return ProcessVideoUseCase(processor_port=processor_port, task_manager=manager)
