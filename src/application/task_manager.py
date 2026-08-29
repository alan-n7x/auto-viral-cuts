"""In-memory task manager for asynchronous background video processing jobs."""

import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.core.schemas import ProcessingResult, TaskState, TaskStatusResponse


class TaskManager:
    """Thread-safe in-memory registry for tracking async video processing tasks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: Dict[str, TaskStatusResponse] = {}

    def create_task(self, task_id: str, file_name: str) -> TaskStatusResponse:
        """Registers a new task with initial QUEUED status."""
        now = datetime.now(timezone.utc).isoformat()
        task = TaskStatusResponse(
            task_id=task_id,
            status=TaskState.QUEUED,
            file_name=file_name,
            created_at=now,
            updated_at=now,
            progress=0,
            result=None,
            error=None,
        )
        with self._lock:
            self._tasks[task_id] = task
        return task

    def get_task(self, task_id: str) -> Optional[TaskStatusResponse]:
        """Retrieves task status by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def update_task(
        self,
        task_id: str,
        status: TaskState,
        progress: int = 0,
        result: Optional[ProcessingResult] = None,
        error: Optional[str] = None,
    ) -> Optional[TaskStatusResponse]:
        """Updates status, progress, result, or error of a task."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None

            updated_task = TaskStatusResponse(
                task_id=task.task_id,
                status=status,
                file_name=task.file_name,
                created_at=task.created_at,
                updated_at=now,
                progress=progress,
                result=result or task.result,
                error=error or task.error,
            )
            self._tasks[task_id] = updated_task
            return updated_task

    def list_tasks(self) -> List[TaskStatusResponse]:
        """Lists all registered tasks."""
        with self._lock:
            return list(self._tasks.values())


# Global singleton instance for in-memory tracking
task_manager = TaskManager()
