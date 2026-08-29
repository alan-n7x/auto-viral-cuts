"""Tests for Clean Architecture (Ports and Adapters), async aiofiles streaming, and BackgroundTasks."""

import io
import os
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from src.application.task_manager import TaskManager
from src.application.use_cases.process_video_use_case import ProcessVideoUseCase
from src.core.schemas import (
    ClipMetadata,
    CropMode,
    ProcessedClip,
    ProcessingOptions,
    ProcessingResult,
    TaskState,
)
from src.domain.ports.video_processor_port import VideoProcessorPort
from src.infrastructure.adapters.local_processor_adapter import LocalProcessorAdapter
from src.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_task_manager_lifecycle():
    """Validates creating, updating, retrieving and listing background tasks."""
    tm = TaskManager()
    task = tm.create_task("task_123", "video_sample.mp4")

    assert task.task_id == "task_123"
    assert task.status == TaskState.QUEUED
    assert task.progress == 0
    assert task.file_name == "video_sample.mp4"

    # Update to processing
    updated = tm.update_task("task_123", status=TaskState.PROCESSING, progress=45)
    assert updated.status == TaskState.PROCESSING
    assert updated.progress == 45

    # Retrieve
    retrieved = tm.get_task("task_123")
    assert retrieved is not None
    assert retrieved.status == TaskState.PROCESSING

    # Complete with dummy result
    dummy_result = ProcessingResult(
        source_video="video_sample.mp4",
        clips=[],
        total_clips=0,
        execution_time_seconds=1.2,
        status="success",
    )
    completed = tm.update_task("task_123", status=TaskState.COMPLETED, progress=100, result=dummy_result)
    assert completed.status == TaskState.COMPLETED
    assert completed.result.total_clips == 0

    # Nonexistent task
    assert tm.get_task("nonexistent") is None
    assert tm.update_task("nonexistent", status=TaskState.FAILED) is None


def test_process_video_use_case_success(tmp_path):
    """Verifies that ProcessVideoUseCase processes video via port, updates status to completed, and cleans up."""
    tm = TaskManager()
    mock_port = MagicMock(spec=VideoProcessorPort)

    dummy_result = ProcessingResult(
        source_video="test.mp4",
        clips=[],
        total_clips=1,
        execution_time_seconds=2.5,
        status="success",
    )
    mock_port.process_cuts.return_value = dummy_result

    # Create dummy video file on disk
    video_file = str(tmp_path / "temp_upload.mp4")
    with open(video_file, "w") as f:
        f.write("fake video content")

    task_id = "task_success_1"
    tm.create_task(task_id, "temp_upload.mp4")

    use_case = ProcessVideoUseCase(processor_port=mock_port, task_manager=tm)
    options = ProcessingOptions(crop_mode=CropMode.CENTER_CROP)

    result = use_case.execute(task_id, video_file, options)

    assert result.status == "success"
    task = tm.get_task(task_id)
    assert task.status == TaskState.COMPLETED
    assert task.progress == 100
    assert task.result is not None
    # Temporary video must be deleted from disk
    assert not os.path.exists(video_file)


def test_process_video_use_case_failure(tmp_path):
    """Verifies that ProcessVideoUseCase handles exceptions, marks task as failed, and removes temp file."""
    tm = TaskManager()
    mock_port = MagicMock(spec=VideoProcessorPort)
    mock_port.process_cuts.side_effect = RuntimeError("FFmpeg error rendering clip")

    video_file = str(tmp_path / "temp_upload_err.mp4")
    with open(video_file, "w") as f:
        f.write("fake video content")

    task_id = "task_fail_1"
    tm.create_task(task_id, "temp_upload_err.mp4")

    use_case = ProcessVideoUseCase(processor_port=mock_port, task_manager=tm)
    options = ProcessingOptions()

    with pytest.raises(RuntimeError, match="FFmpeg error rendering clip"):
        use_case.execute(task_id, video_file, options)

    task = tm.get_task(task_id)
    assert task.status == TaskState.FAILED
    assert "FFmpeg error rendering clip" in task.error
    # Cleanup should still happen in finally
    assert not os.path.exists(video_file)


def test_local_processor_adapter_implements_port():
    """Validates that LocalProcessorAdapter strictly adheres to VideoProcessorPort contract."""
    assert issubclass(LocalProcessorAdapter, VideoProcessorPort)

    mock_vp = MagicMock()
    mock_analyzer = MagicMock()
    mock_transcriber = MagicMock()

    adapter = LocalProcessorAdapter(
        video_processor=mock_vp,
        analyzer=mock_analyzer,
        transcriber=mock_transcriber,
    )

    # Test extract_audio delegation
    adapter.extract_audio("input.mp4", "output.wav")
    mock_transcriber.extract_audio.assert_called_once_with("input.mp4", output_wav="output.wav")

    # Test process_cuts delegation
    options = ProcessingOptions()
    mock_analyzer.analyze_video.return_value = MagicMock(clips=[])
    adapter.process_cuts("input.mp4", options)
    mock_analyzer.analyze_video.assert_called_once_with("input.mp4", options)
    mock_vp.process_all_clips.assert_called_once()


def test_async_process_endpoint_returns_202_accepted(client):
    """Verifies that POST /api/v1/process-video-async returns 202 Accepted and creates task."""
    fake_content = b"X" * (128 * 1024)  # 128 KB (2 chunks of 64KB)
    files = {"file": ("big_movie.mp4", io.BytesIO(fake_content), "video/mp4")}
    data = {
        "max_clips": 3,
        "crop_mode": "center_crop",
        "custom_prompt": "Cortes mais impactantes",
    }

    with patch("src.application.use_cases.process_video_use_case.ProcessVideoUseCase.execute") as mock_execute:
        response = client.post("/api/v1/process-video-async", files=files, data=data)

        assert response.status_code == 202
        body = response.json()
        assert "task_id" in body
        assert body["status"] == "queued"
        assert body["file_name"] == "big_movie.mp4"
        assert "enfileirado com sucesso" in body["message"]

        # Polling status endpoint
        task_id = body["task_id"]
        status_res = client.get(f"/api/v1/tasks/{task_id}")
        assert status_res.status_code == 200
        status_body = status_res.json()
        assert status_body["task_id"] == task_id
        assert status_body["file_name"] == "big_movie.mp4"


def test_task_status_not_found(client):
    """Verifies 404 when querying an invalid task_id."""
    response = client.get("/api/v1/tasks/nonexistent-task-id-12345")
    assert response.status_code == 404
    assert "não encontrada" in response.json()["detail"]
