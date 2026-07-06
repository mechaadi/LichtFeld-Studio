# SPDX-License-Identifier: GPL-3.0-or-later
"""REST API for the video/photos -> gaussian splat pipeline.

    uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints:
    POST   /jobs              upload a video OR a set of photos, start a pipeline job
    GET    /jobs              list all jobs
    GET    /jobs/{id}         job status
    GET    /jobs/{id}/result  download the trained .ply
    GET    /jobs/{id}/log     pipeline log (for debugging failures)
    DELETE /jobs/{id}         delete a finished job and its working files
    GET    /health            liveness + tool availability check
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from pipeline import (
    COLMAP,
    FFMPEG,
    LICHTFELD,
    PipelineError,
    PipelineParams,
    run_pipeline,
)

WORK_ROOT = Path(os.environ.get("VIDEO2SPLAT_WORK_DIR", Path(__file__).parent / "jobs"))
MAX_UPLOAD_BYTES = int(os.environ.get("VIDEO2SPLAT_MAX_UPLOAD_MB", "2048")) * 1024 * 1024
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
ALLOWED_PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp",
                          ".webp", ".heic", ".heif"}
MAX_PHOTOS = int(os.environ.get("VIDEO2SPLAT_MAX_PHOTOS", "500"))

app = FastAPI(title="video2splat", version="0.1.0")


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued | extracting_frames | reconstructing_poses | training | done | failed
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    num_frames: Optional[int] = None
    num_registered_images: Optional[int] = None
    ply_path: Optional[Path] = None
    params: Optional[PipelineParams] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "num_frames": self.num_frames,
            "num_registered_images": self.num_registered_images,
            "result_ready": self.status == "done",
        }


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()
# one pipeline at a time: COLMAP and training each want the whole GPU
_pipeline_semaphore = threading.Semaphore(int(os.environ.get("VIDEO2SPLAT_CONCURRENCY", "1")))


def _job_dir(job_id: str) -> Path:
    return WORK_ROOT / job_id


def _worker(job: Job, video_path: Optional[Path], photo_paths: Optional[list[Path]]) -> None:
    with _pipeline_semaphore:
        def on_stage(stage: str) -> None:
            with _jobs_lock:
                job.status = stage

        try:
            result = run_pipeline(_job_dir(job.id), job.params, on_stage,
                                  video=video_path, photos=photo_paths)
            with _jobs_lock:
                job.status = "done"
                job.ply_path = result.ply_path
                job.num_frames = result.num_frames
                job.num_registered_images = result.num_registered_images
        except PipelineError as e:
            with _jobs_lock:
                job.status = "failed"
                job.error = str(e)
        except Exception as e:  # noqa: BLE001 - surface anything unexpected in the job status
            with _jobs_lock:
                job.status = "failed"
                job.error = f"unexpected error: {e}"
        finally:
            with _jobs_lock:
                job.finished_at = time.time()


async def _save_upload(upload: UploadFile, dest: Path, budget: list[int]) -> int:
    """Stream one upload to disk against a shared byte budget. Returns bytes written."""
    written = 0
    with open(dest, "wb") as f:
        while chunk := await upload.read(8 * 1024 * 1024):
            written += len(chunk)
            budget[0] += len(chunk)
            if budget[0] > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "upload exceeds size limit")
            f.write(chunk)
    return written


@app.post("/jobs", status_code=202)
async def create_job(
    video: Optional[UploadFile] = File(None),
    photos: Optional[list[UploadFile]] = File(None),
    iterations: int = Form(30000, ge=100, le=200000),
    strategy: str = Form("mcmc"),
    max_frames: int = Form(300, ge=10, le=2000),
    resize_factor: str = Form("auto"),
) -> dict:
    if (video is None) == (not photos):
        raise HTTPException(400, "provide exactly one of: 'video' (single file) or 'photos' (multiple files)")
    if strategy not in {"mcmc", "mrnf", "igs+"}:
        raise HTTPException(400, "strategy must be one of: mcmc, mrnf, igs+")
    if resize_factor not in {"auto", "1", "2", "4", "8"}:
        raise HTTPException(400, "resize_factor must be one of: auto, 1, 2, 4, 8")

    if video is not None:
        suffix = Path(video.filename or "video.mp4").suffix.lower()
        if suffix not in ALLOWED_VIDEO_SUFFIXES:
            raise HTTPException(400, f"unsupported video format '{suffix}'")
    else:
        if not (5 <= len(photos) <= MAX_PHOTOS):
            raise HTTPException(400, f"need between 5 and {MAX_PHOTOS} photos, got {len(photos)}")
        for p in photos:
            s = Path(p.filename or "").suffix.lower()
            if s not in ALLOWED_PHOTO_SUFFIXES:
                raise HTTPException(400, f"unsupported photo format '{s}' ({p.filename})")

    job_id = uuid.uuid4().hex[:12]
    job_dir = _job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path: Optional[Path] = None
    photo_paths: Optional[list[Path]] = None

    try:
        budget = [0]
        if video is not None:
            video_path = job_dir / f"input{Path(video.filename or 'v.mp4').suffix.lower()}"
            if await _save_upload(video, video_path, budget) == 0:
                raise HTTPException(400, "empty upload")
        else:
            raw_dir = job_dir / "photos_raw"
            raw_dir.mkdir()
            photo_paths = []
            for i, p in enumerate(photos, start=1):
                dest = raw_dir / f"photo_{i:05d}{Path(p.filename or '').suffix.lower()}"
                if await _save_upload(p, dest, budget) == 0:
                    raise HTTPException(400, f"empty photo upload ({p.filename})")
                photo_paths.append(dest)
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    job = Job(
        id=job_id,
        params=PipelineParams(
            iterations=iterations,
            strategy=strategy,
            max_frames=max_frames,
            resize_factor=resize_factor,
        ),
    )
    with _jobs_lock:
        _jobs[job_id] = job

    threading.Thread(target=_worker, args=(job, video_path, photo_paths), daemon=True).start()
    return job.to_dict()


@app.get("/jobs")
def list_jobs() -> list[dict]:
    with _jobs_lock:
        return [j.to_dict() for j in sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)]


def _get_job(job_id: str) -> Job:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    return _get_job(job_id).to_dict()


@app.get("/jobs/{job_id}/result")
def job_result(job_id: str) -> FileResponse:
    job = _get_job(job_id)
    if job.status == "failed":
        raise HTTPException(409, f"job failed: {job.error}")
    if job.status != "done" or job.ply_path is None:
        raise HTTPException(409, f"job not finished (status: {job.status})")
    return FileResponse(job.ply_path, media_type="application/octet-stream",
                        filename=f"{job_id}.ply")


@app.get("/jobs/{job_id}/log", response_class=PlainTextResponse)
def job_log(job_id: str) -> str:
    _get_job(job_id)
    log = _job_dir(job_id) / "pipeline.log"
    if not log.exists():
        raise HTTPException(404, "no log yet")
    return log.read_text(encoding="utf-8", errors="replace")


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    job = _get_job(job_id)
    if job.status not in {"done", "failed", "queued"}:
        raise HTTPException(409, "job is running; wait for it to finish")
    with _jobs_lock:
        _jobs.pop(job_id, None)
    shutil.rmtree(_job_dir(job_id), ignore_errors=True)
    return {"deleted": job_id}


@app.get("/health")
def health() -> dict:
    def tool_ok(cmd: list[str]) -> bool:
        try:
            return subprocess.run(cmd, capture_output=True, timeout=30).returncode == 0
        except Exception:
            return False

    return {
        "status": "ok",
        "ffmpeg": tool_ok([FFMPEG, "-version"]),
        "colmap": tool_ok([COLMAP, "--help"]),
        "lichtfeld": Path(LICHTFELD).exists(),
    }
