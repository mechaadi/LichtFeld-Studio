# SPDX-License-Identifier: GPL-3.0-or-later
"""Video/photos -> Gaussian Splat pipeline.

Stages:
  1. ffmpeg   - extract frames from the input video (or ingest a photo set)
  2. COLMAP   - feature extraction, matching (sequential for video frames,
                exhaustive for unordered photos), sparse reconstruction
  3. LichtFeld-Studio (headless) - train the splat and write the final .ply

Each stage runs as a subprocess; all stdout/stderr is appended to a per-job
log file so failures are diagnosable after the fact.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

def _default_lichtfeld_bin() -> str:
    # repo checkout layout: <repo>/tools/video2splat/pipeline.py -> <repo>/build/
    parents = Path(__file__).resolve().parents
    if len(parents) > 2:
        return str(parents[2] / "build" / "LichtFeld-Studio.exe")
    return "LichtFeld-Studio"


FFMPEG = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE_BIN", "ffprobe")
COLMAP = os.environ.get("COLMAP_BIN", "colmap")
LICHTFELD = os.environ.get("LICHTFELD_BIN") or _default_lichtfeld_bin()
COLMAP_USE_GPU = os.environ.get("COLMAP_USE_GPU", "1")


class PipelineError(RuntimeError):
    """A stage failed; the message points at the job log for details."""


@dataclass
class PipelineParams:
    iterations: int = 30000
    strategy: str = "mcmc"  # mcmc | mrnf | igs+
    max_frames: int = 300
    resize_factor: str = "auto"  # auto | 1 | 2 | 4 | 8
    output_name: str = "splat"
    extra_train_args: list[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    ply_path: Path
    num_frames: int
    num_registered_images: int


def _run(cmd: list[str], log_file: Path, cwd: Optional[Path] = None) -> None:
    with open(log_file, "a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n$ {' '.join(str(c) for c in cmd)}\n")
        log.flush()
        proc = subprocess.run(
            [str(c) for c in cmd],
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(cwd) if cwd else None,
        )
    if proc.returncode != 0:
        raise PipelineError(
            f"command failed (exit {proc.returncode}): {Path(str(cmd[0])).name} "
            f"- see log {log_file}"
        )


def _video_duration_seconds(video: Path) -> Optional[float]:
    try:
        out = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(video)],
            capture_output=True, text=True, timeout=60,
        )
        return float(json.loads(out.stdout)["format"]["duration"])
    except Exception:
        return None


def extract_frames(video: Path, images_dir: Path, max_frames: int, log_file: Path) -> int:
    """Extract up to max_frames evenly spaced frames as high-quality JPEGs."""
    images_dir.mkdir(parents=True, exist_ok=True)
    duration = _video_duration_seconds(video)
    if duration and duration > 0:
        fps = max(max_frames / duration, 0.01)
    else:
        fps = 2.0  # duration unknown; a sane default for handheld capture
    _run(
        [FFMPEG, "-y", "-i", video, "-vf", f"fps={fps:.6f}",
         "-frames:v", str(max_frames), "-qmin", "1", "-qscale:v", "1",
         str(images_dir / "frame_%05d.jpg")],
        log_file,
    )
    count = len(list(images_dir.glob("frame_*.jpg")))
    if count < 5:
        raise PipelineError(f"only {count} frames extracted - video too short or unreadable")
    return count


# formats COLMAP reads directly; anything else is converted via ffmpeg
PHOTO_DIRECT_SUFFIXES = {".jpg", ".jpeg", ".png"}


def ingest_photos(photos: list[Path], images_dir: Path, log_file: Path) -> int:
    """Copy a photo set into the dataset images dir, converting exotic formats."""
    images_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(sorted(photos), start=1):
        suffix = src.suffix.lower()
        if suffix in PHOTO_DIRECT_SUFFIXES:
            shutil.copy2(src, images_dir / f"photo_{i:05d}{suffix}")
        else:
            try:
                _run([FFMPEG, "-y", "-i", src, "-frames:v", "1",
                      "-qmin", "1", "-qscale:v", "1",
                      str(images_dir / f"photo_{i:05d}.jpg")], log_file)
            except PipelineError as e:
                raise PipelineError(
                    f"could not convert photo '{src.name}' ({suffix}) to JPEG - "
                    f"format not supported by this ffmpeg build: {e}"
                ) from e
    count = len(list(images_dir.iterdir()))
    if count < 5:
        raise PipelineError(f"only {count} photos - need at least 5 for reconstruction")
    return count


def run_colmap(work_dir: Path, log_file: Path, sequential: bool = True) -> int:
    """Sparse reconstruction into <work_dir>/sparse/0. Returns registered image count."""
    db = work_dir / "database.db"
    images_dir = work_dir / "images"
    sparse_dir = work_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    _run(
        [COLMAP, "feature_extractor",
         "--database_path", db, "--image_path", images_dir,
         "--ImageReader.camera_model", "OPENCV",
         "--ImageReader.single_camera", "1",
         "--SiftExtraction.use_gpu", COLMAP_USE_GPU],
        log_file,
    )
    # video frames are ordered, so sequential matching is faster and more
    # reliable; photo sets have no guaranteed order, so match exhaustively.
    # Wider overlap + guided matching help feature-poor scenes (studio
    # backdrops, indoor walls) hold a pose chain together
    if sequential:
        matcher_args = ["sequential_matcher", "--SequentialMatching.overlap", "20"]
    else:
        matcher_args = ["exhaustive_matcher"]
    _run(
        [COLMAP, *matcher_args,
         "--database_path", db,
         "--SiftMatching.guided_matching", "1",
         "--SiftMatching.use_gpu", COLMAP_USE_GPU],
        log_file,
    )
    _run(
        [COLMAP, "mapper",
         "--database_path", db, "--image_path", images_dir,
         "--output_path", sparse_dir],
        log_file,
    )

    model_dir = sparse_dir / "0"
    if not (model_dir / "cameras.bin").exists():
        raise PipelineError("COLMAP mapper produced no model - not enough overlap between frames")

    # if COLMAP split the scene into several models, keep only the largest in sparse/0
    models = sorted(d for d in sparse_dir.iterdir() if d.is_dir() and d.name.isdigit())
    if len(models) > 1:
        largest = max(models, key=lambda d: (d / "images.bin").stat().st_size)
        if largest.name != "0":
            shutil.rmtree(model_dir)
            largest.rename(model_dir)

    return _count_registered_images(model_dir, log_file)


def _count_registered_images(model_dir: Path, log_file: Path) -> int:
    try:
        out = subprocess.run(
            [COLMAP, "model_analyzer", "--path", str(model_dir)],
            capture_output=True, text=True, timeout=120,
        )
        for line in (out.stdout + out.stderr).splitlines():
            if "Registered images" in line:
                return int(line.split(":")[1].strip())
    except Exception:
        pass
    return -1


def train_splat(work_dir: Path, params: PipelineParams, log_file: Path) -> Path:
    output_dir = work_dir / "output"
    _run(
        [LICHTFELD, "--headless",
         "--data-path", work_dir,
         "--output-path", output_dir,
         "--output-name", params.output_name,
         "--iter", str(params.iterations),
         "--strategy", params.strategy,
         "--resize_factor", params.resize_factor,
         # COLMAP's OPENCV camera model has distortion params; LichtFeld refuses
         # to train on distorted cameras without this
         "--undistort",
         "--log-level", "info",
         *params.extra_train_args],
        log_file,
    )
    ply = output_dir / f"{params.output_name}.ply"
    if not ply.exists():
        raise PipelineError(f"training finished but {ply.name} was not written - see log {log_file}")
    return ply


def run_pipeline(
    work_dir: Path,
    params: PipelineParams,
    on_stage: Callable[[str], None] = lambda stage: None,
    video: Optional[Path] = None,
    photos: Optional[list[Path]] = None,
) -> PipelineResult:
    """Run the full video/photos -> ply pipeline inside work_dir."""
    if (video is None) == (photos is None):
        raise ValueError("provide exactly one of video or photos")
    work_dir.mkdir(parents=True, exist_ok=True)
    log_file = work_dir / "pipeline.log"

    on_stage("extracting_frames")
    if video is not None:
        num_frames = extract_frames(video, work_dir / "images", params.max_frames, log_file)
    else:
        num_frames = ingest_photos(photos, work_dir / "images", log_file)

    on_stage("reconstructing_poses")
    registered = run_colmap(work_dir, log_file, sequential=video is not None)
    if 0 <= registered < max(5, num_frames // 10):
        source = "frames" if video is not None else "photos"
        raise PipelineError(
            f"COLMAP registered only {registered} of {num_frames} {source} - too few for a "
            "usable splat. The capture likely has too little camera movement (parallax), too "
            "little overlap between shots, too little texture, or motion blur. Capture advice: "
            "move around the subject (orbit, don't just pan/rotate in place), overlap "
            "consecutive shots by ~70%, keep surfaces well-lit and textured."
        )

    on_stage("training")
    ply = train_splat(work_dir, params, log_file)

    return PipelineResult(ply_path=ply, num_frames=num_frames, num_registered_images=registered)
