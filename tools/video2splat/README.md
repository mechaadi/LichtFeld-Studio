# video2splat

REST API that turns a video **or a set of photos** into a trained gaussian splat (`.ply`):

```text
video ──ffmpeg──▶ frames ─┐
                          ├─COLMAP──▶ camera poses ──LichtFeld-Studio──▶ splat.ply
photos ──────────────────┘
```

Video frames are matched sequentially (ordered); photo sets are matched
exhaustively (order-independent).

## Prerequisites

- `ffmpeg` / `ffprobe` on PATH (or set `FFMPEG_BIN` / `FFPROBE_BIN`)
- `colmap` on PATH (or set `COLMAP_BIN`) — CUDA build recommended
- A built `LichtFeld-Studio` binary (defaults to `../../build/LichtFeld-Studio.exe`,
  override with `LICHTFELD_BIN`)
- Python 3.10+

## Run

```sh
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

## Docker (all-in-one)

[Dockerfile](Dockerfile) builds everything into a single image: LichtFeld-Studio
(portable build, runs on any SM ≥ 75 GPU), COLMAP with CUDA, ffmpeg, and the API.
Build from the **repo root** with submodules initialized:

```sh
git submodule update --init --recursive
docker build -f tools/video2splat/Dockerfile -t video2splat .
```

The first build compiles LichtFeld's vcpkg dependencies and COLMAP — expect it
to take a while and want ~16 GB RAM. Run with the NVIDIA container toolkit
(host driver ≥ 570):

```sh
docker run --gpus all -p 8000:8000 -v splat-jobs:/data/jobs video2splat
```

Then use the API as below (`curl http://localhost:8000/health` should report
all three tools available). Job data lives in the `/data/jobs` volume.
Image knobs (`--build-arg`): `COLMAP_VERSION` (default 3.11.1),
`COLMAP_CUDA_ARCHS` (default `75;80;86;89;90;120`), `CUDA_IMAGE_TAG` (default 12.8.0).

### GUI in the browser (gui-web)

`gui-web` runs the full LichtFeld GUI on a virtual display inside the container
and streams it to your browser via noVNC — no display setup needed:

```sh
docker run --rm --gpus all -p 6080:6080 -v splat-jobs:/data/jobs \
  video2splat gui-web /data/jobs/<job-id>/output/splat.ply
# then open http://localhost:6080/vnc.html
```

Resolution via `-e GUI_RESOLUTION=2560x1440` (default 1920x1080).

**Requires a native Linux host with the NVIDIA container toolkit** (which
injects the NVIDIA Vulkan driver): LichtFeld's renderer needs CUDA↔Vulkan
external-semaphore interop, which only the native NVIDIA ICD provides.
**This does not work under Docker Desktop on Windows/WSL** — containers there
get no NVIDIA Vulkan driver, and Mesa's llvmpipe fails LichtFeld's interop
check ("Vulkan external timeline-semaphore interop is required"). On Windows,
view results with the native Windows LichtFeld build instead: download the
PLY (`GET /jobs/<id>/result`) and open it with `build\LichtFeld-Studio.exe`.

### GUI mode (native display)

The container's first argument selects the mode: `api` (default), `gui`, or `gui-web`.
`gui` launches the full LichtFeld Studio interface; pass a `.ply` to open the
viewer directly — handy for inspecting a finished job from the shared volume:

```sh
# Windows / Docker Desktop (WSLg provides the display):
docker run --rm --gpus all \
  -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=:0 \
  -v splat-jobs:/data/jobs \
  video2splat gui /data/jobs/<job-id>/output/splat.ply

# Linux host with X11:
docker run --rm --gpus all \
  -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY \
  -v splat-jobs:/data/jobs \
  video2splat gui
```

Notes:

- On a **Linux host**, the NVIDIA container toolkit injects the native Vulkan
  driver (`NVIDIA_DRIVER_CAPABILITIES` already includes `graphics`) — full-speed GUI.
- On **Windows/WSLg** there is no native NVIDIA Vulkan driver inside containers;
  the image ships Mesa's Vulkan fallbacks (dozen/llvmpipe). Treat GUI-in-container
  on Windows as experimental — expect reduced rendering performance. The Windows
  build of LichtFeld Studio is the better viewer on this machine.
- Any other first argument is exec'd verbatim (e.g. `docker run -it video2splat sh`).

## RunPod (serverless)

A RunPod serverless worker — send a video **or** an image sequence, get a `.ply`
back — lives in [runpod/](runpod/). It builds as a thin layer on top of this
image and reuses the identical ffmpeg → COLMAP → LichtFeld pipeline. Build,
deploy, and the input/output schema are documented in
[runpod/README.md](runpod/README.md).

## Usage

```sh
# start a job from a video (returns {"id": "...", "status": "queued", ...})
curl -F "video=@capture.mp4" -F "iterations=30000" -F "strategy=mcmc" \
     -F "max_frames=300" http://localhost:8000/jobs

# ...or from a photo set (5-500 images; jpg/png/tiff/bmp/webp/heic)
curl -F "photos=@img_001.jpg" -F "photos=@img_002.jpg" -F "photos=@img_003.jpg" \
     ... http://localhost:8000/jobs
# tip: in PowerShell build the args from a folder:
#   $f = (ls .\shots\*.jpg | % { "-F", "photos=@$($_.FullName)" }); curl.exe @f http://localhost:8000/jobs

# poll status: queued -> extracting_frames -> reconstructing_poses -> training -> done
curl http://localhost:8000/jobs/<id>

# download the splat when done
curl -o splat.ply http://localhost:8000/jobs/<id>/result

# inspect the full ffmpeg/colmap/training log (useful on failure)
curl http://localhost:8000/jobs/<id>/log

# clean up
curl -X DELETE http://localhost:8000/jobs/<id>
```

### Job parameters (multipart form fields)

| field           | default | notes                                                      |
|-----------------|---------|------------------------------------------------------------|
| `video`         | —       | mp4/mov/avi/mkv/webm/m4v (exactly one of `video`/`photos`) |
| `photos`        | —       | repeatable field, 5–500 images                             |
| `iterations`    | 30000   | training iterations                                        |
| `strategy`      | mcmc    | `mcmc`, `mrnf`, or `igs+`                                  |
| `max_frames`    | 300     | frames extracted evenly across the video (video only)      |
| `resize_factor` | auto    | image downscale: `auto`, `1`, `2`, `4`, `8`                |

### Environment variables

| variable                     | default                               |
|------------------------------|---------------------------------------|
| `LICHTFELD_BIN`              | `../../build/LichtFeld-Studio.exe`    |
| `COLMAP_BIN`                 | `colmap`                              |
| `FFMPEG_BIN` / `FFPROBE_BIN` | `ffmpeg` / `ffprobe`                  |
| `COLMAP_USE_GPU`             | `1`                                   |
| `VIDEO2SPLAT_WORK_DIR`       | `./jobs`                              |
| `VIDEO2SPLAT_CONCURRENCY`    | `1` (COLMAP + training are GPU-heavy) |
| `VIDEO2SPLAT_MAX_UPLOAD_MB`  | `2048`                                |
