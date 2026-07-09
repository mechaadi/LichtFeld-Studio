# video2splat — RunPod serverless worker

A [RunPod serverless](https://docs.runpod.io/serverless/overview) worker that
takes **a video or an image sequence and returns a `.ply`**. It's a thin image
layered on top of the full video2splat image
(`fitloopnet/lichtfeld-studio:0.1`) — the base already contains LichtFeld,
COLMAP-CUDA, ffmpeg, the Python venv, and `pipeline.py`, so this
[Dockerfile](Dockerfile) only adds the RunPod SDK and
[runpod-handler.py](runpod-handler.py). It builds in seconds instead of
recompiling the GPU toolchain, and reuses the identical
ffmpeg → COLMAP → LichtFeld pipeline as the [REST API](../README.md).

## Build & push

This folder is self-contained — build with it as the context (the last argument).
Keep it on one line (the `\` line-continuation below is bash; **on Windows
PowerShell put it all on a single line** or use a backtick `` ` `` to continue):

```sh
docker build -f tools/video2splat/runpod/Dockerfile \
  -t fitloopnet/video2splat-runpod:0.1 tools/video2splat/runpod
docker push fitloopnet/video2splat-runpod:0.1
```

Override the base with `--build-arg BASE_IMAGE=<image:tag>` if you retag it.
The handler imports the base image's `pipeline.py`; if you change the pipeline,
rebuild the base image (`tools/video2splat/Dockerfile`) and bump the tag.

## Deploy on RunPod

Create a **Serverless Endpoint** from the pushed image on a GPU with ≥ 16 GB
VRAM. No start-command override is needed — the image runs the handler directly.

For download URLs (recommended, since a `.ply` is typically tens of MB), set an
S3-compatible bucket in the endpoint environment:

```
BUCKET_ENDPOINT_URL, BUCKET_ACCESS_KEY_ID, BUCKET_SECRET_ACCESS_KEY
```

Without a bucket, results under `VIDEO2SPLAT_MAX_INLINE_MB` (default 10) come
back base64-encoded; larger ones return an error asking you to configure a bucket.

## Input

Send the job payload under `input`, providing **exactly one** input source:

| field           | notes                                                       |
|-----------------|-------------------------------------------------------------|
| `video_url`     | single video, downloaded over HTTP(S)                       |
| `video_base64`  | single video, base64 (a `data:` URI prefix is accepted)     |
| `video`         | either of the above (auto-detected)                         |
| `images`        | ordered list; each item an HTTP(S) URL **or** base64 string |
| `image_urls`    | ordered list of HTTP(S) URLs                                |
| `images_base64` | ordered list of base64 strings                              |
| `iterations`    | training iterations (default 30000)                         |
| `strategy`      | `mcmc` (default), `mrnf`, or `igs+`                         |
| `max_frames`    | frames extracted evenly across the video (video only, 300)  |
| `resize_factor` | image downscale: `auto` (default), `1`, `2`, `4`, `8`       |
| `output_name`   | base name of the `.ply` (default `splat`)                   |

## Output

`{ "ply_url": ... }` when a bucket is configured, otherwise
`{ "ply_base64": ... }`, plus `num_frames`, `num_registered_images`, and
`filename`. Failures return `{ "error": ..., "log": ... }` (a tail of the
pipeline log). Pipeline stages are reported via RunPod progress updates
(`extracting_frames` → `reconstructing_poses` → `training`).

## Call it

```sh
# video from a URL (blocking)
curl -X POST https://api.runpod.ai/v2/<endpoint-id>/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"input": {"video_url": "https://example.com/capture.mp4",
                 "iterations": 30000, "strategy": "mcmc", "max_frames": 300}}'

# image sequence from URLs (5+ images; async, poll /status/<id>)
curl -X POST https://api.runpod.ai/v2/<endpoint-id>/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"input": {"image_urls": ["https://.../001.jpg", "https://.../002.jpg", "..."]}}'
```

Use `/run` (async) for long jobs; `/runsync` blocks and is subject to RunPod's
response-size limits, so pair it with a bucket for anything but tiny results.

## Environment variables

| variable                     | default | notes                                    |
|------------------------------|---------|------------------------------------------|
| `BUCKET_ENDPOINT_URL` + `BUCKET_ACCESS_KEY_ID` + `BUCKET_SECRET_ACCESS_KEY` | — | S3-compatible bucket for result URLs (recommended) |
| `VIDEO2SPLAT_MAX_INLINE_MB`  | `10`    | max `.ply` size returned inline as base64 when no bucket |
| `VIDEO2SPLAT_KEEP_WORK`      | `0`     | `1` keeps the per-job work dir on disk (debugging) |
| `VIDEO2SPLAT_WORK_DIR`       | `/data/jobs` | scratch dir for frames/poses/output (from the base image) |
