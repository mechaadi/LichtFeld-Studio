# syntax=docker/dockerfile:1
# SPDX-License-Identifier: GPL-3.0-or-later
#
# RunPod serverless worker for video2splat, built as a thin layer on top of the
# already-published video2splat image (LichtFeld + COLMAP-CUDA + ffmpeg + the
# Python venv, with pipeline.py already at /app). The base does all the heavy
# GPU-toolchain building; this stage only adds the RunPod SDK and the handler,
# so it builds in seconds.
#
# Build with this folder as the context (it is self-contained). Single line -
# on Windows PowerShell a bash-style "\" line break splits the command:
#
#   docker build -f tools/video2splat/runpod/Dockerfile -t fitloopnet/video2splat-runpod:0.1 tools/video2splat/runpod
#
# Then create a RunPod Serverless Endpoint from the pushed image (GPU with
# >= 16 GB VRAM). No start-command override is needed - this image runs the
# handler directly. For result URLs, set BUCKET_ENDPOINT_URL /
# BUCKET_ACCESS_KEY_ID / BUCKET_SECRET_ACCESS_KEY in the endpoint env.

ARG BASE_IMAGE=fitloopnet/lichtfeld-studio:0.1
FROM ${BASE_IMAGE}

# The base venv (from the 0.1 tag) predates the RunPod dependency.
COPY requirements.txt ./
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Add the handler. WORKDIR /app, pipeline.py, and all the tool env vars
# (LICHTFELD_BIN, COLMAP_BIN, COLMAP_USE_GPU, VIDEO2SPLAT_WORK_DIR) already come
# from the base image - the handler imports the base's pipeline.py.
WORKDIR /app
COPY runpod-handler.py ./

# RunPod just launches the container's start process; go straight to the
# serverless worker (the base 0.1 entrypoint.sh has no "runpod" mode).
ENTRYPOINT []
CMD ["/opt/venv/bin/python", "/app/runpod-handler.py"]
