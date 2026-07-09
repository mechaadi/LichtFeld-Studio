# syntax=docker/dockerfile:1
# SPDX-License-Identifier: GPL-3.0-or-later
#
# video2splat: video -> frames (ffmpeg) -> poses (COLMAP, CUDA) -> splat (LichtFeld-Studio) -> .ply
#
# Build from the REPO ROOT (submodules must be initialized on the host,
# i.e. clone with --recursive or run `git submodule update --init --recursive`):
#
#   docker build -f tools/video2splat/Dockerfile -t video2splat .
#
# Run (needs the NVIDIA container toolkit, driver >= 570):
#
#   docker run --gpus all -p 8000:8000 -v splat-jobs:/data/jobs video2splat
#
# Build recipe mirrors .github/workflows/ubuntu.yml (the Linux CI build).
# BUILD_PORTABLE=ON compiles PTX-only for SM >= 75 (Turing and newer), so the
# image runs on any reasonably modern GPU regardless of the build machine.

ARG CUDA_IMAGE_TAG=12.8.0
ARG COLMAP_VERSION=3.11.1
# real SM targets for COLMAP (LichtFeld handles portability itself via PTX)
ARG COLMAP_CUDA_ARCHS=75;80;86;89;90;120
# parallelism caps for RAM-constrained build hosts (0 = use all cores)
ARG VCPKG_MAX_CONCURRENCY=0
ARG BUILD_JOBS=0

########################################################################
# Stage 1: build LichtFeld-Studio (portable, self-contained dist)
########################################################################
FROM nvidia/cuda:${CUDA_IMAGE_TAG}-devel-ubuntu24.04 AS lichtfeld-build

ENV DEBIAN_FRONTEND=noninteractive \
    VCPKG_ROOT=/opt/vcpkg \
    VCPKG_FORCE_SYSTEM_BINARIES=1 \
    VCPKG_DEFAULT_BINARY_CACHE=/vcpkg-cache \
    CC=gcc-14 CXX=g++-14

# toolchain + the X11/Wayland dev stack (vcpkg builds SDL3/vulkan-loader against it)
RUN apt-get update && apt-get install -y ca-certificates gpg wget && \
    wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc \
      | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg && \
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ noble main' \
      > /etc/apt/sources.list.d/kitware.list && \
    apt-get update && apt-get install -y \
      git curl unzip cmake gcc-14 g++-14 ninja-build zip tar pkg-config \
      python3 python3-dev \
      libxinerama-dev libxcursor-dev xorg-dev libglu1-mesa-dev \
      libwayland-dev libxkbcommon-dev libegl-dev libdecor-0-dev \
      libibus-1.0-dev libdbus-1-dev libsystemd-dev libgtk-3-dev \
      nasm autoconf autoconf-archive automake libtool && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 100 \
                        --slave /usr/bin/g++ g++ /usr/bin/g++-14 && \
    rm -rf /var/lib/apt/lists/*

# full clone: manifest-mode baseline resolution needs the vcpkg git history.
# Pin per-port build parallelism in the triplet. 0 => resolve to nproc: inside
# cgroup-limited containers std::thread::hardware_concurrency() can report 0,
# which recent vcpkg treats as a hard error ("VCPKG_MAX_CONCURRENCY is 0").
ARG VCPKG_MAX_CONCURRENCY
RUN git clone https://github.com/microsoft/vcpkg.git "$VCPKG_ROOT" && \
    "$VCPKG_ROOT/bootstrap-vcpkg.sh" -disableMetrics && \
    printf 'set(VCPKG_BUILD_TYPE release)\n' >> "$VCPKG_ROOT/triplets/x64-linux.cmake" && \
    CONC="$VCPKG_MAX_CONCURRENCY"; [ "$CONC" = "0" ] && CONC="$(nproc 2>/dev/null || echo 1)"; \
    case "$CONC" in ''|*[!0-9]*) CONC=1;; esac; \
    printf 'set(VCPKG_MAX_CONCURRENCY %s)\n' "$CONC" >> "$VCPKG_ROOT/triplets/x64-linux.cmake"

WORKDIR /src

# warm the vcpkg binary cache from the manifest alone, so source-only changes
# don't rebuild every dependency (.github/overlays = overlay ports referenced
# by vcpkg-configuration.json)
COPY vcpkg.json vcpkg-configuration.json ./
COPY .github/overlays .github/overlays
# vcpkg-seed: source tarballs pre-downloaded on the host for servers the build
# container can't reach (e.g. code.videolan.org TLS fails from this network);
# vcpkg picks them up from its downloads dir after SHA512 verification
COPY tools/video2splat/vcpkg-seed/ /vcpkg-seed/
# cache mounts survive failed builds, so a flaky download (videolan, gnu.org...)
# only costs a retry, not a from-scratch dependency rebuild
RUN --mount=type=cache,target=/vcpkg-cache \
    --mount=type=cache,target=/opt/vcpkg/downloads \
    cp -n /vcpkg-seed/* "$VCPKG_ROOT/downloads/" 2>/dev/null; \
    CONC="$VCPKG_MAX_CONCURRENCY"; [ "$CONC" = "0" ] && CONC="$(nproc 2>/dev/null || echo 1)"; \
    case "$CONC" in ''|*[!0-9]*) CONC=1;; esac; \
    export VCPKG_MAX_CONCURRENCY="$CONC"; \
    ok=0; \
    for i in 1 2 3; do \
      if "$VCPKG_ROOT/vcpkg" install --triplet x64-linux; then ok=1; break; fi; \
      echo "vcpkg install attempt $i failed; retrying in 15s"; sleep 15; \
    done; \
    [ "$ok" = "1" ]

COPY --exclude=tools/video2splat --exclude=docker . .
RUN if [ -d .git ]; then \
      git config --global --add safe.directory '*' && \
      git submodule update --init --recursive; \
    fi

# LFS_ENFORCE_LINUX_GUI_BACKENDS=OFF: skip the display-backend preflight; the
# container only ever runs with --headless
ARG BUILD_JOBS
ARG VCPKG_MAX_CONCURRENCY
RUN --mount=type=cache,target=/vcpkg-cache \
    --mount=type=cache,target=/opt/vcpkg/downloads \
    JOBS="$BUILD_JOBS"; [ "$JOBS" = "0" ] && JOBS="$(nproc)"; \
    CONC="$VCPKG_MAX_CONCURRENCY"; [ "$CONC" = "0" ] && CONC="$(nproc 2>/dev/null || echo 1)"; \
    case "$CONC" in ''|*[!0-9]*) CONC=1;; esac; \
    export VCPKG_MAX_CONCURRENCY="$CONC"; \
    cmake -B build -S . -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DBUILD_PORTABLE=ON \
      -DBUILD_PYTHON_STUBS=OFF \
      -DLFS_DEV_IMPORT_SOURCE_PYTHON=OFF \
      -DLFS_DEV_IMPORT_SOURCE_RESOURCES=OFF \
      -DCUDA_DEVICE_DEBUG=OFF \
      -DLFS_ENFORCE_LINUX_GUI_BACKENDS=OFF && \
    cmake --build build -j"$JOBS" && \
    cmake --install build --prefix /opt/lichtfeld

########################################################################
# Stage 2: build COLMAP with CUDA (the apt package is CPU-only)
########################################################################
FROM nvidia/cuda:${CUDA_IMAGE_TAG}-devel-ubuntu24.04 AS colmap-build

ARG COLMAP_VERSION
ARG COLMAP_CUDA_ARCHS
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
      git cmake ninja-build build-essential \
      libboost-program-options-dev libboost-graph-dev libboost-system-dev \
      libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev \
      libgoogle-glog-dev libsqlite3-dev libglew-dev libceres-dev \
      libcurl4-openssl-dev libcgal-dev && \
    rm -rf /var/lib/apt/lists/*

ARG BUILD_JOBS
RUN JOBS="$BUILD_JOBS"; [ "$JOBS" = "0" ] && JOBS="$(nproc)"; \
    git clone --depth 1 --branch "${COLMAP_VERSION}" https://github.com/colmap/colmap.git /colmap && \
    cmake -S /colmap -B /colmap/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DGUI_ENABLED=OFF \
      -DTESTS_ENABLED=OFF \
      -DCUDA_ENABLED=ON \
      -DCMAKE_CUDA_ARCHITECTURES="${COLMAP_CUDA_ARCHS}" \
      -DCMAKE_INSTALL_PREFIX=/opt/colmap && \
    cmake --build /colmap/build -j"$JOBS" && \
    cmake --install /colmap/build

########################################################################
# Stage 3: runtime — ffmpeg + COLMAP + LichtFeld + the FastAPI service
########################################################################
FROM nvidia/cuda:${CUDA_IMAGE_TAG}-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# ffmpeg + python + COLMAP's shared libs (-dev metapackages: stable names on
# noble, unlike the t64-suffixed runtime packages) + X/GL runtime libs that the
# LichtFeld binary links even when running --headless. Vulkan loader + Mesa
# ICDs enable the optional GUI mode (native NVIDIA ICD is injected by the
# container toolkit on Linux hosts; on WSLg Mesa's dozen/lavapipe are the
# fallback).
# Max-FutureTime: tolerate repo metadata "from the future" - hosts with a
# skewed clock (e.g. WSL VMs inheriting a wrong Windows clock) otherwise fail
# apt-get update with "Release file ... is not valid yet"
RUN echo 'Acquire::Max-FutureTime "25200";' > /etc/apt/apt.conf.d/99clock-skew && \
    apt-get update && apt-get install -y \
      ffmpeg python3 python3-venv curl \
      libboost-program-options-dev libboost-graph-dev libboost-system-dev \
      libflann-dev libfreeimage-dev libmetis-dev libgoogle-glog-dev \
      libglew-dev libceres-dev libcurl4-openssl-dev libsqlite3-dev libcgal-dev \
      libgomp1 \
      libx11-6 libxinerama1 libxcursor1 libxrandr2 libxi6 \
      libglu1-mesa libegl1 libwayland-client0 libxkbcommon0 libdecor-0-0 \
      libdbus-1-3 libgtk-3-0 libibus-1.0-5 \
      libvulkan1 mesa-vulkan-drivers vulkan-tools \
      xvfb x11vnc novnc websockify && \
    rm -rf /var/lib/apt/lists/* /etc/apt/apt.conf.d/99clock-skew

COPY --from=lichtfeld-build /opt/lichtfeld /opt/lichtfeld
COPY --from=colmap-build /opt/colmap /opt/colmap

WORKDIR /app
COPY tools/video2splat/requirements.txt .
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt
COPY tools/video2splat/pipeline.py tools/video2splat/api.py ./
COPY tools/video2splat/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

ENV PATH=/opt/venv/bin:/opt/colmap/bin:$PATH \
    LICHTFELD_BIN=/opt/lichtfeld/bin/run_lichtfeld.sh \
    COLMAP_BIN=/opt/colmap/bin/colmap \
    COLMAP_USE_GPU=1 \
    VIDEO2SPLAT_WORK_DIR=/data/jobs \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

VOLUME /data/jobs
EXPOSE 8000 6080
# healthy = API answering (api mode) or the GUI process alive (gui mode)
HEALTHCHECK --interval=60s --timeout=30s --start-period=30s \
  CMD sh -c 'curl -sf http://localhost:8000/health || pgrep -f LichtFeld-Studio >/dev/null'

# single worker only: job state lives in process memory
ENTRYPOINT ["/entrypoint.sh"]
CMD ["api"]
