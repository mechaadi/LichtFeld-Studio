#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# video2splat container entrypoint.
#   api (default)      serve the REST API on :8000
#   gui [args...]      launch the LichtFeld Studio GUI on the host's display
#                      (mount /tmp/.X11-unix + set DISPLAY - see README)
#   gui-web [args...]  launch the GUI on a virtual display, streamed to the
#                      browser via noVNC on :6080 (GUI_RESOLUTION=1920x1080)
#   anything else      exec'd verbatim (e.g. `sh` for debugging)
set -e
cmd="${1:-api}"
case "$cmd" in
    api)
        cd /app
        exec /opt/venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000
        ;;
    gui)
        shift
        exec /opt/lichtfeld/bin/run_lichtfeld.sh "$@"
        ;;
    gui-web)
        shift
        res="${GUI_RESOLUTION:-1920x1080}"
        Xvfb :99 -screen 0 "${res}x24" &
        i=0; while [ ! -e /tmp/.X11-unix/X99 ] && [ $i -lt 50 ]; do sleep 0.2; i=$((i+1)); done
        export DISPLAY=:99
        export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg}"
        mkdir -p "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR"
        x11vnc -display :99 -forever -shared -nopw -quiet -bg
        websockify --web /usr/share/novnc 6080 localhost:5900 &
        echo "noVNC ready: open http://localhost:6080/vnc.html"
        exec /opt/lichtfeld/bin/run_lichtfeld.sh "$@"
        ;;
    *)
        exec "$@"
        ;;
esac
