#!/usr/bin/env bash
# PLAN.md Phase 1: camera -> NVENC H.264 -> RTP/UDP to the Mac on port 5000.
# Usage: start_gstream_upload.sh [mac_ip]
set -euo pipefail

HOST="${1:-192.168.1.233}"

# -e: send EOS on Ctrl-C so nvarguscamerasrc releases the camera cleanly.
gst-launch-1.0 -e nvarguscamerasrc ! \
  'video/x-raw(memory:NVMM),width=1280,height=720,framerate=60/1,format=NV12' ! \
  nvv4l2h264enc maxperf-enable=1 insert-sps-pps=true idrinterval=30 bitrate=4000000 ! \
  h264parse ! rtph264pay config-interval=1 pt=96 ! \
  udpsink host="$HOST" port=5000 sync=false
