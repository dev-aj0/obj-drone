#!/usr/bin/env bash
# Start ArduCopter SITL for bench testing (requires ArduPilot's sim_vehicle.py in PATH).
#
# SITL is the safest way to exercise the takeoff/track/land flow: it speaks real
# MAVLink so the whole companion stack runs unchanged, with no hardware at risk.
set -euo pipefail

cat <<'EOF'
Starting ArduCopter SITL on UDP 14550.

In another terminal:
  source .venv/bin/activate
  obj-drone --config config/sitl.yaml test           # link + mode list
  obj-drone --config config/sitl.yaml run --skip-preflight

The SITL profile uses camera_backend: usb and the colour tracker, so point a
webcam at something red — or set vision.detector.enabled to exercise the model.
EOF
echo ""

if command -v sim_vehicle.py &>/dev/null; then
  sim_vehicle.py -v ArduCopter -f quad --console --map --out=udp:127.0.0.1:14550
elif command -v arducopter &>/dev/null; then
  arducopter --model quad --speedup 1 \
    --defaults "$HOME/ardupilot/Tools/autotest/default_params/copter.parm" \
    --sim-address=127.0.0.1 -I0
else
  echo "Install ArduPilot SITL: https://ardupilot.org/dev/docs/building-setup-linux.html"
  exit 1
fi
