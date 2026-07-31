#!/usr/bin/env bash
# Start ArduCopter SITL for bench testing (requires ardupilot in PATH or adjust below).
set -euo pipefail

echo "Starting ArduCopter SITL on UDP 14550..."
echo "In another terminal:"
echo "  source .venv/bin/activate"
echo "  obj-drone --config config/sitl.yaml test"
echo "  obj-drone --config config/sitl.yaml run --skip-preflight"
echo ""

if command -v sim_vehicle.py &>/dev/null; then
  sim_vehicle.py -v ArduCopter -f quad --console --map --out=udp:127.0.0.1:14550
elif command -v arducopter &>/dev/null; then
  arducopter --model quad --speedup 1 --defaults "$HOME/ardupilot/Tools/autotest/default_params/copter.parm" \
    --sim-address=127.0.0.1 -I0
else
  echo "Install ArduPilot SITL: https://ardupilot.org/dev/docs/building-setup-linux.html"
  exit 1
fi
