#!/usr/bin/env bash
# Install obj-drone on a Raspberry Pi 5 with a CSI camera (IMX219/IMX519/etc)
# and a UART MAVLink link to a CoreWing F405 Wing V2.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
USER_NAME="${SUDO_USER:-$USER}"
BOOT_CFG="/boot/firmware/config.txt"
BOOT_CMDLINE="/boot/firmware/cmdline.txt"
REBOOT_REQUIRED=0

# ---------------------------------------------------------------- board model
PI_MODEL="$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
echo "==> Detected board: $PI_MODEL"
case "$PI_MODEL" in
  *"Raspberry Pi 5"*) IS_PI5=1 ;;
  *)                  IS_PI5=0 ;;
esac

# ------------------------------------------------------------ system packages
# picamera2 and libcamera have no usable PyPI wheels — they must come from apt,
# and the venv below is created with --system-site-packages so it can see them.
# numpy also comes from apt: apt-built extensions such as simplejpeg (which
# picamera2 imports) are compiled against numpy 1.x and segfault on numpy 2.
#
# OpenCV deliberately does NOT come from apt. Debian Bookworm ships 4.6.0, whose
# DNN importer cannot parse a YOLOv8 ONNX graph ("Assertion failed ... in
# function 'total'"). It is installed from pip below instead.
echo "==> Installing system packages"
sudo apt-get update
sudo apt-get install -y \
  python3 python3-venv python3-pip \
  python3-picamera2 python3-libcamera python3-kms++ \
  python3-numpy \
  libcap-dev git

# ------------------------------------------------------------------ UART setup
echo "==> Configuring UART"
if [[ ! -f "$BOOT_CFG" ]]; then
  echo "WARNING: $BOOT_CFG not found — configure UART manually" >&2
else
  add_cfg() {
    if ! grep -qE "^\s*$1\s*$" "$BOOT_CFG"; then
      echo "$1" | sudo tee -a "$BOOT_CFG" >/dev/null
      echo "    added to config.txt: $1"
      REBOOT_REQUIRED=1
    else
      echo "    already set: $1"
    fi
  }

  add_cfg "enable_uart=1"
  if [[ "$IS_PI5" == "1" ]]; then
    # Pi 5: /dev/ttyAMA0 is the dedicated debug UART connector by default.
    # This overlay puts a real UART on GPIO14/15 (header pins 8 and 10).
    # 'dtoverlay=disable-bt' is Pi 0-4 guidance and does NOT apply here.
    add_cfg "dtoverlay=uart0-pi5"
  else
    add_cfg "dtoverlay=disable-bt"
  fi
fi

# A login console on the same port will fight with MAVLink for the bytes.
echo "==> Disabling serial login console"
for svc in serial-getty@ttyAMA0 serial-getty@ttyS0 serial-getty@serial0; do
  if systemctl list-unit-files | grep -q "^${svc}.service"; then
    sudo systemctl disable --now "${svc}.service" 2>/dev/null || true
  fi
done
if [[ -f "$BOOT_CMDLINE" ]] && grep -qE 'console=(serial0|ttyAMA0|ttyS0)' "$BOOT_CMDLINE"; then
  sudo cp "$BOOT_CMDLINE" "${BOOT_CMDLINE}.objdrone.bak"
  sudo sed -i -E 's/console=(serial0|ttyAMA0|ttyS0),[0-9]+ ?//g' "$BOOT_CMDLINE"
  echo "    removed serial console from cmdline.txt (backup: ${BOOT_CMDLINE}.objdrone.bak)"
  REBOOT_REQUIRED=1
fi

echo "==> Adding $USER_NAME to dialout and video groups"
sudo usermod -aG dialout,video "$USER_NAME"

# --------------------------------------------------------------- python venv
# --system-site-packages is REQUIRED: picamera2/libcamera/numpy live in the
# system site-packages and cannot be pip-installed into an isolated venv.
echo "==> Creating Python virtual environment (--system-site-packages)"
cd "$PROJECT_DIR"
if [[ -d .venv ]] && ! grep -q 'include-system-site-packages = true' .venv/pyvenv.cfg 2>/dev/null; then
  echo "    existing .venv is isolated and cannot see picamera2 — recreating"
  rm -rf .venv
fi
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .

# YOLOv8 ONNX needs OpenCV >= 4.7; Bookworm's apt build is 4.6.0.
# --no-deps is essential: without it pip drags in numpy 2.x, which breaks the
# apt-built simplejpeg that picamera2 imports. The system numpy 1.24 is used
# instead, and OpenCV 4.14 runs fine against it.
echo "==> Installing OpenCV from pip (newer than apt, without touching numpy)"
pip install --no-deps "opencv-python-headless>=4.10,<5"

echo "==> Verifying imports"
python - <<'PY'
import sys
ok = True
for mod, hint in [
    ("serial",     "pyserial — needed by pymavlink for /dev/tty* links"),
    ("yaml",       "pyyaml"),
    ("numpy",      "numpy"),
    ("cv2",        "pip install --no-deps 'opencv-python-headless>=4.10,<5'"),
    ("picamera2",  "sudo apt install python3-picamera2 (and recreate venv with --system-site-packages)"),
]:
    try:
        __import__(mod)
        print(f"    OK   {mod}")
    except Exception as exc:
        ok = False
        print(f"    FAIL {mod}: {exc}\n         -> {hint}")
if not ok:
    sys.exit(1)
PY

echo "==> Installing systemd service"
sudo cp "$PROJECT_DIR/deploy/obj-drone.service" /etc/systemd/system/
sudo sed -i "s|__PROJECT_DIR__|$PROJECT_DIR|g" /etc/systemd/system/obj-drone.service
sudo sed -i "s|__USER__|$USER_NAME|g" /etc/systemd/system/obj-drone.service
sudo systemctl daemon-reload

cat <<EOF

Installation complete.

  1. Wire the Pi UART to an F405 TELEM port:
       Pi pin 8  (GPIO14 TX) -> F405 TELEM RX
       Pi pin 10 (GPIO15 RX) -> F405 TELEM TX
       Pi pin 39 (GND)       -> F405 GND
     Do NOT connect Pi 5V to the flight controller.

  2. On the F405 set, for that TELEM port:
       SERIALx_PROTOCOL = 2   (MAVLink2)
       SERIALx_BAUD     = 57  (57600, must match config/default.yaml)

  3. Serial device on this board: $( [[ "$IS_PI5" == "1" ]] && echo "/dev/ttyAMA0  (Pi 5 — already the default in config/default.yaml)" || echo "/dev/serial0" )
EOF

if [[ "$REBOOT_REQUIRED" == "1" ]]; then
  echo "  4. REBOOT REQUIRED (boot config changed):  sudo reboot"
else
  echo "  4. No reboot needed."
fi

cat <<EOF
  5. Log out and back in so the dialout/video groups apply.
  6. Check the link:   $PROJECT_DIR/.venv/bin/obj-drone test
  7. Check the camera: $PROJECT_DIR/.venv/bin/obj-drone calibrate-color --save
  8. Fly:              $PROJECT_DIR/.venv/bin/obj-drone run

Optional — start on boot:
  sudo systemctl enable --now obj-drone
EOF
