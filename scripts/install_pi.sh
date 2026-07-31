#!/usr/bin/env bash
# Install obj-drone on Raspberry Pi 5 with Camera Module V2 and UART to F405.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
USER_NAME="${SUDO_USER:-$USER}"

echo "==> Installing system packages"
sudo apt-get update
sudo apt-get install -y \
  python3 python3-venv python3-pip \
  libcap-dev libatlas-base-dev \
  libopencv-dev python3-opencv \
  git

echo "==> Enabling UART (requires reboot if changed)"
BOOT_CFG="/boot/firmware/config.txt"
if [[ -f "$BOOT_CFG" ]]; then
  grep -q '^enable_uart=1' "$BOOT_CFG" || echo 'enable_uart=1' | sudo tee -a "$BOOT_CFG"
  grep -q 'disable-bt' "$BOOT_CFG" || echo 'dtoverlay=disable-bt' | sudo tee -a "$BOOT_CFG"
else
  echo "WARNING: $BOOT_CFG not found — enable UART manually"
fi

echo "==> Adding $USER_NAME to dialout group"
sudo usermod -aG dialout "$USER_NAME"

echo "==> Creating Python virtual environment"
cd "$PROJECT_DIR"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[pi]"

echo "==> Installing systemd service"
sudo cp "$PROJECT_DIR/deploy/obj-drone.service" /etc/systemd/system/
sudo sed -i "s|__PROJECT_DIR__|$PROJECT_DIR|g" /etc/systemd/system/obj-drone.service
sudo sed -i "s|__USER__|$USER_NAME|g" /etc/systemd/system/obj-drone.service
sudo systemctl daemon-reload

echo ""
echo "Installation complete."
echo "  1. Wire Pi UART (pin 8 TX, pin 10 RX, pin 6 GND) to F405 TELEM port"
echo "  2. Set ArduPilot SERIALx_PROTOCOL=2, SERIALx_BAUD=57 on the TELEM port"
echo "  3. Reboot if UART was just enabled: sudo reboot"
echo "  4. Test link:  $PROJECT_DIR/.venv/bin/obj-drone test"
echo "  5. Calibrate:   $PROJECT_DIR/.venv/bin/obj-drone calibrate-color"
echo "  6. Run mission: $PROJECT_DIR/.venv/bin/obj-drone run"
echo ""
echo "Optional — enable on boot:"
echo "  sudo systemctl enable --now obj-drone"
