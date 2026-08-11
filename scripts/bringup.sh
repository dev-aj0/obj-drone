#!/usr/bin/env bash
# Non-destructive bring-up check for the Pi 5 + CSI camera + F405 stack.
#
# Run this ON THE PI, after scripts/install_pi.sh. It changes nothing: it only
# inspects the system and exercises the read-only commands, then writes a single
# log file you can send back for diagnosis.
#
#   bash scripts/bringup.sh
#
# Nothing here arms the vehicle or spins a motor.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$PROJECT_DIR/.venv"
OBJ="$VENV/bin/obj-drone"
LOG="$PROJECT_DIR/logs/bringup_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$PROJECT_DIR/logs"

# Everything from here goes to both the terminal and the log.
exec > >(tee "$LOG") 2>&1

FAILURES=0

section() { printf '\n\n=== %s ===\n' "$1"; }
pass()    { printf '  [ OK ] %s\n' "$1"; }
fail()    { printf '  [FAIL] %s\n' "$1"; FAILURES=$((FAILURES + 1)); }
warn()    { printf '  [warn] %s\n' "$1"; }

printf 'obj-drone bring-up check — %s\n' "$(date)"

# ---------------------------------------------------------------------- system
section "1. System"
echo "model:  $(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
echo "kernel: $(uname -srm)"
echo "os:     $(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME")"
echo "user:   $(whoami)"
echo "groups: $(id -nG)"

id -nG | grep -qw dialout && pass "user is in 'dialout' (serial access)" \
  || fail "user NOT in 'dialout' — run: sudo usermod -aG dialout $(whoami), then log out and back in"
id -nG | grep -qw video && pass "user is in 'video' (camera access)" \
  || warn "user not in 'video' — camera access may fail"

# ----------------------------------------------------------------------- UART
section "2. Serial / UART"
BOOT_CFG=/boot/firmware/config.txt
if [[ -f "$BOOT_CFG" ]]; then
  echo "--- relevant lines in $BOOT_CFG ---"
  grep -nE 'enable_uart|uart0|disable-bt|dtoverlay' "$BOOT_CFG" || echo "(none)"
  grep -qE '^\s*enable_uart=1' "$BOOT_CFG" && pass "enable_uart=1 set" \
    || fail "enable_uart=1 missing from $BOOT_CFG"
  if grep -q "Raspberry Pi 5" /proc/device-tree/model 2>/dev/null; then
    grep -qE '^\s*dtoverlay=uart0-pi5' "$BOOT_CFG" \
      && pass "dtoverlay=uart0-pi5 set (GPIO14/15 UART on Pi 5)" \
      || fail "dtoverlay=uart0-pi5 missing — on a Pi 5 GPIO14/15 has no UART without it"
  fi
else
  warn "$BOOT_CFG not found"
fi

echo "--- serial devices ---"
ls -l /dev/ttyAMA* /dev/serial* /dev/ttyS* 2>/dev/null || echo "(none found)"

DEV=$(grep -E '^\s*connection:' "$PROJECT_DIR/config/default.yaml" | head -1 | sed 's/.*"\(.*\)".*/\1/')
echo "configured device: $DEV"
if [[ -e "$DEV" ]]; then
  pass "$DEV exists"
  [[ -r "$DEV" && -w "$DEV" ]] && pass "$DEV is readable/writable by $(whoami)" \
    || fail "$DEV exists but is not accessible — check the 'dialout' group"
else
  fail "$DEV does not exist — check the UART overlay and reboot"
fi

echo "--- serial console (must NOT be active on the MAVLink port) ---"
if systemctl is-active --quiet serial-getty@ttyAMA0 2>/dev/null; then
  fail "serial-getty@ttyAMA0 is ACTIVE and will fight MAVLink for the port"
  echo "       fix: sudo systemctl disable --now serial-getty@ttyAMA0"
else
  pass "no serial console on ttyAMA0"
fi
grep -qE 'console=(serial0|ttyAMA0)' /boot/firmware/cmdline.txt 2>/dev/null \
  && fail "cmdline.txt still puts a console on the serial port" \
  || pass "cmdline.txt has no serial console"

# --------------------------------------------------------------------- python
section "3. Python environment"
if [[ ! -x "$OBJ" ]]; then
  fail "$OBJ not found — run scripts/install_pi.sh first"
  echo; echo "Log written to $LOG"; exit 1
fi
pass "obj-drone installed"

if grep -q 'include-system-site-packages = true' "$VENV/pyvenv.cfg" 2>/dev/null; then
  pass "venv has --system-site-packages (required for picamera2)"
else
  fail "venv is ISOLATED — picamera2/libcamera will not import."
  echo "       fix: rm -rf .venv && bash scripts/install_pi.sh"
fi

echo "--- imports ---"
"$VENV/bin/python" - <<'PY'
for mod, hint in [
    ("serial",    "pip install pyserial"),
    ("yaml",      "pip install pyyaml"),
    ("numpy",     "sudo apt install python3-numpy"),
    ("cv2",       "pip install --no-deps 'opencv-python-headless>=4.10,<5'"),
    ("picamera2", "sudo apt install python3-picamera2 + venv with --system-site-packages"),
    ("pymavlink", "pip install pymavlink"),
]:
    try:
        m = __import__(mod)
        print(f"  [ OK ] {mod:<12} {getattr(m, '__version__', '')}")
    except Exception as exc:
        print(f"  [FAIL] {mod:<12} {exc}\n         -> {hint}")
PY

# --------------------------------------------------------------------- camera
section "4. Camera (CSI)"
if command -v rpicam-hello &>/dev/null; then
  rpicam-hello --list-cameras 2>&1 | head -20
elif command -v libcamera-hello &>/dev/null; then
  libcamera-hello --list-cameras 2>&1 | head -20
else
  warn "rpicam-hello not installed — skipping libcamera probe"
fi
CAM=$(rpicam-hello --list-cameras 2>/dev/null | grep -oE 'imx[0-9]+|ov[0-9]+' | head -1)
if [[ -n "$CAM" ]]; then
  pass "camera detected by libcamera: $CAM"
else
  fail "no camera detected — check the ribbon cable seating and orientation"
fi

# ------------------------------------------------------------------ obj-drone
section "5. MAVLink link  (obj-drone test)"
echo "Sends nothing that moves the aircraft — reads heartbeat and telemetry only."
if "$OBJ" test; then
  pass "MAVLink link OK"
else
  fail "MAVLink link failed — see the error above"
fi

section "6. Camera + detector  (obj-drone detect)"
if [[ -f "$PROJECT_DIR/models/yolov8n.onnx" ]]; then
  echo "Runs the camera and model for 15s, writing annotated frames."
  if "$OBJ" detect --seconds 15 --save; then
    pass "detector ran"
    echo "  annotated frames: $PROJECT_DIR/logs/frames/"
  else
    fail "detector failed — see the error above"
  fi
else
  warn "models/yolov8n.onnx not present — run scripts/fetch_model.sh"
  echo "  Falling back to the colour tracker to at least prove the camera works:"
  "$OBJ" calibrate-color --seconds 10 --save && pass "camera delivered frames" \
    || fail "camera check failed"
fi

# -------------------------------------------------------------------- summary
section "Summary"
if [[ "$FAILURES" -eq 0 ]]; then
  echo "All checks passed."
  echo
  echo "Next, with PROPS REMOVED:"
  echo "  $OBJ hover        # GUIDED + zero-velocity, still disarmed"
  echo "  $OBJ run          # full mission"
else
  echo "$FAILURES check(s) failed — see [FAIL] lines above."
fi
echo
echo "Full log: $LOG"
echo "Send that file back for diagnosis."
