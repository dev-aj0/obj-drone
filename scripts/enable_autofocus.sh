#!/usr/bin/env bash
# Enable autofocus on a CSI camera whose Raspberry Pi OS tuning file has no
# autofocus algorithm.
#
# Raspberry Pi OS ships tuning files with an "rpi.af" block only for a few
# sensors (imx708, ov64a40). Modules like the Arducam IMX519 advertise AfMode
# and LensPosition, but libcamera logs
#
#     WARN IPARPI: Could not set AF_MODE - no AF algorithm
#
# and the lens never moves — neither autofocus nor manual focus works.
#
# This copies the installed tuning file, injects a contrast-detection (CDAF)
# autofocus block, and writes it to config/tuning/. Nothing system-wide is
# modified, so there is nothing to undo: point vision.tuning_file at the result,
# or delete it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$PROJECT_DIR/config/tuning"

SENSOR="${1:-}"
if [[ -z "$SENSOR" ]]; then
  SENSOR=$(rpicam-hello --list-cameras 2>/dev/null | grep -oE 'imx[0-9]+|ov[0-9a-z]+' | head -1 || true)
fi
if [[ -z "$SENSOR" ]]; then
  echo "Could not detect a camera. Pass the sensor name, e.g.: $0 imx519" >&2
  exit 1
fi

# Pi 5 uses the pisp pipeline; Pi 4 and earlier use vc4.
for variant in pisp vc4; do
  CANDIDATE="/usr/share/libcamera/ipa/rpi/$variant/$SENSOR.json"
  [[ -f "$CANDIDATE" ]] && SRC="$CANDIDATE" && break
done
if [[ -z "${SRC:-}" ]]; then
  echo "No tuning file found for '$SENSOR' under /usr/share/libcamera/ipa/rpi/" >&2
  exit 1
fi

echo "==> Sensor:      $SENSOR"
echo "==> Source:      $SRC"
mkdir -p "$OUT_DIR"

python3 - "$SRC" "$OUT_DIR/${SENSOR}_af.json" <<'PY'
import json, sys

src, dst = sys.argv[1], sys.argv[2]
d = json.load(open(src))

names = [list(a)[0] for a in d["algorithms"]]
if "rpi.af" in names:
    print("    Source tuning already has autofocus — copying unchanged.")
else:
    print("    Injecting rpi.af (contrast-detection autofocus).")

d["algorithms"] = [a for a in d["algorithms"] if "rpi.af" not in a]

# PDAF (phase-detect) is disabled: on the IMX519 it needs Arducam's camhelper,
# which Raspberry Pi OS does not ship. CDAF works with the stock stack.
speed = {
    "step_coarse": 1.0,
    "step_fine": 0.25,
    "contrast_ratio": 0.75,
    "retrigger_ratio": 0.75,
    "retrigger_delay": 10,
    "pdaf_gain": -0.02,
    "pdaf_squelch": 0.125,
    "max_slew": 2.0,
    "pdaf_frames": 0,
    "dropout_frames": 0,
    "step_frames": 4,
}
d["algorithms"].append({
    "rpi.af": {
        "ranges": {
            "normal": {"min": 0.0, "max": 12.0, "default": 1.0},
            "macro": {"min": 3.0, "max": 15.0, "default": 4.0},
        },
        "speeds": {
            "normal": speed,
            "fast": {**speed, "step_coarse": 1.25, "step_fine": 0.0},
        },
        "conf_epsilon": 8,
        "conf_thresh": 16,
        "conf_clip": 512,
        "skip_frames": 5,
        "check_for_ir": False,
        # Dioptres -> VCM DAC codes. Uncalibrated for third-party modules, so
        # reported distances are approximate; contrast AF still finds focus.
        "map": [0.0, 445, 15.0, 925],
    }
})
with open(dst, "w") as f:
    json.dump(d, f, indent=4)
print(f"    Wrote {dst}")
PY

cat <<EOF

Done. Point the config at it:

  vision:
    tuning_file: "config/tuning/${SENSOR}_af.json"

Then find the sharpest lens position, aimed at something DETAILED
(text, a keyboard, a patterned surface — never a blank wall):

  obj-drone focus-sweep

To remove: delete $OUT_DIR/${SENSOR}_af.json and clear vision.tuning_file.
EOF
