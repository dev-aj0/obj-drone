#!/usr/bin/env bash
# Produce models/yolov8n.onnx for the detector.
#
# Exporting needs PyTorch, which is heavy — do this once on your Mac/PC and copy
# the resulting .onnx to the Pi, or run it on the Pi if you don't mind the wait.
# The export toolchain is installed into its own throwaway venv so it never
# pollutes the runtime environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_DIR="$PROJECT_DIR/models"
IMGSZ="${IMGSZ:-320}"
MODEL="${MODEL:-yolov8n}"
EXPORT_VENV="${EXPORT_VENV:-/tmp/obj-drone-export}"

mkdir -p "$MODEL_DIR"

if [[ -f "$MODEL_DIR/$MODEL.onnx" ]]; then
  echo "$MODEL_DIR/$MODEL.onnx already exists — delete it to re-export."
  exit 0
fi

echo "==> Creating export environment at $EXPORT_VENV"
python3 -m venv "$EXPORT_VENV"
"$EXPORT_VENV/bin/pip" install --upgrade pip
"$EXPORT_VENV/bin/pip" install ultralytics onnx

echo "==> Exporting $MODEL to ONNX at ${IMGSZ}x${IMGSZ}"
cd "$MODEL_DIR"
"$EXPORT_VENV/bin/python" - <<PY
from ultralytics import YOLO
model = YOLO("${MODEL}.pt")
# opset 12 keeps the graph within what OpenCV's DNN importer supports.
path = model.export(format="onnx", imgsz=${IMGSZ}, opset=12, simplify=True)
print("exported:", path)
PY

echo ""
echo "Done. $MODEL_DIR/$MODEL.onnx"
echo ""
echo "Make sure config/default.yaml matches:"
echo "  vision.detector.model:      models/$MODEL.onnx"
echo "  vision.detector.input_size: $IMGSZ"
echo ""
echo "Verify it end-to-end on the Pi (no flight, props off):"
echo "  obj-drone detect --seconds 20 --save"
echo ""
echo "You can delete the export venv now:  rm -rf $EXPORT_VENV"
