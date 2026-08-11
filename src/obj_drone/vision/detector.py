"""Neural-network object detection via OpenCV's DNN module.

Uses cv2.dnn so the same code path runs on the Raspberry Pi 5 CPU and on a dev
machine with no extra runtime dependency. Supports ONNX models with either
YOLO-style outputs (v5/v8/v11) or SSD-style detection outputs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class DetectorError(RuntimeError):
    """Model could not be loaded or its output could not be decoded."""


@dataclass(frozen=True)
class Detection:
    """A detected object in original-frame pixel coordinates."""

    class_id: int
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x, y, w, h

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return x + w / 2.0, y + h / 2.0

    @property
    def area(self) -> int:
        return self.bbox[2] * self.bbox[3]


def load_labels(path: str | Path | None) -> list[str]:
    """Read a newline-delimited class-name file (e.g. coco.names)."""
    if path is None:
        return []
    p = Path(path)
    if not p.is_file():
        raise DetectorError(f"Label file not found: {p}")
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def letterbox(
    frame: np.ndarray, size: int
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize preserving aspect ratio and pad to a square.

    Returns (image, scale, (pad_x, pad_y)) so boxes can be mapped back.
    """
    h, w = frame.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), 114, dtype=frame.dtype)
    pad_x = (size - new_w) / 2.0
    pad_y = (size - new_h) / 2.0
    top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas, scale, (float(left), float(top))


class ObjectDetector:
    """Run an ONNX object-detection model over camera frames."""

    def __init__(
        self,
        model_path: str | Path,
        model_type: str = "yolo",
        input_size: int = 320,
        confidence: float = 0.5,
        nms_threshold: float = 0.45,
        labels: list[str] | None = None,
        target_classes: list[str] | None = None,
        num_threads: int = 4,
    ) -> None:
        self.model_path = Path(model_path)
        self.model_type = model_type.lower()
        self.input_size = int(input_size)
        self.confidence = float(confidence)
        self.nms_threshold = float(nms_threshold)
        self.labels = labels or []
        self.num_threads = int(num_threads)

        if self.model_type not in ("yolo", "ssd"):
            raise DetectorError(
                f"Unknown detector type {model_type!r} (expected 'yolo' or 'ssd')"
            )
        if not self.model_path.is_file():
            raise DetectorError(
                f"Model file not found: {self.model_path}. "
                "Run scripts/fetch_model.sh to download a default model."
            )

        # Resolve target class names to ids once, so the hot loop compares ints.
        self.target_ids: set[int] | None = None
        if target_classes:
            if not self.labels:
                raise DetectorError(
                    "target_classes is set but no label file was provided"
                )
            unknown = [c for c in target_classes if c not in self.labels]
            if unknown:
                raise DetectorError(
                    f"target_classes not present in the label file: {unknown}"
                )
            self.target_ids = {self.labels.index(c) for c in target_classes}

        try:
            self._net = cv2.dnn.readNet(str(self.model_path))
        except cv2.error as exc:
            raise DetectorError(f"Could not load model {self.model_path}: {exc}") from exc

        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        cv2.setNumThreads(self.num_threads)

        self._last_inference_ms = 0.0
        logger.info(
            "Detector loaded: %s (type=%s size=%d conf=%.2f threads=%d)",
            self.model_path.name,
            self.model_type,
            self.input_size,
            self.confidence,
            self.num_threads,
        )

    @property
    def last_inference_ms(self) -> float:
        return self._last_inference_ms

    def _label_for(self, class_id: int) -> str:
        if 0 <= class_id < len(self.labels):
            return self.labels[class_id]
        return str(class_id)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return detections sorted by descending confidence."""
        start = time.monotonic()
        if self.model_type == "yolo":
            detections = self._detect_yolo(frame)
        else:
            detections = self._detect_ssd(frame)
        self._last_inference_ms = (time.monotonic() - start) * 1000.0

        if self.target_ids is not None:
            detections = [d for d in detections if d.class_id in self.target_ids]
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    # ------------------------------------------------------------------ YOLO
    def _detect_yolo(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        img, scale, (pad_x, pad_y) = letterbox(frame, self.input_size)
        blob = cv2.dnn.blobFromImage(
            img, 1 / 255.0, (self.input_size, self.input_size), swapRB=True, crop=False
        )
        self._net.setInput(blob)
        out = self._net.forward()

        # Strip leading batch dimensions only. np.squeeze would also collapse the
        # detection axis when the model emits a single box.
        pred = out
        while pred.ndim > 2 and pred.shape[0] == 1:
            pred = pred[0]
        if pred.ndim != 2:
            raise DetectorError(f"Unexpected YOLO output shape {out.shape}")

        # v8/v11 emit (4+nc, N); v5 emits (N, 5+nc). Orient to rows=detections.
        # Comparing the two axis lengths breaks when N is small, so prefer
        # matching against the attribute count implied by the label file.
        n_labels = len(self.labels)
        expected = {4 + n_labels, 5 + n_labels} if n_labels else set()
        if expected:
            if pred.shape[0] in expected and pred.shape[1] not in expected:
                pred = pred.T
        elif pred.shape[0] < pred.shape[1]:
            pred = pred.T

        cols = pred.shape[1]
        if n_labels and cols == 4 + n_labels:
            has_objectness = False
        elif n_labels and cols == 5 + n_labels:
            has_objectness = True
        else:
            # No labels to disambiguate: 85 columns is the classic v5 COCO head.
            has_objectness = cols == 85

        if has_objectness:
            class_scores = pred[:, 5:] * pred[:, 4:5]
        else:
            class_scores = pred[:, 4:]

        if class_scores.size == 0:
            return []

        class_ids = class_scores.argmax(axis=1)
        confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]
        keep = confidences >= self.confidence
        if not np.any(keep):
            return []

        boxes_cxcywh = pred[keep, :4]
        class_ids = class_ids[keep]
        confidences = confidences[keep]

        # Undo letterbox: strip padding, then divide by the resize scale.
        cx = (boxes_cxcywh[:, 0] - pad_x) / scale
        cy = (boxes_cxcywh[:, 1] - pad_y) / scale
        bw = boxes_cxcywh[:, 2] / scale
        bh = boxes_cxcywh[:, 3] / scale
        x = cx - bw / 2.0
        y = cy - bh / 2.0

        rects = np.stack([x, y, bw, bh], axis=1)
        return self._nms_to_detections(rects, confidences, class_ids, w, h)

    # ------------------------------------------------------------------- SSD
    def _detect_ssd(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame,
            1 / 127.5,
            (self.input_size, self.input_size),
            (127.5, 127.5, 127.5),
            swapRB=True,
            crop=False,
        )
        self._net.setInput(blob)
        out = np.squeeze(self._net.forward())
        if out.ndim == 1:
            out = out[np.newaxis, :]
        if out.ndim != 2 or out.shape[1] < 7:
            raise DetectorError(f"Unexpected SSD output shape {out.shape}")

        confidences = out[:, 2]
        keep = confidences >= self.confidence
        if not np.any(keep):
            return []

        rows = out[keep]
        # SSD emits normalised corner coordinates.
        x1 = rows[:, 3] * w
        y1 = rows[:, 4] * h
        x2 = rows[:, 5] * w
        y2 = rows[:, 6] * h
        rects = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
        return self._nms_to_detections(
            rects, rows[:, 2], rows[:, 1].astype(int), w, h
        )

    # ------------------------------------------------------------------- util
    def _nms_to_detections(
        self,
        rects: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray,
        frame_w: int,
        frame_h: int,
    ) -> list[Detection]:
        boxes = [[float(v) for v in row] for row in rects]
        scores = [float(c) for c in confidences]
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence, self.nms_threshold)
        if len(indices) == 0:
            return []

        detections: list[Detection] = []
        for i in np.array(indices).flatten():
            i = int(i)
            x, y, bw, bh = rects[i]
            # Clip to the frame so downstream centre maths stays in-image.
            x1 = max(0, min(frame_w - 1, int(round(x))))
            y1 = max(0, min(frame_h - 1, int(round(y))))
            x2 = max(0, min(frame_w, int(round(x + bw))))
            y2 = max(0, min(frame_h, int(round(y + bh))))
            if x2 <= x1 or y2 <= y1:
                continue
            cid = int(class_ids[i])
            detections.append(
                Detection(
                    class_id=cid,
                    label=self._label_for(cid),
                    confidence=float(confidences[i]),
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                )
            )
        return detections
