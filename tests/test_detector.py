"""Detector geometry: letterboxing and output decoding back to frame pixels."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_frame
from obj_drone.vision.detector import (
    Detection,
    DetectorError,
    ObjectDetector,
    letterbox,
    load_labels,
)


def test_letterbox_preserves_aspect_and_pads() -> None:
    frame = make_frame(640, 480)
    img, scale, (pad_x, pad_y) = letterbox(frame, 320)

    assert img.shape == (320, 320, 3)
    assert scale == pytest.approx(0.5)
    # 640x480 * 0.5 -> 320x240, so 40 px of padding top and bottom, none sideways.
    assert pad_x == pytest.approx(0.0)
    assert pad_y == pytest.approx(40.0)


def test_letterbox_square_input_has_no_padding() -> None:
    _, scale, (pad_x, pad_y) = letterbox(make_frame(480, 480), 320)
    assert scale == pytest.approx(320 / 480)
    assert (pad_x, pad_y) == (0.0, 0.0)


def test_load_labels_missing_file_raises(tmp_path) -> None:
    with pytest.raises(DetectorError, match="Label file not found"):
        load_labels(tmp_path / "nope.names")


def test_load_labels_strips_blanks(tmp_path) -> None:
    p = tmp_path / "labels.names"
    p.write_text("person\n\n  car  \ndog\n")
    assert load_labels(p) == ["person", "car", "dog"]


def test_missing_model_raises(tmp_path) -> None:
    with pytest.raises(DetectorError, match="Model file not found"):
        ObjectDetector(model_path=tmp_path / "absent.onnx")


def test_bad_model_type_raises(tmp_path) -> None:
    model = tmp_path / "m.onnx"
    model.write_bytes(b"")
    with pytest.raises(DetectorError, match="Unknown detector type"):
        ObjectDetector(model_path=model, model_type="magic")


class _StubNet:
    """Replaces cv2.dnn's Net so decoding can be tested without a real model."""

    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.blob = None

    def setPreferableBackend(self, _b): ...
    def setPreferableTarget(self, _t): ...
    def setInput(self, blob): self.blob = blob
    def forward(self): return self.output


def _detector_with(monkeypatch, tmp_path, output: np.ndarray, **kwargs) -> ObjectDetector:
    model = tmp_path / "m.onnx"
    model.write_bytes(b"stub")
    monkeypatch.setattr(
        "cv2.dnn.readNet", lambda _p: _StubNet(output), raising=True
    )
    return ObjectDetector(model_path=model, **kwargs)


def _yolov8_output(cx, cy, w, h, scores, num_boxes=1) -> np.ndarray:
    """Build a (1, 4+nc, N) YOLOv8-style tensor with one real box."""
    nc = len(scores)
    out = np.zeros((1, 4 + nc, num_boxes), dtype=np.float32)
    out[0, 0, 0] = cx
    out[0, 1, 0] = cy
    out[0, 2, 0] = w
    out[0, 3, 0] = h
    for i, s in enumerate(scores):
        out[0, 4 + i, 0] = s
    return out


def test_yolo_box_maps_back_to_frame_pixels(monkeypatch, tmp_path) -> None:
    """A box at the centre of the letterboxed image is the centre of the frame."""
    labels = ["person", "car"]
    # 640x480 -> scale 0.5, pad_y 40. Frame centre (320,240) lands at (160,160).
    out = _yolov8_output(cx=160, cy=160, w=40, h=40, scores=[0.9, 0.1])
    det = _detector_with(
        monkeypatch, tmp_path, out, input_size=320, labels=labels, confidence=0.5
    )

    results = det.detect(make_frame(640, 480))
    assert len(results) == 1
    r = results[0]
    assert r.label == "person"
    assert r.confidence == pytest.approx(0.9, abs=1e-5)
    assert r.center[0] == pytest.approx(320, abs=2)
    assert r.center[1] == pytest.approx(240, abs=2)
    # 40 px in a 0.5-scaled image is 80 px in the original.
    assert r.bbox[2] == pytest.approx(80, abs=2)


def test_yolo_below_confidence_is_dropped(monkeypatch, tmp_path) -> None:
    out = _yolov8_output(cx=160, cy=160, w=40, h=40, scores=[0.2, 0.1])
    det = _detector_with(
        monkeypatch, tmp_path, out, input_size=320, labels=["person", "car"], confidence=0.5
    )
    assert det.detect(make_frame()) == []


def test_target_class_filter(monkeypatch, tmp_path) -> None:
    out = _yolov8_output(cx=160, cy=160, w=40, h=40, scores=[0.1, 0.95])
    det = _detector_with(
        monkeypatch,
        tmp_path,
        out,
        input_size=320,
        labels=["person", "car"],
        confidence=0.5,
        target_classes=["person"],
    )
    # The only detection is a car, which is filtered out.
    assert det.detect(make_frame()) == []


def test_unknown_target_class_raises(monkeypatch, tmp_path) -> None:
    out = _yolov8_output(cx=10, cy=10, w=4, h=4, scores=[0.9, 0.1])
    with pytest.raises(DetectorError, match="not present in the label file"):
        _detector_with(
            monkeypatch,
            tmp_path,
            out,
            labels=["person", "car"],
            target_classes=["helicopter"],
        )


def test_target_classes_without_labels_raises(monkeypatch, tmp_path) -> None:
    out = _yolov8_output(cx=10, cy=10, w=4, h=4, scores=[0.9])
    with pytest.raises(DetectorError, match="no label file"):
        _detector_with(monkeypatch, tmp_path, out, target_classes=["person"])


def test_boxes_are_clipped_to_frame(monkeypatch, tmp_path) -> None:
    """A box hanging off the edge must not produce out-of-frame centres."""
    out = _yolov8_output(cx=5, cy=160, w=60, h=40, scores=[0.9])
    det = _detector_with(
        monkeypatch, tmp_path, out, input_size=320, labels=["person"], confidence=0.5
    )
    results = det.detect(make_frame(640, 480))
    assert len(results) == 1
    x, y, w, h = results[0].bbox
    assert x >= 0 and y >= 0
    assert x + w <= 640 and y + h <= 480


def test_detection_center_and_area() -> None:
    d = Detection(class_id=0, label="person", confidence=0.8, bbox=(10, 20, 30, 40))
    assert d.center == (25.0, 40.0)
    assert d.area == 1200


def test_inference_time_is_recorded(monkeypatch, tmp_path) -> None:
    out = _yolov8_output(cx=160, cy=160, w=40, h=40, scores=[0.9])
    det = _detector_with(monkeypatch, tmp_path, out, input_size=320, labels=["person"])
    det.detect(make_frame())
    assert det.last_inference_ms >= 0.0
