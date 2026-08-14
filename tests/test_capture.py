"""Detection capture store: rate limiting, listing, download bundling."""

from __future__ import annotations

import zipfile
from io import BytesIO

import numpy as np
import pytest

from obj_drone.capture import CaptureStore
from obj_drone.vision.detector import Detection


def _frame(w: int = 320, h: int = 240) -> np.ndarray:
    return np.full((h, w, 3), 60, dtype=np.uint8)


def _det(label: str = "bottle", conf: float = 0.8) -> Detection:
    return Detection(class_id=39, label=label, confidence=conf, bbox=(10, 10, 40, 90))


def _store(tmp_path, **kw) -> CaptureStore:
    opts = dict(min_interval_s=0.0, min_confidence=0.45)
    opts.update(kw)
    return CaptureStore(output_dir=tmp_path / "captures", **opts)


def test_saves_a_frame_for_a_confident_detection(tmp_path) -> None:
    store = _store(tmp_path)
    capture = store.maybe_capture(_frame(), [_det()], now=0.0)
    assert capture is not None
    assert capture.path.is_file()
    assert capture.path.read_bytes().startswith(b"\xff\xd8")  # JPEG
    assert capture.labels == ("bottle",)
    assert store.count == 1


def test_nothing_saved_without_detections(tmp_path) -> None:
    store = _store(tmp_path)
    assert store.maybe_capture(_frame(), [], now=0.0) is None
    assert store.count == 0


def test_low_confidence_is_skipped(tmp_path) -> None:
    store = _store(tmp_path, min_confidence=0.6)
    assert store.maybe_capture(_frame(), [_det(conf=0.5)], now=0.0) is None
    assert store.count == 0


def test_rate_limit_prevents_a_flood(tmp_path) -> None:
    """At 15 fps an unlimited store would write 900 near-identical frames a minute."""
    store = _store(tmp_path, min_interval_s=2.0)
    assert store.maybe_capture(_frame(), [_det()], now=100.0) is not None
    # Half a second later — still inside the window.
    assert store.maybe_capture(_frame(), [_det()], now=100.5) is None
    assert store.maybe_capture(_frame(), [_det()], now=101.9) is None
    # Past the interval.
    assert store.maybe_capture(_frame(), [_det()], now=102.1) is not None
    assert store.count == 2


def test_max_captures_is_enforced(tmp_path) -> None:
    store = _store(tmp_path, max_captures=3)
    for i in range(10):
        store.maybe_capture(_frame(), [_det()], now=float(i))
    assert store.count == 3


def test_disabled_store_writes_nothing(tmp_path) -> None:
    store = CaptureStore(output_dir=tmp_path / "off", enabled=False, min_interval_s=0.0)
    assert store.maybe_capture(_frame(), [_det()], now=0.0) is None
    assert store.count == 0


def test_records_every_distinct_label(tmp_path) -> None:
    store = _store(tmp_path)
    capture = store.maybe_capture(
        _frame(), [_det("bottle"), _det("cup"), _det("bottle")], now=0.0
    )
    assert capture.labels == ("bottle", "cup")
    assert capture.count == 3


def test_best_confidence_is_the_highest(tmp_path) -> None:
    store = _store(tmp_path)
    capture = store.maybe_capture(
        _frame(), [_det(conf=0.55), _det(conf=0.91), _det(conf=0.62)], now=0.0
    )
    assert capture.best_confidence == pytest.approx(0.91)


def test_list_is_newest_first(tmp_path) -> None:
    store = _store(tmp_path)
    store.maybe_capture(_frame(), [_det("bottle")], now=0.0)
    store.maybe_capture(_frame(), [_det("cup")], now=1.0)
    assert store.list()[0].labels == ("cup",)
    assert store.list(newest_first=False)[0].labels == ("bottle",)


def test_get_by_name_and_unknown_name(tmp_path) -> None:
    store = _store(tmp_path)
    capture = store.maybe_capture(_frame(), [_det()], now=0.0)
    assert store.get(capture.name) is capture
    assert store.get("nope.jpg") is None


def test_get_rejects_path_traversal(tmp_path) -> None:
    """Names are matched against recorded captures, never used as a path."""
    store = _store(tmp_path)
    store.maybe_capture(_frame(), [_det()], now=0.0)
    for evil in ("../../etc/passwd", "/etc/passwd", "..%2F..%2Fsecret"):
        assert store.get(evil) is None


def test_clear_deletes_files(tmp_path) -> None:
    store = _store(tmp_path)
    a = store.maybe_capture(_frame(), [_det()], now=0.0)
    b = store.maybe_capture(_frame(), [_det()], now=1.0)
    assert store.clear() == 2
    assert store.count == 0
    assert not a.path.exists() and not b.path.exists()


def test_clear_resets_the_rate_limiter(tmp_path) -> None:
    store = _store(tmp_path, min_interval_s=5.0)
    store.maybe_capture(_frame(), [_det()], now=100.0)
    store.clear()
    assert store.maybe_capture(_frame(), [_det()], now=100.1) is not None


def test_zip_contains_every_capture(tmp_path) -> None:
    store = _store(tmp_path)
    names = []
    for i in range(3):
        names.append(store.maybe_capture(_frame(), [_det()], now=float(i)).name)
    with zipfile.ZipFile(BytesIO(store.as_zip())) as archive:
        assert sorted(archive.namelist()) == sorted(names)
        assert archive.read(names[0]).startswith(b"\xff\xd8")


def test_zip_of_empty_store_is_valid(tmp_path) -> None:
    with zipfile.ZipFile(BytesIO(_store(tmp_path).as_zip())) as archive:
        assert archive.namelist() == []


def test_as_dict_is_json_safe(tmp_path) -> None:
    import json

    store = _store(tmp_path)
    capture = store.maybe_capture(_frame(), [_det()], now=0.0)
    payload = json.loads(json.dumps(capture.as_dict()))
    assert payload["url"] == f"/captures/{capture.name}"
    assert payload["labels"] == ["bottle"]
