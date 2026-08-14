"""Live viewer: capture loop, status payload, and HTTP endpoints."""

from __future__ import annotations

import json
import threading
import time
import urllib.request

import numpy as np
import pytest

from conftest import FakeCamera, draw_bgr_box, make_frame
from obj_drone.vision.debug import annotate_frame
from obj_drone.vision.detector import Detection
from obj_drone.vision.tracker import NOT_FOUND, TargetTracker, TrackingResult
from obj_drone.web import VisionStream, make_handler

RED_BGR = (0, 0, 255)


class LoopingCamera:
    """Always returns the same frame, so the stream never starves."""

    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.stopped = False

    def start(self) -> None: ...
    def stop(self) -> None:
        self.stopped = True

    def read(self):
        time.sleep(0.005)
        return self.frame.copy()


def _stream(frame, source="color"):
    tracker = TargetTracker(640, 480)
    return VisionStream(LoopingCamera(frame), tracker, source=source, jpeg_quality=60)


def _wait_running(stream: VisionStream, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if stream.status().get("frames", 0) > 2:
            return True
        time.sleep(0.02)
    return False


# ------------------------------------------------------------------- annotation
def test_annotate_marks_no_target() -> None:
    frame = make_frame()
    out = annotate_frame(frame, NOT_FOUND)
    assert out.shape == frame.shape
    # Something red was drawn for the NO TARGET caption.
    assert out[:60, :300, 2].max() > 200
    # Original frame untouched.
    assert frame.max() == 0


def test_annotate_draws_target_box() -> None:
    result = TrackingResult(
        found=True, center_x=320, center_y=240, bbox=(300, 220, 40, 40),
        label="person", confidence=0.9,
    )
    out = annotate_frame(make_frame(), result, pixel_error=(0.0, 0.0))
    assert out[:, :, 1].max() > 200  # green box/caption


def test_annotate_draws_other_detections_faintly() -> None:
    locked = TrackingResult(
        found=True, center_x=100, center_y=100, bbox=(80, 80, 40, 40),
        label="person", confidence=0.9,
    )
    others = [
        Detection(0, "person", 0.9, (80, 80, 40, 40)),
        Detection(2, "car", 0.6, (400, 300, 60, 60)),
    ]
    out = annotate_frame(make_frame(), locked, (0.0, 0.0), others)
    # The unlocked 'car' region got grey annotation.
    region = out[295:365, 395:465]
    assert region.max() > 100


# ----------------------------------------------------------------------- stream
def test_stream_publishes_frames_and_status() -> None:
    stream = _stream(draw_bgr_box(make_frame(), RED_BGR, 300, 220, 60, 60))
    stream.start()
    try:
        assert _wait_running(stream)
        seq, jpeg, status = stream.wait_for_frame(-1, timeout=3.0)
        assert seq > 0
        assert jpeg is not None and jpeg.startswith(b"\xff\xd8")  # JPEG SOI
        assert status["running"] is True
        assert status["found"] is True
        assert status["source"] == "color"
        assert status["resolution"] == [640, 480]
    finally:
        stream.stop()


def test_stream_reports_no_target_on_empty_frame() -> None:
    stream = _stream(make_frame())
    stream.start()
    try:
        assert _wait_running(stream)
        status = stream.status()
        assert status["found"] is False
        assert status["center"] is None
        assert status["pixel_error"] is None
    finally:
        stream.stop()


def test_stream_status_is_json_serialisable() -> None:
    """The HTTP layer serialises this directly — numpy scalars would break it."""
    stream = _stream(draw_bgr_box(make_frame(), RED_BGR, 300, 220, 60, 60))
    stream.start()
    try:
        assert _wait_running(stream)
        json.dumps(stream.status())
    finally:
        stream.stop()


def test_stream_tracks_hit_rate_and_fps() -> None:
    stream = _stream(draw_bgr_box(make_frame(), RED_BGR, 300, 220, 60, 60))
    stream.start()
    try:
        assert _wait_running(stream)
        time.sleep(0.3)
        status = stream.status()
        assert status["hit_rate"] == pytest.approx(1.0)
        assert status["fps"] > 0
    finally:
        stream.stop()


def test_stop_marks_stream_not_running() -> None:
    stream = _stream(make_frame())
    stream.start()
    assert _wait_running(stream)
    stream.stop()
    assert stream.status()["running"] is False


def test_wait_for_frame_times_out_when_idle() -> None:
    stream = _stream(make_frame())  # never started
    start = time.monotonic()
    seq, jpeg, _ = stream.wait_for_frame(0, timeout=0.2)
    assert time.monotonic() - start < 1.5
    assert jpeg is None


# ------------------------------------------------------------------------- HTTP
@pytest.fixture
def server():
    from http.server import ThreadingHTTPServer

    stream = _stream(draw_bgr_box(make_frame(), RED_BGR, 300, 220, 60, 60))
    stream.start()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(stream))
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    _wait_running(stream)
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.server_close()
    stream.stop()


def test_index_page_served(server) -> None:
    with urllib.request.urlopen(f"{server}/", timeout=5) as r:
        assert r.status == 200
        assert "text/html" in r.headers["Content-Type"]
        body = r.read().decode()
    assert "obj-drone live view" in body
    assert "/stream.mjpg" in body


def test_status_endpoint_returns_json(server) -> None:
    with urllib.request.urlopen(f"{server}/status.json", timeout=5) as r:
        assert r.status == 200
        payload = json.loads(r.read())
    assert payload["running"] is True
    assert "found" in payload and "fps" in payload


def test_snapshot_endpoint_returns_jpeg(server) -> None:
    with urllib.request.urlopen(f"{server}/snapshot.jpg", timeout=5) as r:
        assert r.status == 200
        assert r.headers["Content-Type"] == "image/jpeg"
        data = r.read()
    assert data.startswith(b"\xff\xd8") and data.endswith(b"\xff\xd9")


def test_mjpeg_stream_emits_multipart_frames(server) -> None:
    req = urllib.request.urlopen(f"{server}/stream.mjpg", timeout=5)
    try:
        assert "multipart/x-mixed-replace" in req.headers["Content-Type"]
        chunk = req.read(2048)
        assert b"--objdroneframe" in chunk
        assert b"Content-Type: image/jpeg" in chunk
    finally:
        req.close()


def test_unknown_path_is_404(server) -> None:
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(f"{server}/nope", timeout=5)
    assert exc.value.code == 404


# ------------------------------------------------------------------- captures
def _capture_server(tmp_path):
    """Server with a capture store that saves on every detection."""
    from http.server import ThreadingHTTPServer

    from obj_drone.capture import CaptureStore

    tracker = TargetTracker(640, 480)
    frame = draw_bgr_box(make_frame(), RED_BGR, 300, 220, 60, 60)
    store = CaptureStore(output_dir=tmp_path / "caps", min_interval_s=0.0, min_confidence=0.0)
    stream = VisionStream(
        LoopingCamera(frame), tracker, source="color", jpeg_quality=60, captures=store
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(stream))
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{httpd.server_port}", httpd, stream, store


def test_capture_endpoints(tmp_path) -> None:
    import json as _json

    base, httpd, stream, store = _capture_server(tmp_path)
    try:
        # Seed a capture directly — the colour path has no Detection objects.
        store.maybe_capture(make_frame(), [Detection(39, "bottle", 0.9, (1, 1, 10, 10))])

        with urllib.request.urlopen(f"{base}/captures.json", timeout=5) as r:
            payload = _json.loads(r.read())
        assert len(payload["captures"]) == 1
        entry = payload["captures"][0]
        assert entry["labels"] == ["bottle"]

        # The image itself downloads.
        with urllib.request.urlopen(f"{base}{entry['url']}", timeout=5) as r:
            assert r.headers["Content-Type"] == "image/jpeg"
            assert "attachment" in r.headers["Content-Disposition"]
            assert r.read().startswith(b"\xff\xd8")

        # And the whole set as a zip.
        with urllib.request.urlopen(f"{base}/captures.zip", timeout=5) as r:
            assert r.headers["Content-Type"] == "application/zip"
            assert r.read()[:2] == b"PK"
    finally:
        httpd.shutdown(); httpd.server_close(); stream.stop()


def test_unknown_capture_is_404(tmp_path) -> None:
    base, httpd, stream, _ = _capture_server(tmp_path)
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(f"{base}/captures/does-not-exist.jpg", timeout=5)
        assert exc.value.code == 404
    finally:
        httpd.shutdown(); httpd.server_close(); stream.stop()


def test_capture_path_traversal_is_refused(tmp_path) -> None:
    base, httpd, stream, store = _capture_server(tmp_path)
    try:
        store.maybe_capture(make_frame(), [Detection(39, "bottle", 0.9, (1, 1, 10, 10))])
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(f"{base}/captures/..%2F..%2Fetc%2Fpasswd", timeout=5)
        assert exc.value.code == 404
    finally:
        httpd.shutdown(); httpd.server_close(); stream.stop()


def test_page_includes_the_gallery(tmp_path) -> None:
    base, httpd, stream, _ = _capture_server(tmp_path)
    try:
        with urllib.request.urlopen(f"{base}/", timeout=5) as r:
            body = r.read().decode()
        assert "Detection captures" in body
        assert "/captures.zip" in body
    finally:
        httpd.shutdown(); httpd.server_close(); stream.stop()
