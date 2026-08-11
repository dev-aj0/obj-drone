"""Live camera + detection viewer served over HTTP.

Streams annotated frames as MJPEG (multipart/x-mixed-replace), which every
browser renders natively in an <img>, so this needs no JavaScript video stack
and no third-party dependency — only the standard library and OpenCV's JPEG
encoder.

This is a ground-testing tool. It does not talk to the flight controller and
cannot move the aircraft.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import cv2

from obj_drone.controller.follow import FollowConfig, follow_velocity
from obj_drone.controller.mission import MissionConfig, velocity_for
from obj_drone.vision.camera import Camera
from obj_drone.vision.debug import annotate_frame
from obj_drone.vision.tracker import TargetTracker

logger = logging.getLogger(__name__)

BOUNDARY = "objdroneframe"


class VisionStream:
    """Owns the camera, runs detection, and publishes annotated JPEG frames.

    Exactly one capture thread reads the camera regardless of how many browsers
    are connected — the CSI camera can only have one owner, and re-running
    detection per viewer would multiply the CPU cost for no benefit.
    """

    def __init__(
        self,
        camera: Camera,
        tracker: TargetTracker,
        source: str = "detector",
        jpeg_quality: int = 80,
        mission_config: MissionConfig | None = None,
        follow_config: FollowConfig | None = None,
    ) -> None:
        self.camera = camera
        self.tracker = tracker
        self.source = source
        self.jpeg_quality = int(jpeg_quality)
        # Used only to compute what the controller WOULD command. Nothing is
        # ever sent to the flight controller from this viewer.
        self.mission_config = mission_config or MissionConfig()
        self.follow_config = follow_config

        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._status: dict[str, Any] = {"running": False}
        self._seq = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame_times: deque[float] = deque(maxlen=30)
        self._frames = 0
        self._hits = 0
        self._last_focus: dict[str, Any] = {}
        self._last_distance: float | None = None
        self._last_too_close = False

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="vision-stream", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        with self._cond:
            self._cond.notify_all()

    # ------------------------------------------------------------------ capture
    def _run(self) -> None:
        while not self._stop.is_set():
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.01)
                continue

            detections = None
            if self.source == "detector" and self.tracker.detector is not None:
                detections = self.tracker.detector.detect(frame)
                result = self.tracker.select_target(detections)
            elif self.source == "roi":
                result = self.tracker.track_roi(frame)
            else:
                result = self.tracker.detect_color(frame)

            self._frames += 1
            if result.found:
                self._hits += 1
            self._frame_times.append(time.monotonic())

            err = self.tracker.pixel_error(result) if result.found else None
            norm = self.tracker.normalized_error(result) if result.found else None
            velocity = self._would_command(result)
            annotated = annotate_frame(frame, result, err, detections)

            ok, buf = cv2.imencode(
                ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            )
            if not ok:
                continue

            status = {
                "running": True,
                "source": self.source,
                "found": bool(result.found),
                "label": result.label or None,
                "confidence": round(float(result.confidence), 3),
                "center": [round(result.center_x, 1), round(result.center_y, 1)]
                if result.found
                else None,
                "pixel_error": [round(err[0], 1), round(err[1], 1)] if err else None,
                "normalized_error": [round(norm[0], 3), round(norm[1], 3)] if norm else None,
                "detections": len(detections) if detections is not None else None,
                "inference_ms": round(self.tracker.detector.last_inference_ms, 1)
                if self.tracker.detector is not None
                else None,
                "fps": self._fps(),
                "frames": self._frames,
                "hit_rate": round(self._hits / self._frames, 3) if self._frames else 0.0,
                "resolution": [annotated.shape[1], annotated.shape[0]],
                "velocity": velocity,
                "camera_orientation": self.mission_config.camera_orientation,
                "follow_enabled": self.mission_config.follow_enabled,
                "locked": bool(getattr(self.tracker, "locked", False)),
                "distance_m": round(self._last_distance, 2)
                if self._last_distance is not None
                else None,
                "too_close": self._last_too_close,
                **self._focus_info(),
            }

            with self._cond:
                self._jpeg = buf.tobytes()
                self._status = status
                self._seq += 1
                self._cond.notify_all()

        with self._cond:
            self._status = {**self._status, "running": False}
            self._cond.notify_all()

    def _would_command(self, result) -> list[float] | None:
        """Body-frame velocity the mission controller would send for this frame.

        Computed with the real MissionController maths so what you see here is
        exactly what would be commanded in flight — but this viewer holds no
        MAVLink connection, so nothing is transmitted.
        """
        if not result.found:
            return None
        if self.mission_config.follow_enabled and self.follow_config is not None:
            # Altitude is unknown on the bench, so report the setpoint: this
            # shows the lateral/longitudinal signs, which is what needs checking.
            cmd = follow_velocity(
                result,
                self.tracker.frame_width,
                self.tracker.frame_height,
                self.follow_config.follow_height_m,
                self.follow_config,
            )
            self._last_distance = cmd.distance_m
            self._last_too_close = cmd.too_close
            return [round(cmd.vx, 2), round(cmd.vy, 2), round(cmd.vz, 2)]
        vx, vy, vz = velocity_for(self.tracker, result, self.mission_config)
        return [round(vx, 2), round(vy, 2), round(vz, 2)]

    def _focus_info(self) -> dict[str, Any]:
        """Focus telemetry, polled occasionally — capture_metadata() is not free."""
        if not hasattr(self.camera, "focus_info"):
            return {}
        if self._frames % 15 != 0 and self._last_focus:
            return self._last_focus
        try:
            self._last_focus = self.camera.focus_info()
        except Exception:
            self._last_focus = {}
        return self._last_focus

    def _fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return round((len(self._frame_times) - 1) / span, 1) if span > 0 else 0.0

    # -------------------------------------------------------------- subscribers
    def wait_for_frame(
        self, last_seq: int, timeout: float = 5.0
    ) -> tuple[int, bytes | None, dict[str, Any]]:
        """Block until a frame newer than ``last_seq`` is published."""
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout=timeout)
            return self._seq, self._jpeg, dict(self._status)

    def status(self) -> dict[str, Any]:
        with self._cond:
            return dict(self._status)


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>obj-drone live view</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 1.5rem;
    background: #0e1116; color: #e6edf3;
    font: 15px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  h1 { font-size: 1.1rem; margin: 0 0 .25rem; font-weight: 600; }
  .sub { color: #8b949e; font-size: .85rem; margin-bottom: 1.25rem; }
  .wrap { display: grid; gap: 1.25rem; grid-template-columns: minmax(0,1fr) 260px; align-items: start; }
  @media (max-width: 820px) { .wrap { grid-template-columns: minmax(0,1fr); } }
  .feed { background:#000; border:1px solid #30363d; border-radius:10px; overflow:hidden; line-height:0; }
  .feed img { width: 100%; height: auto; display: block; }
  .badge {
    display:flex; align-items:center; justify-content:center; gap:.5rem;
    padding:.8rem; border-radius:10px; font-weight:700; letter-spacing:.04em;
    font-size:1.05rem; border:1px solid; transition: background .15s, color .15s;
  }
  .badge.on  { background:#0f2f1b; color:#3fb950; border-color:#238636; }
  .badge.off { background:#2d1618; color:#f85149; border-color:#a5252b; }
  .badge.idle{ background:#161b22; color:#8b949e; border-color:#30363d; }
  .dot { width:.6rem; height:.6rem; border-radius:50%; background:currentColor; }
  .badge.on .dot { animation: pulse 1.2s ease-in-out infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.35} }
  table { width:100%; border-collapse:collapse; margin-top:1rem; font-size:.85rem; }
  td { padding:.35rem 0; border-bottom:1px solid #21262d; }
  td:first-child { color:#8b949e; }
  td:last-child { text-align:right; font-variant-numeric:tabular-nums; }
  .note { margin-top:1rem; font-size:.75rem; color:#6e7681; }
</style>
</head>
<body>
  <h1>obj-drone live view</h1>
  <div class="sub">Ground testing only — this page cannot command the aircraft.</div>
  <div class="wrap">
    <div class="feed"><img src="/stream.mjpg" alt="Live camera feed"></div>
    <div>
      <div id="badge" class="badge idle"><span class="dot"></span><span id="badgetext">CONNECTING</span></div>
      <table>
        <tr><td>Source</td><td id="source">—</td></tr>
        <tr><td>Confidence</td><td id="conf">—</td></tr>
        <tr><td>Detections</td><td id="dets">—</td></tr>
        <tr><td>Centre (px)</td><td id="center">—</td></tr>
        <tr><td>Error (px)</td><td id="err">—</td></tr>
        <tr><td>Error (norm)</td><td id="nerr">—</td></tr>
        <tr><td>Inference</td><td id="inf">—</td></tr>
        <tr><td>FPS</td><td id="fps">—</td></tr>
        <tr><td>Frames</td><td id="frames">—</td></tr>
        <tr><td>Hit rate</td><td id="hit">—</td></tr>
        <tr><td>Resolution</td><td id="res">—</td></tr>
        <tr><td>Cmd vx (fwd)</td><td id="vx">—</td></tr>
        <tr><td>Cmd vy (right)</td><td id="vy">—</td></tr>
        <tr><td>Cmd vz (down)</td><td id="vz">—</td></tr>
        <tr><td>Mounting</td><td id="orient">—</td></tr>
        <tr><td>Locked on</td><td id="locked">—</td></tr>
        <tr><td>Range</td><td id="dist">—</td></tr>
        <tr><td>Lens</td><td id="lens">—</td></tr>
        <tr><td>AF state</td><td id="af">—</td></tr>
        <tr><td>Sharpness</td><td id="fom">—</td></tr>
      </table>
      <div class="note"><b>Cmd vx/vy/vz</b> is what the controller WOULD send in flight —
      nothing is transmitted from this page. Move the target right of centre:
      <b>vy must go positive</b>. If it goes negative, set mission.invert_lateral.<br><br>
      Sharpness (FocusFoM) near zero at every lens position means
      the scene has no detail to focus on — aim at something textured.<br><br>
      Green box = locked target. Grey = other detections.
      Line runs from frame centre to target — that offset is what drives the velocity command.</div>
    </div>
  </div>
<script>
const $ = id => document.getElementById(id);
const fmt = v => (v === null || v === undefined) ? "—" : v;

async function poll() {
  try {
    const r = await fetch("/status.json", { cache: "no-store" });
    const s = await r.json();
    const badge = $("badge"), text = $("badgetext");

    if (!s.running)      { badge.className = "badge idle"; text.textContent = "STOPPED"; }
    else if (s.found)    { badge.className = "badge on";   text.textContent = "TARGET"; }
    else                 { badge.className = "badge off";  text.textContent = "NO TARGET"; }

    $("source").textContent = fmt(s.source);
    $("conf").textContent   = s.found && s.confidence ? (s.confidence * 100).toFixed(0) + "%" : "—";
    $("dets").textContent   = fmt(s.detections);
    $("center").textContent = s.center ? s.center[0] + ", " + s.center[1] : "—";
    $("err").textContent    = s.pixel_error ? s.pixel_error[0] + ", " + s.pixel_error[1] : "—";
    $("nerr").textContent   = s.normalized_error ? s.normalized_error[0] + ", " + s.normalized_error[1] : "—";
    $("inf").textContent    = s.inference_ms !== null && s.inference_ms !== undefined ? s.inference_ms + " ms" : "—";
    $("fps").textContent    = fmt(s.fps);
    $("frames").textContent = fmt(s.frames);
    $("hit").textContent    = s.hit_rate !== undefined ? (s.hit_rate * 100).toFixed(0) + "%" : "—";
    $("res").textContent    = s.resolution ? s.resolution[0] + "x" + s.resolution[1] : "—";
    const v = s.velocity;
    $("vx").textContent     = v ? v[0].toFixed(2) + " m/s" : "—";
    $("vy").textContent     = v ? v[1].toFixed(2) + " m/s" : "—";
    $("vz").textContent     = v ? v[2].toFixed(2) + " m/s" : "—";
    $("orient").textContent = fmt(s.camera_orientation);
    $("locked").textContent = s.locked ? "YES" : "no";
    $("dist").textContent   = s.distance_m !== null && s.distance_m !== undefined
                              ? s.distance_m.toFixed(2) + " m" + (s.too_close ? " TOO CLOSE" : "")
                              : "—";
    $("lens").textContent   = (s.lens_position !== null && s.lens_position !== undefined)
                              ? s.lens_position.toFixed(2) + " dpt" : "—";
    $("af").textContent     = fmt(s.af_state);
    $("fom").textContent    = fmt(s.focus_fom);
  } catch (e) {
    $("badge").className = "badge idle";
    $("badgetext").textContent = "DISCONNECTED";
  }
}
poll();
setInterval(poll, 250);
</script>
</body>
</html>
"""


def make_handler(stream: VisionStream) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.debug("%s - %s", self.address_string(), fmt % args)

        def _no_store(self) -> None:
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send_page()
            elif path == "/stream.mjpg":
                self._send_stream()
            elif path == "/status.json":
                self._send_status()
            elif path == "/snapshot.jpg":
                self._send_snapshot()
            else:
                self.send_error(404)

        def _send_page(self) -> None:
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._no_store()
            self.end_headers()
            self.wfile.write(body)

        def _send_status(self) -> None:
            body = json.dumps(stream.status()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._no_store()
            self.end_headers()
            self.wfile.write(body)

        def _send_snapshot(self) -> None:
            _seq, jpeg, _status = stream.wait_for_frame(-1, timeout=5.0)
            if jpeg is None:
                self.send_error(503, "No frame available")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self._no_store()
            self.end_headers()
            self.wfile.write(jpeg)

        def _send_stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            self._no_store()
            self.end_headers()

            last = -1
            try:
                while True:
                    last, jpeg, status = stream.wait_for_frame(last, timeout=5.0)
                    if not status.get("running") and jpeg is None:
                        break
                    if jpeg is None:
                        continue
                    header = (
                        f"--{BOUNDARY}\r\n"
                        f"Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(jpeg)}\r\n\r\n"
                    ).encode()
                    self.wfile.write(header)
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                # Browser navigated away or refreshed — entirely normal.
                logger.debug("Stream client disconnected")

    return Handler


def serve(
    stream: VisionStream,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> None:
    """Run the viewer until interrupted."""
    server = ThreadingHTTPServer((host, port), make_handler(stream))
    server.daemon_threads = True

    stream.start()
    shown = "localhost" if host in ("0.0.0.0", "") else host
    logger.info("Live view on http://%s:%d/  (Ctrl+C to stop)", shown, port)
    if host == "0.0.0.0":
        logger.warning(
            "Listening on all interfaces with no authentication — anyone on this "
            "network can watch the camera. Use --host 127.0.0.1 to restrict it."
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping")
    finally:
        server.shutdown()
        server.server_close()
        stream.stop()
