"""Saved detection frames — evidence that the detector actually saw something.

The camera runs at ~15 fps, so writing every frame with a detection would fill
the SD card in minutes and produce hundreds of near-identical images. This keeps
a rate-limited, size-capped set of annotated stills instead, each one a usable
piece of proof.
"""

from __future__ import annotations

import io
import logging
import threading
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from obj_drone.vision.detector import Detection

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Capture:
    """One saved frame."""

    name: str
    path: Path
    timestamp: float
    labels: tuple[str, ...]
    count: int
    best_confidence: float

    @property
    def when(self) -> str:
        return datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S")

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "when": self.when,
            "labels": list(self.labels),
            "count": self.count,
            "confidence": round(self.best_confidence, 3),
            "url": f"/captures/{self.name}",
        }


class CaptureStore:
    """Rate-limited store of annotated detection frames.

    Thread-safe: the vision loop writes, HTTP handlers read.
    """

    def __init__(
        self,
        output_dir: str | Path,
        enabled: bool = True,
        min_interval_s: float = 2.0,
        min_confidence: float = 0.45,
        max_captures: int = 300,
        jpeg_quality: int = 90,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        self.min_interval_s = float(min_interval_s)
        self.min_confidence = float(min_confidence)
        self.max_captures = int(max_captures)
        self.jpeg_quality = int(jpeg_quality)

        self._lock = threading.Lock()
        self._captures: list[Capture] = []
        self._last_capture_at = 0.0
        self._counter = 0

        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Detection captures will be saved to %s", self.output_dir)

    # ------------------------------------------------------------------ writing
    def maybe_capture(
        self,
        annotated: np.ndarray,
        detections: list[Detection],
        now: float | None = None,
    ) -> Capture | None:
        """Save this frame if it shows a confident detection and enough time has passed.

        ``annotated`` should already have boxes drawn on it — the point is a
        picture a human can look at and immediately see what was found.
        """
        if not self.enabled or not detections:
            return None

        now = time.monotonic() if now is None else now
        best = max(d.confidence for d in detections)
        if best < self.min_confidence:
            return None

        with self._lock:
            if now - self._last_capture_at < self.min_interval_s:
                return None
            if len(self._captures) >= self.max_captures:
                return None
            self._last_capture_at = now
            self._counter += 1
            index = self._counter

        labels = tuple(sorted({d.label for d in detections}))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{stamp}_{index:04d}_{'-'.join(labels)[:40]}.jpg"
        path = self.output_dir / name

        ok, buf = cv2.imencode(
            ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            logger.warning("Could not encode capture %s", name)
            return None
        path.write_bytes(buf.tobytes())

        capture = Capture(
            name=name,
            path=path,
            timestamp=time.time(),
            labels=labels,
            count=len(detections),
            best_confidence=best,
        )
        with self._lock:
            self._captures.append(capture)
        logger.info(
            "Captured %s — %d detection(s), best %.0f%%", name, len(detections), best * 100
        )
        return capture

    # ------------------------------------------------------------------ reading
    def list(self, newest_first: bool = True) -> list[Capture]:
        with self._lock:
            items = list(self._captures)
        return list(reversed(items)) if newest_first else items

    def get(self, name: str) -> Capture | None:
        """Look up a capture by name.

        Compares against known names rather than touching the filesystem, so a
        crafted name cannot escape the output directory.
        """
        with self._lock:
            for capture in self._captures:
                if capture.name == name:
                    return capture
        return None

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._captures)

    def clear(self) -> int:
        """Delete every capture from disk and forget them."""
        with self._lock:
            items, self._captures = list(self._captures), []
            self._last_capture_at = 0.0
        for capture in items:
            try:
                capture.path.unlink(missing_ok=True)
            except OSError:
                logger.debug("Could not delete %s", capture.path, exc_info=True)
        return len(items)

    def as_zip(self) -> bytes:
        """Bundle every capture into a zip, for one-click download."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for capture in self.list(newest_first=False):
                if capture.path.is_file():
                    archive.write(capture.path, arcname=capture.name)
        return buffer.getvalue()
