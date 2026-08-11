"""Frame annotation, shared by the debug frame writer and the web viewer."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from obj_drone.paths import project_root
from obj_drone.vision.detector import Detection
from obj_drone.vision.tracker import TrackingResult

logger = logging.getLogger(__name__)

GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
GREY = (160, 160, 160)


def annotate_frame(
    frame: np.ndarray,
    result: TrackingResult,
    pixel_error: tuple[float, float] | None = None,
    detections: list[Detection] | None = None,
) -> np.ndarray:
    """Return a copy of ``frame`` with tracking state drawn on it.

    Other detections are drawn faintly so it is obvious which one the tracker
    locked onto and which it passed over.
    """
    annotated = frame.copy()
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Frame centre — the point the controller drives the target towards.
    cv2.drawMarker(annotated, (cx, cy), WHITE, cv2.MARKER_CROSS, 20, 2)

    if detections:
        locked = result.bbox if result.found else None
        for det in detections:
            if locked is not None and det.bbox == locked:
                continue
            x, y, bw, bh = det.bbox
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), GREY, 1)
            cv2.putText(
                annotated,
                f"{det.label} {det.confidence:.2f}",
                (x, max(12, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                GREY,
                1,
            )

    if result.found and result.bbox is not None:
        x, y, bw, bh = result.bbox
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), GREEN, 2)
        cv2.circle(annotated, (int(result.center_x), int(result.center_y)), 5, RED, -1)
        # Line from frame centre to the target: the error the controller sees.
        cv2.line(annotated, (cx, cy), (int(result.center_x), int(result.center_y)), GREEN, 1)

        if result.label:
            caption = result.label
            if result.confidence > 0:
                caption += f" {result.confidence:.2f}"
            cv2.putText(
                annotated, caption, (x, max(15, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1,
            )
        if pixel_error is not None:
            cv2.putText(
                annotated,
                f"err=({pixel_error[0]:+.0f},{pixel_error[1]:+.0f})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2,
            )
    else:
        cv2.putText(
            annotated, "NO TARGET", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2,
        )

    return annotated


class DebugWriter:
    """Save annotated camera frames to disk."""

    def __init__(
        self,
        enabled: bool = False,
        interval: int = 30,
        output_dir: str | Path | None = None,
    ) -> None:
        self.enabled = enabled
        self.interval = max(1, interval)
        self.output_dir = Path(output_dir) if output_dir else project_root() / "logs" / "frames"
        self._counter = 0
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Debug frames will be saved to %s", self.output_dir)

    def write(
        self,
        frame: np.ndarray,
        result: TrackingResult,
        pixel_error: tuple[float, float] | None,
        detections: list[Detection] | None = None,
    ) -> None:
        if not self.enabled:
            return

        annotated = annotate_frame(frame, result, pixel_error, detections)
        self._counter += 1
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        path = self.output_dir / f"frame_{ts}_{self._counter:06d}.jpg"
        cv2.imwrite(str(path), annotated)
