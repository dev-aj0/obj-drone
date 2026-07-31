"""Optional debug frame writer for vision tuning."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from obj_drone.config import project_root
from obj_drone.vision.tracker import TrackingResult

logger = logging.getLogger(__name__)


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
    ) -> None:
        if not self.enabled:
            return

        annotated = frame.copy()
        cx = frame.shape[1] // 2
        cy = frame.shape[0] // 2
        cv2.drawMarker(annotated, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

        if result.found and result.bbox is not None:
            x, y, w, h = result.bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(
                annotated,
                (int(result.center_x), int(result.center_y)),
                5,
                (0, 0, 255),
                -1,
            )
            if pixel_error is not None:
                text = f"err=({pixel_error[0]:+.0f},{pixel_error[1]:+.0f})"
                cv2.putText(
                    annotated,
                    text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
        else:
            cv2.putText(
                annotated,
                "NO TARGET",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        self._counter += 1
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        path = self.output_dir / f"frame_{ts}_{self._counter:06d}.jpg"
        cv2.imwrite(str(path), annotated)
