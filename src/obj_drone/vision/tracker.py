"""Visual target detection and tracking."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class TrackingResult:
    """Target position in image coordinates."""

    found: bool
    center_x: float
    center_y: float
    bbox: tuple[int, int, int, int] | None = None


def _create_csrt_tracker() -> cv2.Tracker:
    if hasattr(cv2, "TrackerCSRT") and hasattr(cv2.TrackerCSRT, "create"):
        return cv2.TrackerCSRT.create()
    return cv2.TrackerCSRT_create()


class TargetTracker:
    """Track a colored blob or user-selected ROI in the camera frame."""

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        hsv_lower: tuple[int, int, int] = (0, 120, 70),
        hsv_upper: tuple[int, int, int] = (10, 255, 255),
        min_blob_area: int = 100,
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.min_blob_area = min_blob_area
        self._csrt: cv2.Tracker | None = None

    @property
    def image_center(self) -> tuple[float, float]:
        return self.frame_width / 2.0, self.frame_height / 2.0

    def detect_color(self, frame: np.ndarray) -> TrackingResult:
        """Find the largest blob matching the HSV color range."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        # Red wraps around hue 0 — merge upper range if tracking red
        if self.hsv_lower[0] <= 10 and self.hsv_upper[0] <= 10:
            lower2 = np.array([170, self.hsv_lower[1], self.hsv_lower[2]], dtype=np.uint8)
            upper2 = np.array([180, self.hsv_upper[1], self.hsv_upper[2]], dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower2, upper2))

        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return TrackingResult(found=False, center_x=0.0, center_y=0.0)

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.min_blob_area:
            return TrackingResult(found=False, center_x=0.0, center_y=0.0)

        x, y, w, h = cv2.boundingRect(largest)
        cx = x + w / 2.0
        cy = y + h / 2.0
        return TrackingResult(found=True, center_x=cx, center_y=cy, bbox=(x, y, w, h))

    def init_roi_tracker(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        """Start CSRT tracking on a manually selected region."""
        self._csrt = _create_csrt_tracker()
        self._csrt.init(frame, bbox)

    def track_roi(self, frame: np.ndarray) -> TrackingResult:
        if self._csrt is None:
            return TrackingResult(found=False, center_x=0.0, center_y=0.0)

        ok, bbox = self._csrt.update(frame)
        if not ok:
            self._csrt = None
            return TrackingResult(found=False, center_x=0.0, center_y=0.0)

        x, y, w, h = (int(v) for v in bbox)
        return TrackingResult(
            found=True,
            center_x=x + w / 2.0,
            center_y=y + h / 2.0,
            bbox=(x, y, w, h),
        )

    def pixel_error(self, result: TrackingResult) -> tuple[float, float]:
        """Return (horizontal, vertical) pixel offset from image center."""
        cx, cy = self.image_center
        return result.center_x - cx, result.center_y - cy

    def preview_mask(self, frame: np.ndarray) -> np.ndarray:
        """Return the binary mask for color calibration."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        if self.hsv_lower[0] <= 10 and self.hsv_upper[0] <= 10:
            lower2 = np.array([170, self.hsv_lower[1], self.hsv_lower[2]], dtype=np.uint8)
            upper2 = np.array([180, self.hsv_upper[1], self.hsv_upper[2]], dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower2, upper2))
        return mask
