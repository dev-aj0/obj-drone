"""Visual target detection and tracking."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np

from obj_drone.vision.detector import Detection, ObjectDetector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackingResult:
    """Target position in image coordinates."""

    found: bool
    center_x: float
    center_y: float
    bbox: tuple[int, int, int, int] | None = None
    label: str = ""
    confidence: float = 0.0


NOT_FOUND = TrackingResult(found=False, center_x=0.0, center_y=0.0)

# How the subject is chosen when nothing is locked yet.
#   largest    - biggest bounding box, i.e. the person closest to the camera.
#                The operator is normally the nearest person, so this is the
#                most reliable default and ignores bystanders in the background.
#   centre     - nearest the middle of frame: stand in front of the drone.
#   confidence - whatever the detector scored highest. Arbitrary in practice.
ACQUISITION_POLICIES = ("largest", "centre", "confidence")


# Best first. Which of these exist depends heavily on the OpenCV build: CSRT and
# KCF live in the contrib 'tracking' module, so plain opencv-python wheels often
# ship only MIL/Vit/Nano, while Raspberry Pi OS's apt python3-opencv has CSRT.
_ROI_TRACKERS = ("TrackerCSRT", "TrackerKCF", "TrackerVit", "TrackerMIL")


def create_roi_tracker() -> tuple[cv2.Tracker, str]:
    """Build the best available single-object ROI tracker.

    Returns (tracker, name) so callers can log which algorithm was picked.
    """
    for name in _ROI_TRACKERS:
        cls = getattr(cv2, name, None)
        if cls is not None and hasattr(cls, "create"):
            try:
                return cls.create(), name
            except cv2.error:
                # e.g. TrackerVit without its model file — try the next one.
                continue
        factory = getattr(cv2, f"{name}_create", None)
        if factory is not None:
            return factory(), name
        legacy = getattr(cv2, "legacy", None)
        legacy_factory = getattr(legacy, f"{name}_create", None) if legacy else None
        if legacy_factory is not None:
            return legacy_factory(), f"legacy.{name}"

    raise RuntimeError(
        f"No usable ROI tracker in OpenCV {cv2.__version__}. Install a build with "
        "the contrib tracking module: 'pip install opencv-contrib-python>=4.8,<5.0', "
        "or on Raspberry Pi OS 'sudo apt install python3-opencv'."
    )


class TargetTracker:
    """Select and follow a single target across frames.

    Two detection sources are supported:
      * a neural-network object detector (preferred), and
      * an HSV colour-blob detector (fallback / no-model operation).

    Either way a single target is *locked*: once a target is picked, later
    frames prefer the detection nearest the previous centre so the drone does
    not swap between two objects mid-flight.
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        hsv_lower: tuple[int, int, int] = (0, 120, 70),
        hsv_upper: tuple[int, int, int] = (10, 255, 255),
        min_blob_area: int = 100,
        detector: ObjectDetector | None = None,
        max_track_jump_px: float = 160.0,
        reid_enabled: bool = True,
        reid_threshold: float = 0.30,
        reid_weight: float = 0.6,
        lock_memory_frames: int = 0,
        acquisition: str = "largest",
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.min_blob_area = min_blob_area
        self.detector = detector
        self.max_track_jump_px = max_track_jump_px
        self.reid_enabled = reid_enabled
        self.reid_threshold = reid_threshold
        self.reid_weight = reid_weight
        # 0 = never release the lock; keep looking for the same subject.
        self.lock_memory_frames = lock_memory_frames
        if acquisition not in ACQUISITION_POLICIES:
            raise ValueError(
                f"Unknown acquisition {acquisition!r} "
                f"(expected one of {ACQUISITION_POLICIES})"
            )
        self.acquisition = acquisition
        # Set by the operator clicking the live view; consumed on the next frame.
        self._pending_selection: tuple[float, float] | None = None
        self._roi_tracker: cv2.Tracker | None = None
        self._last_center: tuple[float, float] | None = None
        # Appearance signature of the locked person, so we re-acquire the SAME
        # one after an occlusion instead of whoever happens to score highest.
        self._lock_appearance: np.ndarray | None = None
        self._lock_missing_frames = 0
        self.locked = False

    @property
    def image_center(self) -> tuple[float, float]:
        return self.frame_width / 2.0, self.frame_height / 2.0

    def reset_lock(self) -> None:
        """Forget the currently locked target entirely."""
        self._last_center = None
        self._lock_appearance = None
        self._lock_missing_frames = 0
        self.locked = False

    # ------------------------------------------------------------- appearance
    @staticmethod
    def _appearance(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        """Hue/saturation histogram of a detection — a cheap identity signature.

        Robust enough to tell two differently-dressed people apart, and costs
        well under a millisecond, unlike a real re-ID network.
        """
        x, y, w, h = bbox
        roi = frame[max(0, y) : y + h, max(0, x) : x + w]
        if roi.size == 0 or roi.shape[0] < 4 or roi.shape[1] < 4:
            return None
        # Use the torso: heads and legs vary far more between frames.
        top = roi.shape[0] // 5
        bottom = max(top + 1, roi.shape[0] * 3 // 5)
        roi = roi[top:bottom]
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0.0, 1.0, cv2.NORM_MINMAX)
        return hist.flatten()

    @staticmethod
    def _similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
        """Histogram correlation in [0, 1]; 1.0 is identical."""
        if a is None or b is None:
            return 0.0
        score = float(cv2.compareHist(a.astype("float32"), b.astype("float32"), cv2.HISTCMP_CORREL))
        return max(0.0, min(1.0, score))

    # --------------------------------------------------------------- detector
    def detect_objects(self, frame: np.ndarray) -> TrackingResult:
        """Run the NN detector and return the locked target."""
        if self.detector is None:
            raise RuntimeError("No object detector configured")
        return self.select_target(self.detector.detect(frame), frame)

    def select_target(
        self, detections: list[Detection], frame: np.ndarray | None = None
    ) -> TrackingResult:
        """Pick which detection to follow, holding onto the same subject.

        Once locked, the tracker stays with that individual for the rest of the
        flight: candidates are scored on both proximity to the last known
        position and appearance similarity, so after an occlusion it re-acquires
        the person it was following rather than whoever scores highest.

        ``frame`` is needed to compute appearance signatures; without it the
        tracker falls back to proximity only.
        """
        if not detections:
            return self._handle_no_candidates()

        # --- operator clicked a subject in the live view: honour that ---
        if self._pending_selection is not None:
            sx, sy = self._pending_selection
            self._pending_selection = None
            picked = self._detection_at(detections, sx, sy)
            if picked is not None:
                self.reset_lock()
                self._acquire(picked, frame)
                logger.info(
                    "Operator selected %s at (%.0f,%.0f) — locked",
                    picked.label or "target", picked.center[0], picked.center[1],
                )
                return self._result(picked)
            logger.warning("No detection at (%.0f,%.0f) — selection ignored", sx, sy)

        # --- no lock yet: pick a subject and keep them ---
        if not self.locked:
            chosen = self._acquire_candidate(detections)
            self._acquire(chosen, frame)
            logger.info(
                "LOCKED onto %s (%.0f%%) at (%.0f,%.0f) — staying with this "
                "subject for the rest of the flight",
                chosen.label or "target",
                chosen.confidence * 100,
                chosen.center[0],
                chosen.center[1],
            )
            return self._result(chosen)

        # --- locked: score every candidate on position AND appearance ---
        lx, ly = self._last_center if self._last_center else self.image_center
        # Allow a wider search the longer the target has been missing.
        radius = self.max_track_jump_px * (1 + self._lock_missing_frames / 15.0)

        best: Detection | None = None
        best_score = -1.0
        for det in detections:
            dist = math.hypot(det.center[0] - lx, det.center[1] - ly)
            proximity = max(0.0, 1.0 - dist / radius) if radius > 0 else 0.0

            if self.reid_enabled and frame is not None:
                similarity = self._similarity(
                    self._lock_appearance, self._appearance(frame, det.bbox)
                )
                score = self.reid_weight * similarity + (1 - self.reid_weight) * proximity
                # An appearance that clearly is not our target is rejected even
                # if it is standing exactly where we last saw them.
                if similarity < self.reid_threshold and self._lock_missing_frames > 0:
                    continue
            else:
                score = proximity

            if score > best_score:
                best_score, best = score, det

        if best is None or best_score <= 0.0:
            return self._handle_no_candidates()

        self._acquire(best, frame, blend=True)
        return self._result(best)

    def select_at(self, x: float, y: float) -> None:
        """Ask to lock whichever detection contains this image point.

        Applied on the next processed frame, so this is safe to call from
        another thread (the web viewer) while the control loop is running.
        """
        self._pending_selection = (float(x), float(y))

    @staticmethod
    def _detection_at(
        detections: list[Detection], x: float, y: float
    ) -> Detection | None:
        """Smallest detection whose box contains the point, if any."""
        hits = [
            d for d in detections
            if d.bbox[0] <= x <= d.bbox[0] + d.bbox[2]
            and d.bbox[1] <= y <= d.bbox[1] + d.bbox[3]
        ]
        return min(hits, key=lambda d: d.area) if hits else None

    def _acquire_candidate(self, detections: list[Detection]) -> Detection:
        """Choose the subject to follow, per the configured policy."""
        if self.acquisition == "largest":
            # Biggest box = nearest person = almost always the operator.
            return max(detections, key=lambda d: d.area)
        if self.acquisition == "centre":
            cx, cy = self.image_center
            return min(
                detections,
                key=lambda d: math.hypot(d.center[0] - cx, d.center[1] - cy),
            )
        return max(detections, key=lambda d: d.confidence)

    def _handle_no_candidates(self) -> TrackingResult:
        """Target not visible this frame — remember it rather than forgetting."""
        if self.locked:
            self._lock_missing_frames += 1
            # lock_memory_frames <= 0 means "hold this subject for the entire
            # flight" — we keep searching for them and never adopt someone else.
            if 0 < self.lock_memory_frames < self._lock_missing_frames:
                logger.info(
                    "Target lost for %d frames — releasing lock",
                    self._lock_missing_frames,
                )
                self.reset_lock()
        return NOT_FOUND

    def _acquire(
        self, det: Detection, frame: np.ndarray | None, blend: bool = False
    ) -> None:
        self._last_center = det.center
        self._lock_missing_frames = 0
        self.locked = True
        if frame is None:
            return
        appearance = self._appearance(frame, det.bbox)
        if appearance is None:
            return
        if blend and self._lock_appearance is not None:
            # Slow exponential update tracks lighting and pose drift without
            # letting the signature wander onto a different person.
            self._lock_appearance = 0.9 * self._lock_appearance + 0.1 * appearance
        else:
            self._lock_appearance = appearance

    @staticmethod
    def _result(det: Detection) -> TrackingResult:
        cx, cy = det.center
        return TrackingResult(
            found=True,
            center_x=cx,
            center_y=cy,
            bbox=det.bbox,
            label=det.label,
            confidence=det.confidence,
        )

    # ------------------------------------------------------------------ colour
    def color_mask(self, frame: np.ndarray) -> np.ndarray:
        """Binary mask of pixels inside the configured HSV range."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        # Red straddles hue 0, so a low-hue range needs the top of the wheel too.
        if self.hsv_lower[0] <= 10 and self.hsv_upper[0] <= 10:
            lower2 = np.array([170, self.hsv_lower[1], self.hsv_lower[2]], dtype=np.uint8)
            upper2 = np.array([180, self.hsv_upper[1], self.hsv_upper[2]], dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower2, upper2))
        return mask

    def detect_color(self, frame: np.ndarray) -> TrackingResult:
        """Find the largest blob matching the HSV colour range."""
        mask = self.color_mask(frame)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self._last_center = None
            return NOT_FOUND

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.min_blob_area:
            self._last_center = None
            return NOT_FOUND

        x, y, w, h = cv2.boundingRect(largest)
        cx = x + w / 2.0
        cy = y + h / 2.0
        self._last_center = (cx, cy)
        return TrackingResult(
            found=True, center_x=cx, center_y=cy, bbox=(x, y, w, h), label="color"
        )

    def preview_mask(self, frame: np.ndarray) -> np.ndarray:
        """Return the binary mask for colour calibration."""
        return self.color_mask(frame)

    # -------------------------------------------------------------------- ROI
    def init_roi_tracker(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        """Start ROI tracking on a manually selected region."""
        self._roi_tracker, name = create_roi_tracker()
        self._roi_tracker.init(frame, bbox)
        logger.info("ROI tracking started with %s", name)

    def track_roi(self, frame: np.ndarray) -> TrackingResult:
        if self._roi_tracker is None:
            return NOT_FOUND

        ok, bbox = self._roi_tracker.update(frame)
        if not ok:
            self._roi_tracker = None
            self._last_center = None
            return NOT_FOUND

        x, y, w, h = (int(v) for v in bbox)
        cx = x + w / 2.0
        cy = y + h / 2.0
        self._last_center = (cx, cy)
        return TrackingResult(
            found=True, center_x=cx, center_y=cy, bbox=(x, y, w, h), label="roi"
        )

    # ------------------------------------------------------------------- error
    def pixel_error(self, result: TrackingResult) -> tuple[float, float]:
        """Return (horizontal, vertical) pixel offset from image centre."""
        cx, cy = self.image_center
        return result.center_x - cx, result.center_y - cy

    def normalized_error(self, result: TrackingResult) -> tuple[float, float]:
        """Pixel error scaled to [-1, 1] so gains are resolution-independent."""
        err_x, err_y = self.pixel_error(result)
        return err_x / (self.frame_width / 2.0), err_y / (self.frame_height / 2.0)
