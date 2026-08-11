"""Follow geometry: hold station above and behind a tracked person.

Distance is estimated from the apparent height of the person's bounding box
using a pinhole model. That makes it only as accurate as the calibration in
:class:`FollowConfig` — see ``person_height_m`` and ``camera_vfov_deg``. Treat
the result as a usable estimate for closed-loop station keeping, not a
measurement, and never as an obstacle-detection substitute.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from obj_drone.vision.tracker import TrackingResult

logger = logging.getLogger(__name__)


@dataclass
class FollowConfig:
    """Station-keeping geometry and gains."""

    # --- desired station ---
    # Horizontal distance to hold behind the subject, in metres.
    follow_distance_m: float = 3.0
    # Altitude to hold above the launch point (the subject is on the ground).
    follow_height_m: float = 3.0
    # Don't correct distance errors smaller than this.
    distance_tolerance_m: float = 0.5
    # Don't correct altitude errors smaller than this.
    altitude_tolerance_m: float = 0.3

    # --- calibration for the distance estimate ---
    person_height_m: float = 1.7
    # Vertical field of view of the camera in degrees. MUST be calibrated for
    # your lens and sensor mode, or distance estimates will be badly wrong.
    camera_vfov_deg: float = 48.0

    # --- gains (m/s per unit error) ---
    # 1.2 chosen from closed-loop simulation: at 0.6 the drone settled 1.7 m
    # behind a 1 m/s walker; at 1.2 that lag drops to 0.8 m with no
    # oscillation. Kept below 1.6 because the sim applies velocity instantly
    # and cannot show the oscillation that real control latency would add.
    distance_gain: float = 1.2
    # Backing away from a too-close subject uses its own, higher gain: at the
    # normal gain the drone retreats slower than a person walks, so it could
    # never escape someone approaching it.
    retreat_gain: float = 1.5
    # Floor on retreat speed once inside min_distance_m, m/s.
    min_retreat_speed_m_s: float = 0.8
    lateral_gain: float = 1.5
    altitude_gain: float = 0.5

    # --- how horizontal error is corrected ---
    #   "yaw"    - rotate to face the subject (natural follow-me; the drone
    #              always points at them so "behind" stays meaningful)
    #   "strafe" - slide sideways, heading fixed (no rotation at all)
    #   "both"   - yaw, plus a reduced strafe to converge faster
    # "both" by default: simulation showed pure yaw lags a subject walking
    # across the view (a pursuit curve), while pure strafe never turns so
    # "behind them" stops meaning anything. Both together track best.
    lateral_mode: str = "both"
    # rad/s commanded at full-frame horizontal error.
    yaw_gain: float = 1.2
    max_yaw_rate_deg_s: float = 45.0
    # Strafe gain is scaled by this in "both" mode.
    strafe_blend: float = 0.35

    # --- limits ---
    max_horizontal_speed_m_s: float = 2.0
    max_vertical_speed_m_s: float = 1.0
    # Never fly closer than this, whatever the estimate says. The drone has no
    # obstacle sensing and this is a person.
    min_distance_m: float = 2.0
    # Ignore lateral errors inside this fraction of the half-frame.
    lateral_deadband: float = 0.05

    invert_lateral: bool = False
    invert_longitudinal: bool = False


def focal_length_px(frame_height_px: int, vfov_deg: float) -> float:
    """Pinhole focal length in pixels from vertical field of view."""
    if vfov_deg <= 0 or vfov_deg >= 180:
        raise ValueError(f"camera_vfov_deg must be in (0, 180), got {vfov_deg}")
    return (frame_height_px / 2.0) / math.tan(math.radians(vfov_deg / 2.0))


def estimate_distance_m(
    bbox_height_px: float,
    frame_height_px: int,
    person_height_m: float,
    vfov_deg: float,
) -> float | None:
    """Estimate SLANT range (line of sight) to a person from their apparent height.

    This is the straight-line camera-to-subject distance, not the horizontal
    separation — see :func:`ground_range_m`. Returns None when the box is too
    small for the estimate to mean anything.

    Accuracy degrades badly if the person is crouching, partly occluded, or
    clipped by the frame edge — the pinhole model assumes the full body.
    """
    if bbox_height_px < 8:
        return None
    focal = focal_length_px(frame_height_px, vfov_deg)
    return (person_height_m * focal) / bbox_height_px


def ground_range_m(slant_range_m: float, altitude_m: float) -> float:
    """Horizontal separation, from slant range and height above the subject.

    Flying 3 m above someone at a 3 m slant range puts you directly over their
    head with zero horizontal standoff. The controller holds a *horizontal*
    distance, so the altitude has to come out of the measurement first.
    """
    if altitude_m <= 0:
        return max(0.0, slant_range_m)
    return math.sqrt(max(0.0, slant_range_m**2 - altitude_m**2))


@dataclass
class FollowCommand:
    """Body-frame velocity plus the reasoning behind it, for logging/UI."""

    vx: float
    vy: float
    vz: float
    # Positive turns right (clockwise seen from above), rad/s.
    yaw_rate: float
    # Horizontal separation, altitude removed. None if not estimable.
    distance_m: float | None
    # Straight-line camera-to-subject range.
    slant_range_m: float | None
    distance_error_m: float | None
    altitude_m: float
    altitude_error_m: float
    too_close: bool


def follow_velocity(
    result: TrackingResult,
    frame_width: int,
    frame_height: int,
    current_altitude_m: float,
    config: FollowConfig,
) -> FollowCommand:
    """Body-frame velocity to hold station above and behind the subject.

    Axes are ArduPilot BODY_NED: +vx forward, +vy right, +vz DOWN.

    Altitude is closed on the flight controller's own altitude estimate rather
    than on image geometry — it is far more reliable, and the subject is on the
    ground so height above launch is height above them.
    """
    # ---- altitude: hold the configured height above launch ----
    altitude_error = current_altitude_m - config.follow_height_m
    if abs(altitude_error) <= config.altitude_tolerance_m:
        vz = 0.0
    else:
        # +vz is DOWN, so a positive error (too high) means descend.
        vz = _clamp(altitude_error * config.altitude_gain, config.max_vertical_speed_m_s)

    if not result.found or result.bbox is None:
        return FollowCommand(
            vx=0.0, vy=0.0, vz=vz, yaw_rate=0.0,
            distance_m=None, slant_range_m=None, distance_error_m=None,
            altitude_m=current_altitude_m, altitude_error_m=altitude_error,
            too_close=False,
        )

    # ---- horizontal: turn to face the subject, and/or slide sideways ----
    err_x = (result.center_x - frame_width / 2.0) / (frame_width / 2.0)
    if abs(err_x) < config.lateral_deadband:
        err_x = 0.0

    max_yaw = math.radians(config.max_yaw_rate_deg_s)
    if config.lateral_mode == "strafe":
        yaw_rate = 0.0
        vy = _clamp(err_x * config.lateral_gain, config.max_horizontal_speed_m_s)
    elif config.lateral_mode == "both":
        # Subject to the right -> positive yaw turns right, toward them.
        yaw_rate = _clamp(err_x * config.yaw_gain, max_yaw)
        vy = _clamp(
            err_x * config.lateral_gain * config.strafe_blend,
            config.max_horizontal_speed_m_s,
        )
    else:  # "yaw" — rotate only; the distance controller closes the gap
        yaw_rate = _clamp(err_x * config.yaw_gain, max_yaw)
        vy = 0.0

    # ---- longitudinal: hold HORIZONTAL standoff ----
    slant = estimate_distance_m(
        result.bbox[3], frame_height, config.person_height_m, config.camera_vfov_deg
    )
    vx = 0.0
    distance = None
    distance_error = None
    too_close = False
    if slant is not None:
        # The subject is on the ground and we are above them, so strip the
        # altitude out to get true horizontal separation.
        distance = ground_range_m(slant, current_altitude_m)
        distance_error = distance - config.follow_distance_m
        if distance < config.min_distance_m:
            # Hard floor. Retreat decisively: at the normal gain this would back
            # off slower than a person walks forward and could never open the gap.
            too_close = True
            retreat = max(
                (config.min_distance_m - distance) * config.retreat_gain,
                config.min_retreat_speed_m_s,
            )
            vx = -_clamp(retreat, config.max_horizontal_speed_m_s)
        elif abs(distance_error) > config.distance_tolerance_m:
            vx = _clamp(distance_error * config.distance_gain, config.max_horizontal_speed_m_s)

    if config.invert_lateral:
        vy = -vy
        yaw_rate = -yaw_rate
    if config.invert_longitudinal:
        vx = -vx

    return FollowCommand(
        vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate,
        distance_m=distance, slant_range_m=slant, distance_error_m=distance_error,
        altitude_m=current_altitude_m, altitude_error_m=altitude_error,
        too_close=too_close,
    )


class StationaryDetector:
    """Decide whether the subject has stopped moving.

    Image-plane position alone is not enough: a well-tracked walking subject
    stays near the centre of frame the whole time. The drone's own ground speed
    is the discriminator — if the drone is not translating and the subject is
    not drifting in frame, the subject is standing still.
    """

    def __init__(
        self,
        seconds: float = 10.0,
        max_ground_speed_m_s: float = 0.3,
        max_image_drift_px_s: float = 40.0,
    ) -> None:
        self.seconds = seconds
        self.max_ground_speed_m_s = max_ground_speed_m_s
        self.max_image_drift_px_s = max_image_drift_px_s
        self._still_since: float | None = None
        self._last: tuple[float, float, float] | None = None  # t, x, y

    def reset(self) -> None:
        self._still_since = None
        self._last = None

    @property
    def still_for(self) -> float:
        """Seconds the subject has been continuously stationary.

        Measured on the same clock passed to update(), not wall-clock, so the
        caller controls the time source and this stays testable.
        """
        if self._still_since is None or self._last is None:
            return 0.0
        return max(0.0, self._last[0] - self._still_since)

    def update(
        self,
        result: TrackingResult,
        ground_speed_m_s: float,
        now: float,
    ) -> bool:
        """Feed one frame. Returns True once the stationary timeout is met."""
        if not result.found:
            # Can't judge a subject we cannot see.
            self.reset()
            return False

        drift_rate = 0.0
        if self._last is not None:
            dt = now - self._last[0]
            if dt > 0:
                drift = math.hypot(
                    result.center_x - self._last[1], result.center_y - self._last[2]
                )
                drift_rate = drift / dt
        self._last = (now, result.center_x, result.center_y)

        moving = (
            ground_speed_m_s > self.max_ground_speed_m_s
            or drift_rate > self.max_image_drift_px_s
        )
        if moving:
            self._still_since = None
            return False

        if self._still_since is None:
            self._still_since = now
            return False
        return (now - self._still_since) >= self.seconds


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))
