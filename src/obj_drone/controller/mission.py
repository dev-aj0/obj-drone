"""High-level mission: vision-driven velocity commands to ArduPilot."""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto

from obj_drone.controller.follow import FollowConfig, StationaryDetector, follow_velocity
from obj_drone.mavlink.commands import FlightController
from obj_drone.mavlink.telemetry import TelemetryMonitor
from obj_drone.vision.camera import Camera
from obj_drone.vision.debug import DebugWriter
from obj_drone.vision.tracker import TargetTracker, TrackingResult

logger = logging.getLogger(__name__)


class MissionPhase(Enum):
    IDLE = auto()
    GUIDED = auto()
    TAKEOFF = auto()
    TRACKING = auto()
    FAILSAFE = auto()
    LANDING = auto()


@dataclass
class MissionConfig:
    takeoff_altitude_m: float = 3.0
    max_horizontal_speed_m_s: float = 2.0
    max_vertical_speed_m_s: float = 1.0
    position_tolerance_m: float = 0.5
    control_rate_hz: float = 20.0
    # Metres/second commanded at full-frame error (normalised error of 1.0).
    track_gain: float = 1.5
    # Ignore errors this small (fraction of half-frame) to stop hunting.
    track_deadband: float = 0.05
    lost_target_grace_frames: int = 15
    lost_target_action: str = "hover"
    link_loss_action: str = "rtl"
    # Consecutive failed camera reads before treating the camera as dead.
    # Without this the control loop spins forever on a stalled camera.
    max_camera_failures: int = 60
    # "down" = nadir camera (image up = nose). "forward" = forward-facing.
    camera_orientation: str = "down"
    # Set if your camera is mounted rotated; verify on the bench, props off.
    invert_lateral: bool = False
    invert_longitudinal: bool = False
    # --- follow mode: hold station above and behind the subject ---
    follow_enabled: bool = False
    follow: FollowConfig | None = None
    # Land automatically once the subject has stood still this long (0 = never).
    land_when_stationary_s: float = 0.0
    # Stop commanding the moment the pilot switches out of GUIDED. Without this
    # the companion keeps streaming setpoints and would resume the instant the
    # pilot switched back — surprising, and not what "I took control" means.
    abort_on_mode_change: bool = True


class MissionController:
    """Orchestrates takeoff, visual tracking, and landing via MAVLink only."""

    def __init__(
        self,
        fc: FlightController,
        telemetry: TelemetryMonitor,
        camera: Camera,
        tracker: TargetTracker,
        config: MissionConfig,
        debug: DebugWriter | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.fc = fc
        self.telemetry = telemetry
        self.camera = camera
        self.tracker = tracker
        self.config = config
        self.debug = debug
        self._stop_event = stop_event or threading.Event()
        self.phase = MissionPhase.IDLE
        self._running = False
        self._lost_frames = 0
        self._frame_count = 0
        self._acquired = False
        self._camera_failures = 0
        self._follow = config.follow or FollowConfig()
        self._guided_mode: str | None = None
        self.pilot_override = False
        self._stationary = StationaryDetector(seconds=config.land_when_stationary_s or 1e9)
        self._last_follow = None

    # ------------------------------------------------------------------ setup
    def check_vehicle_supported(self) -> None:
        """Refuse to visually servo an airframe that ignores velocity setpoints."""
        if not self.fc.supports_velocity_setpoints:
            raise RuntimeError(
                f"This vehicle reports as '{self.fc.vehicle_class}'. Fixed-wing "
                "ArduPlane ignores SET_POSITION_TARGET_LOCAL_NED velocity setpoints "
                "in GUIDED, so visual tracking by velocity will not steer it. "
                "Use ArduCopter or a QuadPlane, or rewrite the tracking loop to "
                "command GUIDED position targets."
            )

    def prepare_guided(self) -> None:
        """Enter GUIDED mode and arm. Does not take off."""
        self.phase = MissionPhase.GUIDED
        self.fc.set_mode("GUIDED", wait=True, timeout=10.0)
        self.fc.arm(wait=True, timeout=10.0)
        logger.info("Armed and in GUIDED mode")

    def takeoff_and_hover(self) -> None:
        self.phase = MissionPhase.TAKEOFF
        alt = self.config.takeoff_altitude_m
        self.fc.takeoff(alt)
        if not self.fc.wait_altitude(alt, tolerance_m=self.config.position_tolerance_m):
            raise RuntimeError(f"Takeoff failed to reach {alt:.1f} m")
        self.fc.hover()
        self.phase = MissionPhase.TRACKING
        logger.info("Takeoff complete, hovering at %.1f m", alt)

    # ------------------------------------------------------------------- loop
    def track_target_loop(self, source: str = "detector") -> None:
        """Continuously centre the target using body-frame velocity commands.

        source: "detector" (neural net), "color" (HSV blob), or "roi" (CSRT).
        """
        if source == "detector" and self.tracker.detector is None:
            logger.warning("No detector configured — falling back to colour tracking")
            source = "color"

        self._running = True
        self.phase = MissionPhase.TRACKING
        period = 1.0 / self.config.control_rate_hz
        try:
            self._guided_mode = self.fc.resolve_mode("GUIDED")
        except RuntimeError:
            self._guided_mode = "GUIDED"
        logger.info("Starting target tracking loop (source=%s)", source)

        while self._running and not self._stop_event.is_set():
            if not self._check_link_health():
                break
            if self._check_pilot_override():
                break

            loop_start = time.monotonic()
            frame = self.camera.read()
            if frame is None:
                self._camera_failures += 1
                if self._camera_failures >= self.config.max_camera_failures:
                    self.trigger_failsafe(
                        f"Camera returned no frame {self._camera_failures} times",
                        self.config.link_loss_action,
                    )
                    break
                self.fc.hover()
                time.sleep(period)
                continue
            self._camera_failures = 0

            if source == "detector":
                result = self.tracker.detect_objects(frame)
            elif source == "roi":
                result = self.tracker.track_roi(frame)
            else:
                result = self.tracker.detect_color(frame)

            self._apply_tracking(result)

            if self.debug is not None and self.debug.enabled:
                self._frame_count += 1
                if self._frame_count % self.debug.interval == 0:
                    err = self.tracker.pixel_error(result) if result.found else None
                    self.debug.write(frame, result, err)

            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.0, period - elapsed))

    def request_stop(self) -> None:
        self._stop_event.set()
        self._running = False

    def stop_tracking(self) -> None:
        self._running = False
        if self.phase not in (MissionPhase.FAILSAFE, MissionPhase.LANDING):
            self.fc.hover()

    def land(self) -> None:
        self.phase = MissionPhase.LANDING
        self._running = False
        self.fc.land()

    def trigger_failsafe(self, reason: str, action: str | None = None) -> None:
        action = action or self.config.link_loss_action
        self.phase = MissionPhase.FAILSAFE
        self._running = False
        logger.error("Failsafe triggered: %s — action=%s", reason, action)
        if action == "land":
            self.fc.land()
        elif action == "loiter":
            self.fc.loiter()
        else:
            self.fc.rtl()

    def _check_pilot_override(self) -> bool:
        """Stop commanding if the pilot has switched out of GUIDED.

        ArduPilot already ignores our setpoints outside GUIDED, so this does not
        wrestle the pilot for control — it makes the companion stand down and
        stay down, rather than resuming the instant GUIDED is reselected.
        """
        if not self.config.abort_on_mode_change or self._guided_mode is None:
            return False
        mode = self.telemetry.snapshot().mode
        if mode in (self._guided_mode, "UNKNOWN"):
            return False
        logger.warning(
            "Flight mode changed to %s — pilot has taken control. Companion "
            "standing down; it will not resume on its own.",
            mode,
        )
        self.pilot_override = True
        self._running = False
        return True

    def _check_link_health(self) -> bool:
        if self.telemetry.link_healthy():
            return True
        self.trigger_failsafe(
            f"MAVLink link lost ({self.telemetry.seconds_since_heartbeat():.1f}s)",
            self.config.link_loss_action,
        )
        return False

    # --------------------------------------------------------------- control
    def _apply_tracking(self, result: TrackingResult) -> None:
        if not result.found:
            if not self._acquired:
                # Never locked on yet — hold still rather than drift.
                self.fc.hover()
                return

            self._lost_frames += 1
            if self._lost_frames < self.config.lost_target_grace_frames:
                self.fc.hover()
                return

            logger.warning("Target lost for %d frames", self._lost_frames)
            action = self.config.lost_target_action
            if action == "land":
                self.land()
            elif action == "rtl":
                self.trigger_failsafe("Target lost", "rtl")
            else:
                self.fc.hover()
            return

        if not self._acquired:
            logger.info(
                "Target acquired: %s (%.0f%%)", result.label or "target", result.confidence * 100
            )
        self._acquired = True
        self._lost_frames = 0

        if self.config.follow_enabled:
            self._apply_follow(result)
            return

        vx, vy, vz = self._velocity_for(result)
        self.fc.send_velocity_body(vx=vx, vy=vy, vz=vz, yaw_rate=0.0)

    def _apply_follow(self, result: TrackingResult) -> None:
        """Station-keeping follow: hold distance, lateral centring, and altitude."""
        state = self.telemetry.snapshot()
        cmd = follow_velocity(
            result,
            self.tracker.frame_width,
            self.tracker.frame_height,
            state.relative_alt_m,
            self._follow,
        )
        self._last_follow = cmd

        if cmd.too_close:
            logger.warning(
                "Subject at %.1f m — inside the %.1f m minimum, backing off",
                cmd.distance_m or 0.0,
                self._follow.min_distance_m,
            )

        # Landing on a stationary subject is checked here so it only ever fires
        # while we can actually see them.
        ground_speed = math.hypot(state.vx, state.vy)
        if self.config.land_when_stationary_s > 0 and self._stationary.update(
            result, ground_speed, time.monotonic()
        ):
            logger.info(
                "Subject stationary for %.0fs — landing", self._stationary.still_for
            )
            self.land()
            return

        self.fc.send_velocity_body(
            vx=cmd.vx, vy=cmd.vy, vz=cmd.vz, yaw_rate=cmd.yaw_rate
        )

    def _velocity_for(self, result: TrackingResult) -> tuple[float, float, float]:
        return velocity_for(self.tracker, result, self.config)

def velocity_for(tracker, result: TrackingResult, config: MissionConfig) -> tuple[float, float, float]:
    """Map image-plane error to a body-frame velocity command.

    Standalone so the live viewer can show exactly what would be commanded
    without holding a MAVLink connection. Normalised error is used so the gain
    does not change with resolution. Image axes: +x right, +y *down*.
    """
    err_x, err_y = tracker.normalized_error(result)
    err_x = _deadband(err_x, config.track_deadband)
    err_y = _deadband(err_y, config.track_deadband)

    gain = config.track_gain
    h_limit = config.max_horizontal_speed_m_s
    v_limit = config.max_vertical_speed_m_s

    if config.camera_orientation == "forward":
        # Forward-facing: horizontal error slides the drone sideways, vertical
        # error changes altitude. Body-NED vz is positive DOWN, and a target low
        # in the image means descend, so the sign carries through.
        vx = 0.0
        vy = _clamp(err_x * gain, h_limit)
        vz = _clamp(err_y * gain, v_limit)
    else:
        # Nadir camera, image top = nose. A target low in the image (+err_y) is
        # behind the aircraft, so fly backwards to centre it.
        vx = _clamp(-err_y * gain, h_limit)
        vy = _clamp(err_x * gain, h_limit)
        vz = 0.0

    if config.invert_lateral:
        vy = -vy
    if config.invert_longitudinal:
        vx = -vx
    return vx, vy, vz


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _deadband(value: float, band: float) -> float:
    return 0.0 if abs(value) < band else value
