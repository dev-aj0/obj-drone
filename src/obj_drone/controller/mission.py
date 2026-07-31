"""High-level mission: vision-driven velocity commands to ArduPilot."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto

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
    track_gain: float = 0.002
    lost_target_grace_frames: int = 15
    lost_target_action: str = "hover"
    link_loss_action: str = "rtl"


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

    def prepare_guided(self) -> None:
        """Enter GUIDED mode and arm. Does not take off."""
        self.phase = MissionPhase.GUIDED
        self.fc.set_mode("GUIDED")
        if not self.fc.link.wait_for_mode("GUIDED", timeout=10.0):
            raise RuntimeError("Flight controller did not enter GUIDED mode")
        self.fc.arm()
        if not self.fc.link.wait_for_arm(armed=True, timeout=10.0):
            raise RuntimeError("Flight controller did not arm")
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

    def track_target_loop(self, use_color: bool = True) -> None:
        """Continuously center the target using body-frame velocity commands."""
        self._running = True
        self.phase = MissionPhase.TRACKING
        period = 1.0 / self.config.control_rate_hz
        logger.info("Starting target tracking loop")

        while self._running and not self._stop_event.is_set():
            if not self._check_link_health():
                break

            loop_start = time.monotonic()
            frame = self.camera.read()
            if frame is None:
                time.sleep(period)
                continue

            result = (
                self.tracker.detect_color(frame)
                if use_color
                else self.tracker.track_roi(frame)
            )
            self._apply_tracking(result)

            if self.debug is not None:
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
        self.stop_tracking()
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

    def _check_link_health(self) -> bool:
        if self.telemetry.link_healthy():
            return True
        self.trigger_failsafe(
            f"MAVLink link lost ({self.telemetry.seconds_since_heartbeat():.1f}s)",
            self.config.link_loss_action,
        )
        return False

    def _apply_tracking(self, result: TrackingResult) -> None:
        if not result.found:
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

        self._lost_frames = 0
        err_x, err_y = self.tracker.pixel_error(result)
        vx = _clamp(err_y * self.config.track_gain, self.config.max_horizontal_speed_m_s)
        vy = _clamp(err_x * self.config.track_gain, self.config.max_horizontal_speed_m_s)
        self.fc.send_velocity_body(vx=vx, vy=vy, vz=0.0, yaw_rate=0.0)


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))
