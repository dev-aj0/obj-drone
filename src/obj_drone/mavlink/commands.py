"""High-level ArduPilot commands sent over MAVLink."""

from __future__ import annotations

import logging
import time
from enum import IntFlag

from pymavlink import mavutil

from obj_drone.mavlink.connection import MavlinkConnection, VehicleClass
from obj_drone.mavlink.telemetry import TelemetryMonitor

logger = logging.getLogger(__name__)


# Logical intent -> candidate ArduPilot mode names, best first. The vehicle
# classes genuinely differ here: plain ArduPlane has no LAND mode at all, and a
# QuadPlane's VTOL equivalents are the Q-prefixed modes.
_MODE_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    VehicleClass.COPTER: {
        "GUIDED": ("GUIDED",),
        "LAND": ("LAND",),
        "LOITER": ("LOITER",),
        "RTL": ("RTL",),
    },
    VehicleClass.QUADPLANE: {
        "GUIDED": ("GUIDED",),
        "LAND": ("QLAND", "LAND"),
        "LOITER": ("QLOITER", "LOITER"),
        "RTL": ("QRTL", "RTL"),
    },
    VehicleClass.PLANE: {
        "GUIDED": ("GUIDED",),
        "LAND": ("RTL",),  # no LAND mode on plain ArduPlane
        "LOITER": ("LOITER",),
        "RTL": ("RTL",),
    },
    VehicleClass.UNKNOWN: {
        "GUIDED": ("GUIDED",),
        "LAND": ("LAND", "QLAND", "RTL"),
        "LOITER": ("LOITER", "QLOITER"),
        "RTL": ("RTL", "QRTL"),
    },
}


class PositionTargetMask(IntFlag):
    """Fields to ignore in SET_POSITION_TARGET_LOCAL_NED."""

    POSITION = (
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
    )
    VELOCITY = (
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE
    )
    ACCEL = (
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
    )
    YAW = mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
    YAW_RATE = mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE


class FlightController:
    """Send standard ArduPilot MAVLink commands to the flight controller.

    Motor mixing, stabilization, and failsafes remain on the F405.
    """

    def __init__(
        self,
        link: MavlinkConnection,
        telemetry: TelemetryMonitor,
        command_rate_hz: float = 10.0,
    ) -> None:
        self.link = link
        self.telemetry = telemetry
        self._min_command_interval = 1.0 / command_rate_hz if command_rate_hz > 0 else 0.0
        self._last_command_time = 0.0

    # ------------------------------------------------------------------ modes
    @property
    def vehicle_class(self) -> str:
        return self.link.vehicle_class

    @property
    def supports_velocity_setpoints(self) -> bool:
        """Whether SET_POSITION_TARGET_LOCAL_NED velocities steer this airframe.

        ArduCopter and a QuadPlane in VTOL flight follow velocity setpoints.
        Fixed-wing ArduPlane GUIDED accepts position targets only and ignores
        velocity, so visual servoing by velocity would silently do nothing.
        """
        return self.vehicle_class in (VehicleClass.COPTER, VehicleClass.QUADPLANE)

    def resolve_mode(self, logical: str) -> str:
        """Map a logical intent (GUIDED/LAND/LOITER/RTL) to a real mode name."""
        mapping = self.link.mode_mapping()
        if not mapping:
            raise RuntimeError("Flight controller did not report a mode mapping")

        candidates = _MODE_ALIASES.get(self.vehicle_class, _MODE_ALIASES[VehicleClass.UNKNOWN])
        for name in candidates.get(logical, (logical,)):
            if name in mapping:
                if name != logical:
                    logger.warning(
                        "%s is not available on this %s — using %s instead",
                        logical,
                        self.vehicle_class,
                        name,
                    )
                return name
        raise RuntimeError(
            f"No mode available for {logical!r} on this {self.vehicle_class}. "
            f"Vehicle reports: {sorted(mapping)}"
        )

    def set_mode(self, logical: str, wait: bool = False, timeout: float = 10.0) -> str:
        """Request a mode by logical name. Returns the actual mode requested."""
        actual = self.resolve_mode(logical)
        mapping = self.link.mode_mapping()
        assert mapping is not None
        self.link.set_mode_raw(mapping[actual])
        logger.info("Requested mode %s", actual)
        if wait and not self.telemetry.wait_for_mode(actual, timeout=timeout):
            raise RuntimeError(
                f"Flight controller did not enter {actual} within {timeout:.0f}s "
                f"(current mode: {self.telemetry.snapshot().mode})"
            )
        return actual

    # ------------------------------------------------------------------- arming
    def arm(self, force: bool = False, wait: bool = False, timeout: float = 10.0) -> None:
        param2 = 21196 if force else 0
        self.link.send_command_long(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1, param2
        )
        logger.info("Arm command sent")
        if wait and not self.telemetry.wait_for_armed(True, timeout=timeout):
            raise RuntimeError(
                f"Flight controller did not arm within {timeout:.0f}s. "
                "Check the FC: messages logged above for the pre-arm reason."
            )

    def disarm(self, force: bool = False) -> None:
        param2 = 21196 if force else 0
        self.link.send_command_long(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, param2
        )
        logger.info("Disarm command sent")

    # ------------------------------------------------------------------ flight
    def takeoff(self, altitude_m: float) -> None:
        """Command takeoff in GUIDED mode to the given altitude (meters AGL)."""
        self.set_mode("GUIDED", wait=True, timeout=5.0)

        if self.vehicle_class == VehicleClass.QUADPLANE:
            command = mavutil.mavlink.MAV_CMD_NAV_VTOL_TAKEOFF
        else:
            command = mavutil.mavlink.MAV_CMD_NAV_TAKEOFF
        self.link.send_command_long(command, 0, 0, 0, 0, 0, 0, altitude_m)
        logger.info("Takeoff to %.1f m commanded", altitude_m)

    def land(self) -> str:
        return self.set_mode("LAND")

    def rtl(self) -> str:
        return self.set_mode("RTL")

    def loiter(self) -> str:
        return self.set_mode("LOITER")

    def send_velocity_ned(
        self, vx: float, vy: float, vz: float, yaw_rate: float = 0.0
    ) -> bool:
        """Send velocity setpoint in LOCAL_NED frame (m/s, z positive down)."""
        type_mask = int(
            PositionTargetMask.POSITION | PositionTargetMask.ACCEL | PositionTargetMask.YAW
        )
        return self._send_position_target_local_ned(
            type_mask=type_mask, vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate
        )

    def send_velocity_body(
        self, vx: float, vy: float, vz: float, yaw_rate: float = 0.0
    ) -> bool:
        """Send velocity setpoint in BODY_NED frame (forward/right/down)."""
        type_mask = int(
            PositionTargetMask.POSITION | PositionTargetMask.ACCEL | PositionTargetMask.YAW
        )
        return self._send_position_target_local_ned(
            type_mask=type_mask,
            vx=vx,
            vy=vy,
            vz=vz,
            yaw_rate=yaw_rate,
            frame=mavutil.mavlink.MAV_FRAME_BODY_NED,
        )

    def send_position_ned(
        self, north_m: float, east_m: float, down_m: float, yaw: float = 0.0
    ) -> bool:
        """Send position setpoint in LOCAL_NED frame (meters, z positive down)."""
        type_mask = int(
            PositionTargetMask.VELOCITY | PositionTargetMask.ACCEL | PositionTargetMask.YAW_RATE
        )
        return self._send_position_target_local_ned(
            type_mask=type_mask, x=north_m, y=east_m, z=down_m, yaw=yaw
        )

    def goto_local_ned(self, north_m: float, east_m: float, down_m: float) -> None:
        """Fly to a local NED offset in GUIDED mode."""
        self.set_mode("GUIDED")
        self.send_position_ned(north_m, east_m, down_m)

    def hover(self) -> bool:
        """Zero velocity setpoint — hold position in GUIDED mode."""
        return self.send_velocity_ned(0.0, 0.0, 0.0, yaw_rate=0.0)

    # -------------------------------------------------------------- telemetry
    def wait_altitude(
        self, target_alt_m: float, tolerance_m: float = 0.5, timeout: float = 60.0
    ) -> bool:
        """Wait until relative altitude is within tolerance of target."""
        ok = self.telemetry.wait_for_altitude(target_alt_m, tolerance_m, timeout)
        if ok:
            logger.info("Reached altitude %.1f m", self.telemetry.snapshot().relative_alt_m)
        else:
            logger.warning("Altitude wait timed out (target %.1f m)", target_alt_m)
        return ok

    def get_relative_altitude(self) -> float:
        return self.telemetry.snapshot().relative_alt_m

    # ------------------------------------------------------------------ sender
    def _send_position_target_local_ned(
        self,
        type_mask: int,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        vx: float = 0.0,
        vy: float = 0.0,
        vz: float = 0.0,
        afx: float = 0.0,
        afy: float = 0.0,
        afz: float = 0.0,
        yaw: float = 0.0,
        yaw_rate: float = 0.0,
        frame: int = mavutil.mavlink.MAV_FRAME_LOCAL_NED,
    ) -> bool:
        """Rate-limited setpoint send. Returns False if dropped by the limiter.

        The limiter drops rather than sleeps: blocking here would stall the
        vision control loop and delay the next frame.
        """
        now = time.monotonic()
        if self._min_command_interval > 0:
            if now - self._last_command_time < self._min_command_interval:
                return False
        self._last_command_time = now

        master = self.link.master
        msg = master.mav.set_position_target_local_ned_encode(
            0,
            master.target_system,
            master.target_component,
            frame,
            type_mask,
            x,
            y,
            z,
            vx,
            vy,
            vz,
            afx,
            afy,
            afz,
            yaw,
            yaw_rate,
        )
        self.link.send(msg)
        return True
