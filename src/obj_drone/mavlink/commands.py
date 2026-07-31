"""High-level ArduPilot commands sent over MAVLink."""

from __future__ import annotations

import logging
import time
from enum import IntFlag

from pymavlink import mavutil

from obj_drone.mavlink.connection import MavlinkConnection
from obj_drone.mavlink.telemetry import TelemetryMonitor

logger = logging.getLogger(__name__)


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
        telemetry: TelemetryMonitor | None = None,
        command_rate_hz: float = 10.0,
    ) -> None:
        self.link = link
        self.telemetry = telemetry
        self._min_command_interval = 1.0 / command_rate_hz if command_rate_hz > 0 else 0.0
        self._last_command_time = 0.0

    @property
    def _target(self) -> tuple[int, int]:
        m = self.link.master
        return m.target_system, m.target_component

    def set_mode(self, mode: str) -> None:
        mapping = self.link.master.mode_mapping()
        if mapping is None or mode not in mapping:
            raise ValueError(f"Mode {mode!r} not available on this vehicle")
        self.link.master.set_mode(mapping[mode])
        logger.info("Requested mode %s", mode)

    def arm(self, force: bool = False) -> None:
        param2 = 21196 if force else 0
        self.link.master.mav.command_long_send(
            self._target[0],
            self._target[1],
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1,
            param2,
            0,
            0,
            0,
            0,
            0,
        )
        logger.info("Arm command sent")

    def disarm(self, force: bool = False) -> None:
        param2 = 21196 if force else 0
        self.link.master.mav.command_long_send(
            self._target[0],
            self._target[1],
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            0,
            param2,
            0,
            0,
            0,
            0,
            0,
        )
        logger.info("Disarm command sent")

    def takeoff(self, altitude_m: float) -> None:
        """Command takeoff in GUIDED mode to the given altitude (meters AGL)."""
        self.set_mode("GUIDED")
        if not self.link.wait_for_mode("GUIDED", timeout=5.0):
            raise RuntimeError("Failed to enter GUIDED mode for takeoff")

        self.link.master.mav.command_long_send(
            self._target[0],
            self._target[1],
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            altitude_m,
        )
        logger.info("Takeoff to %.1f m commanded", altitude_m)

    def land(self) -> None:
        self.set_mode("LAND")
        logger.info("LAND mode requested")

    def rtl(self) -> None:
        self.set_mode("RTL")
        logger.info("RTL mode requested")

    def loiter(self) -> None:
        self.set_mode("LOITER")
        logger.info("LOITER mode requested")

    def send_velocity_ned(
        self,
        vx: float,
        vy: float,
        vz: float,
        yaw_rate: float = 0.0,
    ) -> None:
        """Send velocity setpoint in LOCAL_NED frame (m/s, z positive down)."""
        type_mask = int(
            PositionTargetMask.POSITION
            | PositionTargetMask.ACCEL
            | PositionTargetMask.YAW
        )
        self._send_position_target_local_ned(
            type_mask=type_mask,
            vx=vx,
            vy=vy,
            vz=vz,
            yaw_rate=yaw_rate,
        )

    def send_velocity_body(
        self,
        vx: float,
        vy: float,
        vz: float,
        yaw_rate: float = 0.0,
    ) -> None:
        """Send velocity setpoint in BODY_NED frame (forward/right/down)."""
        type_mask = int(
            PositionTargetMask.POSITION
            | PositionTargetMask.ACCEL
            | PositionTargetMask.YAW
        )
        self._send_position_target_local_ned(
            type_mask=type_mask,
            vx=vx,
            vy=vy,
            vz=vz,
            yaw_rate=yaw_rate,
            frame=mavutil.mavlink.MAV_FRAME_BODY_NED,
        )

    def send_position_ned(
        self,
        north_m: float,
        east_m: float,
        down_m: float,
        yaw: float = 0.0,
    ) -> None:
        """Send position setpoint in LOCAL_NED frame (meters, z positive down)."""
        type_mask = int(
            PositionTargetMask.VELOCITY
            | PositionTargetMask.ACCEL
            | PositionTargetMask.YAW_RATE
        )
        self._send_position_target_local_ned(
            type_mask=type_mask,
            x=north_m,
            y=east_m,
            z=down_m,
            yaw=yaw,
        )

    def goto_local_ned(
        self,
        north_m: float,
        east_m: float,
        down_m: float,
    ) -> None:
        """Fly to a local NED offset in GUIDED mode."""
        self.set_mode("GUIDED")
        self.send_position_ned(north_m, east_m, down_m)

    def hover(self) -> None:
        """Zero velocity setpoint — hold position in GUIDED mode."""
        self.send_velocity_ned(0.0, 0.0, 0.0, yaw_rate=0.0)

    def wait_altitude(
        self,
        target_alt_m: float,
        tolerance_m: float = 0.5,
        timeout: float = 60.0,
    ) -> bool:
        """Wait until relative altitude is within tolerance of target."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.telemetry is not None:
                alt_m = self.telemetry.snapshot().relative_alt_m
            else:
                msg = self.link.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=1.0)
                if msg is None:
                    continue
                alt_m = msg.relative_alt / 1000.0
            if abs(alt_m - target_alt_m) <= tolerance_m:
                logger.info("Reached altitude %.1f m", alt_m)
                return True
            time.sleep(0.1)
        logger.warning("Altitude wait timed out (target %.1f m)", target_alt_m)
        return False

    def get_relative_altitude(self) -> float | None:
        if self.telemetry is None:
            return None
        return self.telemetry.snapshot().relative_alt_m

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
    ) -> None:
        if self._min_command_interval > 0:
            now = time.monotonic()
            elapsed = now - self._last_command_time
            if elapsed < self._min_command_interval:
                time.sleep(self._min_command_interval - elapsed)
            self._last_command_time = time.monotonic()

        self.link.master.mav.set_position_target_local_ned_send(
            0,
            self.link.master.target_system,
            self.link.master.target_component,
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
