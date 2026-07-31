"""UART / UDP MAVLink connection to the CoreWing F405 Wing V2."""

from __future__ import annotations

import logging
import time
from typing import Any

from pymavlink import mavutil

logger = logging.getLogger(__name__)

# MAVLink message IDs for telemetry stream requests
_MSG_GLOBAL_POSITION_INT = 33
_MSG_ATTITUDE = 30
_MSG_SYS_STATUS = 1


class MavlinkConnection:
    """Maintains a MAVLink link to ArduPilot.

    The companion computer never sends motor or actuator commands directly.
    All outputs are high-level setpoints consumed by the flight controller.
    """

    def __init__(
        self,
        connection: str = "/dev/serial0",
        baud: int = 57600,
        heartbeat_timeout: float = 30.0,
    ) -> None:
        self.connection_string = connection
        self.baud = baud
        self.heartbeat_timeout = heartbeat_timeout
        self._master: mavutil.mavfile | None = None

    @property
    def master(self) -> mavutil.mavfile:
        if self._master is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._master

    @property
    def is_udp(self) -> bool:
        return self.connection_string.startswith("udp:")

    def connect(self) -> None:
        """Open the link and wait for the first heartbeat from the flight controller."""
        logger.info("Connecting to flight controller at %s", self.connection_string)
        if self.is_udp:
            self._master = mavutil.mavlink_connection(self.connection_string)
        else:
            self._master = mavutil.mavlink_connection(
                self.connection_string,
                baud=self.baud,
            )
        self._master.wait_heartbeat(timeout=self.heartbeat_timeout)
        logger.info(
            "Heartbeat from system %s component %s",
            self.master.target_system,
            self.master.target_component,
        )

    def configure_telemetry_streams(
        self,
        global_position_hz: float = 5.0,
        attitude_hz: float = 10.0,
        sys_status_hz: float = 1.0,
    ) -> None:
        """Request periodic telemetry from the flight controller."""
        self.request_message_interval(_MSG_GLOBAL_POSITION_INT, global_position_hz)
        self.request_message_interval(_MSG_ATTITUDE, attitude_hz)
        self.request_message_interval(_MSG_SYS_STATUS, sys_status_hz)
        logger.info(
            "Requested telemetry streams: position=%.0f Hz attitude=%.0f Hz sys=%.0f Hz",
            global_position_hz,
            attitude_hz,
            sys_status_hz,
        )

    def close(self) -> None:
        if self._master is not None:
            self._master.close()
            self._master = None

    def __enter__(self) -> MavlinkConnection:
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def recv_match(
        self,
        type: str | None = None,
        blocking: bool = False,
        timeout: float | None = None,
    ) -> Any | None:
        return self.master.recv_match(type=type, blocking=blocking, timeout=timeout)

    def send(self, msg: Any) -> None:
        self.master.mav.send(msg)

    def request_message_interval(self, message_id: int, rate_hz: float) -> None:
        interval_us = int(1_000_000 / rate_hz) if rate_hz > 0 else -1
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            message_id,
            interval_us,
            0,
            0,
            0,
            0,
            0,
        )

    def wait_for_mode(self, mode_name: str, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            msg = self.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
            if msg is None:
                continue
            if msg.get_srcSystem() == 0:
                continue
            current = mavutil.mode_string_v10(msg)
            if current == mode_name:
                return True
        return False

    def wait_for_arm(self, armed: bool = True, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            msg = self.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
            if msg is None:
                continue
            if msg.get_srcSystem() == 0:
                continue
            is_armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            if is_armed == armed:
                return True
        return False
