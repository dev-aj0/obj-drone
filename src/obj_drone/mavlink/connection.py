"""UART / UDP MAVLink connection to the CoreWing F405 Wing V2."""

from __future__ import annotations

import logging
import threading
from typing import Any

from pymavlink import mavutil

logger = logging.getLogger(__name__)

# MAVLink message IDs for telemetry stream requests
MSG_SYS_STATUS = 1
MSG_ATTITUDE = 30
MSG_GLOBAL_POSITION_INT = 33
MSG_GPS_RAW_INT = 24

_COPTER_TYPES = {
    mavutil.mavlink.MAV_TYPE_QUADROTOR,
    mavutil.mavlink.MAV_TYPE_HEXAROTOR,
    mavutil.mavlink.MAV_TYPE_OCTOROTOR,
    mavutil.mavlink.MAV_TYPE_TRICOPTER,
    mavutil.mavlink.MAV_TYPE_COAXIAL,
    mavutil.mavlink.MAV_TYPE_HELICOPTER,
    mavutil.mavlink.MAV_TYPE_DODECAROTOR,
}

_VTOL_TYPES = {
    getattr(mavutil.mavlink, name)
    for name in (
        "MAV_TYPE_VTOL_DUOROTOR",
        "MAV_TYPE_VTOL_QUADROTOR",
        "MAV_TYPE_VTOL_TILTROTOR",
        "MAV_TYPE_VTOL_FIXEDROTOR",
        "MAV_TYPE_VTOL_TAILSITTER",
        "MAV_TYPE_VTOL_TILTWING",
        "MAV_TYPE_VTOL_RESERVED2",
        "MAV_TYPE_VTOL_RESERVED3",
        "MAV_TYPE_VTOL_RESERVED4",
        "MAV_TYPE_VTOL_RESERVED5",
    )
    if hasattr(mavutil.mavlink, name)
}


class VehicleClass:
    """Coarse airframe class derived from the heartbeat."""

    COPTER = "copter"
    PLANE = "plane"
    QUADPLANE = "quadplane"
    UNKNOWN = "unknown"


def classify_vehicle(mav_type: int) -> str:
    if mav_type in _COPTER_TYPES:
        return VehicleClass.COPTER
    if mav_type in _VTOL_TYPES:
        return VehicleClass.QUADPLANE
    if mav_type == mavutil.mavlink.MAV_TYPE_FIXED_WING:
        return VehicleClass.PLANE
    return VehicleClass.UNKNOWN


class MavlinkConnection:
    """Maintains a MAVLink link to ArduPilot.

    The companion computer never sends motor or actuator commands directly.
    All outputs are high-level setpoints consumed by the flight controller.

    Threading contract: a pymavlink ``mavfile`` is not safe for concurrent use.
    Exactly one reader is allowed — in this application that is
    :class:`~obj_drone.mavlink.telemetry.TelemetryMonitor`. Everything else must
    go through :meth:`send`, which is serialised by a lock. Do not call
    :meth:`recv_match` once the telemetry monitor is running: the two readers
    steal messages from each other, which previously made mode changes and arm
    confirmations time out.
    """

    def __init__(
        self,
        connection: str = "/dev/ttyAMA0",
        baud: int = 57600,
        heartbeat_timeout: float = 30.0,
        source_system: int = 255,
        source_component: int = 190,
    ) -> None:
        self.connection_string = connection
        self.baud = baud
        self.heartbeat_timeout = heartbeat_timeout
        self.source_system = source_system
        self.source_component = source_component
        self._master: mavutil.mavfile | None = None
        self._send_lock = threading.Lock()
        self._reader_active = False
        self.vehicle_class: str = VehicleClass.UNKNOWN
        self.mav_type: int | None = None

    @property
    def master(self) -> mavutil.mavfile:
        if self._master is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._master

    @property
    def is_udp(self) -> bool:
        return self.connection_string.startswith(("udp:", "udpin:", "udpout:", "tcp:"))

    def connect(self) -> None:
        """Open the link and wait for the first heartbeat from the flight controller."""
        logger.info("Connecting to flight controller at %s", self.connection_string)
        try:
            if self.is_udp:
                self._master = mavutil.mavlink_connection(
                    self.connection_string,
                    source_system=self.source_system,
                    source_component=self.source_component,
                )
            else:
                self._master = mavutil.mavlink_connection(
                    self.connection_string,
                    baud=self.baud,
                    source_system=self.source_system,
                    source_component=self.source_component,
                )
        except ImportError as exc:
            raise RuntimeError(
                f"Could not open {self.connection_string}: {exc}. "
                "Serial links need pyserial — install it with 'pip install pyserial'."
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"Could not open {self.connection_string}: {exc}. "
                "Check the device exists (ls -l /dev/ttyAMA0 /dev/serial*), that your "
                "user is in the 'dialout' group, and that no serial console is using it."
            ) from exc

        hb = self._master.wait_heartbeat(timeout=self.heartbeat_timeout)
        if hb is None:
            if self.is_udp:
                hint = (
                    "Check the simulator or telemetry bridge is running and "
                    "forwarding to this address."
                )
            else:
                hint = (
                    "Check TX/RX are not swapped, that the baud rate matches "
                    f"SERIALx_BAUD ({self.baud}), that SERIALx_PROTOCOL=2, and that "
                    "no serial console is holding the port."
                )
            raise RuntimeError(
                f"No MAVLink heartbeat on {self.connection_string} within "
                f"{self.heartbeat_timeout:.0f}s. {hint}"
            )

        self.mav_type = hb.type
        self.vehicle_class = classify_vehicle(hb.type)
        logger.info(
            "Heartbeat from system %s component %s — vehicle=%s (MAV_TYPE=%d)",
            self.master.target_system,
            self.master.target_component,
            self.vehicle_class,
            hb.type,
        )

    def configure_telemetry_streams(
        self,
        global_position_hz: float = 5.0,
        attitude_hz: float = 10.0,
        sys_status_hz: float = 1.0,
        gps_raw_hz: float = 1.0,
    ) -> None:
        """Request periodic telemetry from the flight controller."""
        self.request_message_interval(MSG_GLOBAL_POSITION_INT, global_position_hz)
        self.request_message_interval(MSG_ATTITUDE, attitude_hz)
        self.request_message_interval(MSG_SYS_STATUS, sys_status_hz)
        # Without GPS_RAW_INT the satellite count stays at 0 and preflight always fails.
        self.request_message_interval(MSG_GPS_RAW_INT, gps_raw_hz)
        logger.info(
            "Requested telemetry: position=%.0fHz attitude=%.0fHz sys=%.0fHz gps=%.0fHz",
            global_position_hz,
            attitude_hz,
            sys_status_hz,
            gps_raw_hz,
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

    # ----------------------------------------------------------------- reading
    def mark_reader_active(self, active: bool) -> None:
        """Record that a dedicated reader thread owns recv_match()."""
        self._reader_active = active

    def recv_match(
        self,
        type: str | None = None,
        blocking: bool = False,
        timeout: float | None = None,
    ) -> Any | None:
        """Read one message.

        Only the owning reader thread may call this — see the class docstring.
        """
        return self.master.recv_match(type=type, blocking=blocking, timeout=timeout)

    # ----------------------------------------------------------------- writing
    def send(self, msg: Any) -> None:
        with self._send_lock:
            self.master.mav.send(msg)

    def send_command_long(self, command: int, *params: float) -> None:
        """Send COMMAND_LONG with up to 7 parameters (missing ones default to 0)."""
        args = list(params) + [0.0] * (7 - len(params))
        with self._send_lock:
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                command,
                0,
                *args[:7],
            )

    def set_mode_raw(self, mode_id: int) -> None:
        with self._send_lock:
            self.master.set_mode(mode_id)

    def request_message_interval(self, message_id: int, rate_hz: float) -> None:
        interval_us = int(1_000_000 / rate_hz) if rate_hz > 0 else -1
        self.send_command_long(
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            message_id,
            interval_us,
        )

    def mode_mapping(self) -> dict[str, int] | None:
        return self.master.mode_mapping()
