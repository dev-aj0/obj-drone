"""Background MAVLink telemetry reader and vehicle state cache.

This module owns the *only* reader on the MAVLink connection. Anything that
needs to wait on vehicle state must use the ``wait_*`` helpers here rather than
calling ``recv_match`` itself — two concurrent readers steal messages from each
other and produce spurious timeouts.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from pymavlink import mavutil

from obj_drone.mavlink.connection import MavlinkConnection

logger = logging.getLogger(__name__)


@dataclass
class VehicleState:
    mode: str = "UNKNOWN"
    armed: bool = False
    lat: float = 0.0
    lon: float = 0.0
    alt_msl_m: float = 0.0
    relative_alt_m: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    gps_satellites: int = 0
    gps_fix_type: int = 0
    battery_voltage: float = 0.0
    last_heartbeat_monotonic: float = field(default_factory=time.monotonic)
    connected: bool = False
    # Whether the corresponding message has ever arrived. Distinguishes
    # "reported as zero" from "never received", which preflight needs.
    have_position: bool = False
    have_gps: bool = False
    have_sys_status: bool = False


class TelemetryMonitor:
    """Continuously read MAVLink messages and expose thread-safe vehicle state."""

    def __init__(self, link: MavlinkConnection, link_loss_timeout_s: float = 3.0) -> None:
        self.link = link
        self.link_loss_timeout_s = link_loss_timeout_s
        self.state = VehicleState()
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_statustext = ""

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self.link.mark_reader_active(True)
        self._thread = threading.Thread(
            target=self._run, name="mavlink-telemetry", daemon=True
        )
        self._thread.start()
        logger.debug("Telemetry monitor started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.link.mark_reader_active(False)

    def snapshot(self) -> VehicleState:
        with self._cond:
            return replace(self.state)

    def link_healthy(self) -> bool:
        with self._cond:
            age = time.monotonic() - self.state.last_heartbeat_monotonic
            return self.state.connected and age <= self.link_loss_timeout_s

    def seconds_since_heartbeat(self) -> float:
        with self._cond:
            return time.monotonic() - self.state.last_heartbeat_monotonic

    # ------------------------------------------------------------------- waits
    def wait_for(
        self,
        predicate: Callable[[VehicleState], bool],
        timeout: float = 10.0,
    ) -> bool:
        """Block until ``predicate(state)`` holds or the timeout expires."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while not predicate(self.state):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(timeout=min(remaining, 0.25))
            return True

    def wait_for_mode(self, mode_name: str, timeout: float = 10.0) -> bool:
        return self.wait_for(lambda s: s.mode == mode_name, timeout)

    def wait_for_armed(self, armed: bool = True, timeout: float = 10.0) -> bool:
        return self.wait_for(lambda s: s.armed == armed, timeout)

    def wait_for_altitude(
        self,
        target_alt_m: float,
        tolerance_m: float = 0.5,
        timeout: float = 60.0,
    ) -> bool:
        return self.wait_for(
            lambda s: s.have_position and abs(s.relative_alt_m - target_alt_m) <= tolerance_m,
            timeout,
        )

    def wait_for_telemetry(self, timeout: float = 10.0) -> bool:
        """Wait until the streams preflight depends on have produced a value."""
        return self.wait_for(
            lambda s: s.connected and s.have_gps and s.have_sys_status,
            timeout,
        )

    # ------------------------------------------------------------------ reader
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                msg = self.link.recv_match(blocking=True, timeout=0.5)
            except Exception:
                if self._stop.is_set():
                    return
                logger.exception("Telemetry read failed")
                time.sleep(0.1)
                continue
            if msg is None:
                # Still wake waiters so their timeouts are evaluated promptly.
                with self._cond:
                    self._cond.notify_all()
                continue
            self._handle(msg)

    def _handle(self, msg: object) -> None:
        msg_type = msg.get_type()
        if msg_type == "BAD_DATA":
            return

        statustext: str | None = None
        with self._cond:
            if msg_type == "HEARTBEAT":
                if msg.get_srcSystem() != 0:
                    self.state.mode = mavutil.mode_string_v10(msg)
                    self.state.armed = bool(
                        msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                    )
                    self.state.last_heartbeat_monotonic = time.monotonic()
                    self.state.connected = True
            elif msg_type == "GLOBAL_POSITION_INT":
                self.state.lat = msg.lat / 1e7
                self.state.lon = msg.lon / 1e7
                self.state.alt_msl_m = msg.alt / 1000.0
                self.state.relative_alt_m = msg.relative_alt / 1000.0
                self.state.vx = msg.vx / 100.0
                self.state.vy = msg.vy / 100.0
                self.state.vz = msg.vz / 100.0
                self.state.have_position = True
            elif msg_type == "ATTITUDE":
                self.state.roll = msg.roll
                self.state.pitch = msg.pitch
                self.state.yaw = msg.yaw
            elif msg_type == "GPS_RAW_INT":
                self.state.gps_satellites = int(msg.satellites_visible)
                self.state.gps_fix_type = int(msg.fix_type)
                self.state.have_gps = True
            elif msg_type == "SYS_STATUS":
                self.state.battery_voltage = msg.voltage_battery / 1000.0
                self.state.have_sys_status = True
            elif msg_type == "STATUSTEXT":
                text = msg.text.decode() if isinstance(msg.text, bytes) else str(msg.text)
                text = text.strip()
                if text and text != self._last_statustext:
                    self._last_statustext = text
                    statustext = text
            elif msg_type == "COMMAND_ACK":
                if msg.result != mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    logger.warning(
                        "Command %s rejected by flight controller (result=%d)",
                        msg.command,
                        msg.result,
                    )
            self._cond.notify_all()

        # ArduPilot reports arming/pre-arm failures here — surface them verbatim,
        # they are far more useful than a generic "did not arm" timeout.
        if statustext is not None:
            logger.info("FC: %s", statustext)
