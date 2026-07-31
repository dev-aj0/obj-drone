"""Background MAVLink telemetry reader and vehicle state cache."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

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
    battery_voltage: float = 0.0
    last_heartbeat_monotonic: float = field(default_factory=time.monotonic)
    connected: bool = False


class TelemetryMonitor:
    """Continuously read MAVLink messages and expose thread-safe vehicle state."""

    def __init__(self, link: MavlinkConnection, link_loss_timeout_s: float = 3.0) -> None:
        self.link = link
        self.link_loss_timeout_s = link_loss_timeout_s
        self.state = VehicleState()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="mavlink-telemetry", daemon=True)
        self._thread.start()
        logger.debug("Telemetry monitor started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def snapshot(self) -> VehicleState:
        with self._lock:
            return VehicleState(**self.state.__dict__)

    def link_healthy(self) -> bool:
        with self._lock:
            age = time.monotonic() - self.state.last_heartbeat_monotonic
            return self.state.connected and age <= self.link_loss_timeout_s

    def seconds_since_heartbeat(self) -> float:
        with self._lock:
            return time.monotonic() - self.state.last_heartbeat_monotonic

    def _run(self) -> None:
        while not self._stop.is_set():
            msg = self.link.recv_match(blocking=True, timeout=0.5)
            if msg is None:
                continue
            self._handle(msg)

    def _handle(self, msg: object) -> None:
        msg_type = msg.get_type()
        with self._lock:
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
            elif msg_type == "ATTITUDE":
                self.state.roll = msg.roll
                self.state.pitch = msg.pitch
                self.state.yaw = msg.yaw
            elif msg_type == "GPS_RAW_INT":
                self.state.gps_satellites = int(msg.satellites_visible)
            elif msg_type == "SYS_STATUS":
                self.state.battery_voltage = msg.voltage_battery / 1000.0
