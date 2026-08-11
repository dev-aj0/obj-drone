"""End-to-end MAVLink test against a fake ArduPilot speaking real MAVLink over UDP.

This exercises the actual pymavlink parse path and the telemetry reader thread,
which is where the two-concurrent-readers bug lived.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest
from pymavlink import mavutil

from obj_drone.mavlink.commands import FlightController
from obj_drone.mavlink.connection import MavlinkConnection, VehicleClass
from obj_drone.mavlink.telemetry import TelemetryMonitor


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeVehicle:
    """Streams ArduCopter-like MAVLink to a UDP port until stopped."""

    def __init__(self, port: int, mav_type: int = mavutil.mavlink.MAV_TYPE_QUADROTOR) -> None:
        self.port = port
        self.mav_type = mav_type
        self.custom_mode = 4  # GUIDED on ArduCopter
        self.armed = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        conn = mavutil.mavlink_connection(
            f"udpout:127.0.0.1:{self.port}", source_system=1, source_component=1
        )
        boot = time.monotonic()
        while not self._stop.is_set():
            # time_boot_ms is uint32 — wall-clock milliseconds would overflow it.
            boot_ms = int((time.monotonic() - boot) * 1000) % 2**32
            base_mode = mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
            if self.armed:
                base_mode |= mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            conn.mav.heartbeat_send(
                self.mav_type,
                mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA,
                base_mode,
                self.custom_mode,
                mavutil.mavlink.MAV_STATE_STANDBY,
            )
            conn.mav.gps_raw_int_send(
                boot_ms * 1000, 3, 475000000, 85000000, 100000, 100, 100, 0, 0, 11
            )
            conn.mav.sys_status_send(
                0, 0, 0, 500, 12100, -1, 75, 0, 0, 0, 0, 0, 0
            )
            conn.mav.global_position_int_send(
                boot_ms, 475000000, 85000000, 100000, 3200, 10, 20, 30, 0
            )
            time.sleep(0.05)
        conn.close()


@pytest.fixture
def vehicle():
    port = _free_port()
    v = FakeVehicle(port)
    v.start()
    yield port, v
    v.stop()


def test_connects_and_classifies_copter(vehicle) -> None:
    port, _ = vehicle
    link = MavlinkConnection(f"udpin:127.0.0.1:{port}", heartbeat_timeout=10.0)
    link.connect()
    try:
        assert link.vehicle_class == VehicleClass.COPTER
        assert link.mav_type == mavutil.mavlink.MAV_TYPE_QUADROTOR
    finally:
        link.close()


def test_telemetry_populates_from_real_messages(vehicle) -> None:
    port, _ = vehicle
    link = MavlinkConnection(f"udpin:127.0.0.1:{port}", heartbeat_timeout=10.0)
    link.connect()
    monitor = TelemetryMonitor(link, link_loss_timeout_s=5.0)
    monitor.start()
    try:
        assert monitor.wait_for_telemetry(timeout=5.0)
        state = monitor.snapshot()
        assert state.connected
        assert state.mode == "GUIDED"
        assert state.gps_satellites == 11
        assert state.gps_fix_type == 3
        assert state.battery_voltage == pytest.approx(12.1, abs=0.05)
        assert state.relative_alt_m == pytest.approx(3.2, abs=0.05)
        assert monitor.link_healthy()
    finally:
        monitor.stop()
        link.close()


def test_arm_state_change_is_observed_while_monitor_runs(vehicle) -> None:
    """The regression test for the two-readers race.

    Previously wait_for_arm() did its own recv_match() while the telemetry
    thread was also reading, so it usually missed the heartbeat and timed out.
    """
    port, v = vehicle
    link = MavlinkConnection(f"udpin:127.0.0.1:{port}", heartbeat_timeout=10.0)
    link.connect()
    monitor = TelemetryMonitor(link, link_loss_timeout_s=5.0)
    monitor.start()
    try:
        assert monitor.wait_for_armed(False, timeout=5.0)
        v.armed = True
        assert monitor.wait_for_armed(True, timeout=5.0)
        assert monitor.snapshot().armed
    finally:
        monitor.stop()
        link.close()


def test_mode_wait_succeeds_with_monitor_running(vehicle) -> None:
    port, v = vehicle
    link = MavlinkConnection(f"udpin:127.0.0.1:{port}", heartbeat_timeout=10.0)
    link.connect()
    monitor = TelemetryMonitor(link, link_loss_timeout_s=5.0)
    monitor.start()
    try:
        assert monitor.wait_for_mode("GUIDED", timeout=5.0)
        v.custom_mode = 9  # LAND
        assert monitor.wait_for_mode("LAND", timeout=5.0)
    finally:
        monitor.stop()
        link.close()


def test_setpoints_serialise_over_a_real_link(vehicle) -> None:
    """Velocity setpoints must encode and send without raising on a live link."""
    port, _ = vehicle
    link = MavlinkConnection(f"udpin:127.0.0.1:{port}", heartbeat_timeout=10.0)
    link.connect()
    monitor = TelemetryMonitor(link, link_loss_timeout_s=5.0)
    monitor.start()
    fc = FlightController(link, monitor, command_rate_hz=0.0)
    try:
        for _ in range(10):
            assert fc.send_velocity_body(0.5, -0.25, 0.0) is True
        assert fc.hover() is True
    finally:
        monitor.stop()
        link.close()


def test_no_heartbeat_raises_actionable_error() -> None:
    port = _free_port()
    link = MavlinkConnection(f"udpin:127.0.0.1:{port}", heartbeat_timeout=0.5)
    with pytest.raises(RuntimeError, match="No MAVLink heartbeat"):
        link.connect()
    link.close()


def test_fixed_wing_vehicle_is_classified_as_plane() -> None:
    port = _free_port()
    v = FakeVehicle(port, mav_type=mavutil.mavlink.MAV_TYPE_FIXED_WING)
    v.start()
    try:
        link = MavlinkConnection(f"udpin:127.0.0.1:{port}", heartbeat_timeout=10.0)
        link.connect()
        monitor = TelemetryMonitor(link)
        monitor.start()
        try:
            fc = FlightController(link, monitor)
            assert link.vehicle_class == VehicleClass.PLANE
            assert not fc.supports_velocity_setpoints
        finally:
            monitor.stop()
            link.close()
    finally:
        v.stop()
