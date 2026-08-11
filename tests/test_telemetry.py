"""Telemetry state cache, wait helpers, and preflight gating."""

from __future__ import annotations

import threading
import time
import types

import pytest
from pymavlink import mavutil

from conftest import FakeLink, FakeTelemetry
from obj_drone.controller.preflight import PreflightCheck, PreflightConfig
from obj_drone.mavlink.commands import FlightController
from obj_drone.mavlink.connection import VehicleClass, classify_vehicle
from obj_drone.mavlink.telemetry import TelemetryMonitor, VehicleState


def _msg(msg_type: str, src_system: int = 1, **fields):
    ns = types.SimpleNamespace(**fields)
    ns.get_type = lambda: msg_type
    ns.get_srcSystem = lambda: src_system
    return ns


class _ScriptedLink(FakeLink):
    """Feeds a fixed message list to the telemetry reader, then blocks."""

    def __init__(self, messages) -> None:
        super().__init__()
        self._messages = list(messages)
        self._index = 0

    def recv_match(self, type=None, blocking=False, timeout=None):
        if self._index < len(self._messages):
            msg = self._messages[self._index]
            self._index += 1
            return msg
        time.sleep(0.01)
        return None


def _run_monitor(messages, settle: float = 0.3) -> TelemetryMonitor:
    monitor = TelemetryMonitor(_ScriptedLink(messages), link_loss_timeout_s=5.0)
    monitor.start()
    deadline = time.monotonic() + settle
    while time.monotonic() < deadline:
        if monitor.snapshot().connected:
            break
        time.sleep(0.01)
    time.sleep(0.05)
    return monitor


def test_heartbeat_sets_mode_and_armed() -> None:
    hb = _msg(
        "HEARTBEAT",
        type=mavutil.mavlink.MAV_TYPE_QUADROTOR,
        autopilot=mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA,
        base_mode=mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
        | mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        custom_mode=4,
        system_status=4,
        mavlink_version=3,
    )
    monitor = _run_monitor([hb])
    try:
        state = monitor.snapshot()
        assert state.connected
        assert state.armed
        assert state.mode == "GUIDED"
    finally:
        monitor.stop()


def test_gps_raw_populates_satellites_and_fix() -> None:
    msg = _msg("GPS_RAW_INT", satellites_visible=11, fix_type=3)
    monitor = _run_monitor([msg])
    try:
        state = monitor.snapshot()
        assert state.gps_satellites == 11
        assert state.gps_fix_type == 3
        assert state.have_gps
    finally:
        monitor.stop()


def test_have_flags_distinguish_zero_from_missing() -> None:
    monitor = _run_monitor([])
    try:
        state = monitor.snapshot()
        assert state.gps_satellites == 0
        assert not state.have_gps
        assert not state.have_sys_status
        assert not state.have_position
    finally:
        monitor.stop()


def test_global_position_converts_units() -> None:
    msg = _msg(
        "GLOBAL_POSITION_INT",
        lat=475000000,
        lon=85000000,
        alt=120000,
        relative_alt=5500,
        vx=150,
        vy=-50,
        vz=25,
    )
    monitor = _run_monitor([msg])
    try:
        state = monitor.snapshot()
        assert state.lat == pytest.approx(47.5)
        assert state.relative_alt_m == pytest.approx(5.5)
        assert state.vx == pytest.approx(1.5)
        assert state.have_position
    finally:
        monitor.stop()


def test_sys_status_converts_millivolts() -> None:
    monitor = _run_monitor([_msg("SYS_STATUS", voltage_battery=11400)])
    try:
        assert monitor.snapshot().battery_voltage == pytest.approx(11.4)
    finally:
        monitor.stop()


def test_bad_data_is_ignored() -> None:
    monitor = _run_monitor([_msg("BAD_DATA")])
    try:
        assert not monitor.snapshot().connected
    finally:
        monitor.stop()


def test_wait_for_times_out_without_blocking_forever() -> None:
    monitor = TelemetryMonitor(_ScriptedLink([]))
    start = time.monotonic()
    assert monitor.wait_for(lambda s: s.armed, timeout=0.2) is False
    assert time.monotonic() - start < 1.0


def test_wait_for_returns_immediately_when_already_true() -> None:
    monitor = TelemetryMonitor(_ScriptedLink([]))
    monitor.state.mode = "GUIDED"
    assert monitor.wait_for_mode("GUIDED", timeout=0.1) is True


def test_wait_for_is_woken_by_state_change() -> None:
    """The waiter must not spin until timeout when the state changes."""
    monitor = TelemetryMonitor(_ScriptedLink([]))
    result: list[bool] = []

    def waiter() -> None:
        result.append(monitor.wait_for_armed(True, timeout=3.0))

    thread = threading.Thread(target=waiter)
    thread.start()
    time.sleep(0.05)
    with monitor._cond:
        monitor.state.armed = True
        monitor._cond.notify_all()
    thread.join(timeout=2.0)
    assert result == [True]


def test_link_healthy_expires_after_timeout() -> None:
    monitor = TelemetryMonitor(_ScriptedLink([]), link_loss_timeout_s=0.05)
    monitor.state.connected = True
    monitor.state.last_heartbeat_monotonic = time.monotonic()
    assert monitor.link_healthy()
    time.sleep(0.1)
    assert not monitor.link_healthy()


def test_classify_vehicle_covers_airframes() -> None:
    assert classify_vehicle(mavutil.mavlink.MAV_TYPE_QUADROTOR) == VehicleClass.COPTER
    assert classify_vehicle(mavutil.mavlink.MAV_TYPE_HEXAROTOR) == VehicleClass.COPTER
    assert classify_vehicle(mavutil.mavlink.MAV_TYPE_FIXED_WING) == VehicleClass.PLANE
    assert (
        classify_vehicle(mavutil.mavlink.MAV_TYPE_VTOL_TILTROTOR) == VehicleClass.QUADPLANE
    )
    assert classify_vehicle(mavutil.mavlink.MAV_TYPE_GROUND_ROVER) == VehicleClass.UNKNOWN


# ------------------------------------------------------------------- preflight
def _preflight(state: VehicleState, config: PreflightConfig | None = None):
    link = FakeLink(VehicleClass.COPTER)
    telemetry = FakeTelemetry(state)
    fc = FlightController(link, telemetry)
    check = PreflightCheck(fc, telemetry, config or PreflightConfig())
    return check


def _good_state() -> VehicleState:
    return VehicleState(
        mode="GUIDED",
        connected=True,
        gps_satellites=10,
        gps_fix_type=3,
        battery_voltage=12.0,
        have_gps=True,
        have_sys_status=True,
    )


def test_preflight_passes_on_healthy_vehicle() -> None:
    assert _preflight(_good_state()).run_detailed().ok


def test_preflight_fails_when_gps_never_received() -> None:
    """Regression: GPS_RAW_INT was never requested, so sats sat at 0 forever."""
    state = _good_state()
    state.have_gps = False
    state.gps_satellites = 0
    report = _preflight(state).run_detailed()
    assert not report.ok
    assert any("No GPS_RAW_INT" in e for e in report.errors)


def test_preflight_fails_on_too_few_satellites() -> None:
    state = _good_state()
    state.gps_satellites = 3
    report = _preflight(state).run_detailed()
    assert any("Insufficient GPS satellites" in e for e in report.errors)


def test_preflight_fails_without_3d_fix() -> None:
    state = _good_state()
    state.gps_fix_type = 1
    report = _preflight(state).run_detailed()
    assert any("No 3D GPS fix" in e for e in report.errors)


def test_preflight_fails_on_low_battery() -> None:
    state = _good_state()
    state.battery_voltage = 9.0
    report = _preflight(state).run_detailed()
    assert any("Battery voltage low" in e for e in report.errors)


def test_zero_battery_is_a_warning_not_an_error() -> None:
    """A disabled battery monitor should not be mistaken for a flat pack."""
    state = _good_state()
    state.battery_voltage = 0.0
    report = _preflight(state).run_detailed()
    assert report.ok
    assert any("0 V" in w for w in report.warnings)


def test_gps_check_can_be_disabled() -> None:
    state = _good_state()
    state.have_gps = False
    state.gps_satellites = 0
    config = PreflightConfig(min_gps_satellites=0, min_battery_voltage=0.0)
    assert _preflight(state, config).run_detailed().ok
