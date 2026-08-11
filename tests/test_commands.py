"""Mode resolution per airframe and setpoint rate limiting."""

from __future__ import annotations

import pytest
from pymavlink import mavutil

from conftest import PLANE_MODES, FakeLink, FakeTelemetry
from obj_drone.mavlink.commands import FlightController
from obj_drone.mavlink.connection import VehicleClass


def _fc(vehicle_class=VehicleClass.COPTER, command_rate_hz=10.0, modes=None):
    link = FakeLink(vehicle_class, modes)
    telemetry = FakeTelemetry()
    return FlightController(link, telemetry, command_rate_hz=command_rate_hz), link, telemetry


def test_copter_modes_resolve_directly() -> None:
    fc, _, _ = _fc(VehicleClass.COPTER)
    assert fc.resolve_mode("GUIDED") == "GUIDED"
    assert fc.resolve_mode("LAND") == "LAND"
    assert fc.resolve_mode("LOITER") == "LOITER"
    assert fc.resolve_mode("RTL") == "RTL"


def test_quadplane_prefers_vtol_modes() -> None:
    fc, _, _ = _fc(VehicleClass.QUADPLANE)
    assert fc.resolve_mode("LAND") == "QLAND"
    assert fc.resolve_mode("LOITER") == "QLOITER"
    assert fc.resolve_mode("RTL") == "QRTL"


def test_plane_has_no_land_mode_and_falls_back_to_rtl() -> None:
    """Plain ArduPlane has no LAND mode — LAND must not raise, it must degrade."""
    fc, _, _ = _fc(VehicleClass.PLANE)
    assert "LAND" not in PLANE_MODES
    assert fc.resolve_mode("LAND") == "RTL"


def test_unresolvable_mode_raises_with_available_modes() -> None:
    fc, _, _ = _fc(VehicleClass.COPTER, modes={"STABILIZE": 0})
    with pytest.raises(RuntimeError, match="No mode available for 'GUIDED'"):
        fc.resolve_mode("GUIDED")


def test_velocity_setpoint_support_by_airframe() -> None:
    assert _fc(VehicleClass.COPTER)[0].supports_velocity_setpoints
    assert _fc(VehicleClass.QUADPLANE)[0].supports_velocity_setpoints
    # Fixed-wing GUIDED ignores velocity targets.
    assert not _fc(VehicleClass.PLANE)[0].supports_velocity_setpoints
    assert not _fc(VehicleClass.UNKNOWN)[0].supports_velocity_setpoints


def test_set_mode_sends_correct_id() -> None:
    fc, link, _ = _fc(VehicleClass.COPTER)
    fc.set_mode("GUIDED")
    assert link.mode_requests == [4]


def test_set_mode_wait_raises_on_timeout() -> None:
    fc, _, telemetry = _fc(VehicleClass.COPTER)
    telemetry.wait_results["mode"] = False
    with pytest.raises(RuntimeError, match="did not enter GUIDED"):
        fc.set_mode("GUIDED", wait=True, timeout=0.01)


def test_arm_sends_component_arm_disarm() -> None:
    fc, link, _ = _fc()
    fc.arm()
    command, params = link.commands[-1]
    assert command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM
    assert params[0] == 1


def test_arm_force_uses_magic_value() -> None:
    fc, link, _ = _fc()
    fc.arm(force=True)
    assert link.commands[-1][1][1] == 21196


def test_arm_wait_failure_mentions_prearm() -> None:
    fc, _, telemetry = _fc()
    telemetry.wait_results["armed"] = False
    with pytest.raises(RuntimeError, match="pre-arm"):
        fc.arm(wait=True, timeout=0.01)


def test_quadplane_takeoff_uses_vtol_command() -> None:
    fc, link, telemetry = _fc(VehicleClass.QUADPLANE)
    telemetry.wait_results["mode"] = True
    fc.takeoff(5.0)
    commands = [c for c, _ in link.commands]
    assert mavutil.mavlink.MAV_CMD_NAV_VTOL_TAKEOFF in commands


def test_copter_takeoff_uses_nav_takeoff() -> None:
    fc, link, telemetry = _fc(VehicleClass.COPTER)
    telemetry.wait_results["mode"] = True
    fc.takeoff(3.0)
    command, params = link.commands[-1]
    assert command == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF
    assert params[6] == pytest.approx(3.0)


def test_rate_limiter_drops_rather_than_blocks() -> None:
    """Excess setpoints must be dropped — sleeping here would stall the vision loop."""
    fc, link, _ = _fc(command_rate_hz=1.0)
    assert fc.send_velocity_body(1.0, 0.0, 0.0) is True
    # Immediately after, still inside the 1 s interval.
    assert fc.send_velocity_body(1.0, 0.0, 0.0) is False
    assert len(link.sent) == 1


def test_zero_rate_disables_limiting() -> None:
    fc, link, _ = _fc(command_rate_hz=0.0)
    for _ in range(5):
        assert fc.send_velocity_body(1.0, 0.0, 0.0) is True
    assert len(link.sent) == 5


def test_velocity_body_uses_body_frame_and_ignores_position() -> None:
    fc, link, _ = _fc(command_rate_hz=0.0)
    fc.send_velocity_body(1.5, -0.5, 0.25)
    args = link.master.mav.encoded[-1]["args"]
    # set_position_target_local_ned(time, sys, comp, frame, mask, x,y,z, vx,vy,vz, ...)
    assert args[3] == mavutil.mavlink.MAV_FRAME_BODY_NED
    assert args[8:11] == (1.5, -0.5, 0.25)

    mask = args[4]
    assert mask & mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
    assert not mask & mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE


def test_hover_sends_zero_velocity_in_local_ned() -> None:
    fc, link, _ = _fc(command_rate_hz=0.0)
    fc.hover()
    args = link.master.mav.encoded[-1]["args"]
    assert args[3] == mavutil.mavlink.MAV_FRAME_LOCAL_NED
    assert args[8:11] == (0.0, 0.0, 0.0)


def test_position_setpoint_ignores_velocity() -> None:
    fc, link, _ = _fc(command_rate_hz=0.0)
    fc.send_position_ned(10.0, 5.0, -3.0)
    args = link.master.mav.encoded[-1]["args"]
    assert args[5:8] == (10.0, 5.0, -3.0)
    mask = args[4]
    assert mask & mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE
    assert not mask & mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
