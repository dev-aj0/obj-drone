"""Vision-to-velocity mapping and lost-target / failsafe behaviour."""

from __future__ import annotations

import threading
import time

import pytest

from conftest import (
    FakeCamera,
    FakeTelemetry,
    RecordingFlightController,
    draw_bgr_box,
    make_frame,
)
from obj_drone.controller.mission import MissionConfig, MissionController, MissionPhase
from obj_drone.mavlink.connection import VehicleClass
from obj_drone.vision.tracker import TargetTracker, TrackingResult

RED_BGR = (0, 0, 255)


def _mission(config: MissionConfig | None = None, frames=None, vehicle=VehicleClass.COPTER):
    fc = RecordingFlightController(vehicle)
    telemetry = FakeTelemetry()
    camera = FakeCamera(frames or [])
    tracker = TargetTracker(640, 480)
    mission = MissionController(
        fc=fc,
        telemetry=telemetry,
        camera=camera,
        tracker=tracker,
        config=config or MissionConfig(),
        debug=None,
        stop_event=threading.Event(),
    )
    return mission, fc, telemetry


def _found_at(x: float, y: float) -> TrackingResult:
    return TrackingResult(found=True, center_x=x, center_y=y, bbox=(int(x), int(y), 10, 10))


def test_centered_target_commands_no_motion() -> None:
    mission, _, _ = _mission()
    vx, vy, vz = mission._velocity_for(_found_at(320, 240))
    assert (vx, vy, vz) == (0.0, 0.0, 0.0)


def test_deadband_suppresses_small_errors() -> None:
    mission, _, _ = _mission(MissionConfig(track_deadband=0.2, track_gain=2.0))
    # 10% of half-frame is inside a 20% deadband.
    vx, vy, _ = mission._velocity_for(_found_at(320 + 32, 240 + 24))
    assert (vx, vy) == (0.0, 0.0)


def test_target_right_of_center_commands_right() -> None:
    """Body-NED +y is right, and the target is to the right, so vy > 0."""
    mission, _, _ = _mission(MissionConfig(track_deadband=0.0, track_gain=2.0))
    _, vy, _ = mission._velocity_for(_found_at(640, 240))
    assert vy > 0


def test_target_low_in_nadir_image_commands_backward() -> None:
    """Nadir camera, image top = nose: a target low in frame is behind the drone."""
    mission, _, _ = _mission(
        MissionConfig(camera_orientation="down", track_deadband=0.0, track_gain=2.0)
    )
    vx, _, vz = mission._velocity_for(_found_at(320, 480))
    assert vx < 0
    assert vz == 0.0


def test_forward_camera_maps_vertical_error_to_climb_rate() -> None:
    mission, _, _ = _mission(
        MissionConfig(camera_orientation="forward", track_deadband=0.0, track_gain=2.0)
    )
    vx, vy, vz = mission._velocity_for(_found_at(320, 480))
    assert vx == 0.0
    assert vy == 0.0
    # Target below centre -> descend -> vz positive (NED down).
    assert vz > 0


def test_velocity_is_clamped_to_limits() -> None:
    mission, _, _ = _mission(
        MissionConfig(
            track_gain=100.0,
            track_deadband=0.0,
            max_horizontal_speed_m_s=1.5,
        )
    )
    vx, vy, _ = mission._velocity_for(_found_at(640, 480))
    assert abs(vx) == pytest.approx(1.5)
    assert abs(vy) == pytest.approx(1.5)


def test_inversion_flags_flip_signs() -> None:
    cfg = MissionConfig(track_deadband=0.0, track_gain=2.0)
    normal, _, _ = _mission(cfg)
    base_vx, base_vy, _ = normal._velocity_for(_found_at(640, 480))

    flipped, _, _ = _mission(
        MissionConfig(
            track_deadband=0.0, track_gain=2.0, invert_lateral=True, invert_longitudinal=True
        )
    )
    inv_vx, inv_vy, _ = flipped._velocity_for(_found_at(640, 480))
    assert inv_vx == pytest.approx(-base_vx)
    assert inv_vy == pytest.approx(-base_vy)


def test_never_drifts_before_first_acquisition() -> None:
    """No target has ever been seen — hold still, do not command velocity."""
    mission, fc, _ = _mission()
    for _ in range(50):
        mission._apply_tracking(TrackingResult(found=False, center_x=0, center_y=0))
    assert fc.velocities == []
    assert fc.hovers == 50


def test_lost_target_hovers_during_grace_then_acts() -> None:
    cfg = MissionConfig(lost_target_grace_frames=3, lost_target_action="hover")
    mission, fc, _ = _mission(cfg)
    mission._apply_tracking(_found_at(320, 240))
    assert fc.velocities  # acquired

    lost = TrackingResult(found=False, center_x=0, center_y=0)
    for _ in range(5):
        mission._apply_tracking(lost)
    assert fc.hovers == 5


def test_lost_target_land_action_lands() -> None:
    cfg = MissionConfig(lost_target_grace_frames=2, lost_target_action="land")
    mission, fc, _ = _mission(cfg)
    mission._apply_tracking(_found_at(320, 240))
    lost = TrackingResult(found=False, center_x=0, center_y=0)
    mission._apply_tracking(lost)
    mission._apply_tracking(lost)
    assert "LAND" in fc.modes
    assert mission.phase is MissionPhase.LANDING


def test_link_loss_triggers_configured_failsafe() -> None:
    mission, fc, telemetry = _mission(MissionConfig(link_loss_action="rtl"))
    telemetry.set_healthy(False)
    assert mission._check_link_health() is False
    assert "RTL" in fc.modes
    assert mission.phase is MissionPhase.FAILSAFE


def test_failsafe_loiter_action() -> None:
    mission, fc, _ = _mission()
    mission.trigger_failsafe("test", "loiter")
    assert "LOITER" in fc.modes


def test_fixed_wing_is_rejected() -> None:
    mission, _, _ = _mission(vehicle=VehicleClass.PLANE)
    with pytest.raises(RuntimeError, match="ignores SET_POSITION_TARGET_LOCAL_NED"):
        mission.check_vehicle_supported()


def test_copter_is_accepted() -> None:
    mission, _, _ = _mission(vehicle=VehicleClass.COPTER)
    mission.check_vehicle_supported()


def test_tracking_loop_stops_on_stop_event() -> None:
    frames = [draw_bgr_box(make_frame(), RED_BGR, 300, 220, 40, 40) for _ in range(3)]
    mission, fc, _ = _mission(MissionConfig(control_rate_hz=1000.0), frames=frames)
    mission.request_stop()
    mission.track_target_loop(source="color")
    assert fc.velocities == []


def test_tracking_loop_follows_color_target() -> None:
    frames = [draw_bgr_box(make_frame(), RED_BGR, 500, 220, 40, 40) for _ in range(3)]
    mission, fc, _ = _mission(
        MissionConfig(control_rate_hz=1000.0, track_deadband=0.0, track_gain=2.0),
        frames=frames,
    )
    mission.track_target_loop(source="color")
    assert len(fc.velocities) == 3
    # Target is right of centre, so every command steers right.
    assert all(v[1] > 0 for v in fc.velocities)


def test_tracking_loop_falls_back_to_color_without_detector() -> None:
    frames = [draw_bgr_box(make_frame(), RED_BGR, 300, 220, 40, 40)]
    mission, fc, _ = _mission(
        MissionConfig(control_rate_hz=1000.0, max_camera_failures=2), frames=frames
    )
    assert mission.tracker.detector is None
    mission.track_target_loop(source="detector")  # must not raise
    assert fc.velocities or fc.hovers


def test_dead_camera_triggers_failsafe_instead_of_spinning() -> None:
    """A camera that stops delivering frames must not leave the loop hovering forever."""
    mission, fc, _ = _mission(
        MissionConfig(
            control_rate_hz=1000.0, max_camera_failures=3, link_loss_action="rtl"
        ),
        frames=[],
    )
    mission.track_target_loop(source="color")
    assert mission.phase is MissionPhase.FAILSAFE
    assert "RTL" in fc.modes


# ------------------------------------- shared velocity maths (used by the viewer)
def test_velocity_for_matches_controller() -> None:
    """The viewer and the controller must agree exactly, or the bench test lies."""
    from obj_drone.controller.mission import velocity_for

    cfg = MissionConfig(camera_orientation="forward", track_deadband=0.0, track_gain=2.0)
    mission, _, _ = _mission(cfg)
    result = _found_at(500, 400)
    assert velocity_for(mission.tracker, result, cfg) == mission._velocity_for(result)


def test_velocity_for_forward_right_of_centre_is_positive_vy() -> None:
    """The single most important sign: target right -> move right."""
    from obj_drone.controller.mission import velocity_for
    from obj_drone.vision.tracker import TargetTracker

    tracker = TargetTracker(640, 480)
    cfg = MissionConfig(camera_orientation="forward", track_deadband=0.0, track_gain=2.0)
    vx, vy, vz = velocity_for(tracker, _found_at(640, 240), cfg)
    assert vy > 0
    assert vx == 0.0


def test_velocity_for_respects_invert_flags() -> None:
    from obj_drone.controller.mission import velocity_for
    from obj_drone.vision.tracker import TargetTracker

    tracker = TargetTracker(640, 480)
    base = MissionConfig(camera_orientation="forward", track_deadband=0.0, track_gain=2.0)
    flipped = MissionConfig(
        camera_orientation="forward", track_deadband=0.0, track_gain=2.0, invert_lateral=True
    )
    _, vy_a, _ = velocity_for(tracker, _found_at(640, 240), base)
    _, vy_b, _ = velocity_for(tracker, _found_at(640, 240), flipped)
    assert vy_a == pytest.approx(-vy_b)


# ------------------------------------------------------------------ follow mode
def _follow_mission(land_after=0.0, **follow_kw):
    from obj_drone.controller.follow import FollowConfig

    cfg = MissionConfig(
        follow_enabled=True,
        follow=FollowConfig(**follow_kw),
        land_when_stationary_s=land_after,
    )
    return _mission(cfg)


def test_follow_mode_commands_station_keeping() -> None:
    mission, fc, tel = _follow_mission(follow_distance_m=3.0, follow_height_m=3.0)
    tel.state.relative_alt_m = 3.0
    # Small box = subject far away -> fly forward.
    mission._apply_tracking(
        TrackingResult(found=True, center_x=320, center_y=240, bbox=(300, 190, 40, 40))
    )
    assert fc.velocities
    assert fc.velocities[-1][0] > 0


def test_follow_mode_climbs_when_below_setpoint() -> None:
    mission, fc, tel = _follow_mission(follow_height_m=5.0)
    tel.state.relative_alt_m = 1.0
    mission._apply_tracking(
        TrackingResult(found=True, center_x=320, center_y=240, bbox=(300, 190, 40, 100))
    )
    # +vz is DOWN, so climbing is negative.
    assert fc.velocities[-1][2] < 0


def test_follow_lands_after_subject_stands_still() -> None:
    mission, fc, tel = _follow_mission(land_after=0.01)
    tel.state.relative_alt_m = 3.0
    tel.state.vx = 0.0
    tel.state.vy = 0.0
    target = TrackingResult(found=True, center_x=320, center_y=240, bbox=(300, 190, 40, 100))
    for _ in range(5):
        mission._apply_tracking(target)
        time.sleep(0.01)
    assert "LAND" in fc.modes
    assert mission.phase is MissionPhase.LANDING


def test_follow_does_not_land_while_subject_is_moving() -> None:
    mission, fc, tel = _follow_mission(land_after=0.01)
    tel.state.relative_alt_m = 3.0
    tel.state.vx = 2.0  # drone translating -> subject is walking
    tel.state.vy = 0.0
    target = TrackingResult(found=True, center_x=320, center_y=240, bbox=(300, 190, 40, 100))
    for _ in range(5):
        mission._apply_tracking(target)
        time.sleep(0.01)
    assert "LAND" not in fc.modes


def test_follow_never_lands_when_disabled() -> None:
    mission, fc, tel = _follow_mission(land_after=0.0)
    tel.state.relative_alt_m = 3.0
    target = TrackingResult(found=True, center_x=320, center_y=240, bbox=(300, 190, 40, 100))
    for _ in range(10):
        mission._apply_tracking(target)
        time.sleep(0.005)
    assert "LAND" not in fc.modes


# --------------------------------------------------------------- pilot override
def test_pilot_override_stops_the_loop() -> None:
    """Switching out of GUIDED must make the companion stand down."""
    frames = [draw_bgr_box(make_frame(), RED_BGR, 300, 220, 40, 40) for _ in range(50)]
    mission, fc, tel = _mission(MissionConfig(control_rate_hz=1000.0), frames=frames)
    tel.state.mode = "LOITER"
    mission.track_target_loop(source="color")
    assert mission.pilot_override is True
    assert fc.velocities == []
    assert fc.modes == [], "must not command LAND on top of the pilot"


def test_no_override_while_in_guided() -> None:
    frames = [draw_bgr_box(make_frame(), RED_BGR, 500, 220, 40, 40) for _ in range(3)]
    mission, fc, tel = _mission(
        MissionConfig(control_rate_hz=1000.0, track_deadband=0.0), frames=frames
    )
    tel.state.mode = "GUIDED"
    mission.track_target_loop(source="color")
    assert mission.pilot_override is False
    assert fc.velocities


def test_override_can_be_disabled() -> None:
    frames = [draw_bgr_box(make_frame(), RED_BGR, 500, 220, 40, 40) for _ in range(3)]
    mission, fc, tel = _mission(
        MissionConfig(
            control_rate_hz=1000.0, track_deadband=0.0, abort_on_mode_change=False
        ),
        frames=frames,
    )
    tel.state.mode = "ALT_HOLD"
    mission.track_target_loop(source="color")
    assert mission.pilot_override is False
