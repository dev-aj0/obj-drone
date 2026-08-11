"""Follow geometry, distance estimation, and stationary detection."""

from __future__ import annotations

import math

import pytest

from obj_drone.controller.follow import (
    FollowConfig,
    StationaryDetector,
    estimate_distance_m,
    focal_length_px,
    follow_velocity,
)
from obj_drone.vision.tracker import NOT_FOUND, TrackingResult

W, H = 640, 480


def _person(cx=W / 2, cy=H / 2, box_h=100, box_w=40) -> TrackingResult:
    return TrackingResult(
        found=True,
        center_x=cx,
        center_y=cy,
        bbox=(int(cx - box_w / 2), int(cy - box_h / 2), box_w, box_h),
        label="person",
        confidence=0.9,
    )


# ------------------------------------------------------------------- geometry
def test_focal_length_from_fov() -> None:
    # A 90 deg vfov puts the focal length at exactly half the frame height.
    assert focal_length_px(480, 90.0) == pytest.approx(240.0)


def test_focal_length_rejects_impossible_fov() -> None:
    for bad in (0.0, -10.0, 180.0, 200.0):
        with pytest.raises(ValueError):
            focal_length_px(480, bad)


def test_distance_halves_when_person_appears_twice_as_tall() -> None:
    near = estimate_distance_m(200, H, 1.7, 48.0)
    far = estimate_distance_m(100, H, 1.7, 48.0)
    assert far == pytest.approx(2 * near)


def test_distance_matches_pinhole_by_hand() -> None:
    # focal = (480/2)/tan(24 deg); distance = 1.7 * focal / 120
    focal = 240.0 / math.tan(math.radians(24.0))
    assert estimate_distance_m(120, H, 1.7, 48.0) == pytest.approx(1.7 * focal / 120)


def test_distance_none_for_tiny_boxes() -> None:
    assert estimate_distance_m(4, H, 1.7, 48.0) is None


# -------------------------------------------------------------------- control
def test_centred_subject_at_correct_distance_and_height_is_stationary() -> None:
    cfg = FollowConfig(follow_distance_m=3.0, follow_height_m=3.0)
    focal = focal_length_px(H, cfg.camera_vfov_deg)
    # 3 m horizontal at 3 m altitude is a 4.243 m slant range.
    slant = math.hypot(3.0, 3.0)
    box_h = cfg.person_height_m * focal / slant
    cmd = follow_velocity(_person(box_h=box_h), W, H, 3.0, cfg)
    assert cmd.vx == pytest.approx(0.0, abs=0.05)
    assert cmd.vy == pytest.approx(0.0)
    assert cmd.vz == pytest.approx(0.0)


def test_subject_too_far_commands_forward() -> None:
    cfg = FollowConfig(follow_distance_m=3.0)
    # Small box = far away.
    cmd = follow_velocity(_person(box_h=40), W, H, 3.0, cfg)
    assert cmd.distance_m > 3.0
    assert cmd.vx > 0


def test_subject_too_close_commands_backward() -> None:
    cfg = FollowConfig(follow_distance_m=5.0, min_distance_m=0.5)
    cmd = follow_velocity(_person(box_h=300), W, H, 3.0, cfg)
    assert cmd.distance_m < 5.0
    assert cmd.vx < 0


def test_hard_minimum_distance_overrides_setpoint() -> None:
    """Never close inside min_distance_m — there is a person there."""
    cfg = FollowConfig(follow_distance_m=0.2, min_distance_m=2.0)
    # box_h=700 puts the subject at ~1.3 m, well inside the 2 m hard floor.
    cmd = follow_velocity(_person(box_h=700), W, H, 3.0, cfg)
    assert cmd.too_close is True
    assert cmd.vx < 0


def test_yaw_only_mode_rotates_without_strafing() -> None:
    cfg = FollowConfig(lateral_mode="yaw")
    right = follow_velocity(_person(cx=W - 20), W, H, 3.0, cfg)
    left = follow_velocity(_person(cx=20), W, H, 3.0, cfg)
    assert right.yaw_rate > 0 and right.vy == 0.0
    assert left.yaw_rate < 0 and left.vy == 0.0


def test_default_mode_turns_toward_the_subject() -> None:
    """Whatever the default is, it must rotate toward the subject."""
    right = follow_velocity(_person(cx=W - 20), W, H, 3.0, FollowConfig())
    left = follow_velocity(_person(cx=20), W, H, 3.0, FollowConfig())
    assert right.yaw_rate > 0
    assert left.yaw_rate < 0


def test_strafe_mode_slides_instead_of_turning() -> None:
    cfg = FollowConfig(lateral_mode="strafe")
    right = follow_velocity(_person(cx=W - 20), W, H, 3.0, cfg)
    left = follow_velocity(_person(cx=20), W, H, 3.0, cfg)
    assert right.vy > 0 and left.vy < 0
    assert right.yaw_rate == 0.0 and left.yaw_rate == 0.0


def test_both_mode_yaws_and_strafes() -> None:
    cfg = FollowConfig(lateral_mode="both")
    cmd = follow_velocity(_person(cx=W - 20), W, H, 3.0, cfg)
    assert cmd.yaw_rate > 0 and cmd.vy > 0
    # Strafe is deliberately reduced in this mode.
    strafe_only = follow_velocity(
        _person(cx=W - 20), W, H, 3.0, FollowConfig(lateral_mode="strafe")
    )
    assert abs(cmd.vy) < abs(strafe_only.vy)


def test_yaw_rate_is_clamped() -> None:
    cfg = FollowConfig(yaw_gain=100.0, max_yaw_rate_deg_s=30.0)
    cmd = follow_velocity(_person(cx=W - 1), W, H, 3.0, cfg)
    assert abs(cmd.yaw_rate) <= math.radians(30.0) + 1e-9


# ------------------------------------------------------- slant vs ground range
def test_ground_range_removes_altitude() -> None:
    """Slant range equal to altitude means directly overhead, zero standoff."""
    from obj_drone.controller.follow import ground_range_m

    assert ground_range_m(3.0, 3.0) == pytest.approx(0.0)
    assert ground_range_m(5.0, 3.0) == pytest.approx(4.0)
    assert ground_range_m(2.0, 3.0) == pytest.approx(0.0)  # clamped, not NaN
    assert ground_range_m(5.0, 0.0) == pytest.approx(5.0)


def test_overhead_subject_is_treated_as_too_close() -> None:
    """Regression: hovering on someone's head used to satisfy a 3 m setpoint."""
    cfg = FollowConfig(follow_distance_m=3.0, follow_height_m=3.0, min_distance_m=2.0)
    focal = focal_length_px(H, cfg.camera_vfov_deg)
    # Box height for a 3.0 m SLANT range while flying at 3.0 m altitude.
    cmd = follow_velocity(
        _person(box_h=cfg.person_height_m * focal / 3.0), W, H, 3.0, cfg
    )
    assert cmd.distance_m == pytest.approx(0.0, abs=0.1)
    assert cmd.too_close is True
    assert cmd.vx < 0, "must retreat when directly above the subject"


def test_retreat_outruns_a_walking_person() -> None:
    """At the plain distance gain the drone backed off slower than a walk."""
    cfg = FollowConfig(min_distance_m=2.0)
    focal = focal_length_px(H, cfg.camera_vfov_deg)
    # 1.5 m horizontal at 0 m altitude -> inside the floor but not on top.
    cmd = follow_velocity(
        _person(box_h=cfg.person_height_m * focal / 1.5), W, H, 0.0, cfg
    )
    assert cmd.too_close is True
    assert abs(cmd.vx) >= 0.8, f"retreat of {cmd.vx:.2f} m/s is slower than walking"


def test_lateral_deadband_suppresses_small_errors() -> None:
    cfg = FollowConfig(lateral_deadband=0.3)
    cmd = follow_velocity(_person(cx=W / 2 + 20), W, H, 3.0, cfg)
    assert cmd.vy == 0.0


def test_too_low_commands_climb() -> None:
    """+vz is DOWN, so climbing is negative."""
    cfg = FollowConfig(follow_height_m=5.0)
    cmd = follow_velocity(_person(), W, H, 2.0, cfg)
    assert cmd.vz < 0


def test_too_high_commands_descend() -> None:
    cfg = FollowConfig(follow_height_m=2.0)
    cmd = follow_velocity(_person(), W, H, 6.0, cfg)
    assert cmd.vz > 0


def test_altitude_held_even_without_a_target() -> None:
    """Losing the subject must not mean losing altitude control."""
    cfg = FollowConfig(follow_height_m=5.0)
    cmd = follow_velocity(NOT_FOUND, W, H, 2.0, cfg)
    assert cmd.vx == 0.0 and cmd.vy == 0.0
    assert cmd.vz < 0


def test_speeds_are_clamped() -> None:
    cfg = FollowConfig(
        follow_distance_m=50.0, distance_gain=100.0, lateral_gain=100.0,
        altitude_gain=100.0, max_horizontal_speed_m_s=1.5, max_vertical_speed_m_s=0.8,
        follow_height_m=50.0,
    )
    cmd = follow_velocity(_person(cx=W - 5, box_h=20), W, H, 1.0, cfg)
    assert abs(cmd.vx) <= 1.5 and abs(cmd.vy) <= 1.5
    assert abs(cmd.vz) <= 0.8


def test_invert_flags_flip_signs() -> None:
    base = FollowConfig()
    flip = FollowConfig(invert_lateral=True, invert_longitudinal=True)
    a = follow_velocity(_person(cx=W - 20, box_h=40), W, H, 3.0, base)
    b = follow_velocity(_person(cx=W - 20, box_h=40), W, H, 3.0, flip)
    assert a.vy == pytest.approx(-b.vy)
    assert a.vx == pytest.approx(-b.vx)


# ----------------------------------------------------------------- stationary
def test_stationary_triggers_after_timeout() -> None:
    det = StationaryDetector(seconds=10.0)
    t = 1000.0
    assert det.update(_person(), 0.0, t) is False
    assert det.update(_person(), 0.0, t + 5) is False
    assert det.update(_person(), 0.0, t + 10.1) is True


def test_walking_subject_never_triggers() -> None:
    """A tracked walker stays centred in frame — ground speed is the giveaway."""
    det = StationaryDetector(seconds=5.0, max_ground_speed_m_s=0.3)
    t = 1000.0
    for i in range(30):
        # Perfectly centred every frame, but the drone is translating at 1.5 m/s.
        assert det.update(_person(), 1.5, t + i) is False


def test_image_drift_alone_prevents_trigger() -> None:
    det = StationaryDetector(seconds=3.0, max_image_drift_px_s=10.0)
    t = 1000.0
    det.update(_person(cx=100), 0.0, t)
    for i in range(1, 10):
        # Drifting 100 px/s across the frame while the drone is still.
        assert det.update(_person(cx=100 + i * 100), 0.0, t + i) is False


def test_timer_resets_when_subject_moves_again() -> None:
    det = StationaryDetector(seconds=5.0)
    t = 1000.0
    det.update(_person(), 0.0, t)
    det.update(_person(), 0.0, t + 4)
    det.update(_person(), 2.0, t + 5)      # moved
    assert det.update(_person(), 0.0, t + 9) is False   # timer restarted
    assert det.still_for < 5.0


def test_losing_subject_resets_timer() -> None:
    det = StationaryDetector(seconds=5.0)
    t = 1000.0
    det.update(_person(), 0.0, t)
    det.update(NOT_FOUND, 0.0, t + 2)
    assert det.update(_person(), 0.0, t + 6) is False


def test_still_for_reports_elapsed() -> None:
    det = StationaryDetector(seconds=100.0)
    assert det.still_for == 0.0
