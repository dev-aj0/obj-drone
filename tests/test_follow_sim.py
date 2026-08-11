"""Closed-loop simulation: does the controller actually follow a walking person?

The unit tests check individual signs. These integrate the controller against a
simple kinematic model of the aircraft to answer the question that matters — put
a person in front of it and let them walk, does the drone converge on the
standoff distance and stay there?

The model is deliberately crude (velocity commands take effect immediately, no
wind, no attitude dynamics). It cannot predict real flight performance. What it
does catch is a controller that diverges, oscillates, or drives the wrong way —
which is exactly the class of bug that would cause a flyaway.
"""

from __future__ import annotations

import math

import pytest

from obj_drone.controller.follow import FollowConfig, focal_length_px, follow_velocity
from obj_drone.vision.tracker import NOT_FOUND, TrackingResult

W, H = 640, 480
DT = 1.0 / 15.0  # the Pi manages ~15 fps


class Sim:
    """Drone and person on a flat plane, with a forward-facing camera.

    Drone state: position (x, y), altitude, heading. Person: position (x, y).
    Axes are world NED-ish: +x north, +y east, heading 0 = facing +x.
    """

    def __init__(self, config: FollowConfig, drone=(0.0, 0.0), person=(1.0, 0.0), alt=3.0):
        self.cfg = config
        self.dx, self.dy = drone
        self.px, self.py = person
        self.alt = alt
        self.heading = 0.0
        self.focal = focal_length_px(H, config.camera_vfov_deg)

    # -- what the camera sees -------------------------------------------------
    def observe(self) -> TrackingResult:
        rx, ry = self.px - self.dx, self.py - self.dy
        # Rotate into the body frame: +forward, +right.
        fwd = rx * math.cos(self.heading) + ry * math.sin(self.heading)
        right = -rx * math.sin(self.heading) + ry * math.cos(self.heading)
        if fwd <= 0.3:
            return NOT_FOUND  # behind us or on top of us

        ground_range = math.hypot(fwd, right)
        slant = math.hypot(ground_range, self.alt)
        # Horizontal angle off the optical axis -> pixels.
        cx = W / 2.0 + self.focal * (right / fwd)
        if not (0 <= cx <= W):
            return NOT_FOUND  # outside the frame

        box_h = self.cfg.person_height_m * self.focal / slant
        return TrackingResult(
            found=True, center_x=cx, center_y=H / 2.0,
            bbox=(int(cx - 20), int(H / 2 - box_h / 2), 40, int(box_h)),
            label="person", confidence=0.9,
        )

    @property
    def range_m(self) -> float:
        return math.hypot(self.px - self.dx, self.py - self.dy)

    # -- one control step -----------------------------------------------------
    def step(self, person_vx=0.0, person_vy=0.0) -> None:
        cmd = follow_velocity(self.observe(), W, H, self.alt, self.cfg)

        self.heading += cmd.yaw_rate * DT
        # Body-frame velocity back into world axes.
        vx_w = cmd.vx * math.cos(self.heading) - cmd.vy * math.sin(self.heading)
        vy_w = cmd.vx * math.sin(self.heading) + cmd.vy * math.cos(self.heading)
        self.dx += vx_w * DT
        self.dy += vy_w * DT
        self.alt -= cmd.vz * DT  # +vz is DOWN

        self.px += person_vx * DT
        self.py += person_vy * DT

    def run(self, seconds: float, person_vx=0.0, person_vy=0.0) -> None:
        for _ in range(int(seconds / DT)):
            self.step(person_vx, person_vy)


def _cfg(**kw) -> FollowConfig:
    base = dict(follow_distance_m=3.0, follow_height_m=3.0, camera_vfov_deg=35.5,
                min_distance_m=2.0, lateral_mode="yaw")
    base.update(kw)
    return FollowConfig(**base)


# --------------------------------------------------------------- convergence
def test_converges_to_standoff_from_too_far() -> None:
    sim = Sim(_cfg(), drone=(0.0, 0.0), person=(12.0, 0.0))
    sim.run(40)
    assert sim.range_m == pytest.approx(3.0, abs=0.6), f"settled at {sim.range_m:.2f} m"


def test_backs_off_when_starting_too_close() -> None:
    sim = Sim(_cfg(), drone=(0.0, 0.0), person=(1.2, 0.0))
    sim.run(30)
    assert sim.range_m > 1.9, f"stayed at {sim.range_m:.2f} m — inside the floor"


def test_climbs_to_the_configured_height() -> None:
    sim = Sim(_cfg(follow_height_m=4.0), person=(3.0, 0.0), alt=0.2)
    sim.run(30)
    assert sim.alt == pytest.approx(4.0, abs=0.4)


# -------------------------------------------------------------------- following
def test_follows_a_person_walking_straight_away() -> None:
    sim = Sim(_cfg(), drone=(0.0, 0.0), person=(3.0, 0.0))
    sim.run(30, person_vx=1.0)  # brisk walk, 1 m/s
    assert sim.range_m < 5.0, f"fell behind to {sim.range_m:.2f} m"
    assert sim.observe().found, "lost sight of the subject"


def test_turns_to_follow_a_person_walking_sideways() -> None:
    """The whole point of yaw control: the drone should rotate to face them."""
    sim = Sim(_cfg(), drone=(0.0, 0.0), person=(3.0, 0.0))
    start_heading = sim.heading
    sim.run(25, person_vy=0.8)  # walks east, across the drone's view
    assert abs(sim.heading - start_heading) > 0.5, "drone never turned"
    assert sim.observe().found, "lost them out of frame"
    assert sim.range_m < 6.0


def test_follows_a_person_walking_a_curve() -> None:
    sim = Sim(_cfg(), drone=(0.0, 0.0), person=(3.0, 0.0))
    for i in range(int(35 / DT)):
        angle = i * DT * 0.25
        sim.step(person_vx=0.8 * math.cos(angle), person_vy=0.8 * math.sin(angle))
    assert sim.observe().found, "lost the subject on the curve"
    assert sim.range_m < 6.0, f"drifted to {sim.range_m:.2f} m"


def test_only_yawing_modes_turn_to_face_the_subject() -> None:
    """The point of yaw is heading, not range.

    Simulation showed pure yaw actually holds a *looser* range than strafe on a
    sideways walker, because it can only close along its own rotating heading.
    What it buys is that the aircraft points at the subject, so "3 m behind
    them" keeps meaning something and the camera holds them centred.
    """
    yaw = Sim(_cfg(lateral_mode="yaw"), person=(3.0, 0.0))
    strafe = Sim(_cfg(lateral_mode="strafe"), person=(3.0, 0.0))
    both = Sim(_cfg(lateral_mode="both"), person=(3.0, 0.0))
    for _ in range(int(25 / DT)):
        for sim in (yaw, strafe, both):
            sim.step(person_vy=1.0)

    assert abs(yaw.heading) > 0.5, "yaw mode must rotate toward the subject"
    assert abs(both.heading) > 0.5, "both mode must rotate toward the subject"
    assert strafe.heading == pytest.approx(0.0), "strafe mode must not rotate"
    # Whatever the mode, the subject must stay in frame.
    for sim in (yaw, strafe, both):
        assert sim.observe().found


# ------------------------------------------------------------------- stability
def test_holds_station_without_oscillating() -> None:
    """A stationary subject must not make the drone hunt back and forth."""
    sim = Sim(_cfg(), person=(3.0, 0.0))
    sim.run(20)
    ranges = []
    for _ in range(int(10 / DT)):
        sim.step()
        ranges.append(sim.range_m)
    assert max(ranges) - min(ranges) < 0.5, "oscillating around the setpoint"


def test_never_closes_inside_the_minimum_distance() -> None:
    """The safety floor must hold through a whole run, not just at one instant."""
    sim = Sim(_cfg(follow_distance_m=2.5), drone=(0.0, 0.0), person=(6.0, 0.0))
    worst = 99.0
    for _ in range(int(45 / DT)):
        # Person walks straight at the drone, then stops.
        sim.step(person_vx=-0.5 if sim.range_m > 2.0 else 0.0)
        worst = min(worst, sim.range_m)
    assert worst > 1.5, f"closed to {worst:.2f} m — inside the safety floor"


def test_speeds_stay_within_configured_limits() -> None:
    cfg = _cfg(max_horizontal_speed_m_s=1.5, max_yaw_rate_deg_s=30.0)
    sim = Sim(cfg, drone=(0.0, 0.0), person=(20.0, 8.0))
    for _ in range(int(20 / DT)):
        cmd = follow_velocity(sim.observe(), W, H, sim.alt, cfg)
        assert abs(cmd.vx) <= 1.5 + 1e-9
        assert abs(cmd.yaw_rate) <= math.radians(30.0) + 1e-9
        sim.step()
