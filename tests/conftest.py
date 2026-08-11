"""Shared fakes so the suite runs on any machine — no FC, no camera, no model."""

from __future__ import annotations

import threading
import types

import numpy as np
import pytest

from obj_drone.mavlink.connection import VehicleClass

COPTER_MODES = {
    "STABILIZE": 0,
    "GUIDED": 4,
    "LOITER": 5,
    "RTL": 6,
    "LAND": 9,
}

PLANE_MODES = {
    "MANUAL": 0,
    "FBWA": 5,
    "GUIDED": 15,
    "LOITER": 12,
    "RTL": 11,
}

QUADPLANE_MODES = {
    **PLANE_MODES,
    "QLOITER": 19,
    "QLAND": 20,
    "QRTL": 21,
}


class FakeMav:
    """Stands in for pymavlink's message encoder/sender."""

    def __init__(self) -> None:
        self.encoded: list[dict] = []

    def set_position_target_local_ned_encode(self, *args):
        msg = types.SimpleNamespace(args=args)
        self.encoded.append({"args": args})
        return msg


class FakeMaster:
    def __init__(self, modes: dict[str, int]) -> None:
        self.target_system = 1
        self.target_component = 1
        self.mav = FakeMav()
        self._modes = modes
        self.sent_modes: list[int] = []

    def mode_mapping(self):
        return self._modes

    def set_mode(self, mode_id: int) -> None:
        self.sent_modes.append(mode_id)


class FakeLink:
    """Minimal MavlinkConnection stand-in."""

    def __init__(
        self,
        vehicle_class: str = VehicleClass.COPTER,
        modes: dict[str, int] | None = None,
    ) -> None:
        self.vehicle_class = vehicle_class
        if modes is None:
            modes = {
                VehicleClass.COPTER: COPTER_MODES,
                VehicleClass.PLANE: PLANE_MODES,
                VehicleClass.QUADPLANE: QUADPLANE_MODES,
            }.get(vehicle_class, COPTER_MODES)
        self.master = FakeMaster(modes)
        self.commands: list[tuple] = []
        self.sent: list[object] = []
        self.mode_requests: list[int] = []

    def mode_mapping(self):
        return self.master.mode_mapping()

    def set_mode_raw(self, mode_id: int) -> None:
        self.mode_requests.append(mode_id)
        self.master.set_mode(mode_id)

    def send_command_long(self, command: int, *params: float) -> None:
        self.commands.append((command, params))

    def send(self, msg: object) -> None:
        self.sent.append(msg)

    def mark_reader_active(self, active: bool) -> None:
        pass


class FakeTelemetry:
    """TelemetryMonitor stand-in with directly settable state."""

    def __init__(self, state=None) -> None:
        from obj_drone.mavlink.telemetry import VehicleState

        self.state = state or VehicleState()
        self._healthy = True
        self.wait_results: dict[str, bool] = {}

    def snapshot(self):
        from dataclasses import replace

        return replace(self.state)

    def link_healthy(self) -> bool:
        return self._healthy

    def set_healthy(self, healthy: bool) -> None:
        self._healthy = healthy

    def seconds_since_heartbeat(self) -> float:
        return 0.0 if self._healthy else 99.0

    def wait_for(self, predicate, timeout=10.0) -> bool:
        return bool(predicate(self.state))

    def wait_for_mode(self, mode_name: str, timeout: float = 10.0) -> bool:
        return self.wait_results.get("mode", self.state.mode == mode_name)

    def wait_for_armed(self, armed: bool = True, timeout: float = 10.0) -> bool:
        return self.wait_results.get("armed", self.state.armed == armed)

    def wait_for_altitude(self, target, tolerance=0.5, timeout=60.0) -> bool:
        return self.wait_results.get(
            "altitude", abs(self.state.relative_alt_m - target) <= tolerance
        )

    def wait_for_telemetry(self, timeout: float = 10.0) -> bool:
        return self.wait_results.get(
            "telemetry", self.state.have_gps and self.state.have_sys_status
        )


class FakeCamera:
    """Yields a fixed list of frames then returns None."""

    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames_list = frames
        self.index = 0
        self.stopped = False

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self.stopped = True

    def read(self):
        if self.index >= len(self.frames_list):
            return None
        frame = self.frames_list[self.index]
        self.index += 1
        return frame


class RecordingFlightController:
    """Records velocity commands and mode changes instead of flying."""

    def __init__(self, vehicle_class: str = VehicleClass.COPTER) -> None:
        self.vehicle_class = vehicle_class
        self.velocities: list[tuple[float, float, float]] = []
        self.modes: list[str] = []
        self.hovers = 0

    @property
    def supports_velocity_setpoints(self) -> bool:
        return self.vehicle_class in (VehicleClass.COPTER, VehicleClass.QUADPLANE)

    def resolve_mode(self, logical: str) -> str:
        # Copter mode names map straight through; that is what the real
        # FlightController does for this airframe.
        return logical

    def send_velocity_body(self, vx, vy, vz, yaw_rate=0.0) -> bool:
        self.velocities.append((vx, vy, vz))
        return True

    def hover(self) -> bool:
        self.hovers += 1
        return True

    def land(self) -> str:
        self.modes.append("LAND")
        return "LAND"

    def rtl(self) -> str:
        self.modes.append("RTL")
        return "RTL"

    def loiter(self) -> str:
        self.modes.append("LOITER")
        return "LOITER"


@pytest.fixture
def stop_event() -> threading.Event:
    return threading.Event()


def make_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """A black BGR frame."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def draw_bgr_box(
    frame: np.ndarray,
    color: tuple[int, int, int],
    x: int,
    y: int,
    w: int,
    h: int,
) -> np.ndarray:
    frame[y : y + h, x : x + w] = color
    return frame
