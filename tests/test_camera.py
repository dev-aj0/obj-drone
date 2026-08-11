"""Camera construction and autofocus configuration."""

from __future__ import annotations

import pytest

from obj_drone.config import load_config, vision_config_from_dict
from obj_drone.paths import project_root
from obj_drone.vision.camera import FOCUS_MODES, PiCamera, create_camera


class FakePicam:
    """Stands in for a Picamera2 object with configurable control support."""

    def __init__(self, controls: dict | None = None) -> None:
        self.camera_controls = (
            controls
            if controls is not None
            else {"AfMode": (0, 2, 0), "LensPosition": (0.0, 32.0, 1.0)}
        )
        self.applied: dict = {}
        self.cycles = 0

    def set_controls(self, ctrls: dict) -> None:
        self.applied.update(ctrls)

    def autofocus_cycle(self) -> None:
        self.cycles += 1


def _cam(focus_mode="continuous", lens_position=0.0, controls=None) -> tuple[PiCamera, FakePicam]:
    cam = PiCamera(640, 480, 30, focus_mode, lens_position)
    fake = FakePicam(controls)
    cam._camera = fake
    return cam, fake


def test_rejects_unknown_focus_mode() -> None:
    with pytest.raises(ValueError, match="Unknown focus_mode"):
        PiCamera(640, 480, 30, focus_mode="telekinesis")


def test_all_documented_focus_modes_are_accepted() -> None:
    for mode in FOCUS_MODES:
        PiCamera(640, 480, 30, focus_mode=mode)


def test_continuous_sets_af_mode() -> None:
    cam, fake = _cam("continuous")
    cam._configure_focus()
    assert "AfMode" in fake.applied
    assert cam.focus_applied == "continuous"


def test_auto_runs_a_single_focus_cycle() -> None:
    """One-shot focus avoids the lens hunting mid-flight."""
    cam, fake = _cam("auto")
    cam._configure_focus()
    assert fake.cycles == 1
    assert cam.focus_applied == "auto (one-shot)"


def test_manual_sets_lens_position() -> None:
    cam, fake = _cam("manual", lens_position=2.5)
    cam._configure_focus()
    assert fake.applied["LensPosition"] == pytest.approx(2.5)
    assert "0.40 m" in cam.focus_applied


def test_manual_at_zero_reports_infinity() -> None:
    cam, _ = _cam("manual", lens_position=0.0)
    cam._configure_focus()
    assert "infinity" in cam.focus_applied


def test_fixed_focus_camera_is_handled_gracefully() -> None:
    """A module with no AfMode control must not raise."""
    cam, fake = _cam("continuous", controls={"ExposureTime": (0, 1000, 100)})
    cam._configure_focus()
    assert cam.focus_applied == "fixed"
    assert fake.applied == {}


def test_focus_failure_does_not_crash_startup() -> None:
    class Exploding(FakePicam):
        def set_controls(self, ctrls):
            raise RuntimeError("lens jammed")

    cam = PiCamera(640, 480, 30, "continuous")
    cam._camera = Exploding()
    cam._configure_focus()
    assert cam.focus_applied == "unset"


def test_set_lens_position_switches_to_manual() -> None:
    cam, fake = _cam("continuous")
    cam.set_lens_position(4.0)
    assert fake.applied["LensPosition"] == pytest.approx(4.0)
    assert cam.lens_position == pytest.approx(4.0)


def test_set_lens_position_noop_on_fixed_focus() -> None:
    cam, fake = _cam("continuous", controls={"ExposureTime": (0, 1000, 100)})
    cam.set_lens_position(4.0)
    assert "LensPosition" not in fake.applied


def test_unknown_backend_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown camera backend"):
        create_camera(backend="holographic")


def test_shipped_configs_expose_focus_settings() -> None:
    for name in ("default.yaml", "sitl.yaml"):
        v = vision_config_from_dict(load_config(project_root() / "config" / name))
        assert v.focus_mode in FOCUS_MODES
        assert 0.0 <= v.lens_position <= 32.0
