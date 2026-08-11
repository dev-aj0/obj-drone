"""Tuning-file plumbing and focus telemetry."""

from __future__ import annotations

import pytest

from obj_drone.config import load_config, vision_config_from_dict
from obj_drone.paths import project_root
from obj_drone.vision.camera import AF_STATE_NAMES, PiCamera


class FakePicamWithMeta:
    def __init__(self, meta: dict) -> None:
        self.camera_controls = {"AfMode": (0, 2, 0), "LensPosition": (0.0, 32.0, 1.0)}
        self._meta = meta
        self.applied: dict = {}

    def set_controls(self, ctrls):
        self.applied.update(ctrls)

    def capture_metadata(self):
        return self._meta


def test_focus_info_decodes_af_state() -> None:
    cam = PiCamera(640, 480, 30)
    cam._camera = FakePicamWithMeta({"LensPosition": 2.5, "AfState": 2, "FocusFoM": 412})
    info = cam.focus_info()
    assert info["lens_position"] == pytest.approx(2.5)
    assert info["af_state"] == "focused"
    assert info["focus_fom"] == 412


def test_focus_info_handles_missing_metadata() -> None:
    cam = PiCamera(640, 480, 30)
    cam._camera = FakePicamWithMeta({})
    info = cam.focus_info()
    assert info["lens_position"] is None
    assert info["af_state"] is None


def test_focus_info_empty_without_camera() -> None:
    assert PiCamera(640, 480, 30).focus_info() == {}


def test_focus_info_survives_metadata_failure() -> None:
    class Exploding(FakePicamWithMeta):
        def capture_metadata(self):
            raise RuntimeError("camera busy")

    cam = PiCamera(640, 480, 30)
    cam._camera = Exploding({})
    assert cam.focus_info() == {}


def test_af_state_names_cover_libcamera_values() -> None:
    assert AF_STATE_NAMES == {0: "idle", 1: "scanning", 2: "focused", 3: "failed"}


def test_missing_tuning_file_is_not_fatal(tmp_path, caplog) -> None:
    """A missing focus file must not ground the aircraft."""
    import logging

    cam = PiCamera(640, 480, 30, tuning_file=tmp_path / "nope.json")
    with caplog.at_level(logging.WARNING):
        try:
            cam.start()
        except Exception as exc:
            # Off-Pi there is no picamera2 at all; that is a different failure.
            assert "picamera2" in str(exc)
    assert cam.tuning_file.name == "nope.json"


def test_tuning_file_is_optional() -> None:
    assert PiCamera(640, 480, 30).tuning_file is None


def test_default_config_declares_a_tuning_file() -> None:
    """The IMX519 needs one; without it autofocus silently does nothing."""
    v = vision_config_from_dict(load_config(project_root() / "config" / "default.yaml"))
    assert v.tuning_file is not None
    assert v.tuning_file.endswith(".json")
