"""Config loading, defaults, and the shipped YAML profiles."""

from __future__ import annotations

import pytest
import yaml

from obj_drone.config import (
    detector_config_from_dict,
    flight_config_from_dict,
    load_config,
    logging_config_from_dict,
    mavlink_config_from_dict,
    mission_config_from_dict,
    preflight_config_from_dict,
    vision_config_from_dict,
)
from obj_drone.paths import project_root, resolve_path


def test_project_root_contains_expected_dirs() -> None:
    root = project_root()
    assert (root / "config").is_dir()
    assert (root / "src" / "obj_drone").is_dir()


def test_resolve_path_relative_and_absolute(tmp_path) -> None:
    assert resolve_path("models/x.onnx") == project_root() / "models" / "x.onnx"
    assert resolve_path(tmp_path / "x.onnx") == tmp_path / "x.onnx"
    assert resolve_path(None) is None


def test_default_config_loads() -> None:
    cfg = load_config()
    assert cfg["mavlink"]["connection"]
    assert cfg["vision"]["width"] > 0


def test_default_config_targets_a_real_device() -> None:
    """The link must be USB (/dev/ttyACM*) or the Pi 5 GPIO UART (/dev/ttyAMA0).

    /dev/serial0 is Pi 0-4 guidance and is wrong on a Pi 5. Currently ttyACM0,
    because the TELEM wiring is physically broken and USB is the working link.
    """
    conn = load_config()["mavlink"]["connection"]
    assert conn.startswith("/dev/ttyACM") or conn == "/dev/ttyAMA0", conn


def test_default_config_requests_gps_stream() -> None:
    cfg = load_config()
    assert cfg["mavlink"]["telemetry"]["gps_raw_hz"] > 0


def test_sitl_config_loads() -> None:
    cfg = load_config(project_root() / "config" / "sitl.yaml")
    assert cfg["mavlink"]["connection"].startswith("udp:")
    assert cfg["flight"]["min_gps_satellites"] == 0


def test_shipped_configs_are_valid_yaml() -> None:
    for name in ("default.yaml", "sitl.yaml"):
        path = project_root() / "config" / name
        assert yaml.safe_load(path.read_text()) is not None


def test_all_config_sections_parse_for_shipped_profiles() -> None:
    for name in ("default.yaml", "sitl.yaml"):
        cfg = load_config(project_root() / "config" / name)
        mavlink_config_from_dict(cfg)
        flight_config_from_dict(cfg)
        mission_config_from_dict(cfg)
        preflight_config_from_dict(cfg)
        vision_config_from_dict(cfg)
        logging_config_from_dict(cfg)


def test_empty_config_yields_defaults() -> None:
    cfg: dict = {}
    mav = mavlink_config_from_dict(cfg)
    assert mav.connection == "/dev/ttyAMA0"
    assert mav.gps_raw_hz == 1

    vision = vision_config_from_dict(cfg)
    assert vision.camera_backend == "auto"
    assert vision.detector.enabled is False

    mission = mission_config_from_dict(cfg)
    assert mission.camera_orientation == "down"
    assert mission.track_gain > 0


def test_detector_config_reads_nested_section() -> None:
    cfg = {
        "vision": {
            "detector": {
                "enabled": True,
                "model": "models/y.onnx",
                "type": "ssd",
                "input_size": 300,
                "confidence": 0.6,
                "target_classes": ["person", "car"],
            }
        }
    }
    d = detector_config_from_dict(cfg)
    assert d.enabled
    assert d.model_type == "ssd"
    assert d.input_size == 300
    assert d.confidence == pytest.approx(0.6)
    assert d.target_classes == ["person", "car"]


def test_mission_config_reads_camera_orientation_from_vision() -> None:
    cfg = {"vision": {"camera_orientation": "forward"}}
    assert mission_config_from_dict(cfg).camera_orientation == "forward"


def test_hsv_bounds_become_tuples() -> None:
    cfg = {"vision": {"hsv_lower": [5, 100, 100], "hsv_upper": [15, 255, 255]}}
    vision = vision_config_from_dict(cfg)
    assert vision.hsv_lower == (5, 100, 100)
    assert vision.hsv_upper == (15, 255, 255)


def test_build_detector_returns_none_when_disabled() -> None:
    from obj_drone.config import build_detector

    assert build_detector(detector_config_from_dict({})) is None


def test_build_detector_requires_model_path() -> None:
    from obj_drone.config import build_detector
    from obj_drone.vision.detector import DetectorError

    dcfg = detector_config_from_dict({"vision": {"detector": {"enabled": True}}})
    with pytest.raises(DetectorError, match="no model path"):
        build_detector(dcfg)


def test_coco_labels_file_is_complete() -> None:
    from obj_drone.vision.detector import load_labels

    labels = load_labels(project_root() / "models" / "coco.names")
    assert len(labels) == 80
    assert labels[0] == "person"
