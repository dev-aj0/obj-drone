"""CLI wiring — argument parsing and tracker construction."""

from __future__ import annotations

import pytest

from obj_drone.main import build_parser, build_tracker
from obj_drone.vision.detector import DetectorError

# A config that enables the detector but points at a model that does not exist.
CFG_MISSING_MODEL = {
    "vision": {
        "width": 640,
        "height": 480,
        "detector": {
            "enabled": True,
            "model": "models/definitely-not-here.onnx",
            "labels": "models/coco.names",
        },
    }
}


def test_colour_only_tracker_does_not_need_a_model() -> None:
    """Regression: colour calibration failed on a Pi that had no model yet.

    build_tracker() used to load the detector unconditionally, so every
    colour-only command died with 'Model file not found'.
    """
    tracker, _ = build_tracker(CFG_MISSING_MODEL, with_detector=False)
    assert tracker.detector is None


def test_missing_model_still_raises_when_detector_is_wanted() -> None:
    with pytest.raises(DetectorError, match="Model file not found"):
        build_tracker(CFG_MISSING_MODEL, with_detector=True)


def test_colour_tracking_works_without_a_model() -> None:
    from conftest import draw_bgr_box, make_frame

    tracker, _ = build_tracker(CFG_MISSING_MODEL, with_detector=False)
    frame = draw_bgr_box(make_frame(), (0, 0, 255), 300, 200, 60, 60)
    assert tracker.detect_color(frame).found


def test_parser_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_run_defaults() -> None:
    args = build_parser().parse_args(["run"])
    assert args.skip_takeoff is False
    assert args.skip_preflight is False
    assert args.source is None


def test_run_accepts_source_choices() -> None:
    for source in ("detector", "color", "roi"):
        assert build_parser().parse_args(["run", "--source", source]).source == source


def test_run_rejects_unknown_source() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--source", "telepathy"])


def test_detect_and_calibrate_accept_save() -> None:
    parser = build_parser()
    assert parser.parse_args(["detect", "--save"]).save is True
    assert parser.parse_args(["calibrate-color", "--save"]).save is True


def test_config_flag_is_global() -> None:
    args = build_parser().parse_args(["--config", "config/sitl.yaml", "test"])
    assert args.config == "config/sitl.yaml"
    assert args.command == "test"
