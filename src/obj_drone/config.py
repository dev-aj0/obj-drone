"""Load YAML configuration and typed settings objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from obj_drone.controller.follow import FollowConfig
from obj_drone.controller.mission import MissionConfig
from obj_drone.controller.preflight import PreflightConfig
# Re-exported so existing callers can keep using obj_drone.config.project_root.
from obj_drone.paths import project_root, resolve_path  # noqa: F401


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        path = project_root() / "config" / "default.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class MavlinkConfig:
    connection: str
    baud: int
    heartbeat_timeout: float
    command_rate_hz: float
    link_loss_timeout_s: float
    global_position_hz: float
    attitude_hz: float
    sys_status_hz: float
    gps_raw_hz: float


@dataclass
class FlightConfig:
    takeoff_altitude_m: float
    max_horizontal_speed_m_s: float
    max_vertical_speed_m_s: float
    position_tolerance_m: float
    min_gps_satellites: int
    min_battery_voltage: float
    link_loss_action: str


@dataclass
class DetectorConfig:
    enabled: bool
    model: str | None
    model_type: str
    input_size: int
    confidence: float
    nms_threshold: float
    labels: str | None
    target_classes: list[str]
    num_threads: int


@dataclass
class VisionConfig:
    width: int
    height: int
    fps: int
    camera_backend: str
    focus_mode: str
    lens_position: float
    tuning_file: str | None
    hsv_lower: tuple[int, int, int]
    hsv_upper: tuple[int, int, int]
    min_blob_area: int
    max_track_jump_px: float
    acquisition: str
    debug_overlay: bool
    debug_frame_interval: int
    detector: DetectorConfig


@dataclass
class LoggingConfig:
    directory: str
    level: str


def mavlink_config_from_dict(cfg: dict[str, Any]) -> MavlinkConfig:
    m = cfg.get("mavlink", {})
    tel = m.get("telemetry", {})
    return MavlinkConfig(
        connection=m.get("connection", "/dev/ttyAMA0"),
        baud=m.get("baud", 57600),
        heartbeat_timeout=m.get("heartbeat_timeout", 30),
        command_rate_hz=m.get("command_rate_hz", 10),
        link_loss_timeout_s=m.get("link_loss_timeout_s", 3.0),
        global_position_hz=tel.get("global_position_hz", 5),
        attitude_hz=tel.get("attitude_hz", 10),
        sys_status_hz=tel.get("sys_status_hz", 1),
        gps_raw_hz=tel.get("gps_raw_hz", 1),
    )


def flight_config_from_dict(cfg: dict[str, Any]) -> FlightConfig:
    f = cfg.get("flight", {})
    return FlightConfig(
        takeoff_altitude_m=f.get("takeoff_altitude_m", 3.0),
        max_horizontal_speed_m_s=f.get("max_horizontal_speed_m_s", 2.0),
        max_vertical_speed_m_s=f.get("max_vertical_speed_m_s", 1.0),
        position_tolerance_m=f.get("position_tolerance_m", 0.5),
        min_gps_satellites=f.get("min_gps_satellites", 6),
        min_battery_voltage=f.get("min_battery_voltage", 10.5),
        link_loss_action=f.get("link_loss_action", "rtl"),
    )


def mission_config_from_dict(cfg: dict[str, Any]) -> MissionConfig:
    flight = cfg.get("flight", {})
    mission = cfg.get("mission", {})
    vision = cfg.get("vision", {})
    follow = cfg.get("follow", {})
    return MissionConfig(
        takeoff_altitude_m=flight.get("takeoff_altitude_m", 3.0),
        max_horizontal_speed_m_s=flight.get("max_horizontal_speed_m_s", 2.0),
        max_vertical_speed_m_s=flight.get("max_vertical_speed_m_s", 1.0),
        position_tolerance_m=flight.get("position_tolerance_m", 0.5),
        control_rate_hz=mission.get("control_rate_hz", 20.0),
        track_gain=mission.get("track_gain", 1.5),
        track_deadband=mission.get("track_deadband", 0.05),
        lost_target_grace_frames=mission.get("lost_target_grace_frames", 15),
        lost_target_action=mission.get("lost_target_action", "hover"),
        max_camera_failures=mission.get("max_camera_failures", 60),
        link_loss_action=flight.get("link_loss_action", "rtl"),
        camera_orientation=vision.get("camera_orientation", "down"),
        invert_lateral=mission.get("invert_lateral", False),
        invert_longitudinal=mission.get("invert_longitudinal", False),
        follow_enabled=follow.get("enabled", False),
        follow=follow_config_from_dict(cfg),
        land_when_stationary_s=follow.get("land_when_stationary_s", 0.0),
        abort_on_mode_change=mission.get("abort_on_mode_change", True),
    )


def follow_config_from_dict(cfg: dict[str, Any]) -> FollowConfig:
    f = cfg.get("follow", {})
    flight = cfg.get("flight", {})
    mission = cfg.get("mission", {})
    return FollowConfig(
        follow_distance_m=f.get("distance_m", 3.0),
        follow_height_m=f.get("height_m", 3.0),
        distance_tolerance_m=f.get("distance_tolerance_m", 0.5),
        altitude_tolerance_m=f.get("altitude_tolerance_m", 0.3),
        person_height_m=f.get("person_height_m", 1.7),
        camera_vfov_deg=f.get("camera_vfov_deg", 48.0),
        distance_gain=f.get("distance_gain", 1.2),
        retreat_gain=f.get("retreat_gain", 1.5),
        min_retreat_speed_m_s=f.get("min_retreat_speed_m_s", 0.8),
        lateral_mode=f.get("lateral_mode", "both"),
        yaw_gain=f.get("yaw_gain", 1.2),
        max_yaw_rate_deg_s=f.get("max_yaw_rate_deg_s", 45.0),
        strafe_blend=f.get("strafe_blend", 0.35),
        lateral_gain=f.get("lateral_gain", 1.5),
        altitude_gain=f.get("altitude_gain", 0.5),
        max_horizontal_speed_m_s=flight.get("max_horizontal_speed_m_s", 2.0),
        max_vertical_speed_m_s=flight.get("max_vertical_speed_m_s", 1.0),
        min_distance_m=f.get("min_distance_m", 2.0),
        lateral_deadband=mission.get("track_deadband", 0.05),
        invert_lateral=mission.get("invert_lateral", False),
        invert_longitudinal=mission.get("invert_longitudinal", False),
    )


def preflight_config_from_dict(cfg: dict[str, Any]) -> PreflightConfig:
    f = cfg.get("flight", {})
    return PreflightConfig(
        min_gps_satellites=f.get("min_gps_satellites", 6),
        min_battery_voltage=f.get("min_battery_voltage", 10.5),
        require_gps_fix=f.get("require_gps_fix", True),
    )


def detector_config_from_dict(cfg: dict[str, Any]) -> DetectorConfig:
    d = cfg.get("vision", {}).get("detector", {})
    return DetectorConfig(
        enabled=d.get("enabled", False),
        model=d.get("model"),
        model_type=d.get("type", "yolo"),
        input_size=d.get("input_size", 320),
        confidence=d.get("confidence", 0.5),
        nms_threshold=d.get("nms_threshold", 0.45),
        labels=d.get("labels"),
        target_classes=list(d.get("target_classes", []) or []),
        num_threads=d.get("num_threads", 4),
    )


def vision_config_from_dict(cfg: dict[str, Any]) -> VisionConfig:
    v = cfg.get("vision", {})
    return VisionConfig(
        width=v.get("width", 640),
        height=v.get("height", 480),
        fps=v.get("fps", 30),
        camera_backend=v.get("camera_backend", "auto"),
        focus_mode=v.get("focus_mode", "continuous"),
        lens_position=float(v.get("lens_position", 0.0)),
        tuning_file=v.get("tuning_file"),
        hsv_lower=tuple(v.get("hsv_lower", [0, 120, 70])),
        hsv_upper=tuple(v.get("hsv_upper", [10, 255, 255])),
        min_blob_area=v.get("min_blob_area", 100),
        max_track_jump_px=v.get("max_track_jump_px", 160.0),
        acquisition=v.get("acquisition", "largest"),
        debug_overlay=v.get("debug_overlay", False),
        debug_frame_interval=v.get("debug_frame_interval", 30),
        detector=detector_config_from_dict(cfg),
    )


def logging_config_from_dict(cfg: dict[str, Any]) -> LoggingConfig:
    lg = cfg.get("logging", {})
    return LoggingConfig(
        directory=lg.get("directory", "logs"),
        level=lg.get("level", "INFO"),
    )


def build_detector(dcfg: DetectorConfig):
    """Instantiate an ObjectDetector from config, or None when disabled."""
    if not dcfg.enabled:
        return None

    from obj_drone.vision.detector import DetectorError, ObjectDetector, load_labels

    if not dcfg.model:
        raise DetectorError("vision.detector.enabled is true but no model path is set")

    model_path = resolve_path(dcfg.model)
    labels_path = resolve_path(dcfg.labels)
    return ObjectDetector(
        model_path=model_path,
        model_type=dcfg.model_type,
        input_size=dcfg.input_size,
        confidence=dcfg.confidence,
        nms_threshold=dcfg.nms_threshold,
        labels=load_labels(labels_path) if labels_path else [],
        target_classes=dcfg.target_classes,
        num_threads=dcfg.num_threads,
    )


def setup_logging(log_cfg: LoggingConfig, verbose: bool = False) -> None:
    import logging
    from logging.handlers import RotatingFileHandler

    log_dir = project_root() / log_cfg.directory
    log_dir.mkdir(parents=True, exist_ok=True)

    level = logging.DEBUG if verbose else getattr(logging, log_cfg.level.upper(), logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        log_dir / "obj_drone.log",
        maxBytes=5_000_000,
        backupCount=3,
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)
