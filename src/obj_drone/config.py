"""Load YAML configuration and typed settings objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from obj_drone.controller.mission import MissionConfig
from obj_drone.controller.preflight import PreflightConfig


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        path = project_root() / "config" / "default.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


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
class VisionConfig:
    width: int
    height: int
    fps: int
    hsv_lower: tuple[int, int, int]
    hsv_upper: tuple[int, int, int]
    min_blob_area: int
    debug_overlay: bool
    debug_frame_interval: int


@dataclass
class LoggingConfig:
    directory: str
    level: str


def mavlink_config_from_dict(cfg: dict[str, Any]) -> MavlinkConfig:
    m = cfg.get("mavlink", {})
    tel = m.get("telemetry", {})
    return MavlinkConfig(
        connection=m.get("connection", "/dev/serial0"),
        baud=m.get("baud", 57600),
        heartbeat_timeout=m.get("heartbeat_timeout", 30),
        command_rate_hz=m.get("command_rate_hz", 10),
        link_loss_timeout_s=m.get("link_loss_timeout_s", 3.0),
        global_position_hz=tel.get("global_position_hz", 5),
        attitude_hz=tel.get("attitude_hz", 10),
        sys_status_hz=tel.get("sys_status_hz", 1),
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
    return MissionConfig(
        takeoff_altitude_m=flight.get("takeoff_altitude_m", 3.0),
        max_horizontal_speed_m_s=flight.get("max_horizontal_speed_m_s", 2.0),
        max_vertical_speed_m_s=flight.get("max_vertical_speed_m_s", 1.0),
        position_tolerance_m=flight.get("position_tolerance_m", 0.5),
        control_rate_hz=mission.get("control_rate_hz", 20.0),
        track_gain=mission.get("track_gain", 0.002),
        lost_target_grace_frames=mission.get("lost_target_grace_frames", 15),
        lost_target_action=mission.get("lost_target_action", "hover"),
        link_loss_action=flight.get("link_loss_action", "rtl"),
    )


def preflight_config_from_dict(cfg: dict[str, Any]) -> PreflightConfig:
    f = cfg.get("flight", {})
    return PreflightConfig(
        min_gps_satellites=f.get("min_gps_satellites", 6),
        min_battery_voltage=f.get("min_battery_voltage", 10.5),
    )


def vision_config_from_dict(cfg: dict[str, Any]) -> VisionConfig:
    v = cfg.get("vision", {})
    return VisionConfig(
        width=v.get("width", 640),
        height=v.get("height", 480),
        fps=v.get("fps", 30),
        hsv_lower=tuple(v.get("hsv_lower", [0, 120, 70])),
        hsv_upper=tuple(v.get("hsv_upper", [10, 255, 255])),
        min_blob_area=v.get("min_blob_area", 100),
        debug_overlay=v.get("debug_overlay", False),
        debug_frame_interval=v.get("debug_frame_interval", 30),
    )


def logging_config_from_dict(cfg: dict[str, Any]) -> LoggingConfig:
    lg = cfg.get("logging", {})
    return LoggingConfig(
        directory=lg.get("directory", "logs"),
        level=lg.get("level", "INFO"),
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
