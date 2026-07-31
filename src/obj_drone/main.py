"""Application entry point and CLI commands."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time

from obj_drone.config import (
    flight_config_from_dict,
    load_config,
    logging_config_from_dict,
    mavlink_config_from_dict,
    mission_config_from_dict,
    preflight_config_from_dict,
    setup_logging,
    vision_config_from_dict,
)
from obj_drone.controller.mission import MissionController
from obj_drone.controller.preflight import PreflightCheck
from obj_drone.mavlink.commands import FlightController
from obj_drone.mavlink.connection import MavlinkConnection
from obj_drone.mavlink.telemetry import TelemetryMonitor
from obj_drone.vision.camera import Camera, create_camera
from obj_drone.vision.debug import DebugWriter
from obj_drone.vision.tracker import TargetTracker

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ArduPilot companion computer — vision and MAVLink control",
    )
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")

    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Full mission: preflight → takeoff → track")
    run_p.add_argument(
        "--skip-takeoff",
        action="store_true",
        help="Already airborne in GUIDED; start tracking immediately",
    )
    run_p.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip preflight checks (not recommended)",
    )

    sub.add_parser("test", help="Verify MAVLink heartbeat and telemetry")

    cal_p = sub.add_parser("calibrate-color", help="Tune HSV color tracking with live preview")
    cal_p.add_argument("--seconds", type=int, default=30, help="Preview duration")

    hover_p = sub.add_parser("hover", help="Enter GUIDED and send zero-velocity hover")
    hover_p.add_argument("--arm", action="store_true", help="Arm before hovering")

    sub.add_parser("land", help="Command LAND mode")
    sub.add_parser("rtl", help="Command RTL mode")

    return parser


def connect_stack(cfg: dict) -> tuple[MavlinkConnection, TelemetryMonitor, FlightController]:
    mavlink_cfg = mavlink_config_from_dict(cfg)
    link = MavlinkConnection(
        connection=mavlink_cfg.connection,
        baud=mavlink_cfg.baud,
        heartbeat_timeout=mavlink_cfg.heartbeat_timeout,
    )
    link.connect()
    link.configure_telemetry_streams(
        global_position_hz=mavlink_cfg.global_position_hz,
        attitude_hz=mavlink_cfg.attitude_hz,
        sys_status_hz=mavlink_cfg.sys_status_hz,
    )
    telemetry = TelemetryMonitor(link, link_loss_timeout_s=mavlink_cfg.link_loss_timeout_s)
    telemetry.start()
    fc = FlightController(
        link,
        telemetry=telemetry,
        command_rate_hz=mavlink_cfg.command_rate_hz,
    )
    return link, telemetry, fc


def cmd_test(cfg: dict) -> int:
    link, telemetry, _fc = connect_stack(cfg)
    try:
        time.sleep(2.0)
        state = telemetry.snapshot()
        if not telemetry.link_healthy():
            logger.error("No heartbeat")
            return 1
        logger.info(
            "Connected — mode=%s armed=%s alt=%.1fm sats=%d battery=%.1fV",
            state.mode,
            state.armed,
            state.relative_alt_m,
            state.gps_satellites,
            state.battery_voltage,
        )
        return 0
    finally:
        telemetry.stop()
        link.close()


def cmd_hover(cfg: dict, arm: bool) -> int:
    link, telemetry, fc = connect_stack(cfg)
    try:
        fc.set_mode("GUIDED")
        if arm:
            fc.arm()
            if not link.wait_for_arm(True, timeout=10.0):
                logger.error("Arm failed")
                return 1
        logger.info("Hovering — press Ctrl+C to land")
        while telemetry.link_healthy():
            fc.hover()
            time.sleep(0.1)
        logger.error("Link lost")
        return 1
    except KeyboardInterrupt:
        fc.land()
        return 0
    finally:
        telemetry.stop()
        link.close()


def cmd_land(cfg: dict) -> int:
    link, telemetry, fc = connect_stack(cfg)
    try:
        fc.land()
        time.sleep(1.0)
        logger.info("LAND commanded")
        return 0
    finally:
        telemetry.stop()
        link.close()


def cmd_rtl(cfg: dict) -> int:
    link, telemetry, fc = connect_stack(cfg)
    try:
        fc.rtl()
        time.sleep(1.0)
        logger.info("RTL commanded")
        return 0
    finally:
        telemetry.stop()
        link.close()


def cmd_calibrate_color(cfg: dict, seconds: int) -> int:
    vision_cfg = vision_config_from_dict(cfg)
    camera: Camera | None = None
    try:
        camera = create_camera(vision_cfg.width, vision_cfg.height, vision_cfg.fps)
        tracker = TargetTracker(
            frame_width=vision_cfg.width,
            frame_height=vision_cfg.height,
            hsv_lower=vision_cfg.hsv_lower,
            hsv_upper=vision_cfg.hsv_upper,
            min_blob_area=vision_cfg.min_blob_area,
        )
        logger.info(
            "Color calibration — HSV lower=%s upper=%s (%ds, press q to quit)",
            vision_cfg.hsv_lower,
            vision_cfg.hsv_upper,
            seconds,
        )
        import cv2

        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            frame = camera.read()
            if frame is None:
                continue
            mask = tracker.preview_mask(frame)
            result = tracker.detect_color(frame)
            display = frame.copy()
            if result.found and result.bbox:
                x, y, w, h = result.bbox
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("obj-drone calibrate-color", display)
            cv2.imshow("mask", mask)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        if camera is not None:
            camera.stop()


def cmd_run(cfg: dict, skip_takeoff: bool, skip_preflight: bool) -> int:
    vision_cfg = vision_config_from_dict(cfg)
    log_cfg = logging_config_from_dict(cfg)

    link: MavlinkConnection | None = None
    telemetry: TelemetryMonitor | None = None
    camera: Camera | None = None
    mission: MissionController | None = None
    stop_event = threading.Event()

    def on_signal(signum: int, _frame: object) -> None:
        logger.info("Signal %s received — landing", signum)
        stop_event.set()
        if mission is not None:
            mission.request_stop()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    try:
        link, telemetry, fc = connect_stack(cfg)
        preflight = PreflightCheck(link, telemetry, preflight_config_from_dict(cfg))

        if not skip_preflight:
            errors = preflight.run()
            if errors:
                for err in errors:
                    logger.error(err)
                return 1

        camera = create_camera(vision_cfg.width, vision_cfg.height, vision_cfg.fps)
        tracker = TargetTracker(
            frame_width=vision_cfg.width,
            frame_height=vision_cfg.height,
            hsv_lower=vision_cfg.hsv_lower,
            hsv_upper=vision_cfg.hsv_upper,
            min_blob_area=vision_cfg.min_blob_area,
        )
        debug = DebugWriter(
            enabled=vision_cfg.debug_overlay,
            interval=vision_cfg.debug_frame_interval,
            output_dir=log_cfg.directory + "/frames",
        )
        mission = MissionController(
            fc=fc,
            telemetry=telemetry,
            camera=camera,
            tracker=tracker,
            config=mission_config_from_dict(cfg),
            debug=debug,
            stop_event=stop_event,
        )

        if not skip_takeoff:
            mission.prepare_guided()
            mission.takeoff_and_hover()

        mission.track_target_loop(use_color=True)

        if stop_event.is_set() and mission.phase.name != "LANDING":
            mission.land()
        return 0

    except KeyboardInterrupt:
        if mission is not None:
            mission.land()
        return 0
    except Exception:
        logger.exception("Mission failed")
        if mission is not None:
            try:
                mission.trigger_failsafe("Unhandled exception")
            except Exception:
                pass
        return 1
    finally:
        if telemetry is not None:
            telemetry.stop()
        if link is not None:
            link.close()
        if camera is not None:
            camera.stop()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(logging_config_from_dict(cfg), verbose=args.verbose)

    if args.command == "test":
        return cmd_test(cfg)
    if args.command == "hover":
        return cmd_hover(cfg, arm=args.arm)
    if args.command == "land":
        return cmd_land(cfg)
    if args.command == "rtl":
        return cmd_rtl(cfg)
    if args.command == "calibrate-color":
        return cmd_calibrate_color(cfg, args.seconds)
    if args.command == "run":
        return cmd_run(cfg, args.skip_takeoff, args.skip_preflight)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
