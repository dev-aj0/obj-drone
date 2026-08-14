"""Application entry point and CLI commands."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time

from obj_drone.config import (
    VisionConfig,
    build_detector,
    capture_config_from_dict,
    follow_config_from_dict,
    load_config,
    logging_config_from_dict,
    mavlink_config_from_dict,
    mission_config_from_dict,
    preflight_config_from_dict,
    setup_logging,
    vision_config_from_dict,
)
from obj_drone.controller.mission import MissionController, MissionPhase
from obj_drone.paths import resolve_path
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
    parser.add_argument(
        "--connection",
        default=None,
        help="Override mavlink.connection, e.g. /dev/ttyACM0 (FC over USB), "
        "/dev/ttyAMA0 (GPIO UART), or udp:127.0.0.1:14550 (SITL)",
    )
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
    run_p.add_argument(
        "--source",
        choices=("detector", "color", "roi"),
        default=None,
        help="Tracking source (default: detector if configured, else color)",
    )

    sub.add_parser("test", help="Verify MAVLink heartbeat and telemetry")

    det_p = sub.add_parser("detect", help="Camera + detector check — no MAVLink, no flight")
    det_p.add_argument("--seconds", type=int, default=20, help="How long to run")
    det_p.add_argument("--save", action="store_true", help="Write annotated frames to logs/frames")

    web_p = sub.add_parser("serve", help="Live camera + detection viewer in a browser")
    web_p.add_argument("--port", type=int, default=8080, help="HTTP port (default 8080)")
    web_p.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address. Default 0.0.0.0 (reachable from other machines); "
        "use 127.0.0.1 to keep it local to the Pi.",
    )
    web_p.add_argument(
        "--source",
        choices=("detector", "color", "roi"),
        default=None,
        help="Tracking source (default: detector if configured, else color)",
    )
    web_p.add_argument("--quality", type=int, default=80, help="JPEG quality 1-100")
    web_p.add_argument(
        "--focus",
        choices=("continuous", "auto", "manual"),
        default=None,
        help="Override vision.focus_mode for this run",
    )
    web_p.add_argument(
        "--lens",
        type=float,
        default=None,
        help="Manual lens position in dioptres (0=infinity, 1.0=1m, 5.0=20cm). "
        "Implies --focus manual.",
    )

    sweep_p = sub.add_parser(
        "focus-sweep",
        help="Find the sharpest lens position — aim at something DETAILED first",
    )
    sweep_p.add_argument("--min", type=float, default=0.0, help="Lowest dioptre")
    sweep_p.add_argument("--max", type=float, default=12.0, help="Highest dioptre")
    sweep_p.add_argument("--steps", type=int, default=13, help="Positions to sample")
    sweep_p.add_argument("--apply", action="store_true", help="Print config to paste in")

    fc_p = sub.add_parser(
        "follow-check",
        help="Calibrate the distance estimate — stand a known distance away",
    )
    fc_p.add_argument("--seconds", type=int, default=20, help="How long to run")
    fc_p.add_argument(
        "--actual",
        type=float,
        default=None,
        help="Your true distance from the camera in metres; prints the corrected "
        "camera_vfov_deg to put in the config",
    )
    fc_p.add_argument(
        "--person-height",
        type=float,
        default=None,
        help="Actual height of the person being measured, in metres. The estimate "
        "scales directly with this — measure it rather than assuming 1.7.",
    )

    cal_p = sub.add_parser("calibrate-color", help="Tune HSV colour tracking")
    cal_p.add_argument("--seconds", type=int, default=30, help="Preview duration")
    cal_p.add_argument(
        "--save",
        action="store_true",
        help="Write frames to logs/frames instead of opening a window (use over SSH)",
    )

    hover_p = sub.add_parser("hover", help="Enter GUIDED and send zero-velocity hover")
    hover_p.add_argument("--arm", action="store_true", help="Arm before hovering")

    sub.add_parser("land", help="Command LAND mode")
    sub.add_parser("rtl", help="Command RTL mode")

    return parser


def connect_stack(cfg: dict) -> tuple[MavlinkConnection, TelemetryMonitor, FlightController]:
    mavlink_cfg = mavlink_config_from_dict(cfg)
    logger.debug("MAVLink endpoint: %s", mavlink_cfg.connection)
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
        gps_raw_hz=mavlink_cfg.gps_raw_hz,
    )
    telemetry = TelemetryMonitor(link, link_loss_timeout_s=mavlink_cfg.link_loss_timeout_s)
    telemetry.start()
    fc = FlightController(
        link,
        telemetry=telemetry,
        command_rate_hz=mavlink_cfg.command_rate_hz,
    )
    return link, telemetry, fc


def build_tracker(cfg: dict, with_detector: bool = True) -> tuple[TargetTracker, VisionConfig]:
    """Create the tracker (with detector if configured) and the vision config.

    with_detector=False skips loading the model entirely, so colour-only
    commands still work on a machine that has no model file yet.
    """
    vision_cfg = vision_config_from_dict(cfg)
    detector = build_detector(vision_cfg.detector) if with_detector else None
    tracker = TargetTracker(
        frame_width=vision_cfg.width,
        frame_height=vision_cfg.height,
        hsv_lower=vision_cfg.hsv_lower,
        hsv_upper=vision_cfg.hsv_upper,
        min_blob_area=vision_cfg.min_blob_area,
        detector=detector,
        max_track_jump_px=vision_cfg.max_track_jump_px,
        acquisition=vision_cfg.acquisition,
    )
    return tracker, vision_cfg


def _gui_available() -> bool:
    import cv2

    try:
        cv2.namedWindow("__probe__")
        cv2.destroyWindow("__probe__")
        return True
    except Exception:
        return False


def _close_windows() -> None:
    """Tear down any OpenCV windows.

    Headless builds raise from destroyAllWindows() rather than no-opping, so this
    must never be called bare in a finally block.
    """
    import cv2

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


def cmd_test(cfg: dict) -> int:
    link, telemetry, _fc = connect_stack(cfg)
    try:
        telemetry.wait_for_telemetry(timeout=5.0)
        state = telemetry.snapshot()
        if not telemetry.link_healthy():
            logger.error("No heartbeat")
            return 1
        logger.info(
            "Connected — vehicle=%s mode=%s armed=%s alt=%.1fm sats=%d fix=%d battery=%.1fV",
            link.vehicle_class,
            state.mode,
            state.armed,
            state.relative_alt_m,
            state.gps_satellites,
            state.gps_fix_type,
            state.battery_voltage,
        )
        mapping = link.mode_mapping() or {}
        logger.info("Available modes: %s", ", ".join(sorted(mapping)))
        return 0
    finally:
        telemetry.stop()
        link.close()


def cmd_detect(cfg: dict, seconds: int, save: bool) -> int:
    """Run the camera and detector only — safe with props off and no FC."""
    import cv2

    tracker, vision_cfg = build_tracker(cfg)
    camera: Camera | None = None
    debug = DebugWriter(enabled=save, interval=1)

    if tracker.detector is None:
        logger.error(
            "No detector configured. Set vision.detector.enabled=true and a model "
            "path in the config, then run scripts/fetch_model.sh."
        )
        return 1

    try:
        camera = create_camera(
            vision_cfg.width,
            vision_cfg.height,
            vision_cfg.fps,
            vision_cfg.camera_backend,
            vision_cfg.focus_mode,
            vision_cfg.lens_position,
            resolve_path(vision_cfg.tuning_file),
        )
        deadline = time.monotonic() + seconds
        frames = 0
        found = 0
        started = time.monotonic()
        while time.monotonic() < deadline:
            frame = camera.read()
            if frame is None:
                continue
            frames += 1
            detections = tracker.detector.detect(frame)
            result = tracker.select_target(detections)
            if result.found:
                found += 1
            if frames % 10 == 0 or result.found:
                logger.info(
                    "frame=%d inference=%.0fms detections=%d target=%s",
                    frames,
                    tracker.detector.last_inference_ms,
                    len(detections),
                    f"{result.label} {result.confidence:.2f} @ "
                    f"({result.center_x:.0f},{result.center_y:.0f})"
                    if result.found
                    else "none",
                )
            if save:
                err = tracker.pixel_error(result) if result.found else None
                debug.write(frame, result, err)

        elapsed = time.monotonic() - started
        logger.info(
            "Done — %d frames in %.1fs (%.1f fps), target present in %d frames",
            frames,
            elapsed,
            frames / elapsed if elapsed > 0 else 0.0,
            found,
        )
        if save:
            logger.info("Annotated frames written to %s", debug.output_dir)
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        if camera is not None:
            camera.stop()
        _close_windows()


def cmd_serve(
    cfg: dict,
    host: str,
    port: int,
    source: str | None,
    quality: int,
    focus: str | None = None,
    lens: float | None = None,
) -> int:
    """Serve the live annotated camera feed over HTTP. No MAVLink, no flight."""
    from obj_drone.capture import CaptureStore
    from obj_drone.web import VisionStream, serve

    tracker, vision_cfg = build_tracker(cfg, with_detector=source != "color")
    cap_cfg = capture_config_from_dict(cfg)

    focus_mode = focus or vision_cfg.focus_mode
    lens_position = vision_cfg.lens_position
    if lens is not None:
        # Asking for a specific lens position only makes sense in manual mode.
        lens_position = lens
        focus_mode = "manual"
    if source is None:
        source = "detector" if tracker.detector is not None else "color"
    elif source == "detector" and tracker.detector is None:
        logger.error(
            "--source detector requested but no model is configured. "
            "Set vision.detector.enabled and run scripts/fetch_model.sh."
        )
        return 1

    camera: Camera | None = None
    try:
        camera = create_camera(
            vision_cfg.width,
            vision_cfg.height,
            vision_cfg.fps,
            vision_cfg.camera_backend,
            focus_mode,
            lens_position,
            resolve_path(vision_cfg.tuning_file),
        )
        stream = VisionStream(
            camera,
            tracker,
            source=source,
            jpeg_quality=quality,
            mission_config=mission_config_from_dict(cfg),
            follow_config=follow_config_from_dict(cfg),
            captures=CaptureStore(
                output_dir=resolve_path(cap_cfg.directory),
                enabled=cap_cfg.enabled,
                min_interval_s=cap_cfg.min_interval_s,
                min_confidence=cap_cfg.min_confidence,
                max_captures=cap_cfg.max_captures,
                jpeg_quality=cap_cfg.jpeg_quality,
            ),
        )
        serve(stream, host=host, port=port)
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        if camera is not None:
            camera.stop()


def cmd_focus_sweep(cfg: dict, lo: float, hi: float, steps: int, apply: bool) -> int:
    """Step the lens through its range and report sharpness at each position."""
    import cv2

    vision_cfg = vision_config_from_dict(cfg)
    camera: Camera | None = None
    try:
        camera = create_camera(
            vision_cfg.width,
            vision_cfg.height,
            vision_cfg.fps,
            vision_cfg.camera_backend,
            "manual",
            lo,
            resolve_path(vision_cfg.tuning_file),
        )
        if not hasattr(camera, "set_lens_position"):
            logger.error("This camera has no controllable lens")
            return 1

        logger.info("Sweeping %.1f to %.1f dioptres in %d steps", lo, hi, steps)
        logger.info("Aim at something DETAILED — a blank wall gives no signal")

        results: list[tuple[float, float]] = []
        for i in range(steps):
            pos = lo + (hi - lo) * i / max(1, steps - 1)
            camera.set_lens_position(pos)
            time.sleep(1.0)
            frame = camera.read()
            if frame is None:
                continue
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Variance of the Laplacian: the standard focus measure. Higher is sharper.
            sharpness = float(cv2.Laplacian(grey, cv2.CV_64F).var())
            results.append((pos, sharpness))
            distance = "infinity" if pos <= 0 else f"{1.0 / pos:.2f} m"
            logger.info("  %5.2f dpt (%-9s) sharpness=%8.1f", pos, distance, sharpness)

        if not results:
            logger.error("No frames captured")
            return 1

        best_pos, best_sharp = max(results, key=lambda r: r[1])
        worst_sharp = min(r[1] for r in results)

        # A flat curve means the scene, not the lens, is the problem.
        if best_sharp < 5.0 or best_sharp < worst_sharp * 1.5:
            logger.warning(
                "Sharpness barely changed across the sweep (%.1f to %.1f). The scene "
                "probably has no detail — aim at text or a patterned surface and retry.",
                worst_sharp,
                best_sharp,
            )
            return 1

        logger.info("Sharpest at %.2f dioptres (sharpness %.1f)", best_pos, best_sharp)
        if apply:
            logger.info(
                "Add to your config:\n\nvision:\n  focus_mode: \"manual\"\n"
                "  lens_position: %.2f\n",
                best_pos,
            )
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        if camera is not None:
            camera.stop()


def cmd_follow_check(
    cfg: dict, seconds: int, actual: float | None, person_height: float | None = None
) -> int:
    """Report the estimated range to the tracked person. No MAVLink, no flight."""
    import math

    from obj_drone.config import follow_config_from_dict
    from obj_drone.controller.follow import estimate_distance_m, focal_length_px

    tracker, vision_cfg = build_tracker(cfg)
    fcfg = follow_config_from_dict(cfg)
    if person_height is not None and person_height > 0:
        fcfg.person_height_m = person_height
    if tracker.detector is None:
        logger.error("follow-check needs the object detector enabled")
        return 1

    camera: Camera | None = None
    samples: list[float] = []
    try:
        camera = create_camera(
            vision_cfg.width,
            vision_cfg.height,
            vision_cfg.fps,
            vision_cfg.camera_backend,
            vision_cfg.focus_mode,
            vision_cfg.lens_position,
            resolve_path(vision_cfg.tuning_file),
        )
        logger.info(
            "Stand in full view of the camera. person_height_m=%.2f vfov=%.1f deg",
            fcfg.person_height_m,
            fcfg.camera_vfov_deg,
        )
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            frame = camera.read()
            if frame is None:
                continue
            result = tracker.detect_objects(frame)
            if not result.found or result.bbox is None:
                continue
            dist = estimate_distance_m(
                result.bbox[3], vision_cfg.height, fcfg.person_height_m, fcfg.camera_vfov_deg
            )
            if dist is None:
                continue
            samples.append(dist)
            if len(samples) % 10 == 0:
                logger.info(
                    "box_height=%dpx  estimated range=%.2f m", result.bbox[3], dist
                )

        if not samples:
            logger.error(
                "No person detected. Stand fully in frame, well lit, and check focus."
            )
            return 1

        samples.sort()
        n = len(samples)
        median = samples[n // 2]
        p10, p90 = samples[int(n * 0.1)], samples[int(n * 0.9)]
        spread_pct = 100.0 * (p90 - p10) / median if median else 0.0
        logger.info(
            "Range over %d samples: median=%.2f m  p10=%.2f  p90=%.2f  spread=%.0f%%",
            n, median, p10, p90, spread_pct,
        )
        if spread_pct > 25:
            logger.warning(
                "Detection boxes are noisy (%.0f%% spread). Stand still, fully in "
                "frame including feet, well lit and in focus, then re-run. A single "
                "noisy calibration will chase that scatter instead of converging.",
                spread_pct,
            )

        if actual is not None and actual > 0:
            # distance is proportional to focal length, so scale the focal length
            # by the ratio of true to estimated range and convert back to an FOV.
            focal = focal_length_px(vision_cfg.height, fcfg.camera_vfov_deg)
            corrected_focal = focal * (actual / median)
            corrected_vfov = 2 * math.degrees(
                math.atan((vision_cfg.height / 2.0) / corrected_focal)
            )
            logger.info(
                "True distance %.2f m, person height %.2f m", actual, fcfg.person_height_m
            )
            logger.info(
                "Put this in config/default.yaml:\n\nfollow:\n  camera_vfov_deg: %.1f\n",
                corrected_vfov,
            )
            logger.info(
                "Run this 2-3 times. If the answers differ by more than a few "
                "degrees, average them — the box jitter is larger than the "
                "calibration error at that point."
            )
        else:
            logger.info(
                "Re-run with --actual <metres> to compute the corrected camera_vfov_deg"
            )
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        if camera is not None:
            camera.stop()


def cmd_calibrate_color(cfg: dict, seconds: int, save: bool) -> int:
    import cv2

    # Colour calibration never uses the model — don't require one to be present.
    tracker, vision_cfg = build_tracker(cfg, with_detector=False)
    camera: Camera | None = None
    use_gui = not save and _gui_available()
    if not save and not use_gui:
        logger.warning(
            "No display available — writing frames to disk instead. "
            "Re-run with --save to silence this."
        )
        save = True
    debug = DebugWriter(enabled=save, interval=1)

    try:
        camera = create_camera(
            vision_cfg.width,
            vision_cfg.height,
            vision_cfg.fps,
            vision_cfg.camera_backend,
            vision_cfg.focus_mode,
            vision_cfg.lens_position,
            resolve_path(vision_cfg.tuning_file),
        )
        logger.info(
            "Colour calibration — HSV lower=%s upper=%s (%ds%s)",
            vision_cfg.hsv_lower,
            vision_cfg.hsv_upper,
            seconds,
            ", press q to quit" if use_gui else "",
        )

        deadline = time.monotonic() + seconds
        frames = 0
        while time.monotonic() < deadline:
            frame = camera.read()
            if frame is None:
                continue
            frames += 1
            result = tracker.detect_color(frame)

            if use_gui:
                display = frame.copy()
                if result.found and result.bbox:
                    x, y, w, h = result.bbox
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow("obj-drone calibrate-color", display)
                cv2.imshow("mask", tracker.preview_mask(frame))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            elif frames % 15 == 0:
                err = tracker.pixel_error(result) if result.found else None
                debug.write(frame, result, err)
                logger.info(
                    "frame=%d blob=%s",
                    frames,
                    f"{result.bbox} @ ({result.center_x:.0f},{result.center_y:.0f})"
                    if result.found
                    else "none",
                )

        if save:
            logger.info("Frames written to %s", debug.output_dir)
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        if camera is not None:
            camera.stop()
        _close_windows()


def cmd_hover(cfg: dict, arm: bool) -> int:
    link, telemetry, fc = connect_stack(cfg)
    try:
        fc.set_mode("GUIDED", wait=True, timeout=10.0)
        if arm:
            fc.arm(wait=True, timeout=10.0)
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


def cmd_run(cfg: dict, skip_takeoff: bool, skip_preflight: bool, source: str | None) -> int:
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
        preflight = PreflightCheck(fc, telemetry, preflight_config_from_dict(cfg))

        if not skip_preflight:
            if not preflight.run_detailed().ok:
                return 1

        # --source color is an explicit opt-out of the model; don't load it.
        tracker, vision_cfg = build_tracker(cfg, with_detector=source != "color")
        if source is None:
            source = "detector" if tracker.detector is not None else "color"

        camera = create_camera(
            vision_cfg.width,
            vision_cfg.height,
            vision_cfg.fps,
            vision_cfg.camera_backend,
            vision_cfg.focus_mode,
            vision_cfg.lens_position,
            resolve_path(vision_cfg.tuning_file),
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
        mission.check_vehicle_supported()

        if not skip_takeoff:
            mission.prepare_guided()
            mission.takeoff_and_hover()

        mission.track_target_loop(source=source)

        if mission.pilot_override:
            logger.info("Exiting without commanding anything — the pilot is flying")
        elif stop_event.is_set() and mission.phase is not MissionPhase.LANDING:
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
                logger.exception("Failsafe also failed")
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
    if args.connection:
        cfg.setdefault("mavlink", {})["connection"] = args.connection
    setup_logging(logging_config_from_dict(cfg), verbose=args.verbose)

    try:
        if args.command == "test":
            return cmd_test(cfg)
        if args.command == "detect":
            return cmd_detect(cfg, args.seconds, args.save)
        if args.command == "hover":
            return cmd_hover(cfg, arm=args.arm)
        if args.command == "land":
            return cmd_land(cfg)
        if args.command == "rtl":
            return cmd_rtl(cfg)
        if args.command == "serve":
            return cmd_serve(
                cfg, args.host, args.port, args.source, args.quality, args.focus, args.lens
            )
        if args.command == "focus-sweep":
            return cmd_focus_sweep(cfg, args.min, args.max, args.steps, args.apply)
        if args.command == "follow-check":
            return cmd_follow_check(
                cfg, args.seconds, args.actual, args.person_height
            )
        if args.command == "calibrate-color":
            return cmd_calibrate_color(cfg, args.seconds, args.save)
        if args.command == "run":
            return cmd_run(cfg, args.skip_takeoff, args.skip_preflight, args.source)
    except RuntimeError as exc:
        # Connection/mode/arming problems already carry an actionable message.
        logger.error("%s", exc)
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
