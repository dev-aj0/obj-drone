"""Pre-arm and pre-takeoff checks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from obj_drone.mavlink.commands import FlightController
from obj_drone.mavlink.telemetry import TelemetryMonitor

logger = logging.getLogger(__name__)

# GPS_FIX_TYPE_3D_FIX
_MIN_USABLE_FIX = 3


@dataclass
class PreflightConfig:
    min_gps_satellites: int = 6
    min_battery_voltage: float = 10.5
    require_gps_fix: bool = True
    # Logical intents, resolved per airframe (a plane has no LAND mode).
    required_modes: tuple[str, ...] = ("GUIDED", "LAND", "RTL", "LOITER")
    telemetry_timeout_s: float = 10.0


@dataclass
class PreflightReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


class PreflightCheck:
    """Validate FC state before arming or takeoff."""

    def __init__(
        self,
        fc: FlightController,
        telemetry: TelemetryMonitor,
        config: PreflightConfig,
    ) -> None:
        self.fc = fc
        self.telemetry = telemetry
        self.config = config

    def run(self) -> list[str]:
        """Return a list of error strings; empty list means all checks passed."""
        return self.run_detailed().errors

    def run_detailed(self) -> PreflightReport:
        report = PreflightReport()

        # Give the requested streams a chance to deliver a first value, otherwise
        # every check below reads its "never received" default of zero.
        if not self.telemetry.wait_for_telemetry(self.config.telemetry_timeout_s):
            report.warnings.append(
                f"Telemetry incomplete after {self.config.telemetry_timeout_s:.0f}s "
                "— some checks are being skipped"
            )

        state = self.telemetry.snapshot()

        if not self.telemetry.link_healthy():
            report.errors.append("No recent heartbeat from flight controller")

        # Modes
        if not self.fc.link.mode_mapping():
            report.errors.append("Could not read mode mapping from flight controller")
        else:
            for mode in self.config.required_modes:
                try:
                    self.fc.resolve_mode(mode)
                except RuntimeError as exc:
                    report.errors.append(str(exc))

        if not self.fc.supports_velocity_setpoints:
            report.errors.append(
                f"Vehicle class '{self.fc.vehicle_class}' does not follow velocity "
                "setpoints in GUIDED — visual tracking cannot steer it"
            )

        # GPS
        if self.config.min_gps_satellites > 0:
            if not state.have_gps:
                report.errors.append(
                    "No GPS_RAW_INT received — cannot verify satellite count"
                )
            else:
                if state.gps_satellites < self.config.min_gps_satellites:
                    report.errors.append(
                        f"Insufficient GPS satellites: {state.gps_satellites} "
                        f"(need {self.config.min_gps_satellites})"
                    )
                if self.config.require_gps_fix and state.gps_fix_type < _MIN_USABLE_FIX:
                    report.errors.append(
                        f"No 3D GPS fix (fix_type={state.gps_fix_type})"
                    )

        # Battery
        if self.config.min_battery_voltage > 0:
            if not state.have_sys_status:
                report.warnings.append("No SYS_STATUS received — battery not checked")
            elif state.battery_voltage <= 0:
                report.warnings.append(
                    "Flight controller reports 0 V — battery monitor may be disabled"
                )
            elif state.battery_voltage < self.config.min_battery_voltage:
                report.errors.append(
                    f"Battery voltage low: {state.battery_voltage:.1f} V "
                    f"(minimum {self.config.min_battery_voltage:.1f} V)"
                )

        if state.armed:
            report.warnings.append("Vehicle is already armed before preflight")

        for warn in report.warnings:
            logger.warning("Preflight: %s", warn)
        for err in report.errors:
            logger.error("Preflight failed: %s", err)
        if report.ok:
            logger.info(
                "Preflight OK — vehicle=%s mode=%s sats=%d fix=%d battery=%.1f V",
                self.fc.vehicle_class,
                state.mode,
                state.gps_satellites,
                state.gps_fix_type,
                state.battery_voltage,
            )
        return report
