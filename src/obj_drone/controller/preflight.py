"""Pre-arm and pre-takeoff checks."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from obj_drone.mavlink.connection import MavlinkConnection
from obj_drone.mavlink.telemetry import TelemetryMonitor

logger = logging.getLogger(__name__)


@dataclass
class PreflightConfig:
    min_gps_satellites: int = 6
    min_battery_voltage: float = 10.5
    required_modes: tuple[str, ...] = ("GUIDED", "LAND", "RTL", "LOITER")


class PreflightCheck:
    """Validate FC state before arming or takeoff."""

    def __init__(
        self,
        link: MavlinkConnection,
        telemetry: TelemetryMonitor,
        config: PreflightConfig,
    ) -> None:
        self.link = link
        self.telemetry = telemetry
        self.config = config

    def run(self) -> list[str]:
        """Return a list of error strings; empty list means all checks passed."""
        errors: list[str] = []
        state = self.telemetry.snapshot()

        if not self.telemetry.link_healthy():
            errors.append("No recent heartbeat from flight controller")

        mapping = self.link.master.mode_mapping()
        if mapping is None:
            errors.append("Could not read mode mapping from flight controller")
        else:
            for mode in self.config.required_modes:
                if mode not in mapping:
                    errors.append(f"Required mode not available: {mode}")

        if self.config.min_gps_satellites > 0:
            if state.gps_satellites < self.config.min_gps_satellites:
                errors.append(
                    f"Insufficient GPS satellites: {state.gps_satellites} "
                    f"(need {self.config.min_gps_satellites})"
                )

        if self.config.min_battery_voltage > 0:
            if state.battery_voltage > 0 and state.battery_voltage < self.config.min_battery_voltage:
                errors.append(
                    f"Battery voltage low: {state.battery_voltage:.1f} V "
                    f"(minimum {self.config.min_battery_voltage:.1f} V)"
                )

        if state.armed:
            logger.warning("Vehicle already armed before preflight")

        for err in errors:
            logger.error("Preflight failed: %s", err)
        if not errors:
            logger.info(
                "Preflight OK — mode=%s sats=%d battery=%.1f V",
                state.mode,
                state.gps_satellites,
                state.battery_voltage,
            )
        return errors
