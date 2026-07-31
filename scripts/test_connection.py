#!/usr/bin/env python3
"""Verify MAVLink connectivity to the flight controller."""

from __future__ import annotations

import sys

from obj_drone.main import cmd_test, load_config, logging_config_from_dict, setup_logging


def main() -> int:
    cfg = load_config()
    setup_logging(logging_config_from_dict(cfg))
    return cmd_test(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
