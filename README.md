# Obj Drone — ArduPilot Companion Computer

Raspberry Pi 5 companion software for a **CoreWing F405 Wing V2** running **ArduPilot**. The Pi performs computer vision and high-level guidance; the flight controller handles all stabilization, sensor fusion, failsafes, and motor mixing.

## Architecture

```
┌─────────────────────┐      UART MAVLink       ┌──────────────────────────┐
│  Raspberry Pi 5     │ ◄──────────────────────►│  CoreWing F405 Wing V2   │
│  (companion only)   │   GUIDED / velocity /   │  ArduPilot firmware      │
│                     │   position commands   │  ESCs, motors, failsafes │
│  • Camera (IMX219)  │                       │  • Stabilization         │
│  • Target tracking  │                       │  • Sensor fusion         │
│  • Mission logic    │                       │  • Motor mixing          │
└─────────────────────┘                       └──────────────────────────┘
```

The Pi **never** drives ESCs or motors directly. All low-level flight control stays on the F405.

## Hardware

| Component | Model |
|-----------|-------|
| Flight controller | CoreWing F405 Wing V2 |
| Firmware | ArduPilot |
| Companion computer | Raspberry Pi 5 (8 GB) |
| Camera | Raspberry Pi Camera Module V2 (IMX219) |
| Link | UART MAVLink (`/dev/serial0`) |
| Power | Separate 5 V supply or BEC |

## Wiring (UART)

Connect the Pi UART to a **TELEM** port on the F405:

| Pi GPIO | Pi pin | F405 TELEM |
|---------|--------|------------|
| TX (GPIO14) | 8 | RX |
| RX (GPIO15) | 10 | TX |
| GND | 6 | GND |

**Do not** connect Pi 5 V to the flight controller unless your wiring diagram explicitly supports it.

### Raspberry Pi serial setup

Enable the primary UART in `/boot/firmware/config.txt`:

```
enable_uart=1
dtoverlay=disable-bt
```

Add your user to the `dialout` group and reboot:

```bash
sudo usermod -aG dialout $USER
```

### ArduPilot parameters

On the F405, configure the TELEM port used for the Pi:

| Parameter | Value |
|-----------|-------|
| `SERIALx_PROTOCOL` | 2 (MAVLink2) |
| `SERIALx_BAUD` | 57 (57600) or match `config/default.yaml` |

Ensure **GUIDED**, **LOITER**, **LAND**, and **RTL** modes are available for your frame type.

## Installation

On the Raspberry Pi:

```bash
cd "obj drone"
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[pi]"
```

For development on a Mac/PC (USB camera fallback, no picamera2):

```bash
pip install -e .
```

## Configuration

Edit [`config/default.yaml`](config/default.yaml):

- `mavlink.connection` — UART device (default `/dev/serial0`)
- `mavlink.baud` — must match ArduPilot `SERIALx_BAUD`
- `flight.takeoff_altitude_m` — GUIDED takeoff height
- `mission.track_gain` — how aggressively the drone centers a visual target

## Usage

Test MAVLink connectivity:

```bash
python scripts/test_connection.py
```

Run the full mission (arm → takeoff → track → land on SIGINT):

```bash
obj-drone
```

If already airborne in GUIDED:

```bash
obj-drone --skip-takeoff
```

## MAVLink commands used

All commands are standard ArduPilot MAVLink messages:

| Operation | Message / mode |
|-----------|----------------|
| Mode change | `SET_MODE` (GUIDED, LAND, RTL, LOITER) |
| Arm / disarm | `MAV_CMD_COMPONENT_ARM_DISARM` |
| Takeoff | `MAV_CMD_NAV_TAKEOFF` in GUIDED |
| Velocity | `SET_POSITION_TARGET_LOCAL_NED` (velocity-only type mask) |
| Position | `SET_POSITION_TARGET_LOCAL_NED` (position type mask) |
| Hover | Zero-velocity `SET_POSITION_TARGET_LOCAL_NED` |

Velocity setpoints use **BODY_NED** during visual tracking (forward/right/down relative to the airframe) and **LOCAL_NED** for absolute commands.

## Project layout

```
config/default.yaml          # Runtime settings
scripts/test_connection.py   # MAVLink link check
src/obj_drone/
  main.py                    # Entry point
  mavlink/
    connection.py            # UART link + heartbeat
    commands.py              # ArduPilot high-level commands
  vision/
    camera.py                # Pi Camera V2 / USB fallback
    tracker.py               # Color blob + CSRT ROI tracking
  controller/
    mission.py               # Takeoff, track, land orchestration
```

## Safety

- Always test on the bench with props removed first.
- Verify failsafe behavior (RC loss, GCS loss, battery) on the F405 independently of this software.
- Use `--test-connection` before arming.
- The companion computer is not a substitute for ArduPilot failsafes.

## License

MIT
