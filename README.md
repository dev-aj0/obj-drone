# Obj Drone — ArduPilot Companion Computer

Raspberry Pi 5 companion software for a **CoreWing F405 Wing V2** running **ArduPilot**. The Pi runs an object-detection model and high-level guidance; the flight controller handles all stabilization, sensor fusion, failsafes, and motor mixing.

## Architecture

```
┌─────────────────────┐      UART MAVLink       ┌──────────────────────────┐
│  Raspberry Pi 5     │ ◄──────────────────────►│  CoreWing F405 Wing V2   │
│  (companion only)   │   GUIDED / velocity /   │  ArduPilot firmware      │
│                     │   position commands     │  ESCs, motors, failsafes │
│  • IMX219 camera    │                         │  • Stabilization         │
│  • Object detection │                         │  • Sensor fusion         │
│  • Mission logic    │                         │  • Motor mixing          │
└─────────────────────┘                         └──────────────────────────┘
```

The Pi **never** drives ESCs or motors directly. All low-level flight control stays on the F405.

## Hardware

| Component | Model |
|-----------|-------|
| Flight controller | CoreWing F405 Wing V2 |
| Firmware | ArduPilot (**ArduCopter or QuadPlane** — see below) |
| Companion computer | Raspberry Pi 5 (8 GB) |
| Camera | Any libcamera-supported CSI camera (IMX219, IMX519, …) |
| Link | UART MAVLink (`/dev/ttyAMA0` on Pi 5) |
| Power | Separate 5 V supply or BEC |

Nothing in the code is sensor-specific — it takes whatever libcamera enumerates. Check yours with `rpicam-hello --list-cameras`; if it isn't auto-detected you may need a `dtoverlay=` line in `config.txt` (e.g. `dtoverlay=imx519,cam0` for the 16 MP Arducam).

### Firmware requirement

This software steers by sending **velocity setpoints** (`SET_POSITION_TARGET_LOCAL_NED`) in GUIDED mode. That works on **ArduCopter** and on a **QuadPlane in VTOL flight**.

Plain fixed-wing **ArduPlane ignores velocity setpoints in GUIDED** and has no `LAND` mode, so visual tracking cannot steer it. The software detects the airframe from the heartbeat and refuses to run rather than silently doing nothing. If you are on fixed wing, the tracking loop needs rewriting around GUIDED *position* targets.

## Wiring (UART)

Connect the Pi UART to a **TELEM** port on the F405:

| Pi GPIO | Pi pin | F405 TELEM |
|---------|--------|------------|
| TX (GPIO14) | 8 | RX |
| RX (GPIO15) | 10 | TX |
| GND | 39 | GND |

**Do not** connect Pi 5 V to the flight controller unless your wiring diagram explicitly supports it.

### Raspberry Pi 5 serial setup

The Pi 5 differs from earlier models and most tutorials are wrong for it:

* `/dev/ttyAMA0` is the **dedicated debug UART connector** by default, not GPIO14/15.
* `dtoverlay=disable-bt` is Pi 0–4 guidance and does **not** apply.
* To get a real UART on GPIO14/15 you need the Pi 5-only overlay:

```
enable_uart=1
dtoverlay=uart0-pi5
```

in `/boot/firmware/config.txt`, after which GPIO14/15 is `/dev/ttyAMA0` — which is what `config/default.yaml` uses.

You must also make sure no login console is holding the port (`scripts/install_pi.sh` does this), and add your user to `dialout`:

```bash
sudo usermod -aG dialout,video $USER
```

Reboot after changing boot config.

### ArduPilot parameters

On the F405, for the TELEM port wired to the Pi:

| Parameter | Value |
|-----------|-------|
| `SERIALx_PROTOCOL` | 2 (MAVLink2) |
| `SERIALx_BAUD` | 57 (57600) or match `config/default.yaml` |

See [`docs/ardupilot_params.txt`](docs/ardupilot_params.txt). Ensure **GUIDED**, **LOITER**, **LAND**, and **RTL** are available for your frame type.

## Installation

On the Raspberry Pi:

```bash
bash scripts/install_pi.sh
```

This installs `python3-picamera2`, `python3-libcamera` and `python3-numpy` from apt, installs OpenCV from pip, configures the UART, and creates the venv with `--system-site-packages`.

Three constraints interact here, and getting any one wrong breaks the stack:

> 1. **`--system-site-packages` is not optional.** `picamera2` and `libcamera` have no usable PyPI wheels — `pip install picamera2` in an isolated venv fails on the missing `libcamera` module.
> 2. **OpenCV must come from pip, not apt.** Debian Bookworm ships OpenCV 4.6.0, whose DNN importer cannot parse a YOLOv8 ONNX graph — it dies with `Assertion failed ... in function 'total'`. You need ≥ 4.7.
> 3. **numpy must stay at the apt 1.x version.** Installing OpenCV normally drags in numpy 2.x, which breaks the apt-built `simplejpeg` that picamera2 imports (`numpy.dtype size changed`). Hence the `--no-deps`:
>
> ```bash
> pip install --no-deps "opencv-python-headless>=4.10,<5"
> ```
>
> OpenCV 4.14 runs fine against numpy 1.24. The install script does all of this for you.

For development on a Mac/PC (USB camera fallback, no picamera2):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Object detection model

```bash
bash scripts/fetch_model.sh
```

This exports `models/yolov8n.onnx` (12 MB) at 320×320 using a throwaway export venv. Exporting needs PyTorch, so it is usually easiest to run on your Mac/PC and copy the `.onnx` across:

```bash
scp models/yolov8n.onnx user@yourpi.local:~/obj-drone/models/
```

Detection runs through OpenCV's DNN module on the Pi 5 CPU — no accelerator required. **Measured on a Pi 5: 14.7 FPS, 35 ms per inference** at 320×320 with `yolov8n`. If you have a Hailo AI HAT, that would need a separate backend.

If ultralytics auto-installs `onnxruntime`/`onnxslim` on first run it will ask you to rerun the command — that is expected; just run it again.

## Bring-up order

Do these in order, **props removed**, before any flight:

```bash
obj-drone test                        # 1. MAVLink link, vehicle class, mode list
obj-drone detect --seconds 20 --save  # 2. camera + model, writes annotated frames
obj-drone hover                       # 3. GUIDED + zero-velocity, still disarmed
obj-drone run                         # 4. full mission
```

Step 2 writes annotated JPEGs to `logs/frames/` — check the boxes land on the right objects and that colours look correct (a red object must read as red).

### Live view

The fastest way to see what the drone sees:

```bash
obj-drone serve
```

Then open `http://<your-pi>.local:8080/` from any machine on the network. It shows the annotated camera feed with a **TARGET / NO TARGET** badge, plus confidence, pixel and normalised error, inference time, FPS, and hit rate.

- **Green box** — the locked target the controller is steering towards
- **Grey boxes** — other detections it passed over
- **Line from centre** — the tracking error that becomes the velocity command

Streaming is MJPEG over the standard library, so there is nothing extra to install and it works in any browser. Endpoints: `/` (page), `/stream.mjpg`, `/snapshot.jpg`, `/status.json`.

```bash
obj-drone serve --port 8080 --host 127.0.0.1   # Pi-local only
obj-drone serve --source color                 # colour tracker instead of the model
obj-drone serve --quality 60                   # lower bandwidth
```

> The viewer is **unauthenticated**. The default `--host 0.0.0.0` means anyone on your network can watch the camera. Use `--host 127.0.0.1` (plus SSH port-forwarding) if that matters. It is read-only — it holds no MAVLink connection and cannot command the aircraft.

Only one process can own the CSI camera, so stop `serve` before running `detect` or `run`.

### Focus

Autofocus modules (IMX519, Camera Module 3, …) power up in **manual focus with the lens parked at 1.0 dioptre — about 1 metre.** Everything at another distance looks blurry until focus is configured. Check whether yours has a movable lens:

```bash
rpicam-hello --list-cameras          # identify the sensor
```

If `AfMode` appears in its controls, `vision.focus_mode` applies:

| Mode | Behaviour | Use for |
|------|-----------|---------|
| `continuous` | Refocuses constantly | Bench testing — default |
| `auto` | Focuses once at startup, then holds | **Flight** — no mid-flight hunting |
| `manual` | Lens fixed at `vision.lens_position` | Flight at a known distance |

`lens_position` is in dioptres — `0.0` is infinity, `1.0` is 1 m, `5.0` is 20 cm. For aerial work, targets are effectively at infinity, so `focus_mode: manual` with `lens_position: 0.0` is usually the right answer: it is sharp where it matters and the lens never hunts while tracking.

Find the right setting interactively against the live view:

```bash
obj-drone serve --focus continuous    # let it hunt, see what's sharp
obj-drone serve --lens 0.0            # infinity
obj-drone serve --lens 2.0            # 50 cm
```

Fixed-focus modules ignore all of this and log `focus=fixed`.

#### If the lens never moves ("no AF algorithm")

Raspberry Pi OS ships tuning files containing an `rpi.af` block for only a handful of sensors (imx708, ov64a40). Third-party autofocus modules — notably the **Arducam IMX519** — advertise `AfMode` and `LensPosition`, but libcamera logs:

```
WARN IPARPI: Could not set AF_MODE - no AF algorithm
```

and the lens is completely unreachable — neither autofocus *nor* manual focus works. Arducam has never upstreamed their tuning. Fix it with:

```bash
bash scripts/enable_autofocus.sh
```

That copies the installed tuning file, injects a contrast-detection AF block, and writes `config/tuning/<sensor>_af.json`. Point `vision.tuning_file` at it. Nothing system-wide is touched, so undoing it is just deleting the file.

Note the `LIBCAMERA_RPI_TUNING_FILE` environment variable does **not** work with picamera2 — picamera2 passes its own tuning through and overrides it. The file has to go through `vision.tuning_file`.

#### Finding the sharpest position

```bash
obj-drone focus-sweep --apply
```

Steps the lens across its range and reports the variance-of-Laplacian at each position. **Aim it at something detailed first** — text, a keyboard, a patterned surface. On a blank wall every position scores the same and the command will tell you so rather than returning a meaningless answer. Contrast autofocus fails on blank scenes for exactly the same reason.

The live view shows `Lens`, `AF state` and `Sharpness` so you can watch focus hunt and settle in real time.

### Verifying control signs

`mission.invert_lateral` / `invert_longitudinal` exist because camera mounting rotation is not knowable from here. On the bench, with **props removed**, run `obj-drone run` and watch the commanded velocities in the log: moving the target right of frame centre must produce a positive `vy`. If the drone would move *away* from the target, flip the corresponding invert flag.

## Usage

```bash
obj-drone test                          # link check
obj-drone serve                         # live view in a browser (see below)
obj-drone detect --seconds 20 --save    # camera + detector, no MAVLink
obj-drone calibrate-color --save        # tune the HSV fallback (headless-safe)
obj-drone run                           # preflight → arm → takeoff → track
obj-drone run --skip-takeoff            # already airborne in GUIDED
obj-drone run --source color            # force the colour tracker
obj-drone hover --arm
obj-drone land
obj-drone rtl
```

`--config config/sitl.yaml` points any command at ArduPilot SITL instead of hardware.

## Configuration

Edit [`config/default.yaml`](config/default.yaml):

- `mavlink.connection` — UART device (`/dev/ttyAMA0` on Pi 5)
- `mavlink.baud` — must match ArduPilot `SERIALx_BAUD`
- `vision.camera_backend` — `picamera2` on the aircraft so a broken CSI stack fails loudly
- `vision.camera_orientation` — `down` (nadir) or `forward`
- `vision.detector.*` — model path, input size, confidence, and `target_classes`
- `mission.track_gain` — m/s commanded when the target sits at the frame edge
- `flight.takeoff_altitude_m` — GUIDED takeoff height

## Testing

The suite runs on any machine — no flight controller, camera, or model needed. It includes an integration test that speaks real MAVLink to a fake ArduPilot over UDP.

```bash
pip install -e ".[dev]"
pytest
```

For a full flight rehearsal, use ArduPilot SITL:

```bash
bash scripts/sitl_test.sh
obj-drone --config config/sitl.yaml run --skip-preflight
```

## MAVLink commands used

All commands are standard ArduPilot MAVLink messages:

| Operation | Message / mode |
|-----------|----------------|
| Mode change | `SET_MODE` (GUIDED, LAND/QLAND, RTL/QRTL, LOITER/QLOITER) |
| Arm / disarm | `MAV_CMD_COMPONENT_ARM_DISARM` |
| Takeoff | `MAV_CMD_NAV_TAKEOFF` (`MAV_CMD_NAV_VTOL_TAKEOFF` on QuadPlane) |
| Velocity | `SET_POSITION_TARGET_LOCAL_NED` (velocity-only type mask) |
| Position | `SET_POSITION_TARGET_LOCAL_NED` (position type mask) |
| Hover | Zero-velocity `SET_POSITION_TARGET_LOCAL_NED` |
| Stream rates | `MAV_CMD_SET_MESSAGE_INTERVAL` |

Velocity setpoints use **BODY_NED** during visual tracking (forward/right/down relative to the airframe) and **LOCAL_NED** for absolute commands. Setpoints above `mavlink.command_rate_hz` are dropped rather than delayed, so the vision loop never stalls on the link.

## Project layout

```
config/default.yaml          # Runtime settings
models/coco.names            # Class labels (model weights are gitignored)
scripts/install_pi.sh        # Pi 5 setup: apt, UART, venv
scripts/fetch_model.sh       # Export yolov8n.onnx
scripts/sitl_test.sh         # ArduPilot SITL launcher
src/obj_drone/
  main.py                    # CLI entry point
  config.py / paths.py       # Settings loading
  mavlink/
    connection.py            # UART link, heartbeat, vehicle classification
    telemetry.py             # Sole MAVLink reader + state cache + wait helpers
    commands.py              # ArduPilot high-level commands, per-airframe modes
  vision/
    camera.py                # IMX219 via picamera2 / USB fallback
    detector.py              # ONNX object detection via cv2.dnn
    tracker.py               # Target selection, locking, colour fallback
    debug.py                 # Annotated frame writer
  controller/
    mission.py               # Takeoff, track, land orchestration
    preflight.py             # Pre-arm checks
tests/                       # Runs anywhere; no hardware required
```

## Safety

- Always test on the bench with **props removed** first, and work through the bring-up order above.
- Verify the control-sign flags before flight — a wrong sign makes the drone accelerate *away* from the target.
- Verify failsafe behavior (RC loss, GCS loss, battery) on the F405 independently of this software.
- The companion computer is not a substitute for ArduPilot failsafes. Keep `ARMING_CHECK` enabled and keep a pilot on the sticks with a mode switch to override GUIDED.
- The systemd unit arms and takes off **on boot** if enabled. Leave it disabled unless you are certain.

## License

MIT
